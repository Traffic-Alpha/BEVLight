'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Frozen DINOv2 patch features for a BEV frame.

The backbone stays frozen: there are ~1.7k decision samples in the full dataset,
which is nowhere near enough to fine-tune a ViT without memorising the synthetic
renderer. Keeping it frozen also buys something valuable downstream — since the
weights never change and MaskPool has no parameters either, the per-lane vectors
`v_i` are a fixed function of the image and can be computed once and cached. The
trainable part then never sees a pixel, so render resolution costs a one-off
extraction pass rather than every epoch.

DINOv2 is patch-14, so an input of 1274 px yields a 91x91 feature grid and each
cell covers 14x14 px. BEVLight renders at a fixed 11.36 px/m, which puts a 3.2 m
lane at 2.6 cells across.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn

DEFAULT_MODEL = "vit_small_patch14_reg4_dinov2.lvd142m"
PATCH_SIZE = 14

# DINOv2 normalisation (ImageNet statistics).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@contextlib.contextmanager
def _usable_proxy_env():
    """Hide an `ALL_PROXY` that httpx cannot parse, for the weight download.

    `socks://host:port` is not a scheme httpx accepts (it wants `socks5://`), and
    huggingface_hub builds its client from the environment, so a shell exporting
    the short form makes every `create_model(pretrained=True)` fail with an
    unrelated-looking ValueError. HTTP_PROXY still applies, so dropping just this
    variable keeps the download working through the same proxy.
    """
    removed = {}
    for name in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value and value.startswith("socks://"):
            removed[name] = os.environ.pop(name)
    try:
        yield
    finally:
        os.environ.update(removed)


class DinoBackbone(nn.Module):
    """Wraps a timm DINOv2 ViT and returns a dense patch feature map."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        weights_root=None,
    ):
        super().__init__()
        import timm

        from ._internal.weights import local_weights

        self.model_name = model_name
        # DINOv2 was pretrained at 518px and timm pins that size unless told
        # otherwise. BEVLight renders each junction at its own size
        # (1274-1834px), so position embeddings are interpolated to the actual
        # grid instead.
        options = dict(num_classes=0, dynamic_img_size=True)

        weights_path = local_weights(model_name, weights_root)
        if weights_path is not None:
            # Offline path: weights already in the project, nothing to fetch.
            self.model = timm.create_model(model_name, pretrained=False, **options)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"{weights_path} does not match {model_name}: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected tensors."
                )
            self.weights_source = str(weights_path)
        else:
            from ..utils.paths import HF_CACHE_ROOT

            HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            with _usable_proxy_env():
                self.model = timm.create_model(
                    model_name, pretrained=True, cache_dir=str(HF_CACHE_ROOT), **options
                )
            self.weights_source = "downloaded (run tools/download_backbone.py to pin it)"
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.embed_dim = int(self.model.embed_dim)
        self.patch_size = int(self.model.patch_embed.patch_size[0])
        # ViT-S/14 with registers prepends 1 CLS + 4 register tokens; they are not
        # spatial and must not be reshaped into the grid.
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` float images in [0, 1] -> normalised."""
        return (images - self.mean.to(images)) / self.std.to(images)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` in [0, 1] -> `(B, D, H/14, W/14)` patch features."""
        if images.shape[-1] % self.patch_size or images.shape[-2] % self.patch_size:
            raise ValueError(
                f"Input {tuple(images.shape[-2:])} is not a whole number of "
                f"{self.patch_size}px patches. BEVLight render sizes are chosen "
                f"to be, so this usually means the image was resized."
            )
        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size

        tokens = self.model.forward_features(self.normalize(images))
        tokens = tokens[:, self.num_prefix_tokens:, :]          # drop CLS + registers
        batch, count, dim = tokens.shape
        if count != height * width:
            raise RuntimeError(
                f"Expected {height * width} patch tokens for a "
                f"{images.shape[-2]}x{images.shape[-1]} input, got {count}."
            )
        return tokens.transpose(1, 2).reshape(batch, dim, height, width)

    def grid_size(self, resolution) -> tuple[int, int]:
        """Feature-map size for a render resolution `(width, height)`."""
        width, height = int(resolution[0]), int(resolution[1])
        return height // self.patch_size, width // self.patch_size
