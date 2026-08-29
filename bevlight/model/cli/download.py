'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Fetch backbone weights once, then run offline.

Training and evaluation should not depend on a reachable Hugging Face, a warm
cache, or a working proxy. This pulls the weights into `runs/backbones/`
where `DinoBackbone` finds them, and after that nothing touches the network.
The directory is gitignored: weights are large and reproducible from here.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ...utils.paths import BACKBONE_ROOT, HF_CACHE_ROOT
from .._internal.weights import KNOWN_BACKBONES, META_NAME, WEIGHTS_NAME, local_dir, local_weights


def download(
    model_name: str,
    root: Path | None = None,
    force: bool = False,
    keep_cache: bool = False,
) -> Path:
    """Download one backbone's weights into the local checkpoint directory."""
    import timm
    import torch

    from ..backbone import _usable_proxy_env

    target = local_dir(model_name, root)
    weights = target / WEIGHTS_NAME
    if weights.is_file() and not force:
        print(f"[skip] {model_name}: already at {weights}")
        return weights

    print(f"[get ] {model_name} ...")
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with _usable_proxy_env():
        model = timm.create_model(
            model_name, pretrained=True, num_classes=0, cache_dir=str(HF_CACHE_ROOT)
        )

    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    (target / META_NAME).write_text(
        json.dumps(
            {
                "model_name": model_name,
                "embed_dim": int(model.embed_dim),
                "patch_size": int(model.patch_embed.patch_size[0]),
                "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 1)),
                "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "source": "timm pretrained weights",
            },
            indent=2,
        )
    )
    size_mb = weights.stat().st_size / 1e6
    print(f"[done] {model_name}: {size_mb:.0f} MB -> {weights}")

    # The staging cache is now a byte-for-byte duplicate of what we extracted.
    if not keep_cache and HF_CACHE_ROOT.is_dir():
        shutil.rmtree(HF_CACHE_ROOT, ignore_errors=True)
    return weights


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download backbone weights for offline use.")
    parser.add_argument("--model", nargs="+", default=["vit_small_patch14_reg4_dinov2.lvd142m"],
                        help="Model names to fetch. Default: the ViT-S/14 pilot backbone.")
    parser.add_argument("--all", action="store_true", help="Fetch every known backbone.")
    parser.add_argument("--root", default=None, help="Destination. Default: runs/backbones.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--keep-cache", action="store_true", help="Keep the Hugging Face staging cache after extracting.")
    parser.add_argument("--list", action="store_true", help="List known backbones and exit.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root) if args.root else BACKBONE_ROOT

    if args.list:
        print("known backbones:")
        for name, description in KNOWN_BACKBONES.items():
            marker = "cached" if local_weights(name, root) else "-"
            print(f"  [{marker:6s}] {name:44s} {description}")
        return 0

    models = list(KNOWN_BACKBONES) if args.all else args.model
    for model_name in models:
        if model_name not in KNOWN_BACKBONES:
            print(f"[warn] {model_name} is not in the known list; fetching anyway")
        download(model_name, root, force=args.force, keep_cache=args.keep_cache)

    print(f"[summary] backbones in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
