'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Where a pinned backbone lives, and which backbones exist.

Pure path arithmetic, no network and no torch. That is the point of the split:
`backbone.py` has to ask "are the weights already here" on every construction,
and it should not have to import the downloader -- a module that reaches for
timm and Hugging Face -- to get an answer.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from pathlib import Path

from ...paths import BACKBONE_ROOT

# The DINOv2 variants worth having. ViT-S is the pilot default; ViT-B is the
# upgrade path if per-lane features turn out to be the bottleneck.
KNOWN_BACKBONES = {
    "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-S/14 with registers (22M, dim 384)",
    "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-B/14 with registers (86M, dim 768)",
    "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-L/14 with registers (300M, dim 1024)",
}

WEIGHTS_NAME = "weights.pt"
META_NAME = "backbone.json"


def local_dir(model_name: str, root: Path | None = None) -> Path:
    return (root or BACKBONE_ROOT) / model_name


def local_weights(model_name: str, root: Path | None = None) -> Path | None:
    """Path to already-downloaded weights, or None."""
    path = local_dir(model_name, root) / WEIGHTS_NAME
    return path if path.is_file() else None
