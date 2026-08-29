'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Lane masks -> per-lane feature vectors, with no parameters.

    v_i = sum_p w_ip * F_p / sum_p w_ip

`F` is the patch feature map and `w_i` is lane i's mask brought down to patch
resolution. The weights are *soft*: a patch is weighted by the fraction of its
14x14 pixels that belong to the lane. At BEVLight's fixed 11.36 px/m a 3.2 m lane
spans 2.6 patches, so its middle column is essentially pure while the edge
patches are shared with the neighbouring lane in proportion to their overlap.
Hard assignment would either drop those edge patches or hand them to one lane
outright.

Because the masks are static per (junction, plan, resolution), the weights are
computed once and reused for every frame of every episode.

This layer is what makes the model indifferent to the drone's heading: the mask
carries the lane->pixel correspondence, so a rotated junction needs a rotated
mask, not a retrained model. It is also why the lane count N never has to be
fixed.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def lane_patch_weights(
    labels: np.ndarray,
    mask_ids: list[int],
    patch_size: int = 14,
    hard: bool = False,
) -> torch.Tensor:
    """`(N, gh, gw)` patch coverage of each lane, in [0, 1].

    Args:
        labels: the uint16 lane-id image.
        mask_ids: lane mask ids, in the order the model will use.
        hard: give each patch outright to the lane covering most of it, the
            alternative this module's docstring argues against. The ablation
            that measures what soft weights are worth -- and it acts here, when
            the feature cache is built, not at training time, so a run using it
            needs its own cache.
    """
    height, width = labels.shape[:2]
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Lane mask {width}x{height} is not a whole number of {patch_size}px patches."
        )
    stack = np.stack([(labels == mask_id) for mask_id in mask_ids]).astype(np.float32)
    tensor = torch.from_numpy(stack).unsqueeze(0)          # (1, N, H, W)
    # Average pooling over a patch = fraction of the patch covered by the lane.
    weights = F.avg_pool2d(tensor, kernel_size=patch_size, stride=patch_size)
    weights = weights.squeeze(0)                           # (N, gh, gw)
    if hard:
        # Winner takes the patch; a patch no lane touches stays at zero, so a
        # lane's coverage can still be near zero and be detected as such.
        best = weights.argmax(dim=0, keepdim=True)
        covered = weights.sum(dim=0, keepdim=True) > 0
        weights = torch.zeros_like(weights).scatter_(0, best, 1.0) * covered
    return weights


class MaskPool(nn.Module):
    """Pool a patch feature map into one vector per lane. No parameters."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """`(B, D, gh, gw)` and `(B, N, gh, gw)` -> `(B, N, D)`."""
        batch, dim, grid_h, grid_w = features.shape
        if weights.dim() == 3:
            weights = weights.unsqueeze(0).expand(batch, -1, -1, -1)
        lanes = weights.shape[1]

        flat_features = features.reshape(batch, dim, grid_h * grid_w)      # (B, D, P)
        flat_weights = weights.reshape(batch, lanes, grid_h * grid_w)      # (B, N, P)

        pooled = torch.bmm(flat_weights, flat_features.transpose(1, 2))    # (B, N, D)
        totals = flat_weights.sum(dim=-1, keepdim=True)                    # (B, N, 1)
        return pooled / totals.clamp_min(self.eps)

    @staticmethod
    def coverage(weights: torch.Tensor) -> torch.Tensor:
        """Total patch weight per lane — near zero means the lane is barely visible."""
        return weights.flatten(-2).sum(-1)
