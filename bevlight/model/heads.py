'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Auxiliary heads that tie visual features to physical quantities.

The main task is choosing a phase, but a phase choice is a weak, sparse signal:
one label per ten seconds, and heavily biased towards "keep the current phase".
Left alone, behaviour cloning can satisfy it without ever learning to read the
image. Regressing a physical per-lane quantity forces the features to mean
something.

Every head is **shared across lanes** and applied per lane vector. That is what
makes an unseen junction with a different lane count just work: there is no
per-lane parameter to be missing.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneHead(nn.Module):
    """Per-lane scalar regression, shared across lanes.

    Input `(B, N, D)` -> output `(B, N)`.
    """

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.0, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        width = dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(width, hidden), nn.GELU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, lane_features: torch.Tensor) -> torch.Tensor:
        return self.net(lane_features).squeeze(-1)


class QueueHead(LaneHead):
    """Queue length on an incoming lane, in vehicles.

    Softplus rather than ReLU to keep the output non-negative. A negative queue
    is not a thing, but ReLU is the wrong way to say so here: most lanes are
    empty most of the time, so the gradient pushes every pre-activation negative,
    and once they are all in ReLU's flat region no gradient comes back and the
    head is stuck emitting exactly zero forever. That failure is easy to miss
    because a collapsed head still posts a fine-looking MAE on a sparse target.
    Softplus is positive everywhere and always passes gradient.
    """

    def forward(self, lane_features: torch.Tensor) -> torch.Tensor:
        return F.softplus(super().forward(lane_features))


def masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    kind: str = "l1",
) -> torch.Tensor:
    """Loss over real lanes only.

    `valid` excludes padding lanes and, for queue targets, lanes whose queue runs
    past the image edge — the label there is a lower bound, and training on it
    teaches the model to under-count exactly when the junction is busiest.
    """
    if valid.sum() == 0:
        return prediction.sum() * 0.0
    error = prediction - target
    per_element = error.abs() if kind == "l1" else error.pow(2)
    return (per_element * valid).sum() / valid.sum()
