'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Fuse a window of frames into one lane vector carrying a trend.

This is the layer that justifies using video rather than a snapshot. A single
frame says how long each queue is, which is exactly what a detector-and-count
pipeline already provides. What a frame cannot say is whether that queue is
building or draining — and that is what separates "this movement needs green
now" from "this movement is already clearing".

So the frame differences are computed explicitly and concatenated, rather than
left for a transformer to infer from positional encodings. `v_t - v_{t-1}` in
feature space is the trend, stated directly.

The transformer is shared across lanes and small: a lane's own history is what
matters here, and interaction between lanes is the movement layer's job. Keeping
each layer to one question is what makes the ablations interpretable.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    """`(B, T, N, D)` frames of lane vectors -> `(B, N, D)` state-plus-trend."""

    def __init__(self, dim: int, heads: int = 4, layers: int = 1, max_frames: int = 8):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        self.position = nn.Parameter(torch.randn(1, max_frames, dim) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 2, batch_first=True, norm_first=True
        )
        # Nested tensors are incompatible with norm_first and would only warn.
        self.encoder = nn.TransformerEncoder(
            encoder, num_layers=layers, enable_nested_tensor=False
        )
        self.out = nn.LayerNorm(dim)

    def forward(self, frames: torch.Tensor, lane_valid: torch.Tensor) -> torch.Tensor:
        batch, time, lanes, dim = frames.shape

        # Trend, stated rather than inferred. The first frame has no predecessor,
        # so its difference is zero: "no information yet", not a wrap-around.
        delta = torch.zeros_like(frames)
        delta[:, 1:] = frames[:, 1:] - frames[:, :-1]
        fused = self.input(torch.cat([frames, delta], dim=-1))

        # Lanes are independent here, so fold them into the batch: one shared
        # temporal model runs over every lane, and N never enters the weights.
        sequence = fused.permute(0, 2, 1, 3).reshape(batch * lanes, time, dim)
        sequence = sequence + self.position[:, :time]
        encoded = self.encoder(sequence)

        # The last position is "now, having seen the window".
        state = self.out(encoded[:, -1]).reshape(batch, lanes, dim)
        return state * lane_valid.unsqueeze(-1)
