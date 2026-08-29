'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Lane features -> movement demand.

A movement is a turn, `l_i -> l_j`: an incoming lane and where its traffic goes.
Its identity comes entirely from the two lanes it joins, never from an index or a
hand-written direction label, which is what lets an unseen junction with a
different turn topology work without retraining.

Three steps, each with one job:

  1. Lane cross-attention. An incoming lane needs to know whether its *exit* is
     blocked, which is information held by a different lane. Without this step
     nothing in the model can see spillback.
  2. Composition `f_M`. Pairs are formed by the connectivity the scenario
     provides, and combined as `concat(in, out, in - out)`. The difference term
     is explicit on purpose: `in - out` is literally the definition of pressure,
     so the layer is handed the physical quantity rather than asked to discover
     subtraction.
  3. Movement cross-attention. Turns compete for the same green time, so a
     movement's claim depends on what the others are asking for.

The pressure auxiliary head hangs on the output of step 2, *before* step 3:
pressure is a property of a movement on its own, and regressing it after the
competition has been mixed in would be regressing a different quantity.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedCrossAttention(nn.Module):
    """Self-attention over a padded set, with the padding excluded.

    Padding leaks in two places if unguarded: attention would attend *to* pad
    slots, and pad slots would produce outputs that later reductions pick up.
    Both are closed here, so a batch's result cannot depend on how much padding
    its neighbours brought.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """`(B, N, D)` and a `(B, N)` validity mask -> `(B, N, D)`."""
        pad = ~valid.bool()
        # A row that is entirely padding has nothing to attend to; softmax over an
        # all-masked row is NaN, so those rows keep a single open slot and are
        # zeroed afterwards instead.
        empty = pad.all(dim=-1, keepdim=True)
        key_padding = pad & ~empty

        normed = self.norm(x)
        attended, _ = self.attention(
            normed, normed, normed, key_padding_mask=key_padding, need_weights=False
        )
        x = x + attended.nan_to_num(0.0)
        x = x + self.ffn(x)
        return x * valid.unsqueeze(-1)


class MovementComposer(nn.Module):
    """`f_M`: an incoming lane and its exit -> that movement's demand."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, incoming: torch.Tensor, outgoing: torch.Tensor) -> torch.Tensor:
        # The difference term states pressure directly: demand minus room ahead.
        return self.net(torch.cat([incoming, outgoing, incoming - outgoing], dim=-1))


def gather_lanes(lane_features: torch.Tensor, index: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    """Average the lanes belonging to each movement.

    `index` is `(B, R, L)` lane positions and `weight` `(B, R, L)` marks which of
    those slots are real, so a movement served by one lane and one served by
    three are handled by the same code path.
    """
    batch, lanes, dim = lane_features.shape
    flat = index.clamp(min=0)
    gathered = torch.gather(
        lane_features.unsqueeze(1).expand(-1, index.shape[1], -1, -1),
        2,
        flat.unsqueeze(-1).expand(-1, -1, -1, dim),
    )
    weighted = gathered * weight.unsqueeze(-1)
    return weighted.sum(dim=2) / weight.sum(dim=2, keepdim=True).clamp_min(1e-6)


class MovementLayer(nn.Module):
    """Lane vectors -> per-movement demand vectors."""

    def __init__(self, dim: int, heads: int = 4, lane_attention: bool = True,
                 movement_attention: bool = True):
        super().__init__()
        self.lane_attention = MaskedCrossAttention(dim, heads) if lane_attention else None
        self.compose = MovementComposer(dim)
        self.movement_attention = (
            MaskedCrossAttention(dim, heads) if movement_attention else None
        )

    def forward(
        self,
        lane_features: torch.Tensor,   # (B, N, D)
        lane_valid: torch.Tensor,      # (B, N)
        in_index: torch.Tensor,        # (B, R, L)
        in_weight: torch.Tensor,       # (B, R, L)
        out_index: torch.Tensor,       # (B, R, L)
        out_weight: torch.Tensor,      # (B, R, L)
        movement_valid: torch.Tensor,  # (B, R)
    ):
        """Returns `(h_before_competition, h_after_competition)`.

        The first is what the pressure head reads; the second feeds the phases.
        """
        if self.lane_attention is not None:
            lane_features = self.lane_attention(lane_features, lane_valid)

        incoming = gather_lanes(lane_features, in_index, in_weight)
        outgoing = gather_lanes(lane_features, out_index, out_weight)
        movements = self.compose(incoming, outgoing) * movement_valid.unsqueeze(-1)

        competed = movements
        if self.movement_attention is not None:
            competed = self.movement_attention(movements, movement_valid)
        return movements, competed
