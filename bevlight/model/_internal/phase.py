'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Movement demands -> one demand vector per candidate phase.

A phase is a *set of movements*, not an index. `P_1 = {m_1, m_4}` means "this
phase serves those two turns", and that is the only thing about it the model may
use. Two consequences follow, and they are the whole point:

  * The aggregation must be **permutation invariant** — the set is unordered, and
    a phase written `{m_4, m_1}` is the same phase.
  * It must be **size invariant** — a phase can serve one movement or three, and
    the number of phases K varies by junction and even between the two plans of
    one junction (Hongkong_YMT: 4 under `easy`, 3 under `normal`).

Attention pooling gives both: one shared query attends over the phase's members
and returns a fixed-width vector whatever the set contains. The same pooler runs
once per candidate phase, which is how a variable K is handled without a
variable-size output head.

Nothing about the current phase enters here. This layer answers only "how much
does this phase's traffic want green", and mixing in "and it happens to be the
one already running" would make the answer un-comparable across candidates.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Pool a variable-size set into one vector with a learned query."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, members: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """`(B, S, D)` members and `(B, S)` validity -> `(B, D)`."""
        batch = members.shape[0]
        pad = ~valid.bool()
        empty = pad.all(dim=-1, keepdim=True)
        key_padding = pad & ~empty

        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attention(
            query, self.norm(members), self.norm(members),
            key_padding_mask=key_padding, need_weights=False,
        )
        pooled = pooled.squeeze(1).nan_to_num(0.0)
        # An empty phase contributes nothing rather than a learned bias.
        return self.out(pooled) * (~empty.squeeze(-1)).float().unsqueeze(-1)


class DeepSetsPooling(nn.Module):
    """Sum-then-transform. The ablation alternative to attention pooling."""

    def __init__(self, dim: int):
        super().__init__()
        self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.rho = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, members: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weighted = self.phi(members) * valid.unsqueeze(-1)
        totals = valid.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.rho(weighted.sum(dim=1) / totals)


POOLERS = {"attention": AttentionPooling, "deepsets": DeepSetsPooling}


class PhaseLayer(nn.Module):
    """Aggregate each candidate phase's movements into one demand vector."""

    def __init__(self, dim: int, heads: int = 4, pooling: str = "attention"):
        super().__init__()
        if pooling not in POOLERS:
            raise ValueError(f"Unknown pooling '{pooling}'. Available: {list(POOLERS)}")
        cls = POOLERS[pooling]
        self.pool = cls(dim, heads) if pooling == "attention" else cls(dim)
        self.pooling = pooling

    def forward(
        self,
        movements: torch.Tensor,        # (B, R, D)
        phase_members: torch.Tensor,    # (B, K, S) movement indices
        member_valid: torch.Tensor,     # (B, K, S)
        phase_valid: torch.Tensor,      # (B, K)
    ) -> torch.Tensor:
        """-> `(B, K, D)`, one vector per candidate phase."""
        batch, phases, members = phase_members.shape
        dim = movements.shape[-1]

        index = phase_members.clamp(min=0)
        gathered = torch.gather(
            movements.unsqueeze(1).expand(-1, phases, -1, -1),
            2,
            index.unsqueeze(-1).expand(-1, -1, -1, dim),
        )                                                   # (B, K, S, D)

        # One pooling call for every (batch, phase) pair: the pooler is shared,
        # which is exactly what makes K free to vary.
        flat = gathered.reshape(batch * phases, members, dim)
        flat_valid = member_valid.reshape(batch * phases, members)
        pooled = self.pool(flat, flat_valid).reshape(batch, phases, dim)
        return pooled * phase_valid.unsqueeze(-1)
