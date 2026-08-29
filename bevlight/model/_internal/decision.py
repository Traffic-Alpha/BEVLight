'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Phase demands -> the phase to run next.

Every candidate phase is scored by the *same* `f_D`, and the action is the
argmax. There is no output layer of fixed width anywhere, which is what lets a
junction with three candidate phases and one with four use the same weights.

The current phase enters as its own demand vector `e_{p_t}`, not as an embedding
of its index. Phase 0 at one junction and phase 0 at another serve completely
different turns, so an index embedding would be learning a symbol with no
meaning across junctions. Representing "where we are" by "what is currently
being served" keeps the whole model free of junction-specific vocabulary.

`f_D` is handed `e_{P_k} - e_{p_t}` explicitly for the same reason the movement
layer is handed `in - out`: the question "is it worth switching" is a comparison,
so the comparison is provided rather than left to be discovered.

Minimum green is a hard constraint, not a learned preference. It is applied by
masking the scores, so it holds exactly, matches the constraint the expert
operated under when the data was recorded, and cannot be traded away for a
marginal gain in the loss.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import math

import torch
import torch.nn as nn

NEG_INF = -1e9


class DurationEncoding(nn.Module):
    """Sinusoidal encoding of how long the current phase has been running.

    A raw scalar in seconds would need the network to learn its scale from few
    samples; a Fourier feature makes "just switched" and "long overdue" linearly
    separable from the start.
    """

    def __init__(self, dim: int, max_period: float = 300.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=tau.device) / half
        )
        angles = tau.unsqueeze(-1) * freqs
        encoded = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if encoded.shape[-1] < self.dim:                     # odd dim
            encoded = torch.cat([encoded, encoded[..., :1]], dim=-1)
        return encoded


class DecisionLayer(nn.Module):
    """Score every candidate phase and pick one."""

    def __init__(self, dim: int, hidden: int = 256, duration_dim: int = 32,
                 min_green_s: float = 10.0, use_phase_context: bool = True):
        super().__init__()
        self.duration = DurationEncoding(duration_dim)
        self.min_green_s = float(min_green_s)
        # Without the context a candidate is scored on its own demand alone: no
        # e_{p_t}, no difference against it, no time-in-phase. The ablation that
        # asks whether the policy is choosing a phase or only ranking demand.
        self.use_phase_context = use_phase_context
        width = dim * 3 + duration_dim if use_phase_context else dim
        self.score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        phase_features: torch.Tensor,   # (B, K, D)
        phase_valid: torch.Tensor,      # (B, K)
        current_phase: torch.Tensor,    # (B,) index into K
        time_in_phase: torch.Tensor,    # (B,) seconds
    ) -> torch.Tensor:
        """-> `(B, K)` logits, with padded phases at -inf."""
        batch, phases, dim = phase_features.shape

        if not self.use_phase_context:
            logits = self.score(phase_features).squeeze(-1)
            return logits.masked_fill(~phase_valid.bool(), NEG_INF)

        current = torch.gather(
            phase_features, 1,
            current_phase.view(batch, 1, 1).expand(-1, 1, dim),
        )                                                    # (B, 1, D)
        current = current.expand(-1, phases, -1)

        tau = self.duration(time_in_phase).unsqueeze(1).expand(-1, phases, -1)
        features = torch.cat(
            [phase_features, current, phase_features - current, tau], dim=-1
        )
        logits = self.score(features).squeeze(-1)            # (B, K)

        # Padded candidates must never be selectable, and must not take softmax
        # mass away from the real ones.
        return logits.masked_fill(~phase_valid.bool(), NEG_INF)

    def act(
        self,
        logits: torch.Tensor,
        current_phase: torch.Tensor,
        time_in_phase: torch.Tensor,
    ) -> torch.Tensor:
        """Apply minimum green, then take the argmax.

        Enforced at action time rather than inside the loss: the training target
        is what the expert did under the same rule, so the scores stay a clean
        statement of demand.
        """
        held = time_in_phase < self.min_green_s
        chosen = logits.argmax(dim=-1)
        return torch.where(held, current_phase, chosen)
