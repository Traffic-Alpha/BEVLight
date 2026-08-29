'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Uniformly random phase choice — the bottom anchor.

Fixed-time is the floor a *controller* has to beat; this is the floor a *reward*
has to recognise. A reward preflight needs controllers whose ordering is already
known and well separated, and two of them cannot produce a rank correlation:
with only fixed-time and max-pressure every candidate reward that gets the pair
the right way round scores a perfect -1, including ones that are right by luck.

Random is also the one controller that violates the cycle structure outright —
it starves approaches at irregular intervals rather than serving them late — so a
reward that tracks control quality has to place it below fixed-time, and a reward
that merely counts switches will not.

Seeded per episode, so a preflight is reproducible.
@LastEditTime: 2026-08-28
'''

from __future__ import annotations

import random

from ..base import BaseController, SignalPlan


class RandomPhase(BaseController):
    """Pick a candidate phase uniformly at random at every decision."""

    name = "random"

    def __init__(self, seed: int = 7) -> None:
        super().__init__()
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def reset(self, plan: SignalPlan) -> None:
        super().reset(plan)
        self._rng = random.Random(self.seed)

    def act(self, obs, plan: SignalPlan) -> int:
        self.last_action = self._rng.choice(plan.phases)
        return self.last_action
