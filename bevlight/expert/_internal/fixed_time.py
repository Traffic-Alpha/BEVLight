'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Fixed-time control: cycle the phases, ignore the traffic.

The floor every other method has to beat. It is also the cleanest check that a
scenario is actually loaded correctly: if a demand pattern is so light that
fixed-time already empties the junction, that scenario cannot separate any two
controllers and should not be used to judge them.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

from ..base import BaseController, SignalPlan

DEFAULT_GREEN_S = 30.0


class FixedTime(BaseController):
    """Serve every phase in turn for a fixed green duration."""

    name = "fixed_time"

    def __init__(self, green_duration: float = DEFAULT_GREEN_S) -> None:
        super().__init__()
        self.green_duration = float(green_duration)

    def reset(self, plan: SignalPlan) -> None:
        super().reset(plan)
        self._cursor = 0

    def act(self, obs, plan: SignalPlan) -> int:
        # Decisions arrive every delta_time seconds; hold a phase until it has had
        # its full green, then advance. This keeps the cycle honest even when the
        # decision interval does not divide the green duration.
        if obs.time_in_phase >= self.green_duration - 1e-6:
            self._cursor = (self._cursor + 1) % plan.num_phases
        self.last_action = plan.phases[self._cursor]
        return self.last_action
