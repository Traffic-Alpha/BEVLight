'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Max-pressure control, restricted to the BEV window.

Pressure of a movement is demand upstream minus room downstream:

    P(m) = mean queue on m's incoming lanes - mean queue on m's outgoing lanes

and the pressure of a phase is the sum over the movements it serves. The phase
with the highest pressure goes green. Releasing into a blocked exit scores low,
so a spilling-back downstream lane suppresses its own movement instead of
feeding the jam.

This is the main expert for behaviour cloning precisely because it is greedy on
the current state: observation and action line up, so "what the image shows" is
enough to explain "what the expert did". A long-horizon RL expert can act on
history the image does not contain, which makes pure visual imitation ill-posed
— that one is kept only as a contrast.

Two variants, differing only in what they are allowed to see:

  * `visible` (the expert that labels the data) counts only vehicles inside the
    BEV window, so the model is never asked to imitate a decision driven by a
    queue outside its input.
  * `full` (the "upper reference" row in the results tables) counts the whole
    lane, which real detectors cannot deliver either.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

from ..base import BaseController, SignalPlan


class MaxPressure(BaseController):
    """Choose the phase with the highest total movement pressure."""

    name = "max_pressure"

    def __init__(self, use_occupancy: bool = False, min_pressure: float = 0.0) -> None:
        super().__init__()
        # Queue counts are the classical formulation; occupancy is the same idea
        # normalised by lane length, which matters when lanes differ a lot.
        self.use_occupancy = use_occupancy
        self.min_pressure = float(min_pressure)

    def movement_pressure(self, obs, plan: SignalPlan, movement: str) -> float:
        in_lanes = plan.movement_in_lanes.get(movement, ())
        out_lanes = plan.movement_out_lanes.get(movement, ())
        if self.use_occupancy:
            return obs.occupancy(in_lanes) - obs.occupancy(out_lanes)

        def mean_queue(lanes) -> float:
            seen = [obs.lanes[l].queued for l in lanes if l in obs.lanes]
            return sum(seen) / len(seen) if seen else 0.0

        return mean_queue(in_lanes) - mean_queue(out_lanes)

    def phase_pressure(self, obs, plan: SignalPlan, phase: int) -> float:
        return sum(
            self.movement_pressure(obs, plan, movement)
            for movement in plan.movements_of(phase)
        )

    def pressures(self, obs, plan: SignalPlan) -> dict[int, float]:
        return {phase: self.phase_pressure(obs, plan, phase) for phase in plan.phases}

    def act(self, obs, plan: SignalPlan) -> int:
        pressures = self.pressures(obs, plan)
        best = max(pressures.values())

        # Nothing is waiting anywhere: hold rather than cycle for no reason. A
        # switch costs a yellow interval, so churning on ties wastes green time.
        if best <= self.min_pressure:
            self.last_action = obs.phase_index
            return self.last_action

        # Ties break towards the current phase, for the same reason.
        if pressures.get(obs.phase_index, float("-inf")) >= best - 1e-9:
            self.last_action = obs.phase_index
            return self.last_action

        self.last_action = max(plan.phases, key=lambda p: (pressures[p], -p))
        return self.last_action
