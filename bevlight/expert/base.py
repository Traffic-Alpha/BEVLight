'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: The controller interface, shared by the expert and the baselines.

Every controller answers exactly one question: given what the junction looks
like now, which phase should be green next?

    a_t = controller.act(obs, plan)   ->   an index into plan.phases

That is the `choose_next_phase` action space, and it is deliberately the same
shape the network produces: the model scores each candidate phase and takes the
argmax. A `next_or_not` controller would only ever answer "keep / advance",
which cannot express "skip an empty phase" and would not match the model's
output space, so nothing here uses it.

The number of phases is a property of the junction's signal plan, not a
constant: K is 3 or 4 depending on the junction, and at Hongkong_YMT it differs
between the two plans of the *same* junction. Controllers therefore read K from
the plan they are handed and never assume a fixed action count.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SignalPlan:
    """The candidate phases of one junction under one signal plan.

    A phase is a *set of movements*, never an index with a fixed meaning. Phase 1
    at one junction and phase 1 at another serve completely different turns, and
    that is exactly what the model must not memorise.
    """

    junction: str
    plan: str
    tls_id: str
    phases: tuple[int, ...]
    phase2movements: dict[int, tuple[str, ...]]
    movement_in_lanes: dict[str, tuple[str, ...]]
    movement_out_lanes: dict[str, tuple[str, ...]]

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    def movements_of(self, phase: int) -> tuple[str, ...]:
        return self.phase2movements[phase]

    @classmethod
    def from_lane_mask(cls, mask) -> "SignalPlan":
        """Build the plan from a loaded lane mask, which is already plan-scoped."""
        phase2movements = {
            int(phase): tuple(movements) for phase, movements in mask.phase2movements.items()
        }
        movement_lane_ids = mask.tls["movement_lane_ids"]

        in_lanes, out_lanes = {}, {}
        for movement in mask.movement_ids:
            lanes = tuple(movement_lane_ids.get(movement, ()))
            in_lanes[movement] = lanes
            # Downstream lanes of a movement: where the traffic it releases goes.
            downstream: list[str] = []
            for lane_id in lanes:
                for to_lane in mask.lane_record(lane_id).get("to_lanes", []):
                    if to_lane not in downstream:
                        downstream.append(to_lane)
            out_lanes[movement] = tuple(downstream)

        return cls(
            junction=mask.junction,
            plan=mask.plan,
            tls_id=mask.tls_id,
            phases=tuple(sorted(phase2movements)),
            phase2movements=phase2movements,
            movement_in_lanes=in_lanes,
            movement_out_lanes=out_lanes,
        )


@runtime_checkable
class Controller(Protocol):
    """Anything that can drive a junction: expert, baseline, student or agent.

    Three members, and everything that is ever scored implements them -- the
    rule-based controllers here, the behaviour-cloned checkpoint in
    `eval.closed_loop`, and a reinforcement-learning agent through
    `eval.policies`. That is what lets one evaluation path produce every row of
    a results table instead of each method arriving with its own.

    `obs_spec` is the third member and the one that is easy to leave out. A
    controller that reads the whole lane and one that reads the BEV window are
    not comparable, and without the declaration the difference is invisible in
    the output -- see `env/obs_spec.py`.
    """

    name: str
    obs_spec: object

    def reset(self, plan: SignalPlan) -> None:
        """Called once per episode, before the first action."""

    def act(self, obs, plan: SignalPlan) -> int:
        """Return the phase to run next. Must be one of `plan.phases`."""


class BaseController:
    """Shared bookkeeping: remembers the plan and the phase it last asked for."""

    name = "base"

    #: What this controller reads. The default is the deployable one -- the BEV
    #: window, as per-lane numbers -- because that is what every rule-based
    #: controller here is restricted to, and a subclass that sees more has to
    #: say so rather than inherit a claim that flatters it.
    @property
    def obs_spec(self):
        from ..env.obs_spec import ObsMode, ObsScope, ObsSpec

        return ObsSpec(scope=ObsScope.WINDOW, mode=ObsMode.STATE, frames=1)

    def __init__(self) -> None:
        self.plan: SignalPlan | None = None
        self.last_action: int = 0

    def reset(self, plan: SignalPlan) -> None:
        self.plan = plan
        self.last_action = plan.phases[0]

    def act(self, obs, plan: SignalPlan) -> int:  # pragma: no cover - interface
        raise NotImplementedError
