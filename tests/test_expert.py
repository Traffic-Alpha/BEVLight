'''Controllers and the visible-window observation they read.

These run without SUMO: the observation is a plain object, which is the reason
controllers take one instead of an environment handle.
'''

from __future__ import annotations

import pytest

from bevlight.collect.observation import JunctionObservation, LaneState
from bevlight.expert import FixedTime, MaxPressure, SignalPlan
from bevlight.scenario.lane_mask import load_lane_mask


def make_plan(junction="Beijing_Beihuan", plan="normal") -> SignalPlan:
    return SignalPlan.from_lane_mask(load_lane_mask(junction, plan))


def make_obs(plan: SignalPlan, queues: dict, phase: int = 0, time_in_phase: float = 10.0):
    """An observation where the named lanes hold the given queues."""
    lanes = {}
    for movement in set(plan.movement_in_lanes) | set(plan.movement_out_lanes):
        for lane_id in plan.movement_in_lanes.get(movement, ()) + plan.movement_out_lanes.get(movement, ()):
            lanes[lane_id] = LaneState(
                lane_id=lane_id,
                role="incoming" if lane_id in sum((list(v) for v in plan.movement_in_lanes.values()), []) else "outgoing",
                visible_m=60.0,
                vehicles=queues.get(lane_id, 0),
                queued=queues.get(lane_id, 0),
                queue_m=7.5 * queues.get(lane_id, 0),
            )
    return JunctionObservation(
        time=100.0, lanes=lanes, phase_index=phase, time_in_phase=time_in_phase, can_act=True
    )


@pytest.mark.needs_scenarios
def test_max_pressure_picks_the_busiest_phase():
    plan = make_plan()
    target = plan.phases[-1]
    busy_lane = plan.movement_in_lanes[plan.movements_of(target)[0]][0]
    obs = make_obs(plan, {busy_lane: 9}, phase=plan.phases[0])
    assert MaxPressure().act(obs, plan) == target


@pytest.mark.needs_scenarios
def test_max_pressure_holds_when_nothing_is_waiting():
    """Switching costs a yellow interval, so an empty junction should not churn."""
    plan = make_plan()
    obs = make_obs(plan, {}, phase=plan.phases[1])
    assert MaxPressure().act(obs, plan) == plan.phases[1]


@pytest.mark.needs_scenarios
def test_max_pressure_prefers_the_current_phase_on_a_tie():
    plan = make_plan()
    lanes = {plan.movement_in_lanes[plan.movements_of(p)[0]][0]: 4 for p in plan.phases}
    obs = make_obs(plan, lanes, phase=plan.phases[1])
    assert MaxPressure().act(obs, plan) == plan.phases[1]


@pytest.mark.needs_scenarios
def test_a_blocked_exit_suppresses_its_own_movement():
    """The point of pressure: releasing into a jam should not win."""
    plan = make_plan()
    a, b = plan.phases[0], plan.phases[1]
    in_a = plan.movement_in_lanes[plan.movements_of(a)[0]][0]
    out_a = plan.movement_out_lanes[plan.movements_of(a)[0]][0]
    in_b = plan.movement_in_lanes[plan.movements_of(b)[0]][0]

    free = make_obs(plan, {in_a: 6, in_b: 5}, phase=b)
    assert MaxPressure().act(free, plan) == a           # 6 beats 5

    blocked = make_obs(plan, {in_a: 6, out_a: 6, in_b: 5}, phase=b)
    assert MaxPressure().act(blocked, plan) == b        # 6-6=0 loses to 5


@pytest.mark.needs_scenarios
@pytest.mark.parametrize("controller", [FixedTime(30.0), MaxPressure()])
def test_controllers_only_return_valid_phases(controller):
    plan = make_plan()
    controller.reset(plan)
    for phase in plan.phases:
        obs = make_obs(plan, {}, phase=phase, time_in_phase=99.0)
        assert controller.act(obs, plan) in plan.phases


@pytest.mark.needs_scenarios
def test_fixed_time_advances_only_after_a_full_green():
    plan = make_plan()
    controller = FixedTime(30.0)
    controller.reset(plan)
    held = make_obs(plan, {}, phase=plan.phases[0], time_in_phase=10.0)
    assert controller.act(held, plan) == plan.phases[0]
    due = make_obs(plan, {}, phase=plan.phases[0], time_in_phase=30.0)
    assert controller.act(due, plan) == plan.phases[1]


@pytest.mark.needs_scenarios
def test_fixed_time_cycles_through_every_phase():
    plan = make_plan()
    controller = FixedTime(30.0)
    controller.reset(plan)
    seen = set()
    for _ in range(plan.num_phases * 2):
        obs = make_obs(plan, {}, time_in_phase=30.0)
        seen.add(controller.act(obs, plan))
    assert seen == set(plan.phases)


@pytest.mark.needs_scenarios
def test_variable_phase_count_is_read_from_the_plan():
    """K differs between the two plans of the same junction at YMT."""
    assert make_plan("Hongkong_YMT", "easy").num_phases == 4
    assert make_plan("Hongkong_YMT", "normal").num_phases == 3
