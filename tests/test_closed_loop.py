'''The policy controller, without paying for a backbone or a simulator.

Three things have to hold before an episode is worth running: the window has to
fill correctly when the episode has only just started, the action has to be a
phase this junction actually has, and the frame-skipping optimisation must never
skip a frame the next decision will read.
'''

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bevlight.data.collate import junction_structure
from bevlight.eval.closed_loop import PolicyController
from bevlight.expert.base import SignalPlan
from bevlight.model.bevlight import BEVLight, BEVLightConfig
from bevlight.scenario.lane_mask import load_lane_mask

EMBED, WINDOW = 384, 5

# One junction of each phase count, so the fixed-width assumption cannot hide.
JUNCTIONS = [("Beijing_Beihuan", "normal"), ("Beijing_Pinganli", "easy")]


class FakeExtractor:
    """Stands in for the frozen backbone: same shape, no download, no GPU."""

    def __init__(self, lanes: int):
        self.lanes = lanes
        self.calls = 0

    def encode_array(self, array, junction, plan):
        self.calls += 1
        return np.full((self.lanes, EMBED), float(self.calls), dtype=np.float32)


def make_controller(junction: str, plan_name: str) -> tuple:
    """A PolicyController wired to a real junction but a stub backbone."""
    mask = load_lane_mask(junction, plan_name)
    controller = object.__new__(PolicyController)
    controller.junction = junction
    controller.plan_name = plan_name
    controller.window = WINDOW
    controller.decision_interval = 10
    controller.structure = junction_structure(mask)
    controller.max_lanes = controller.structure["lane_valid"].shape[0]
    controller.num_lanes = len(controller.structure["lane_order"])
    controller.extractor = FakeExtractor(controller.num_lanes)
    controller.device = torch.device("cpu")

    torch.manual_seed(0)
    controller.model = BEVLight(BEVLightConfig()).eval()
    controller.last_action = 0
    controller.encoded = 0
    controller.last_decision_at = None
    from collections import deque

    controller.frames = deque(maxlen=WINDOW)

    # Through reset(), as run_episode does: it binds self.plan to the SignalPlan
    # object, which must not disturb the plan *name* the lane mask is loaded by.
    signal_plan = SignalPlan.from_lane_mask(mask)
    controller.reset(signal_plan)
    return controller, signal_plan


class Obs:
    def __init__(self, phase_index: int, time_in_phase: float, time: float):
        self.phase_index = phase_index
        self.time_in_phase = time_in_phase
        self.time = time


@pytest.mark.parametrize("junction,plan_name", JUNCTIONS)
def test_action_is_a_phase_this_junction_has(junction, plan_name):
    controller, plan = make_controller(junction, plan_name)
    for _ in range(WINDOW):
        controller.observe(0, {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})

    action = controller.act(Obs(plan.phases[0], 30.0, 30.0), plan)
    assert action in plan.phases


@pytest.mark.parametrize("junction,plan_name", JUNCTIONS)
def test_a_short_window_repeats_the_earliest_frame(junction, plan_name):
    """The first decision of an episode must not be taken on zeros."""
    controller, plan = make_controller(junction, plan_name)
    controller.observe(0, {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})
    controller.observe(1, {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})

    batch = controller._batch(Obs(plan.phases[0], 30.0, 30.0), plan)
    features = batch["lane_features"][0].numpy()
    real = controller.num_lanes
    # Two real frames, so the older one is repeated three times and nothing is zero.
    assert [features[t, 0, 0] for t in range(WINDOW)] == [1.0, 1.0, 1.0, 1.0, 2.0]
    assert not np.any(features[:, :real] == 0.0)
    # Padded lanes stay zero whatever the window did.
    assert np.all(features[:, real:] == 0.0)


def test_min_green_holds_the_current_phase():
    controller, plan = make_controller(*JUNCTIONS[1])
    for _ in range(WINDOW):
        controller.observe(0, {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})

    held = plan.phases[1]
    minimum = controller.model.decision.min_green_s
    assert controller.act(Obs(held, minimum - 1.0, 100.0), plan) == held


def test_frame_skipping_never_skips_a_frame_the_next_decision_reads():
    """The optimisation is only safe if the whole window survives it."""
    controller, plan = make_controller(*JUNCTIONS[1])

    # Cadence unknown before the first decision: render everything.
    assert all(controller.needs_frame(second) for second in range(1, 12))

    controller.last_decision_at = 20.0
    next_decision = 20 + controller.decision_interval
    wanted = [next_decision - offset for offset in range(WINDOW - 1, -1, -1)]
    assert all(controller.needs_frame(second) for second in wanted)
    # And it does skip the seconds no window covers, which is the point.
    assert not controller.needs_frame(next_decision - WINDOW)
