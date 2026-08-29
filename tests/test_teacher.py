'''The teacher must be the same model, the same masks, and an honest expectation.

Three things here would fail silently if they were wrong.

Padding: the teacher reads the same padded tensors the student does, so a mask
that fails to reach one softmax makes its answer depend on batch composition.
`test_padding.py` pins this for the vision path; the lane encoder is a new seam
in front of it and needs the same guarantee.

The masked policy: discrete SAC sums an expectation over candidate actions, and a
padded candidate carrying any probability at all puts weight on a phase that does
not exist. The failure is quiet — the numbers stay finite and the policy just
learns to spend part of its mass on nothing.

The replay buffer: junction wiring is stored once and referenced by index, so a
transition sampled with the wrong structure would describe a junction it never
visited. Nothing raises; the critic just fits noise.
'''

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bevlight.data.collate import junction_structure
from bevlight.model.teacher import LANE_STATE_DIM, TeacherNet, teacher_config
from bevlight.rl.sac import STRUCTURE_KEYS, ReplayBuffer, policy, to_batch
from bevlight.scenario.lane_mask import load_lane_mask

WINDOW = 5


def make_observation(junction: str, plan: str, max_lanes: int = 48,
                     max_movements: int = 16, max_phases: int = 6,
                     seed: int = 0) -> dict:
    mask = load_lane_mask(junction, plan)
    structure = junction_structure(mask, max_lanes, max_movements, max_phases)
    real = int(structure["lane_valid"].sum())

    rng = np.random.default_rng(seed)
    state = np.empty((WINDOW, max_lanes, LANE_STATE_DIM), dtype=np.float32)
    state[:, :real] = rng.uniform(0, 8, (WINDOW, real, LANE_STATE_DIM)).astype(np.float32)
    # Padded lanes carry garbage on purpose: if a mask leaks, it will show.
    state[:, real:] = 1e3
    return {
        **{k: v for k, v in structure.items() if k not in ("lane_order", "phase_order")},
        "lane_state": state, "current_phase": 1, "time_in_phase": 12.0,
    }


@pytest.fixture(scope="module")
def teacher():
    torch.manual_seed(0)
    return TeacherNet(teacher_config()).eval()


@pytest.mark.parametrize("junction,plan", [("Beijing_Pinganli", "easy"),
                                           ("Hongkong_YMT", "normal")])
@pytest.mark.needs_scenarios
def test_extra_padding_does_not_change_the_scores(teacher, junction, plan):
    tight = make_observation(junction, plan, 32, 12, 4)
    loose = make_observation(junction, plan, 48, 16, 6)
    with torch.no_grad():
        a = teacher(to_batch([tight], "cpu"))[:, :4]
        b = teacher(to_batch([loose], "cpu"))[:, :4]
    assert torch.allclose(a, b, atol=1e-4), "padding changed the teacher's answer"


@pytest.mark.needs_scenarios
def test_padded_candidates_hold_no_probability(teacher):
    observation = make_observation("Beijing_Pinganli", "easy")
    batch = to_batch([observation], "cpu")
    with torch.no_grad():
        probabilities, log_probabilities = policy(teacher(batch), batch["phase_valid"])

    valid = batch["phase_valid"].bool()
    assert torch.allclose(probabilities.sum(-1), torch.ones(1)), "mass leaked away"
    assert probabilities[~valid].abs().max() == 0.0, "a phase that does not exist got mass"
    # The entropy sum must not pick up a 0 * -inf, and must respect the real K.
    entropy = -(probabilities * log_probabilities).sum(-1)
    assert torch.isfinite(entropy).all()
    assert entropy.item() <= float(np.log(int(valid.sum()))) + 1e-5


@pytest.mark.needs_scenarios
def test_the_expectation_over_candidates_ignores_padding(teacher):
    """A sum over K must equal a sum over the real candidates, exactly."""
    observation = make_observation("Hongkong_YMT", "normal")
    batch = to_batch([observation], "cpu")
    valid = batch["phase_valid"].bool()
    with torch.no_grad():
        probabilities, _ = policy(teacher(batch), batch["phase_valid"])
        q = teacher(batch)
    full = (probabilities * q).sum(-1)
    real = (probabilities[valid] * q[valid]).sum()
    assert torch.allclose(full.squeeze(), real, atol=1e-5)


@pytest.mark.needs_scenarios
def test_the_buffer_returns_the_structure_the_transition_had():
    """Two junctions in one buffer must not be handed each other's wiring."""
    first = make_observation("Beijing_Pinganli", "easy", seed=1)
    second = make_observation("Hongkong_YMT", "normal", seed=2)
    buffer = ReplayBuffer(64, WINDOW, 48, LANE_STATE_DIM)
    for _ in range(8):
        buffer.add(("Beijing_Pinganli", "easy"), first, 0, -1.0, first, False, 0.857)
        buffer.add(("Hongkong_YMT", "normal"), second, 1, -2.0, second, True, 0.95)

    assert buffer.size == 16
    batch = buffer.sample(16, "cpu")
    for key in STRUCTURE_KEYS:
        assert batch[key].shape[0] == 16
    # Every sampled row must match one of the two junctions it could have come
    # from, and the reward it was stored with must select which.
    reference = {-1.0: torch.as_tensor(first["phase_valid"]),
                 -2.0: torch.as_tensor(second["phase_valid"])}
    for i in range(16):
        expected = reference[round(float(batch["reward"][i]), 1)]
        assert torch.equal(batch["phase_valid"][i], expected), "wrong junction wiring"
    # Terminal is stored per transition, and only the drained one is terminal.
    assert torch.equal(batch["terminal"], (batch["reward"] == -2.0).float())
    # So is the bootstrap discount: a window cut short by an episode end must not
    # be handed the full-length one.
    assert torch.allclose(batch["discount"],
                          torch.where(batch["reward"] == -2.0, 0.95, 0.857), atol=1e-6)


@pytest.mark.needs_scenarios
def test_the_lane_state_window_reaches_the_model(teacher):
    """Changing an earlier second must change the answer, or the window is dead."""
    observation = make_observation("Beijing_Pinganli", "easy", seed=3)
    disturbed = {**observation, "lane_state": observation["lane_state"].copy()}
    real = int(observation["lane_valid"].sum())
    disturbed["lane_state"][0, :real, 0] += 5.0        # the oldest second only
    with torch.no_grad():
        a = teacher(to_batch([observation], "cpu"))
        b = teacher(to_batch([disturbed], "cpu"))
    assert not torch.allclose(a, b, atol=1e-6), "the earliest frame is being ignored"


def test_a_full_window_is_the_discounted_sum():
    """R = r0 + g r1 + g^2 r2, bootstrapping g^3 away."""
    from bevlight.rl.sac import NStepAccumulator

    acc = NStepAccumulator(3, 0.5)
    assert acc.push("s0", 0, 1.0, "s1", False, False) == []
    assert acc.push("s1", 0, 2.0, "s2", False, False) == []
    emitted = acc.push("s2", 0, 4.0, "s3", False, False)
    assert len(emitted) == 1
    observation, action, reward, successor, terminal, discount = emitted[0]
    assert observation == "s0" and successor == "s3"
    assert reward == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    assert discount == pytest.approx(0.125)
    assert terminal is False


def test_the_end_of_an_episode_flushes_every_shorter_window():
    """The last decisions of an episode are real decisions, at their own horizon.

    Dropping them would silently discard the jam-clearing tail — the part of the
    episode where the control problem is hardest.
    """
    from bevlight.rl.sac import NStepAccumulator

    acc = NStepAccumulator(3, 0.5)
    acc.push("s0", 0, 1.0, "s1", False, False)
    emitted = acc.push("s1", 1, 2.0, "sT", True, True)
    assert len(emitted) == 2, "a shorter tail window was dropped"

    first, second = emitted
    assert first[0] == "s0" and first[2] == pytest.approx(1.0 + 0.5 * 2.0)
    assert first[5] == pytest.approx(0.25), "a two-step window bootstrapped as three"
    assert second[0] == "s1" and second[2] == pytest.approx(2.0)
    assert second[5] == pytest.approx(0.5)
    assert first[3] == second[3] == "sT" and first[4] is True
    assert not acc.pending, "the queue must not survive an episode boundary"


def test_no_window_spans_two_episodes():
    """After a reset the queue restarts, so no reward crosses the boundary."""
    from bevlight.rl.sac import NStepAccumulator

    acc = NStepAccumulator(3, 0.9)
    acc.push("a0", 0, 1.0, "a1", False, False)
    acc.push("a1", 0, 1.0, "aT", False, True)          # truncated, not terminal
    emitted = acc.push("b0", 0, 5.0, "b1", False, False)
    assert emitted == [], "the new episode reused the old queue"
    assert [entry[0] for entry in acc.pending] == ["b0"]


def test_one_step_is_plain_td():
    """n=1 must reproduce the single-step target exactly."""
    from bevlight.rl.sac import NStepAccumulator

    acc = NStepAccumulator(1, 0.95)
    emitted = acc.push("s0", 2, -0.4, "s1", False, False)
    assert len(emitted) == 1
    assert emitted[0][2] == pytest.approx(-0.4)
    assert emitted[0][5] == pytest.approx(0.95)


def test_every_decision_reaches_the_buffer_exactly_once():
    """Across episodes, transitions in must equal transitions out."""
    from bevlight.rl.sac import NStepAccumulator

    acc = NStepAccumulator(4, 0.99)
    pushed = emitted = 0
    for episode in range(5):
        length = 3 + episode * 4                       # shorter and longer than n
        for t in range(length):
            done = t == length - 1
            emitted += len(acc.push(f"e{episode}s{t}", 0, -1.0, "next", done, done))
            pushed += 1
    assert emitted == pushed, f"{pushed} decisions in, {emitted} out"
