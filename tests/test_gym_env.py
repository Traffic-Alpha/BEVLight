'''The step-driven environment must be the same world as the callback-driven loop.

`env/episode.py` drives a controller; `env/gym_env.py` hands the steps out so a
learner can drive instead. If the two ever diverge, a policy is optimised against
one world and scored in another, and every closed-loop number in the paper stops
meaning what it says. So the test is direct: run the same controller from the same
seed down both paths and require the control metrics to match exactly.

Rendering is off here — the equivalence being checked is the simulation and the
decision timing, and Panda would make the test cost minutes instead of seconds.
'''

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
tshub = pytest.importorskip("tshub", reason="needs the tshub environment")

from bevlight.env import JunctionEnv, run_episode
from bevlight.eval.compare import build_controller

JUNCTION, PLAN, DEMAND, STEPS, SEED = "Beijing_Beihuan", "normal", "low_density", 300, 7

# The metrics a controller is judged on. Travel time and waiting come from
# per-vehicle bookkeeping, queue from a per-second series, so between them they
# cover both accounting paths.
JUDGED = ("avg_travel_time_s", "avg_waiting_time_s", "avg_queue_veh",
          "throughput", "unfinished", "decisions", "switches")


def drive_step_env(controller_spec: str) -> dict:
    """Roll the step-driven environment out under a rule-based controller."""
    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, num_seconds=STEPS,
                      render=False)
    controller = build_controller(controller_spec)
    controller.reset(env.signal_plan)
    try:
        _, _, done, _ = env.reset()
        while not done:
            # `_pending` is the structured observation the decision is taken on;
            # a rule-based controller reads that rather than the pixel window.
            action = env.signal_plan.phases.index(
                controller.act(env._pending, env.signal_plan)
            )
            _, _, done, _ = env.step(action)
        return env.summary()
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("controller", ["max_pressure", "fixed_time"])
def test_stepping_reproduces_the_callback_loop(controller):
    reference = run_episode(
        junction=JUNCTION, plan=PLAN, demand=DEMAND,
        controller=build_controller(controller), seed=SEED,
        num_seconds=STEPS, progress_interval=0,
    ).metrics
    stepped = drive_step_env(controller)

    for field in JUDGED:
        assert stepped[field] == pytest.approx(reference[field], rel=1e-6), (
            f"{field}: stepping gave {stepped[field]}, the loop gave {reference[field]}"
        )


@pytest.mark.slow
def test_training_refuses_a_test_scenario():
    """The held-out plans and junctions carry results; reaching them must be explicit."""
    with pytest.raises(SystemExit, match="not in the train split"):
        JunctionEnv("SouthKorea_Songdo", "easy", "increasing_demand", render=False)
    # Evaluating on them is legitimate, and says so.
    env = JunctionEnv("SouthKorea_Songdo", "easy", "increasing_demand",
                      render=False, allow_any_scenario=True)
    assert env.junction == "SouthKorea_Songdo"


@pytest.mark.slow
def test_every_decision_reads_consecutive_seconds():
    """The rendered window must be the `window` seconds immediately before a decision.

    Rendering is skipped for the seconds no decision reads, which is most of them.
    The cadence that decides *which* seconds those are is not uniform: a phase
    change adds yellow time, so decisions land at 1, 11, ..., 51, 64, 77. A rule
    based on `second % interval` hands the decision at 64 a window of 56-59 and
    64 — eight seconds with a hole — while the model was trained on five
    consecutive frames. Nothing raises; the input distribution just quietly stops
    matching training.
    """
    from bevlight.env import JunctionEnv

    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, num_seconds=400,
                      render=False)
    controller = build_controller("max_pressure")
    controller.reset(env.signal_plan)

    rendered, decisions = [], []
    try:
        env._open()
        env.second = 0
        env._last_decision_at = None
        done = False
        while not done and env.second < 400:
            env.second += 1
            obs = env._look()
            # `frames` stays empty without a renderer, so stand in for it.
            env.frames.append(object())
            if env._is_in_window(obs):
                rendered.append(env.second)
            if obs.can_act:
                decisions.append(env.second)
                env._last_decision_at = float(env.second)
                action = controller.act(obs, env.signal_plan)
                if action != obs.phase_index:
                    env.phase_started_at = float(env.second)
                env.current_phase = action
            done = env._tick(obs, env.current_phase)
    finally:
        env.close()

    assert len(decisions) > 8, "not enough decisions to exercise a drifting cadence"
    assert len(set(np.diff(decisions))) > 1, "cadence never drifted; test is vacuous"

    rendered_set = set(rendered)
    for t in decisions[1:]:                      # the first has no full window yet
        needed = {t - offset for offset in range(env.window)}
        missing = needed - rendered_set
        assert not missing, f"decision at {t} is missing frames {sorted(missing)}"

    # And it must still be skipping most seconds, or the optimisation is gone.
    assert len(rendered) < 0.75 * env.second


@pytest.mark.slow
def test_the_lane_state_window_is_the_frame_window():
    """A teacher must see the seconds the student sees — not one, and not others.

    The vision model reads `window` consecutive seconds before each decision. A
    structured-state policy trained on a single snapshot is solving a different
    problem, and the shortfall would later be read as a distillation loss rather
    than as the two agents never having had the same input.
    """
    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, num_seconds=200, render=False)
    controller = build_controller("max_pressure")
    controller.reset(env.signal_plan)
    try:
        observation, _, done, _ = env.reset()
        seen = 0
        while not done:
            window = observation["lane_state"]
            assert window.shape[0] == env.window, "window is not `window` seconds deep"
            assert window.shape[-1] == 3, "expected queue, occupancy and validity"
            # Queue counts are non-negative and the validity flag is a flag.
            assert (window[..., 0] >= 0).all()
            assert set(np.unique(window[..., 2])) <= {0.0, 1.0}
            # Padded lane slots must stay empty whatever the traffic does.
            valid = observation["lane_valid"].astype(bool)
            assert not window[:, ~valid].any(), "a padded lane carried state"
            seen += 1
            action = env.signal_plan.phases.index(
                controller.act(env._pending, env.signal_plan)
            )
            observation, _, done, _ = env.step(action)
        assert seen > 10
    finally:
        env.close()


@pytest.mark.slow
def test_the_full_queue_reward_sees_what_the_visible_one_cannot():
    """The reward may read the whole lane; the observation may not.

    Where a queue reaches the edge of the BEV window the visible count stops
    growing, so a policy paid on it is paid the same for a jam that is getting
    worse. The reward is a training-time signal no deployed policy computes, so
    it is allowed to be the honest one.
    """
    summaries = {}
    for reward in ("visible_queue", "full_queue"):
        env = JunctionEnv(JUNCTION, PLAN, "high_density", seed=SEED,
                          num_seconds=400, render=False, reward=reward)
        controller = build_controller("random")
        controller.reset(env.signal_plan)
        try:
            _, _, done, _ = env.reset()
            total = 0.0
            while not done:
                action = env.signal_plan.phases.index(
                    controller.act(env._pending, env.signal_plan)
                )
                _, r, done, _ = env.step(action)
                total += r
            summaries[reward] = total
        finally:
            env.close()

    # Same episode, same actions: the full-lane reward can only be the harsher one.
    assert summaries["full_queue"] <= summaries["visible_queue"] + 1e-9


@pytest.mark.slow
def test_widening_the_observation_changes_nothing_but_the_observation():
    """The control experiment is only a control if it varies one thing.

    `full_lane` widens what the *policy* reads. It must not widen the reward, the
    control metrics, or what a rule-based baseline acts on — max-pressure driven
    off a full-lane observation is a different, stronger controller, and comparing
    against it would silently move the bar at the same time as the treatment.
    """
    rewards, baselines, queues = {}, {}, {}
    for scope in ("window", "full_lane"):
        env = JunctionEnv(JUNCTION, PLAN, "high_density", seed=SEED,
                          num_seconds=400, render=False, observe=scope)
        try:
            observation, _, done, _ = env.reset()
            stream, expert, seen = [], [], []
            step = 0
            while not done:
                # A fixed action sequence, so both runs see identical dynamics.
                action = step % env.signal_plan.num_phases
                # What max-pressure would do here, from what it is handed.
                expert.append(env._pending.queued(env.mask.incoming_lane_ids()))
                seen.append(float(observation["lane_state"][-1, :, 0].sum()))
                observation, reward, done, _ = env.step(action)
                stream.append(reward)
                step += 1
            rewards[scope], baselines[scope], queues[scope] = stream, expert, seen
        finally:
            env.close()

    assert rewards["window"] == pytest.approx(rewards["full_lane"]), (
        "widening the observation moved the reward"
    )
    assert baselines["window"] == baselines["full_lane"], (
        "widening the observation moved what the rule-based baseline reads"
    )
    # And it did do the one thing it is for: a lane measured whole is never
    # shorter than the part of it the camera sees, and here it is sometimes more.
    assert all(f >= w - 1e-6 for w, f in zip(queues["window"], queues["full_lane"]))
    assert sum(queues["full_lane"]) > sum(queues["window"]), (
        "full_lane saw no more traffic than the window; the control is vacuous"
    )
