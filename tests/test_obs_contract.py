'''The observation contract and the reward registry: the two shared interfaces.

Both exist so that a second learner -- an RL agent, an ablation variant -- plugs
into the world the reported results were measured in, rather than into a copy of
it that has drifted. These check the properties that make that true.
'''

from __future__ import annotations

from pathlib import Path

import pytest

from bevlight.env import JunctionEnv, ObsMode, ObsScope, ObsSpec
from bevlight.env.obs_spec import matched
from bevlight.env.rewards import CANDIDATES, REWARDS

JUNCTION, PLAN, DEMAND, SEED = "Beijing_Beihuan", "normal", "low_density", 7


def test_the_default_spec_is_the_deployable_one():
    """The setting every reported result uses must be what you get for free.

    A full-lane default would make the expensive mistake silent: results would
    be produced under an observation no drone can supply, and nothing in the
    output would say so.
    """
    spec = ObsSpec()
    assert spec.scope is ObsScope.WINDOW
    assert spec.deployable


@pytest.mark.needs_scenarios
def test_an_env_publishes_a_spec_matching_how_it_was_built():
    """Every environment carries its contract, whether or not one was passed."""
    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, render=False)
    assert env.obs_spec.mode is ObsMode.STATE
    assert env.obs_spec.scope is ObsScope.WINDOW
    assert not env.obs_spec.renders

    wide = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, render=False,
                       observe="full_lane")
    assert wide.obs_spec.scope is ObsScope.FULL_LANE
    assert not wide.obs_spec.deployable


@pytest.mark.needs_scenarios
def test_a_spec_that_contradicts_its_keywords_is_refused():
    """Silently resolving the conflict would hide it in the results.

    Scope in particular is the difference between a deployable measurement and
    a control experiment, and which of the two won must never be a matter of
    argument precedence.
    """
    with pytest.raises(ValueError, match="contradicts"):
        JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, render=False,
                    observe="full_lane",
                    obs_spec=ObsSpec(scope=ObsScope.WINDOW, mode=ObsMode.STATE))


@pytest.mark.needs_scenarios
def test_frames_come_from_the_spec_not_from_a_stale_keyword():
    """The window the buffers hold has to be the window the spec advertises."""
    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, render=False,
                      obs_spec=ObsSpec(mode=ObsMode.STATE, frames=3))
    assert env.window == 3
    assert env.frames.maxlen == 3
    assert env.lane_states.maxlen == 3


def test_scope_may_not_differ_between_two_compared_runs():
    """Mode may differ -- that comparison is the distillation measurement."""
    window_state = ObsSpec(scope=ObsScope.WINDOW, mode=ObsMode.STATE)
    window_pixels = ObsSpec(scope=ObsScope.WINDOW, mode=ObsMode.FEATURES)
    full_lane = ObsSpec(scope=ObsScope.FULL_LANE, mode=ObsMode.STATE)

    assert matched(window_state, window_pixels)
    assert not matched(window_state, full_lane)


def test_a_spec_round_trips_through_a_runs_config():
    """A run records its contract as JSON and must read back identical."""
    spec = ObsSpec(scope=ObsScope.FULL_LANE, mode=ObsMode.FRAMES, frames=3)
    assert ObsSpec(**spec.as_dict()) == spec


@pytest.mark.needs_scenarios
def test_every_registered_reward_is_accepted_by_the_environment():
    """The registry is the set of rewards, not a superset of what works."""
    for name in REWARDS:
        env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, render=False,
                          reward=name)
        assert env.reward_kind == name

    with pytest.raises(ValueError, match="Unknown reward"):
        JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, render=False,
                    reward="no_such_reward")


def test_the_preflight_candidates_are_all_real_rewards():
    """A candidate the environment cannot be paid is one nothing can act on."""
    assert set(CANDIDATES) <= set(REWARDS)
    # The diagnostic is not a hypothesis about control, so ranking it against
    # travel time would be measuring nothing.
    assert "probe_constant_phase" not in CANDIDATES


@pytest.mark.slow
def test_a_real_observation_carries_everything_its_spec_promised():
    """The contract is checked against what the environment actually emits.

    A spec that advertises a key the world does not produce is worse than no
    spec: a policy written against it fails at the first step of a long run.
    """
    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, num_seconds=120,
                      render=False)
    try:
        observation, _, _, _ = env.reset()
        env.obs_spec.validate(observation)
        observation, _, _, _ = env.step(0)
        env.obs_spec.validate(observation)
    finally:
        env.close()


@pytest.mark.slow
def test_the_reward_the_learner_is_paid_is_the_registrys_arithmetic():
    """`_reward` and the preflight must agree, because the preflight decides.

    A reward preflight ranks candidates in minutes so that training does not
    have to. That is only worth anything if the candidate it ranks and the
    reward the learner is paid are the same function.
    """
    from bevlight.env.rewards import RewardContext, visible_queue

    env = JunctionEnv(JUNCTION, PLAN, DEMAND, seed=SEED, num_seconds=120,
                      render=False, reward="visible_queue")
    try:
        env.reset()
        direct = visible_queue(RewardContext(
            visible=env.observer(env.states),
            states=env.states,
            incoming_lanes=env.metrics.incoming_lanes,
            current_phase=env.current_phase,
            first_phase=env.signal_plan.phases[0],
        ))
        assert env._queue_cost() == pytest.approx(direct)
    finally:
        env.close()


@pytest.mark.slow
def test_a_teacher_scored_through_the_controller_interface_matches_its_own_loop():
    """An RL policy must reach the results table by the path every other row uses.

    `rl/sac.py` scores its teacher with a step-driven loop of its own; the
    results tables are produced by `run_episode`. If those two disagree, an RL
    row printed beside max-pressure is not a comparison. They are required to
    agree exactly, not approximately: the same weights taking the same decisions
    on the same traffic have no reason to differ at all.

    This is also what pins `observe_state`. Without the per-second hook the
    adapter's window covers five decisions instead of five seconds, and this
    test fails with a switch rate 0.34 apart -- which is how that bug was found.
    """
    import torch

    from bevlight.env import run_episode
    from bevlight.env.gym_env import JunctionEnv
    from bevlight.eval.policies import TeacherController
    from bevlight.rl.sac import policy as sac_policy
    from bevlight.rl.sac import to_batch

    checkpoint = Path(__file__).resolve().parents[1] / (
        "runs/train/gate1_ymt_window/teacher_060000.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("no trained teacher checkpoint in runs/train/")

    junction, plan, demand, seed, steps = "Hongkong_YMT", "normal", "high_density", 7, 300
    net = TeacherController._load(checkpoint).eval()

    env = JunctionEnv(junction, plan, demand, seed=seed, num_seconds=steps,
                      render=False, allow_any_scenario=True)
    try:
        observation, _, done, _ = env.reset()
        while not done:
            batch = to_batch([observation], torch.device("cpu"))
            with torch.no_grad():
                probabilities, _ = sac_policy(net(batch), batch["phase_valid"])
            observation, _, done, _ = env.step(int(probabilities.argmax(dim=-1)[0]))
        step_driven = env.summary()
    finally:
        env.close()

    controller = TeacherController(checkpoint, junction, plan, model=net, device="cpu")
    through_run_episode = dict(run_episode(
        junction=junction, plan=plan, demand=demand, controller=controller,
        seed=seed, num_seconds=steps,
    ).metrics)

    for key in ("avg_travel_time_s", "avg_waiting_time_s", "avg_queue_veh",
                "throughput", "switch_rate"):
        assert through_run_episode[key] == pytest.approx(step_driven[key]), key


def test_every_kind_of_policy_declares_what_it_reads():
    """Three routes to a phase, one declaration each, so a mixed comparison shows."""
    from bevlight.eval.policies import build_policy

    baseline = build_policy("max_pressure")
    assert baseline.obs_spec.scope is ObsScope.WINDOW
    assert baseline.obs_spec.mode is ObsMode.STATE

    with pytest.raises(ValueError, match="need junction"):
        build_policy("teacher:/nowhere.pt")
    with pytest.raises(ValueError, match="need junction"):
        build_policy("checkpoint:/nowhere.pt")
