'''Environments in their own processes must behave like ones in this process.

Panda3D's ShowBase is a process-level singleton, so parallel sampling means
parallel processes — and a process boundary is an easy place to lose an episode
boundary, a reward, or a summary without anything raising. These pin the parts
that would fail silently.

Rendering is off: what is under test is the transport and the episode
bookkeeping, and a Panda context per worker would make the test cost minutes.
'''

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("tshub", reason="needs the tshub environment")

from bevlight.env.vector import RemoteEnv
from bevlight.eval.compare import build_controller

SCENARIO = dict(junction="Beijing_Beihuan", plan="normal", demand="low_density",
                num_seconds=200, render=False)


@pytest.mark.slow
def test_a_remote_env_matches_a_local_one():
    """Same seed, same actions, same rewards and the same final summary."""
    from bevlight.env import JunctionEnv

    local = JunctionEnv(seed=7, **SCENARIO)
    remote = RemoteEnv(seed=7, **SCENARIO)
    try:
        local.reset()
        remote.reset()
        local_rewards, remote_rewards = [], []
        done = False
        while not done:
            # A fixed action sequence, so the two are driven identically without
            # needing a policy on either side.
            action = len(local_rewards) % local.signal_plan.num_phases
            _, r_local, done, _ = local.step(action)
            _, r_remote, done_remote, _ = remote.step(action)
            local_rewards.append(r_local)
            remote_rewards.append(r_remote)
            assert done == done_remote, "episodes ended at different steps"

        assert local_rewards == pytest.approx(remote_rewards)
        for field in ("avg_travel_time_s", "avg_waiting_time_s", "throughput"):
            assert remote.summary()[field] == pytest.approx(local.summary()[field])
    finally:
        local.close()
        remote.close()


@pytest.mark.slow
def test_async_stepping_keeps_environments_independent():
    """Two workers on different seeds must not blur into each other."""
    envs = [RemoteEnv(seed=s, **SCENARIO) for s in (7, 11)]
    try:
        for env in envs:
            env.reset()
        for step in range(12):
            for env in envs:
                env.step_async(step % 3)
            results = [env.step_wait() for env in envs]
            assert len(results) == 2
        # Different seeds mean different traffic, so the summaries must differ.
        rewards = [env.step(0)[1] for env in envs]
        assert len(rewards) == 2
    finally:
        for env in envs:
            env.close()


@pytest.mark.slow
def test_the_terminal_observation_survives_auto_reset():
    """A truncated episode must leave behind the state its value bootstraps from.

    Auto-reset overwrites the observation returned with `done` — that observation
    is `s_T`. Distinguishing truncation from termination in `info` buys nothing if
    the state to bootstrap from was dropped in transport, and nothing raises when
    it is: the learner just quietly bootstraps from the *next* episode's first
    state, which is an empty network, so every truncated episode is taught to be
    worth about zero. That is the bias the `truncated` flag exists to prevent.
    """
    from bevlight.env.vector import LocalEnv, RemoteEnv

    for factory in (LocalEnv, RemoteEnv):
        env = factory(seed=7, **{**SCENARIO, "num_seconds": 120})
        try:
            env.reset()
            done, info = False, {}
            while not done:
                _, _, done, info = env.step(0)
            assert info.get("truncated") or info.get("drained"), (
                f"{factory.__name__}: an episode ended as neither"
            )
            assert "terminal_observation" in info, (
                f"{factory.__name__} dropped the terminal observation on auto-reset"
            )
            assert "current_phase" in info["terminal_observation"]
            assert info["episode_summary"]["decisions"] > 0
        finally:
            env.close()


@pytest.mark.slow
def test_one_environment_ends_episodes_like_several_do():
    """`make_envs(1)` and `make_envs(n)` must not have different episode semantics.

    The single-environment configuration is the one a fast debugging loop uses.
    If it alone stops auto-resetting, a learner is tuned against a world that
    ends and never restarts, and the difference only appears when it is scaled up.
    """
    from bevlight.env.vector import make_envs

    single = make_envs(1, **{**SCENARIO, "num_seconds": 120}, seed=7)
    several = make_envs(2, **{**SCENARIO, "num_seconds": 120}, seed=7)
    try:
        for env in single + several:
            env.reset()
        for env in single + several:
            done = False
            while not done:
                _, _, done, info = env.step(0)
            assert "terminal_observation" in info
            # Auto-reset means the next step is the new episode's, not an error.
            _, _, done_again, _ = env.step(0)
            assert done_again is False
    finally:
        for env in single + several:
            env.close()


@pytest.mark.slow
def test_a_structured_environment_needs_no_vision_backbone():
    """`render=False` is the teacher's world; it must not load the backbone.

    `make_envs` used to build a `FeatureExtractor` unconditionally, so asking for
    a structured-state environment pulled in the vision weights, the GPU and the
    checkpoint directory to produce numbers none of them touch.
    """
    import bevlight.data.features as features
    from bevlight.env.vector import make_envs

    def explode(*args, **kwargs):
        raise AssertionError("a structured-state environment built a FeatureExtractor")

    original, features.FeatureExtractor = features.FeatureExtractor, explode
    try:
        envs = make_envs(1, **{**SCENARIO, "num_seconds": 60}, seed=7)
        envs[0].reset()
        envs[0].close()
    finally:
        features.FeatureExtractor = original
