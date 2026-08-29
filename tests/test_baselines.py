'''The off-the-shelf baselines, and the guard that stops one reporting a win it did not have.

Nothing here starts a simulator or trains anything. What is worth pinning is the
registry -- a baseline table whose algorithm names drift is unreadable a month
later -- and the throughput gate, which is the one piece of arithmetic standing
between "DQN beats max-pressure" and the truth that it stopped the traffic.
'''

from __future__ import annotations

import pytest

from bevlight.rl.baselines import ALGORITHMS, comparability, resolve


def test_every_algorithm_names_something_importable():
    """The registry is what `--algo` accepts, so a typo in it is a broken command."""
    for name, algorithm in ALGORITHMS.items():
        assert algorithm.name == name
        assert algorithm.load_class().__name__ == algorithm.attribute


def test_exactly_one_algorithm_can_read_the_action_mask():
    """`MaskablePPO` is why sb3-contrib is a dependency at all.

    If a second one grows the ability, the table's masked/unmasked split stops
    being the thing that explains a deficit, and this test is where that gets
    noticed.
    """
    masked = [name for name, a in ALGORITHMS.items() if a.masked]
    assert masked == ["maskable_ppo"]


def test_an_unknown_algorithm_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        resolve("sac")


def summary(throughput: float, travel: float = 30.0) -> dict:
    return {"throughput": throughput, "avg_travel_time_s": travel}


def test_a_policy_that_clears_the_traffic_is_comparable():
    ok, shortfall = comparability(summary(300), {"max_pressure": summary(307)})
    assert ok and shortfall < 0.05


def test_a_policy_that_strands_the_traffic_is_not():
    """The measured case: 170 cleared against max-pressure's 307.

    On completed trips that policy read 14 s *ahead*; counting the stranded at
    their time so far it was 50 s behind. Without this gate the table reports
    the first number.
    """
    ok, shortfall = comparability(summary(170), {"max_pressure": summary(307)})
    assert not ok
    assert shortfall == pytest.approx(0.446, abs=0.01)


def test_the_gate_reads_the_best_baseline_not_the_worst():
    """Fixed-time is beatable; clearing only as much as it is not a pass."""
    ok, _ = comparability(
        summary(280), {"max_pressure": summary(307), "fixed_time": summary(277)}
    )
    assert not ok


def test_a_baseline_that_cleared_nothing_cannot_gate_anything():
    ok, shortfall = comparability(summary(100), {"max_pressure": summary(0)})
    assert ok and shortfall == 0.0


class FakeVecEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


def built_kwargs(algorithm_name: str, num_envs: int, **overrides) -> dict:
    """What `build` would hand the constructor, without constructing one."""
    import bevlight.rl.baselines as baselines

    captured = {}

    class Recorder:
        def __init__(self, policy, env, **kwargs):
            captured.update(kwargs)

    algorithm = resolve(algorithm_name)
    original = algorithm.load_class
    object.__setattr__(algorithm, "load_class", lambda: Recorder)
    try:
        baselines.build(algorithm, FakeVecEnv(num_envs), seed=7, **overrides)
    finally:
        object.__setattr__(algorithm, "load_class", original)
    return captured


def test_an_off_policy_baseline_keeps_its_update_to_data_ratio_across_the_vector():
    """SB3 counts `train_freq` in vector steps and `num_timesteps` in transitions.

    Left alone, sixteen workers buy a sixteenth of the gradient steps for the
    same number of transitions, and the run reports the vectorisation as the
    algorithm's result. SB3 normalises `target_update_interval` by `n_envs` for
    this reason and does not normalise this.
    """
    assert built_kwargs("dqn", 16)["gradient_steps"] == 16
    assert built_kwargs("dqn", 1)["gradient_steps"] == 1


def test_an_explicit_gradient_steps_is_left_alone():
    assert built_kwargs("dqn", 16, gradient_steps=3)["gradient_steps"] == 3


def test_on_policy_baselines_are_not_touched():
    """PPO and A2C consume the whole rollout, so a wider vector is a wider batch."""
    for name in ("ppo", "a2c", "maskable_ppo"):
        assert "gradient_steps" not in built_kwargs(name, 16)
