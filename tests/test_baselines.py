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


def test_a_controller_episode_is_computed_once_and_remembered(tmp_path, monkeypatch):
    """The largest waste in the grid is re-simulating the same baseline.

    Twelve cells scored against the same two controllers over the same
    scenarios is two thousand episodes where a hundred and eighty distinct ones
    exist, and an episode is ten to fifteen seconds of SUMO whoever is driving.
    """
    import bevlight.rl._internal.rollout as rollout_module
    from bevlight.rl.baselines import controller_rollout

    calls = []

    def fake(spec, junction, plan, demand, seed, reward, steps=None):
        calls.append((spec, junction, plan, demand, seed, reward, steps))
        return {"throughput": 307, "avg_travel_time_s": 29.25}

    monkeypatch.setattr(rollout_module, "rollout_controller", fake)

    first = controller_rollout("max_pressure", "J", "normal", "high", 7,
                               cache_dir=tmp_path)
    second = controller_rollout("max_pressure", "J", "normal", "high", 7,
                                cache_dir=tmp_path)
    assert first == second
    assert len(calls) == 1, "the second call re-simulated instead of reading the cache"


def test_the_reward_is_not_part_of_what_is_remembered(tmp_path, monkeypatch):
    """It looks like an omission in the key, so it is pinned as a decision.

    A rule-based controller never reads the reward, `summary()` never reports
    it, and the traffic is seeded -- so the episode and every metric taken from
    it are identical whichever reward the environment was built with. Only the
    scalar the environment hands back differs, and the cache discards that.
    """
    import bevlight.rl._internal.rollout as rollout_module
    from bevlight.rl.baselines import controller_rollout

    rewards_seen = []

    def fake(spec, junction, plan, demand, seed, reward, steps=None):
        rewards_seen.append(reward)
        return {"throughput": 307}

    monkeypatch.setattr(rollout_module, "rollout_controller", fake)
    controller_rollout("max_pressure", "J", "normal", "high", 7, cache_dir=tmp_path)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert "visible_queue" not in files[0].name
    assert len(rewards_seen) == 1


def test_a_different_seed_is_a_different_episode(tmp_path, monkeypatch):
    import bevlight.rl._internal.rollout as rollout_module
    from bevlight.rl.baselines import controller_rollout

    calls = []
    monkeypatch.setattr(
        rollout_module, "rollout_controller",
        lambda *a, **k: (calls.append(a), {"throughput": 1})[1],
    )
    controller_rollout("max_pressure", "J", "normal", "high", 7, cache_dir=tmp_path)
    controller_rollout("max_pressure", "J", "normal", "high", 8, cache_dir=tmp_path)
    assert len(calls) == 2


def test_a_pooled_return_is_replaced_by_one_that_means_the_same_everywhere():
    """max-pressure's own cost spans 15x across the training split.

    Measured: -0.0640 per decision at Beijing_Beihuan/normal/low_density,
    -0.9692 at Beijing_Beishahe/easy/high_density. An episode return of -6 and
    one of -97 can be the same policy doing equally well, so a curve built from
    the pooled mean moves with which scenarios finished recently rather than
    with the policy.
    """
    from bevlight.rl.baselines import normalised_return

    reference = {"quiet": -0.064, "busy": -0.969}
    matching = [{"scenario": "quiet", "r": -6.4, "l": 100},
                {"scenario": "busy", "r": -96.9, "l": 100}]
    assert normalised_return(matching, reference) == pytest.approx(1.0, abs=1e-3)

    # The raw mean of those two is -51.65, a number about neither scenario.
    better = [{"scenario": "busy", "r": -48.45, "l": 100}]
    assert normalised_return(better, reference) == pytest.approx(0.5, abs=1e-3)


def test_an_episode_from_an_unmeasured_scenario_is_left_out_rather_than_guessed():
    from bevlight.rl.baselines import normalised_return

    reference = {"quiet": -0.064}
    episodes = [{"scenario": "quiet", "r": -6.4, "l": 100},
                {"scenario": "unmeasured", "r": -900.0, "l": 100}]
    assert normalised_return(episodes, reference) == pytest.approx(1.0, abs=1e-3)


def test_without_a_reference_table_no_ratio_is_invented():
    from bevlight.rl.baselines import normalised_return

    assert normalised_return([{"scenario": "quiet", "r": -6.4, "l": 100}], {}) is None


def test_the_reference_table_is_optional(tmp_path):
    """A run on a machine that has not measured one still trains and still logs."""
    from bevlight.rl.baselines import reward_reference

    assert reward_reference("visible_queue", tmp_path / "absent.json") == {}


def test_a_truncated_evaluation_keeps_the_split_it_came_from():
    """Taking the first N separated the groups by the thing under test.

    The splits are ordered by junction and junctions differ in phase count, so
    the first eight of `train` are all three-phase while `cross_plan_test` is
    entirely four-phase. A run truncated that way reported "generalises across
    demand, fails across plan" when what it had measured was "three phases work
    and four do not".
    """
    from bevlight.rl.cli.baseline import sample

    ordered = ["a3", "b3", "c3", "d3", "e3", "f4", "g4", "h4"]
    assert sample(ordered, 4) == ["a3", "c3", "e3", "g4"]
    assert [s[-1] for s in sample(ordered, 4)].count("4") == 1


def test_asking_for_everything_or_more_returns_everything():
    from bevlight.rl.cli.baseline import sample

    ordered = ["a", "b", "c"]
    assert sample(ordered, None) == ordered
    assert sample(ordered, 3) == ordered
    assert sample(ordered, 99) == ordered


def test_an_on_policy_baseline_gets_its_stock_number_of_updates():
    """`n_steps` is per environment, so the rollout buffer scales with the vector.

    PPO's stock 2048 becomes 32768 at sixteen workers, and a 60 000-step budget
    then buys one policy update where a single environment would have bought
    twenty-nine. The first wave of the grid ran that way and measured an
    untrained network.
    """
    assert built_kwargs("ppo", 16)["n_steps"] == 128
    assert built_kwargs("maskable_ppo", 16)["n_steps"] == 128
    assert built_kwargs("ppo", 1)["n_steps"] == 2048


def test_a2c_is_left_at_its_own_default():
    """Its `n_steps` is 5 and cannot be divided by 16 without becoming one-step TD.

    That is a different algorithm rather than the same one configured correctly,
    and A2C does not need the correction: 80 per buffer still buys 750 updates
    out of the same budget.
    """
    assert "n_steps" not in built_kwargs("a2c", 16)


def test_an_explicit_n_steps_is_left_alone():
    assert built_kwargs("ppo", 16, n_steps=512)["n_steps"] == 512


def test_the_key_two_lookups_join_on_is_the_scenarios_own():
    """A training curve is normalised by it and so is the reward divisor.

    Both used to build the string inline. A drift between them and
    `Scenario.key` would not raise -- every lookup would miss and fall back to a
    scale of 1, which looks exactly like normalisation being off.
    """
    from bevlight.env.wrapper import scenario_key
    from bevlight.scenario.selection import Scenario

    scenario = Scenario("Hongkong_YMT", "normal", "high_density", "train")
    assert scenario_key("Hongkong_YMT", "normal", "high_density") == scenario.key


def test_a_phase_id_means_different_lanes_under_a_different_plan():
    """The reason a policy without the wiring cannot cross plans.

    At Hongkong_YMT action 0 releases six lanes under `normal` and three
    entirely different ones under `easy`. A network shown only lane queues and a
    phase index has to learn that association from experience, and what it
    learns is true of one (junction, plan) and wrong for the next -- which is a
    memorised phase id, and is what a cross-plan evaluation collapses on.
    """
    from bevlight.data.collate import junction_structure, phase_lane_incidence
    from bevlight.scenario.lane_mask import load_lane_mask

    matrices = {
        plan: phase_lane_incidence(
            junction_structure(load_lane_mask("Hongkong_YMT", plan))
        )
        for plan in ("normal", "easy")
    }
    served = {plan: {i for i, w in enumerate(m[0]) if w > 0}
              for plan, m in matrices.items()}
    assert served["normal"] != served["easy"], (
        "if action 0 served the same lanes under both plans there would be "
        "nothing for a policy to get wrong"
    )
    assert matrices["normal"].shape == matrices["easy"].shape


def test_the_incidence_covers_only_phases_the_plan_has():
    """Padded phases release nothing, so an unused action row is all zeros."""
    from bevlight.data.collate import junction_structure, phase_lane_incidence
    from bevlight.scenario.lane_mask import load_lane_mask

    mask = load_lane_mask("Hongkong_YMT", "normal")
    structure = junction_structure(mask)
    incidence = phase_lane_incidence(structure)
    for phase in range(mask.num_phases, incidence.shape[0]):
        assert incidence[phase].sum() == 0
    assert incidence[: mask.num_phases].sum() > 0


def structured_observation(batch: int = 2):
    """A batch shaped exactly as the environment publishes one."""
    import numpy as np
    import torch

    from bevlight.data.collate import (
        MAX_LANES,
        MAX_LANES_PER_MOVEMENT,
        MAX_MOVEMENTS,
        MAX_MOVEMENTS_PER_PHASE,
        MAX_PHASES,
    )
    from bevlight.env.gym_env import LANE_STATE_CHANNELS

    rng = np.random.default_rng(7)
    valid_lanes, valid_phases = 12, 3
    lane_valid = np.zeros((batch, MAX_LANES), np.float32)
    lane_valid[:, :valid_lanes] = 1
    phase_valid = np.zeros((batch, MAX_PHASES), np.float32)
    phase_valid[:, :valid_phases] = 1
    movement_valid = np.zeros((batch, MAX_MOVEMENTS), np.float32)
    movement_valid[:, :4] = 1
    return {
        "lane_state": torch.tensor(rng.uniform(
            0, 5, (batch, 5, MAX_LANES, len(LANE_STATE_CHANNELS))).astype(np.float32)),
        "lane_valid": torch.tensor(lane_valid),
        "incoming_valid": torch.tensor(lane_valid),
        "movement_valid": torch.tensor(movement_valid),
        "phase_valid": torch.tensor(phase_valid),
        "current_phase": torch.zeros((batch, 1), dtype=torch.int64),
        "time_in_phase": torch.full((batch, 1), 12.0),
        "movement_in_index": torch.zeros(
            (batch, MAX_MOVEMENTS, MAX_LANES_PER_MOVEMENT), dtype=torch.int64),
        "movement_in_weight": torch.zeros(
            (batch, MAX_MOVEMENTS, MAX_LANES_PER_MOVEMENT)),
        "movement_out_index": torch.zeros(
            (batch, MAX_MOVEMENTS, MAX_LANES_PER_MOVEMENT), dtype=torch.int64),
        "movement_out_weight": torch.zeros(
            (batch, MAX_MOVEMENTS, MAX_LANES_PER_MOVEMENT)),
        "phase_members": torch.zeros(
            (batch, MAX_PHASES, MAX_MOVEMENTS_PER_PHASE), dtype=torch.int64),
        "phase_member_valid": torch.zeros(
            (batch, MAX_PHASES, MAX_MOVEMENTS_PER_PHASE)),
        "phase_lane_in": torch.zeros((batch, MAX_PHASES, MAX_LANES)),
        "phase_lane_out": torch.zeros((batch, MAX_PHASES, MAX_LANES)),
    }


def test_the_structured_trunk_scores_only_the_phases_the_junction_has():
    """Padded candidates leave at -inf, so an unavailable action is never argmax."""
    pytest.importorskip("torch")
    from bevlight.data.collate import MAX_PHASES
    from bevlight.rl._internal.structured import StructuredTrunk

    scores = StructuredTrunk().scores(structured_observation())
    assert scores.shape == (2, MAX_PHASES)
    assert bool((scores[:, 3:] < -1e8).all()), "padded phases took a finite score"
    assert bool((scores[:, :3] > -1e8).all())


def test_the_batch_conversion_keeps_gather_indices_integral():
    """A float `gather` index is a silent wrong answer, not an error."""
    pytest.importorskip("torch")
    from bevlight.rl._internal.structured import as_batch

    observation = structured_observation()
    observation["phase_members"] = observation["phase_members"].float()
    batch = as_batch(observation)
    assert batch["phase_members"].dtype.is_floating_point is False
    assert batch["current_phase"].dim() == 1, "the decision layer indexes with it"
    assert batch["time_in_phase"].dim() == 1


def test_only_what_the_data_asked_for_is_changed():
    """One measured defect, one change; the reflexes are left alone.

    The first pass looked like a policy that would not explore -- entropy 1.02
    against 1.19 for uniform after two hundred thousand steps -- and an entropy
    bonus is the reflex. It would have been the wrong one: the policy had not
    collapsed, it had barely moved, because `policy_gradient_loss` was 0.01
    against a `value_loss` of 27. Paying it to stay uniform fights the
    decisiveness it was failing to acquire.

    So `ent_coef`, `n_epochs` and `batch_size` stay at SB3's values, and the
    single change -- normalising the return so the two halves of the loss are
    comparable -- is applied where the env is built, not here.
    """
    for name in ("ppo", "maskable_ppo"):
        built = built_kwargs(name, 16)
        for key in ("ent_coef", "n_epochs", "batch_size"):
            assert key not in built, f"{name} had {key} set without evidence for it"


def test_the_off_policy_baseline_is_left_at_its_own_defaults():
    """DQN has no policy loss to be drowned by a value loss, so none of it applies."""
    built = built_kwargs("dqn", 16)
    for key in ("ent_coef", "n_epochs", "batch_size"):
        assert key not in built


def test_an_explicit_setting_still_wins():
    assert built_kwargs("ppo", 16, ent_coef=0.05)["ent_coef"] == 0.05
