'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Off-the-shelf reinforcement learning, on the same junction and the same observation.

[11-max-pressure-ceiling.md](../../docs/11-max-pressure-ceiling.md) claims that
within the BEV window max-pressure is effectively the ceiling, and rests that
claim on one learner, at one junction, under one demand. The obvious objection is
that the learner was ours. These are the control for it: four published
implementations, on the same `JunctionEnv`, reading the same `lane_state`, paid
out of the same reward registry, scored by the same `summary()`.

Nothing here is a method. A baseline that lost because it was handed a different
observation, a different reward or a different episode would measure our
plumbing, so the only thing that varies between a row and the arm above it is
the algorithm and the reward name.

Two of them can see which phases exist and two cannot. `MaskablePPO` reads
`JunctionGymEnv.action_masks`; `DQN`, `PPO` and `A2C` have no such hook in SB3,
so at a three-phase junction half of their `Discrete(MAX_PHASES)` resolves to
"hold the current phase". That gap is reported rather than hidden -- it is a
fact about what these libraries offer, and reading a masked and an unmasked run
side by side is the only way to say how much of a baseline's deficit is the
algorithm and how much is the action space.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Algorithm:
    """One published implementation, and what it needs to be run here."""

    name: str
    module: str
    attribute: str
    policy: str
    #: Whether it reads `action_masks`. Only `MaskablePPO` does.
    masked: bool
    #: Off-policy algorithms keep a replay buffer; on-policy ones roll and discard.
    off_policy: bool

    def load_class(self):
        import importlib

        return getattr(importlib.import_module(self.module), self.attribute)


#: Name -> implementation. The name is what `--algo` takes and what a run's
#: config records, so it is part of the experiment record and does not change.
ALGORITHMS = {
    algorithm.name: algorithm
    for algorithm in (
        Algorithm("dqn", "stable_baselines3", "DQN", "MultiInputPolicy",
                  masked=False, off_policy=True),
        Algorithm("ppo", "stable_baselines3", "PPO", "MultiInputPolicy",
                  masked=False, off_policy=False),
        Algorithm("a2c", "stable_baselines3", "A2C", "MultiInputPolicy",
                  masked=False, off_policy=False),
        Algorithm("maskable_ppo", "sb3_contrib", "MaskablePPO", "MultiInputPolicy",
                  masked=True, off_policy=False),
    )
}


def resolve(name: str) -> Algorithm:
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {list(ALGORITHMS)}")
    return ALGORITHMS[name]


def make_env(junction: str, plan: str, demand: str, *, reward: str, seed: int,
             steps: int | None = None, observe: str = "window"):
    """One structured-state environment, built the way every arm here builds it.

    `render=False` is what makes `obs_mode` "state": no Panda context, no
    backbone, per-lane numbers. That is the setting the ceiling was measured in,
    and a baseline rendered instead would be answering a different question at
    fifty times the cost.
    """
    from ..env.wrapper import JunctionGymEnv

    return JunctionGymEnv(
        junction=junction, plan=plan, demand=demand, seed=seed,
        num_seconds=steps, render=False, allow_any_scenario=True,
        reward=reward, observe=observe,
    )


def build(algorithm: Algorithm, env, *, seed: int, **hyperparameters):
    """The published implementation with its published defaults.

    Deliberately not tuned. A baseline tuned by us and beaten by us proves
    nothing about either; one run at stock settings is a statement about what
    the library does out of the box, which is what a reader wants to know.

    One correction, and it is the opposite of tuning: SB3's off-policy defaults
    are calibrated for a single environment, and `train_freq` counts *vector*
    steps while `num_timesteps` counts environment transitions. So at sixteen
    workers, `train_freq=4, gradient_steps=1` trains once per sixty-four
    transitions instead of once per four -- the same budget bought a sixteenth of
    the updates. SB3 normalises `target_update_interval` by `n_envs` for exactly
    this reason (`dqn.py`, "Account for multiple environments") and does not
    normalise this one. Scaling `gradient_steps` with the vector restores the
    library's own update-to-data ratio; leaving it alone would report the
    vectorisation as the algorithm's result.

    On-policy algorithms are untouched: PPO and A2C consume the whole rollout
    every update, so widening the vector widens the batch rather than thinning
    the updates.
    """
    if algorithm.off_policy and "gradient_steps" not in hyperparameters:
        hyperparameters["gradient_steps"] = getattr(env, "num_envs", 1)
    return algorithm.load_class()(
        algorithm.policy, env, seed=seed, verbose=0, **hyperparameters
    )


def predict(model, algorithm: Algorithm, observation, env) -> int:
    """One greedy action, with the mask if the implementation can take one."""
    if algorithm.masked:
        action, _ = model.predict(
            observation, deterministic=True, action_masks=env.action_masks()
        )
    else:
        action, _ = model.predict(observation, deterministic=True)
    return int(action)


def rollout(model, algorithm: Algorithm, junction: str, plan: str, demand: str,
            seed: int, reward: str, steps: int | None = None,
            observe: str = "window") -> dict:
    """One greedy episode, returning the metrics the results tables report.

    The same shape as `rl._internal.rollout.rollout_policy`, and for the same
    reason: what is compared has to be produced by the same `summary()`.
    """
    env = make_env(junction, plan, demand, reward=reward, seed=seed, steps=steps,
                   observe=observe)
    try:
        observation, _ = env.reset()
        done = False
        while not done:
            observation, _, done, _, _ = env.step(
                predict(model, algorithm, observation, env)
            )
        return env.inner.summary()
    finally:
        env.close()


#: How far below the best baseline's throughput a row may sit and still be read
#: as a control result rather than as metering.
THROUGHPUT_TOLERANCE = 0.05


def comparability(policy: dict, references: dict, tolerance: float = THROUGHPUT_TOLERANCE):
    """Did this policy clear the traffic, or did it just stop letting it in?

    Average travel time is taken over trips that *finished*, so a controller that
    holds one phase gives the approach it serves a clear road, strands the rest,
    and wins the metric while failing at the task. It is not a hypothetical: a
    barely-trained DQN does exactly this, and reads 14 s *ahead* of max-pressure
    on completed trips while sitting 50 s behind once the stranded vehicles are
    counted at their time so far.

    So throughput gates the comparison. A row that cleared materially fewer
    vehicles than the best baseline is not a worse controller, it is a different
    experiment, and no travel-time difference against it means anything.

    Returns `(comparable, shortfall)`, the shortfall as a fraction of the best
    baseline's throughput.
    """
    best = max((row["throughput"] for row in references.values()), default=0.0)
    if best <= 0:
        return True, 0.0
    shortfall = 1.0 - policy["throughput"] / best
    return bool(shortfall < tolerance), round(shortfall, 4)


def progress_callback(run_dir, every: int = 5000, started: float | None = None):
    """Record what the run is doing, in the shape the SAC arm already records it.

    A training run that prints nothing cannot be told from a hung one, and a
    reward curve is the first thing anyone asks for. SB3 is silent at
    `verbose=0` and writes no history of its own, so this keeps the same columns
    `runs/train/<sac run>/history.json` carries -- steps, episodes, return,
    elapsed -- and for the same reason: the two arms are meant to be read on one
    pair of axes, and a different schema would mean transposing them by hand
    every time.

    `return` is the mean over the episodes SB3 has in its info buffer, which
    needs the environments to be Monitor-wrapped; without that it reports None
    rather than a plausible-looking zero.
    """
    import json
    import time

    from stable_baselines3.common.callbacks import BaseCallback

    started = started if started is not None else time.time()
    history_path = Path(run_dir) / "history.json"

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history: list[dict] = []
            self.next_at = every

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_at:
                return True
            self.next_at += every
            episodes = list(self.model.ep_info_buffer or [])
            elapsed = time.time() - started
            entry = {
                "steps": int(self.num_timesteps),
                "episodes": len(episodes),
                "return": (round(sum(e["r"] for e in episodes) / len(episodes), 4)
                           if episodes else None),
                "episode_length": (round(sum(e["l"] for e in episodes) / len(episodes), 1)
                                   if episodes else None),
                "elapsed_s": round(elapsed, 1),
                "steps_per_s": round(self.num_timesteps / max(1e-9, elapsed), 1),
            }
            self.history.append(entry)
            history_path.write_text(json.dumps(self.history, indent=2))
            shown = "n/a" if entry["return"] is None else f"{entry['return']:+.3f}"
            print(f"[train] {entry['steps']:>7} steps  {entry['episodes']:>4} eps  "
                  f"return {shown}  {entry['steps_per_s']:.1f}/s  "
                  f"{entry['elapsed_s'] / 60:.1f} min", flush=True)
            return True

    return Progress()


def controller_rollout(spec: str, junction: str, plan: str, demand: str,
                       seed: int, steps: int | None = None,
                       cache_dir: Path | None = None) -> dict:
    """One rule-based episode, remembered on disk.

    The baseline table is a grid of algorithms against rewards, and every cell
    of it is scored against the same `max_pressure` and `fixed_time` on the same
    scenarios and seeds. Recomputing those is the largest single waste in the
    experiment: twelve cells over ninety-one scenarios is two thousand baseline
    episodes where a hundred and eighty distinct ones exist.

    The reward is deliberately not part of the key, which is worth stating
    because it looks like an omission. A rule-based controller does not read the
    reward, `summary()` does not report it, and the traffic is seeded -- so the
    episode and every metric taken from it are identical whichever reward the
    environment was constructed with. Only the scalar the environment hands back
    differs, and this discards it.
    """
    import json

    from ..paths import REPORTS_ROOT
    from ._internal.rollout import rollout_controller

    cache_dir = Path(cache_dir or REPORTS_ROOT / "controller_cache")
    key = f"{spec}__{junction}__{plan}__{demand}__seed{seed}__steps{steps}"
    path = cache_dir / f"{key.replace('/', '_')}.json"
    if path.is_file():
        return json.loads(path.read_text())

    # `reward` reaches the environment but never the result; any registered name
    # would produce this same summary.
    summary = rollout_controller(spec, junction, plan, demand, seed,
                                 "visible_queue", steps)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary
