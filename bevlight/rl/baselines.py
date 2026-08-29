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
    """
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
