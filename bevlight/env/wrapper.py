'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: The gymnasium face of a junction, and the factory a vectorised run needs.

Follows the shape TransSimHub's own RL benchmarks use — a thin `gym.Env` over the
simulator, a `gym.Wrapper` that owns the observation and the reward, and a
`make_env` closure for `SubprocVecEnv` — with one difference that matters here.

Those benchmarks observe `last_step_occupancy`, a vector SUMO hands over for free.
This project observes pixels, so every environment carries a Panda context while
the frozen backbone stays single and shared: N environments render in parallel and
their frames are pooled in one batch, rather than N batch-of-one forwards of the
same weights. `obs_mode` is that switch.

The action space is `Discrete(MAX_PHASES)` rather than this junction's K, because a
vectorised run mixes junctions with different K. Candidates that do not exist are
masked out of the softmax by `phase_valid`, exactly as in training; `action_mask`
is published in `info` for algorithms that can use it directly.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..data.collate import MAX_LANES, MAX_MOVEMENTS, MAX_PHASES
from .gym_env import LANE_STATE_CHANNELS, JunctionEnv


class JunctionGymEnv:
    """The `gymnasium.Env` behaviour over `JunctionEnv`, minus the base class.

    Everything a gymnasium environment has to do is here, but `gymnasium.Env` is
    not a base of it: this module is imported by `bevlight.rl`, and the SAC arm
    that lives there drives `JunctionEnv` directly and has no use for gym. Making
    gym a hard import to reach that arm would put an extra in the way of code
    that does not use it.

    Libraries that type-check their argument -- SB3 raises outright on anything
    that is not a `gymnasium.Env` -- need the real base, so `gym_env_class()`
    builds it on first use and `make_env` hands that out. Both spellings are the
    same behaviour; only the ancestry differs.
    """

    metadata = {"render_modes": []}

    def __init__(self, embed_dim: int = 384, **kwargs):
        import gymnasium as gym

        self.gym = gym
        self.inner = JunctionEnv(**kwargs)
        self.embed_dim = embed_dim
        # Three encodings of one world, chosen by how the env was built rather
        # than by a separate argument that could disagree with it. `state` is
        # what an off-the-shelf learner reads: no renderer, no backbone, the
        # same per-lane numbers the results tables call the structured setting.
        if kwargs.get("extractor") is not None:
            self.obs_mode = "features"
        elif kwargs.get("render", True):
            self.obs_mode = "frames"
        else:
            self.obs_mode = "state"

        self.action_space = gym.spaces.Discrete(MAX_PHASES)
        self.observation_space = self._space()

    def _space(self):
        gym = self.gym
        window = self.inner.window
        spaces = {
            "lane_valid": gym.spaces.Box(0, 1, (MAX_LANES,), np.float32),
            "incoming_valid": gym.spaces.Box(0, 1, (MAX_LANES,), np.float32),
            "movement_valid": gym.spaces.Box(0, 1, (MAX_MOVEMENTS,), np.float32),
            "phase_valid": gym.spaces.Box(0, 1, (MAX_PHASES,), np.float32),
            "current_phase": gym.spaces.Discrete(MAX_PHASES),
            "time_in_phase": gym.spaces.Box(0, np.inf, (1,), np.float32),
        }
        if self.obs_mode == "features":
            spaces["lane_features"] = gym.spaces.Box(
                -np.inf, np.inf, (window, MAX_LANES, self.embed_dim), np.float32
            )
        elif self.obs_mode == "state":
            spaces["lane_state"] = gym.spaces.Box(
                -np.inf, np.inf,
                (window, MAX_LANES, len(LANE_STATE_CHANNELS)), np.float32
            )
        return gym.spaces.Dict(spaces)

    def _pack(self, observation: dict, info: dict) -> tuple:
        """Keep the keys the space declares; the rest travels in `info`."""
        declared = set(self.observation_space.spaces)
        packed = {k: v for k, v in observation.items() if k in declared}
        packed["time_in_phase"] = np.array(
            [observation.get("time_in_phase", 0.0)], dtype=np.float32
        )
        extra = {k: v for k, v in observation.items() if k not in declared}
        return packed, {**info, **extra,
                        "action_mask": observation.get("phase_valid")}

    def reset(self, *, seed=None, options=None):
        observation, _, _, info = self.inner.reset()
        return self._pack(observation, info)

    def step(self, action):
        # A vectorised run shares one Discrete(MAX_PHASES); a junction with K=3
        # never sees phase 3, so an out-of-range action holds the current phase
        # rather than crashing a rollout.
        valid = int(self.inner.signal_plan.num_phases)
        action = int(action) if int(action) < valid else self.inner.signal_plan.phases.index(
            self.inner.current_phase
        )
        observation, reward, done, info = self.inner.step(action)
        packed, info = self._pack(observation, info)
        if done:
            info["episode_summary"] = self.inner.summary()
        return packed, reward, done, done, info

    def action_masks(self) -> np.ndarray:
        """Which of the `MAX_PHASES` actions this junction actually has.

        The name is `sb3-contrib`'s convention: `MaskablePPO` looks for a method
        called exactly this and removes the rest from its distribution. Without
        it a junction with three phases spends half its action space on actions
        that resolve to "hold", which reads as a bad algorithm rather than as a
        mis-specified action space.
        """
        valid = np.zeros(MAX_PHASES, dtype=bool)
        valid[: int(self.inner.signal_plan.num_phases)] = True
        return valid

    def close(self):
        self.inner.close()


@lru_cache(maxsize=1)
def gym_env_class():
    """`JunctionGymEnv` with `gymnasium.Env` genuinely behind it.

    Built once, on first use, so importing this module still costs no gym.
    """
    import gymnasium as gym

    return type("JunctionGymEnv", (JunctionGymEnv, gym.Env), {})


def make_env(rank: int = 0, monitor_dir: str | None = None, **kwargs):
    """A closure that builds one environment, for `SubprocVecEnv([...])`.

    Each subprocess gets its own SUMO and its own Panda context. libsumo is a
    process-global singleton, which is why one environment per process is the
    unit here rather than one per thread.
    """
    def _init():
        env = gym_env_class()(**kwargs)
        if monitor_dir:
            from stable_baselines3.common.monitor import Monitor

            return Monitor(env, filename=f"{monitor_dir}/{rank}")
        return env

    return _init


def make_vec_env(scenarios: list, num_envs: int | None = None,
                 monitor_dir: str | None = None, **kwargs):
    """One environment per scenario, cycled if `num_envs` exceeds the list.

    Mixing junctions across the vector is deliberate: a batch that only ever holds
    one geometry lets the policy drift towards it between updates.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    num_envs = num_envs or len(scenarios)
    chosen = [scenarios[i % len(scenarios)] for i in range(num_envs)]
    # One seed per worker, as `env.vector.make_envs` does. Sharing one seed
    # across the vector is not a smaller experiment, it is sixteen copies of the
    # same episode: the traffic is seeded, so every worker would generate an
    # identical trajectory and the batch would carry one sample's worth of
    # information at sixteen times the cost.
    base_seed = kwargs.pop("seed", 7)
    # `spawn`, not the platform default `fork`, for the reason `env.vector` gives:
    # libsumo is a process-global singleton, and a forked worker inherits a
    # parent that has already touched it.
    return SubprocVecEnv([
        make_env(rank=i, monitor_dir=monitor_dir, seed=base_seed + i,
                 junction=s.junction, plan=s.plan, demand=s.demand, **kwargs)
        for i, s in enumerate(chosen)
    ], start_method="spawn")
