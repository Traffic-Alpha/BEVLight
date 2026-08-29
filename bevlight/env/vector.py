'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: Several junction environments, one process each.

Panda3D's `ShowBase` is a process-level singleton, so a second renderer in the
same process fails outright — parallel sampling here means parallel *processes*,
not threads. That is the same constraint TransSimHub's own RL benchmarks work
under, and it is why `SubprocVecEnv` rather than a thread pool is the shape.

The consequence is that each worker carries its own copy of the frozen backbone.
That is not the arrangement one would choose — a single shared backbone batching
across environments would be cheaper in principle — but it is cheap enough in
practice (~200 MB of the card each) and it recovers most of what batching would
have won anyway: a single batch-of-one forward leaves the GPU far from saturated,
so several issued concurrently overlap rather than queue.

What crosses the process boundary is pooled lane features, about 400 KB per
observation, not the 1680x1680 frames they came from. Sending pixels instead
would put ~5 MB per environment per step through a pipe, which is the one way to
make this slower than doing nothing.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations


def build_env(device=None, **config):
    """One `JunctionEnv`, with a feature extractor only if it renders.

    A structured-state environment (`render=False`) is a legitimate and useful
    configuration — it is what a privileged teacher trains in — and it must not
    be made to load the vision backbone to get there.
    """
    from .gym_env import JunctionEnv

    if config.get("render", True):
        from ..data.features import FeatureExtractor

        # Built here, never pickled: a CUDA module cannot cross a process boundary.
        config["extractor"] = FeatureExtractor(device=device)
    return JunctionEnv(**config)


def step_autoreset(env, action):
    """Step, and start the next episode when this one ends.

    The terminal observation is carried out in `info` rather than dropped. A
    learner that bootstraps a truncated episode needs `V(s_T)`, and `s_T` is the
    observation returned *with* `done` — which auto-reset would otherwise
    overwrite with the first observation of the next episode. Distinguishing
    truncation from termination is pointless if the state to bootstrap from was
    thrown away in transport.
    """
    observation, reward, done, info = env.step(action)
    if not done:
        return observation, reward, done, info
    info = {**info, "terminal_observation": observation,
            "episode_summary": env.summary()}
    observation, _, _, _ = env.reset()
    return observation, reward, done, info


def _worker(remote, config: dict) -> None:
    """One environment, serving reset/step/summary until told to close."""
    if config.get("render", True):
        import torch  # noqa: F401  - initialises CUDA inside this process

    env = build_env(device=config.pop("device", None), **config)
    try:
        while True:
            command, payload = remote.recv()
            if command == "reset":
                remote.send(env.reset())
            elif command == "step":
                observation, reward, done, info = step_autoreset(env, payload)
                summary = info.get("episode_summary") if done else None
                remote.send((observation, reward, done, info, summary))
            elif command == "close":
                remote.send(None)
                break
    finally:
        env.close()


class RemoteEnv:
    """A `JunctionEnv` living in another process, with the same interface.

    `step_async` / `step_wait` are split so a whole vector can be in flight at
    once; stepping them one after another would parallelise nothing.
    """

    def __init__(self, **config):
        import multiprocessing as mp

        # spawn, not fork: CUDA is already initialised in the parent, and a
        # forked child inherits a context it cannot use.
        context = mp.get_context("spawn")
        self.remote, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, config), daemon=True)
        self.process.start()
        child.close()
        self._summary = {}

    def reset(self):
        self.remote.send(("reset", None))
        return self.remote.recv()

    def step_async(self, action: int) -> None:
        self.remote.send(("step", int(action)))

    def step_wait(self):
        observation, reward, done, info, summary = self.remote.recv()
        if summary is not None:
            self._summary = summary
        return observation, reward, done, info

    def step(self, action: int):
        self.step_async(action)
        return self.step_wait()

    def summary(self) -> dict:
        return self._summary

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.remote.send(("close", None))
                self.remote.recv()
            except (EOFError, BrokenPipeError):
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()


class LocalEnv:
    """A `JunctionEnv` in this process, behind the same interface as `RemoteEnv`.

    The interface is the point. One environment does not need a process boundary,
    but it does need to end its episodes the same way N of them do: without this
    the single-environment configuration — the one a fast debugging loop uses —
    silently stops auto-resetting, and a learner tuned there behaves differently
    the moment it is scaled up.
    """

    def __init__(self, device=None, **config):
        self.env = build_env(device=device, **config)
        self._pending = None
        self._summary = {}

    def __getattr__(self, name):
        # signal_plan, junction, close, ... come from the environment itself.
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step_async(self, action: int) -> None:
        self._pending = step_autoreset(self.env, int(action))

    def step_wait(self):
        observation, reward, done, info = self._pending
        if done:
            self._summary = info.get("episode_summary", {})
        return observation, reward, done, info

    def step(self, action: int):
        self.step_async(action)
        return self.step_wait()

    def summary(self) -> dict:
        return self._summary

    def close(self) -> None:
        self.env.close()


def make_envs(num_envs: int, device=None, **config) -> list:
    """`num_envs` environments — in this process if one, in their own if more.

    One environment does not need the machinery, and paying a process boundary
    for it would only add latency to every step.
    """
    if num_envs == 1:
        return [LocalEnv(device=device, **config)]

    base_seed = config.pop("seed", 7)
    return [
        RemoteEnv(seed=base_seed + i, device=device, **config)
        for i in range(num_envs)
    ]
