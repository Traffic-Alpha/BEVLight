'''The simulator-facing layer: SUMO, the renderers, and the junction itself.

Public -- other subpackages read these:
    sumo      TshubEnvironment wiring, and the timing that defines a decision
    render    Panda (in the loop) and Blender (offline) camera rigs
    episode   run_episode - one episode under one controller, callback-driven
    gym_env   JunctionEnv - the same world, step-driven, for a learner to own
    obs_spec  what a policy is allowed to see: scope, mode, window length
    rewards   the per-second costs a reward is built from, in one registry
    vector    RemoteEnv - episodes in worker processes, stepped in parallel
    wrapper   the gymnasium face and the vectorised factory (re-exported by
              `bevlight.rl`, which is the layer that uses it)

_internal/ -- only this package:
    recorder  writing a collected episode to disk

`collect`, `eval` and `rl` are peers on top of this and never import each
other: collection is this loop plus a recorder, evaluation is this loop plus a
scorer, and reinforcement learning is the same loop with the steps handed out
one at a time. `tests/test_gym_env.py` requires the callback and step-driven
paths to produce the same metrics, so a policy is never trained against a world
that differs from the one it is scored in.
'''
from .episode import EpisodeResult, run_episode
from .gym_env import JunctionEnv
from .obs_spec import ObsMode, ObsScope, ObsSpec
from .rewards import REWARDS
from .sumo import DECISION_INTERVAL_S, PANDA_VARIANT, YELLOW_TIME_S

__all__ = ["EpisodeResult", "run_episode", "JunctionEnv",
           "ObsSpec", "ObsScope", "ObsMode", "REWARDS",
           "DECISION_INTERVAL_S", "YELLOW_TIME_S", "PANDA_VARIANT"]
