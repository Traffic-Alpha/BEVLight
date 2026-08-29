'''Reinforcement learning: a different learner on the same world.

`env` is the world and stays below this; nothing here is imported by `env`,
`expert`, `data` or `model`, so the dependency arrow only ever points this way.
That is what lets a new algorithm be added without touching the simulator.

Public -- other subpackages read these:
    sac         discrete SAC. Today it trains the privileged teacher; the
                observation is the BEV window either way, so the same trainer
                takes the pixel policy when that is what is being asked
    preflight   does a reward rank controllers the way the control metrics do,
                measured in minutes instead of by training and finding out

cli/:
    deviation   does deviating from max-pressure at any single decision help
    diagnose    read a teacher run back: entropy, Q spread, evaluation curve
    preflight   the command over `preflight`
    sac         the command over `sac`

The gymnasium face of the world is re-exported here rather than in `env`,
because this is the layer that has a use for it: `JunctionGymEnv` and
`make_vec_env` put an `env.JunctionEnv` behind the API that RL libraries and
`SubprocVecEnv` expect.
'''

from ..env.wrapper import JunctionGymEnv, make_env, make_vec_env

__all__ = ["JunctionGymEnv", "make_env", "make_vec_env"]
