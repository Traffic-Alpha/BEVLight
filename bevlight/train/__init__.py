'''Behaviour cloning with auxiliary lane-state grounding.

Public -- other subpackages read these:
    loop        the training loop, checkpointing, and the run directory
    losses      masked CE over the current candidate phases, the soft
                pressure target, and the masked auxiliary MSE

cli/:
    run         argument parsing and the split that a run trains against

Checkpoints are selected by closed-loop control metrics afterwards, never by
action accuracy -- `bevlight.eval` owns that. Reinforcement learning is not
here: it is a different learner on the same world, and lives in `bevlight.rl`.
'''

from .loop import TrainConfig, action_distribution, train
from .losses import bevlight_loss

__all__ = ["TrainConfig", "action_distribution", "bevlight_loss", "train"]
