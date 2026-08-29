'''The ablation table, as a table rather than as a set of remembered flags.

Public:
    registry    named variants -> config overrides, and what each row is evidence for

cli/:
    summarize   a set of finished runs -> the comparison table

Every knob these turn already existed on `BEVLightConfig`, `TrainConfig` or
`DecisionDataset`; what did not exist was a name for each combination and a
record of what it is supposed to show. A row whose `why` cannot be written is a
row that should not be run.
'''

from .registry import ABLATIONS, Ablation, resolve

__all__ = ["ABLATIONS", "Ablation", "resolve"]
