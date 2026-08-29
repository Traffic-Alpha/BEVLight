'''Controllers: junction state -> the phase to run next.

The same classes serve twice, as the expert that labels the training data and as
the rule-based baselines in the results tables, so they take a plain observation
object and never touch SUMO or the renderer directly.

The action space is `choose_next_phase`: a controller returns an index into the
junction's candidate phases, which is the same space the network scores. K comes
from the signal plan and varies by junction and by plan, so nothing here assumes
a fixed number of actions.

Expert state is restricted to the stretch of approach the BEV window actually
exposes (`LaneMask.visible_length_m`, about 60 m or 8 queued vehicles), which
keeps observation -> action well-defined for behaviour cloning. The same
extractor produces the lane labels, so the expert and the labels cannot disagree.
The `add/e2.add.xml` detectors are not used for this: they are still 45 m / 70 m
from the previous 90 m window. A full-lane variant is recorded alongside as the
"upper reference" expert.

Public:
    base        BaseController / Controller / SignalPlan

_internal/ -- reached only through the CONTROLLERS registry below, so adding a
controller is one file plus one line and never an import somewhere else:
    fixed_time, max_pressure, random_phase
'''

from ._internal.fixed_time import FixedTime
from ._internal.max_pressure import MaxPressure
from ._internal.random_phase import RandomPhase
from .base import BaseController, Controller, SignalPlan

CONTROLLERS = {
    FixedTime.name: FixedTime,
    MaxPressure.name: MaxPressure,
    RandomPhase.name: RandomPhase,
}

__all__ = [
    "CONTROLLERS",
    "BaseController",
    "Controller",
    "FixedTime",
    "MaxPressure",
    "RandomPhase",
    "SignalPlan",
]
