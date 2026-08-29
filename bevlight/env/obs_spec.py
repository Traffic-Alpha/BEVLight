'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: What a policy is allowed to see, as a value rather than a convention.

Three things used to carry this between them -- `JunctionEnv(window=, render=,
observe=)`, a comment in `ObservationExtractor`, and a paragraph in the expert
package docstring -- which meant "the observation is what the BEV image shows"
was true by agreement rather than by construction. This is the agreement,
written down once and passed around.

## scope: how far down the approach a policy may look

`WINDOW` is the deployable setting and what every reported result uses: the
stretch of approach the BEV image actually exposes, about 60 m or 8 queued
vehicles, from `LaneMask.visible_length_m`. The expert reads it, the labels are
built from it, and a pixel policy is physically incapable of exceeding it -- so
observation -> action stays well-defined for behaviour cloning, and a structured
teacher is answering the same question the vision student is.

`FULL_LANE` is the control experiment and nothing else. It asks how much of the
gap to max-pressure is the learner and how much is the window. A result reported
under it is not a result about a deployable policy.

Scope does **not** govern the reward, which may read the whole lane and does --
see `env/rewards.py` for why that asymmetry is deliberate rather than an
oversight.

## mode: in what form

The same world, three encodings, chosen by what the consumer can pool:

    STATE     per-lane numbers, no renderer. What a privileged teacher trains
              on: seconds of SUMO per episode rather than minutes of Panda3D
    FRAMES    the raw BEV frames, pooled by the learner. One shared frozen
              backbone across a batch of environments
    FEATURES  per-lane vectors, pooled inside the environment. One backbone per
              worker process, and only ~400 KB crosses a process boundary

The backbone is frozen, so FRAMES and FEATURES produce identical numbers and
differ only in throughput -- which is why the choice is a deployment detail and
not a semantic one. STATE is a different observation, and the gap between it and
the other two is the measurement the distillation experiment exists to make.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObsScope(str, Enum):
    """How far down the approach a policy may look."""

    WINDOW = "window"
    FULL_LANE = "full_lane"


class ObsMode(str, Enum):
    """In what form the observation arrives."""

    STATE = "state"
    FRAMES = "frames"
    FEATURES = "features"


#: Present whatever the mode is. The junction's clock, and the per-lane numbers
#: -- the vision model ignores `lane_state`, the teacher consumes it, and one
#: environment producing both is what makes them the same world.
COMMON_KEYS = ("current_phase", "time_in_phase", "lane_state")

#: The auxiliary regression targets, free from the simulator at every step.
#: Keeping these losses alive during RL is what stops the trunk drifting into a
#: representation that no longer reads the junction.
TARGET_KEYS = ("queue_target", "occupancy_target", "queue_valid")

#: What each mode adds on top.
_PAYLOAD = {
    ObsMode.STATE: (),
    ObsMode.FRAMES: ("frames",),
    ObsMode.FEATURES: ("lane_features",),
}


def wiring_keys() -> tuple[str, ...]:
    """The junction's wiring and the three padding masks.

    Taken from `data.collate` rather than restated, because a fourth copy of
    this tuple is how the padding masks would eventually disagree about which
    lanes are real.
    """
    from ..data.collate import STRUCTURE_KEYS

    return STRUCTURE_KEYS


@dataclass(frozen=True)
class ObsSpec:
    """The contract between a world and a policy. Hashable, comparable, logged."""

    scope: ObsScope = ObsScope.WINDOW
    mode: ObsMode = ObsMode.FEATURES
    frames: int = 5
    embed_dim: int = 384

    def __post_init__(self):
        # Accept the plain strings a CLI and a run's config.json carry, so a
        # spec round-trips through JSON without the caller converting by hand.
        object.__setattr__(self, "scope", ObsScope(self.scope))
        object.__setattr__(self, "mode", ObsMode(self.mode))
        if self.frames < 1:
            raise ValueError(f"frames must be at least 1, got {self.frames}")

    @property
    def renders(self) -> bool:
        """Whether this spec needs a Panda3D context at all."""
        return self.mode is not ObsMode.STATE

    @property
    def deployable(self) -> bool:
        """Whether a real drone could produce this observation."""
        return self.scope is ObsScope.WINDOW

    def keys(self) -> tuple[str, ...]:
        """Every key an observation under this spec carries."""
        return wiring_keys() + COMMON_KEYS + TARGET_KEYS + _PAYLOAD[self.mode]

    def validate(self, observation: dict) -> None:
        """Raise if an observation does not carry what this spec promised."""
        missing = [k for k in self.keys() if k not in observation]
        if missing:
            raise ValueError(
                f"observation is missing {missing} required by {self}; "
                f"it carries {sorted(observation)}"
            )

    def as_dict(self) -> dict:
        """The form a run's config.json records."""
        return {"scope": self.scope.value, "mode": self.mode.value,
                "frames": self.frames, "embed_dim": self.embed_dim}


def matched(a: ObsSpec, b: ObsSpec) -> bool:
    """Whether two specs are comparable as experiments.

    Mode may differ -- STATE against FEATURES is exactly the distillation
    measurement -- but scope may not. Comparing a full-lane policy against a
    windowed one and reporting the difference as a method result is the mistake
    this exists to make nameable.
    """
    return a.scope is b.scope
