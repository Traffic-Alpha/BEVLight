'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Named ablations: what to change, and what the row is evidence for.

`BEVLightConfig` has said "everything that changes between ablations" since it
was written, and the knobs were all there. What was missing is the part that
makes a table reproducible a month later: which combinations were run, under
which names, and what each one was supposed to demonstrate.

`why` is a required field on purpose. An ablation that removes a component and
reports the number is not yet evidence of anything -- the claim it supports has
to be stated in advance, or the row becomes whatever the result suggests
afterwards. If the `why` cannot be written, the row should not be run.

Three surfaces can be overridden, because that is where the design actually
lives:

    model   BEVLightConfig  -- architecture
    train   TrainConfig     -- objective and optimisation
    data    DecisionDataset -- what a sample is, e.g. how many frames

`rebuild_features` marks the ablations that act when the feature cache is built
rather than at training time. Those cannot be run against an existing dataset,
and saying so here is cheaper than a table row that silently reports the
un-ablated model.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ablation:
    """One row of the table."""

    name: str
    why: str
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    #: Acts on the cached features, so it needs its own dataset build.
    rebuild_features: bool = False

    def __post_init__(self):
        if not self.why.strip():
            raise ValueError(f"ablation {self.name!r} must say what it is evidence for")


_ROWS = (
    Ablation(
        "full", "The reference row. Every other row is read as a difference from this.",
    ),

    # --- temporal: the mechanism behind "video beats a snapshot" -------------
    Ablation(
        "frames1",
        "A single frame is a snapshot, which is what a detector-based baseline "
        "sees. If the multi-frame model is not better than this, the temporal "
        "argument is unsupported and the rest of the table is decoration.",
        data={"window": 1},
    ),
    Ablation(
        "frames2", "Where the temporal gain starts, if it is a trend being read.",
        data={"window": 2},
    ),
    Ablation(
        "frames3", "Whether the gain has saturated before the default of 5.",
        data={"window": 3},
    ),
    Ablation(
        "no_temporal",
        "Removes the temporal transformer while keeping five frames. Separates "
        "'more pixels' from 'reading how the queue is changing' -- frames1 "
        "removes both at once and cannot tell them apart.",
        model={"use_temporal": False},
    ),

    # --- the hierarchy ------------------------------------------------------
    Ablation(
        "no_lane_attention",
        "Lanes stop seeing each other before being combined into movements. "
        "Tests whether competition between approaches is read at the lane level.",
        model={"lane_attention": False},
    ),
    Ablation(
        "no_movement_attention",
        "Movements stop competing before phase pooling. Pressure is a relation "
        "between movements, so this is where that relation is claimed to form.",
        model={"movement_attention": False},
    ),
    Ablation(
        "deepsets",
        "Sum-then-transform instead of attention pooling over a phase's "
        "movements. Both are permutation-invariant and size-independent, so "
        "this isolates the pooling from the property that makes K variable.",
        model={"pooling": "deepsets"},
    ),
    Ablation(
        "no_phase_context",
        "Scores each candidate on its own demand alone, with no current phase "
        "and no time-in-phase. Asks whether the policy chooses a phase or only "
        "ranks demand -- switching cost is invisible without this input.",
        model={"use_phase_context": False},
    ),

    # --- what grounds the representation ------------------------------------
    Ablation(
        "no_aux",
        "Drops both auxiliary regressions. They are the only signal tying the "
        "trunk to per-lane physical state; without them nothing forces the "
        "features to mean anything beyond predicting the expert's argmax.",
        model={"aux_queue": False, "aux_occupancy": False},
        train={"queue_weight": 0.0, "occupancy_weight": 0.0},
    ),
    Ablation(
        "hard_pool",
        "Each patch goes to one lane instead of being shared by coverage. At "
        "11.36 px/m a 3.2 m lane spans 2.6 patches, so the edge patches are the "
        "disputed ones; this measures what dividing them costs.",
        rebuild_features=True,
    ),

    # --- the objective ------------------------------------------------------
    Ablation(
        "soft_targets",
        "Trains on the expert's per-candidate scores rather than its argmax. "
        "The margin between phases says which decisions mattered, where an "
        "argmax label says only what was chosen.",
        train={"soft_weight": 1.0},
    ),
)

#: Name -> Ablation. The name goes into the run directory and the run's config,
#: so it is part of the experiment record and does not get renamed.
ABLATIONS = {row.name: row for row in _ROWS}


def resolve(name: str) -> Ablation:
    if name not in ABLATIONS:
        raise ValueError(
            f"Unknown ablation '{name}'. Available: {sorted(ABLATIONS)}"
        )
    return ABLATIONS[name]


def describe() -> str:
    """The table as text, for `--list-ablations`."""
    width = max(len(n) for n in ABLATIONS)
    lines = []
    for name, row in ABLATIONS.items():
        overrides = {**row.model, **row.train, **row.data}
        setting = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "-"
        mark = "  (needs its own feature cache)" if row.rebuild_features else ""
        lines.append(f"  {name:{width}s}  {setting}{mark}")
    return "\n".join(lines)
