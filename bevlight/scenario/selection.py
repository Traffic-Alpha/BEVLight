'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Read configs/scenario_selection.json into split objects.

The full scenario pool has 120 (junction, plan, demand) combinations; the
reported experiments use the 64-scenario active subset. Every piece of
experiment code asks this module which scenarios it may touch, so no module
ever loops over `scenarios/*.sumocfg` and quietly picks up an inactive one.

    from bevlight.scenario.selection import load_selection

    sel = load_selection()
    sel.train                 # [Scenario, ...]  48
    sel.cross_plan_test       # [Scenario, ...]   8
    sel.cross_structure_test  # [Scenario, ...]   8
    sel.env_names("train")    # ["easy_low_density", ...] per scenario
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ..paths import SCENARIO_SELECTION

# `cross_demand_test` is derived rather than listed: it is the training
# junctions and plans under the two demands held out of training. The three
# generalization tiers are then one split each, in ascending difficulty —
# unseen demand, unseen signal plan, unseen junction.
SPLITS = ("train", "cross_demand_test", "cross_plan_test", "cross_structure_test")


@dataclass(frozen=True)
class Scenario:
    """One (junction, plan, demand) combination in the active set."""

    junction: str
    plan: str
    demand: str
    split: str

    @property
    def env_name(self) -> str:
        """Environment name as used by `loader.load_junction_config`."""
        return f"{self.plan}_{self.demand}"

    @property
    def key(self) -> str:
        """Stable identifier, safe as a directory name."""
        return f"{self.junction}__{self.plan}__{self.demand}"

    def __str__(self) -> str:
        return f"{self.junction}/{self.env_name}"


@dataclass(frozen=True)
class Selection:
    """The active scenario set, split three ways."""

    name: str
    train: tuple[Scenario, ...]
    cross_demand_test: tuple[Scenario, ...]
    cross_plan_test: tuple[Scenario, ...]
    cross_structure_test: tuple[Scenario, ...]
    meta: dict

    def split(self, name: str) -> tuple[Scenario, ...]:
        if name not in SPLITS:
            raise ValueError(f"Unknown split '{name}'. Available: {list(SPLITS)}")
        return getattr(self, name)

    def all(self) -> tuple[Scenario, ...]:
        return (self.train + self.cross_demand_test
                + self.cross_plan_test + self.cross_structure_test)

    def junctions(self, split: str | None = None) -> list[str]:
        """Junction names in a split, or in the whole active set."""
        scenarios = self.all() if split is None else self.split(split)
        seen: list[str] = []
        for scenario in scenarios:
            if scenario.junction not in seen:
                seen.append(scenario.junction)
        return seen

    def of_junction(self, junction: str) -> tuple[Scenario, ...]:
        return tuple(s for s in self.all() if s.junction == junction)

    @property
    def plan_aliases(self) -> dict[str, str]:
        """`{"easy": "plan_A", "normal": "plan_B"}` as used in the paper tables."""
        return dict(self.meta.get("plan_names", {}))


def _expand(entry: dict, split: str, plans_key: str) -> list[Scenario]:
    plans = entry[plans_key] if isinstance(entry.get(plans_key), list) else [entry[plans_key]]
    return [
        Scenario(junction=entry["junction"], plan=plan, demand=demand, split=split)
        for plan in plans
        for demand in entry["demands"]
    ]


@cache
def load_selection(path: Path | str | None = None) -> Selection:
    """Load the active scenario manifest. Cached: the file never changes mid-run."""
    manifest_path = Path(path) if path is not None else SCENARIO_SELECTION
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing scenario manifest: {manifest_path}")
    raw = json.loads(manifest_path.read_text())
    splits = raw["splits"]

    train = [s for e in splits["train"] for s in _expand(e, "train", "plans")]
    # Tier one: the junctions and plans that were trained on, under the demands
    # that were not. Same geometry, same signal plan, traffic never seen.
    test_demands = raw["active_set"]["test_demands"]
    cross_demand = [
        Scenario(junction=s.junction, plan=s.plan, demand=demand,
                 split="cross_demand_test")
        for s in {(x.junction, x.plan): x for x in train}.values()
        for demand in test_demands
    ]
    # Cross-plan entries name the held-out plan explicitly, not a list.
    cross_plan = [
        s for e in splits["cross_plan_test"] for s in _expand(e, "cross_plan_test", "test_plan")
    ]
    cross_structure = [
        s
        for e in splits["cross_structure_test"]
        for s in _expand(e, "cross_structure_test", "plans")
    ]

    selection = Selection(
        name=raw["name"],
        train=tuple(train),
        cross_demand_test=tuple(cross_demand),
        cross_plan_test=tuple(cross_plan),
        cross_structure_test=tuple(cross_structure),
        meta=raw.get("active_set", {}),
    )

    # The manifest counts the splits it declares. `cross_demand_test` is derived
    # from train x the held-out demands, so it is excluded from that count rather
    # than inflating it.
    expected = raw.get("active_set", {}).get("total_scenarios")
    declared = len(train) + len(cross_plan) + len(cross_structure)
    if expected is not None and expected != declared:
        raise ValueError(
            f"{manifest_path}: manifest declares {expected} active scenarios "
            f"but the splits expand to {declared}."
        )
    return selection
