'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: A set of finished ablation runs -> the comparison table.

Reads what the runs already wrote -- no simulation, no model -- and puts the
rows beside each other against a common reference. Paired by scenario, because
the scenario is the largest source of variance in these numbers: a mean over one
set of scenarios against a mean over a different set is not a difference between
two models.

The table reports the control metric, not the offline agreement. Agreeing with
the expert is not controlling well, and an ablation ranked on agreement would be
answering a question nobody asked.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...paths import TRAIN_RUNS_ROOT
from ..registry import ABLATIONS

METRICS = ("avg_travel_time_s", "avg_waiting_time_s", "avg_queue_veh", "throughput")


def load_run(run_dir: Path, split: str, policy: str | None) -> tuple[str | None, dict]:
    """(ablation name, {scenario: row}) for one run's chosen policy."""
    config = run_dir / "config.json"
    name = None
    if config.is_file():
        name = json.loads(config.read_text()).get("ablation")

    results = run_dir / f"closed_loop_{split}.json"
    if not results.is_file():
        return name, {}

    rows = json.loads(results.read_text())["results"]
    controllers = {r["controller"] for r in rows}
    chosen = policy or _best_checkpoint(controllers)
    if chosen is None:
        return name, {}
    return name, {r["scenario"]: r for r in rows if r["controller"] == chosen}


def _best_checkpoint(controllers: set[str]) -> str | None:
    """The last checkpoint present. Ablations are compared at equal budget."""
    checkpoints = sorted(c for c in controllers if c.startswith("checkpoint_"))
    return checkpoints[-1] if checkpoints else None


def compare(runs: dict[str, Path], split: str, policy: str | None,
            reference: str) -> dict:
    loaded = {}
    for label, run_dir in runs.items():
        name, rows = load_run(run_dir, split, policy)
        loaded[label] = {"ablation": name, "rows": rows,
                         "why": ABLATIONS[name].why if name in ABLATIONS else None}

    if reference not in loaded:
        raise SystemExit(f"reference run '{reference}' is not among {sorted(loaded)}")
    base = loaded[reference]["rows"]
    if not base:
        raise SystemExit(
            f"reference run '{reference}' has no closed-loop results for split "
            f"'{split}'. Run bevlight eval closed-loop on it first."
        )

    for label, entry in loaded.items():
        # Only the scenarios both ran: a row averaged over a different set of
        # junctions is not comparable to the reference, however close it looks.
        shared = sorted(set(entry["rows"]) & set(base))
        entry["scenarios"] = len(shared)
        entry["metrics"] = {
            metric: _mean([entry["rows"][s][metric] for s in shared])
            for metric in METRICS
        } if shared else {}
        entry["delta_travel_s"] = _mean([
            entry["rows"][s]["avg_travel_time_s"] - base[s]["avg_travel_time_s"]
            for s in shared
        ]) if shared else None
    return loaded


def _mean(values):
    return round(sum(values) / len(values), 3) if values else None


def tabulate(loaded: dict, reference: str) -> None:
    width = max(len(k) for k in loaded)
    print(f"\n{'run':{width}s} {'ablation':22s} {'n':>4s} {'travel':>9s} "
          f"{'wait':>8s} {'queue':>7s} {'thr':>6s} {'vs ref':>9s}")
    print("-" * (width + 70))
    for label, entry in loaded.items():
        m = entry["metrics"]
        if not m:
            print(f"{label:{width}s} {entry['ablation'] or '-'!s:22s} "
                  f"{'-':>4s}  no closed-loop results for this split")
            continue
        delta = entry["delta_travel_s"]
        mark = "reference" if label == reference else f"{delta:+.2f}s"
        print(f"{label:{width}s} {entry['ablation'] or 'full'!s:22s} "
              f"{entry['scenarios']:4d} {m['avg_travel_time_s']:9.2f} "
              f"{m['avg_waiting_time_s']:8.2f} {m['avg_queue_veh']:7.2f} "
              f"{m['throughput']:6.0f} {mark:>9s}")

    print("\nEach row removes or replaces one component; travel time is the metric that")
    print("counts. A row is evidence only for what its ablation was declared to test:")
    for label, entry in loaded.items():
        if entry["why"]:
            print(f"  {entry['ablation']}: {entry['why']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Put finished ablation runs beside each other.")
    parser.add_argument("--run", nargs="+", required=True,
                        help="Run directory names under runs/train/.")
    parser.add_argument("--reference", default=None,
                        help="The run every other is a difference from. "
                             "Default: the first --run.")
    parser.add_argument("--split", default="train",
                        help="Which closed_loop_<split>.json to read.")
    parser.add_argument("--policy", default=None,
                        help="Controller name in the results. Default: the last "
                             "checkpoint each run has.")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = {name: TRAIN_RUNS_ROOT / name for name in args.run}
    missing = [n for n, d in runs.items() if not d.is_dir()]
    if missing:
        raise SystemExit(f"no such run(s) under {TRAIN_RUNS_ROOT}: {missing}")

    reference = args.reference or args.run[0]
    loaded = compare(runs, args.split, args.policy, reference)
    tabulate(loaded, reference)

    if args.out:
        payload = {label: {k: v for k, v in entry.items() if k != "rows"}
                   for label, entry in loaded.items()}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\n[summary] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
