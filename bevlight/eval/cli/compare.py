'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Run several controllers over the same scenarios and tabulate.

The command over `eval.compare`.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..compare import run, tabulate
from ...paths import REPORTS_ROOT
from ...scenario.selection import SPLITS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare traffic signal controllers.")
    parser.add_argument("--controller", nargs="+", default=["fixed_time", "max_pressure"],
                        help="Controllers to run, e.g. fixed_time:30 max_pressure. The first is the baseline.")
    parser.add_argument("--split", choices=list(SPLITS), default="train",
                        help="Which split to draw scenarios from. Default: train, so test scenarios stay untouched.")
    parser.add_argument("--junction", nargs="+", default=None, help="Restrict to these junctions.")
    parser.add_argument("--demand", nargs="+", default=None, help="Restrict to these demand patterns.")
    parser.add_argument("--seed", nargs="+", type=int, default=[7], help="SUMO seeds. Multiple seeds are averaged.")
    parser.add_argument("--steps", type=int, default=None, help="Episode length. Default: the scenario's num_seconds.")
    parser.add_argument("--out", default=None, help="Write the raw per-episode records here as JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without simulating.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = run(args)
    if not payload:
        return 0

    tabulate(payload["results"], args.controller)

    out = Path(args.out) if args.out else REPORTS_ROOT / "expert_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"\n[summary] {len(payload['results'])} episodes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
