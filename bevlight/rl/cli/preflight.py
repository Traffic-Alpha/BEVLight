'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Does the reward rank controllers the way the control metrics do?

The command over `rl.preflight`. Run it before paying for a training run: a
reward that ranks controllers differently from the metric will train something
that scores well and drives badly.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...paths import REPORTS_ROOT
from ...scenario.selection import SPLITS
from ..preflight import DEFAULT_CONTROLLERS, run, tabulate


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the RL reward ranks controllers as the metrics do.")
    parser.add_argument("--controller", nargs="+", default=list(DEFAULT_CONTROLLERS),
                        help="Controllers spanning known-bad to known-good. At least three.")
    parser.add_argument("--split", choices=list(SPLITS), default="train")
    parser.add_argument("--junction", nargs="+", default=None)
    parser.add_argument("--demand", nargs="+", default=None)
    parser.add_argument("--seed", nargs="+", type=int, default=[7])
    parser.add_argument("--steps", type=int, default=None,
                        help="Episode length. Default: the scenario's num_seconds.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if len(args.controller) < 3:
        raise SystemExit("At least three controllers: two cannot separate a reward "
                         "that is right from one that is right by construction.")
    payload = run(args)
    if not payload:
        return 0
    tabulate(payload["report"], list(args.controller))
    out = Path(args.out) if args.out else REPORTS_ROOT / "reward_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[summary] {len(payload['results'])} episodes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
