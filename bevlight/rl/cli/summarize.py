'''
@Author: WANG Maonan
@Date: 2026-08-30
@Description: The finished baseline runs, as one grid of algorithm against reward.

Reads what `rl baseline` already wrote -- no simulation, no model -- and puts
the cells beside each other, one row per (algorithm, reward, split).

Reports the median rather than the mean, because a single scenario where a
policy camps on one phase moves a mean by a hundred seconds and says nothing
about the other forty-four. And it carries `comparable` through: a cell that
cleared materially fewer vehicles than max-pressure did not lose a control
comparison, it declined to hold one, and reading its travel time as a result is
the mistake this column exists to prevent.
'''

from __future__ import annotations

import argparse
import json

from ...paths import TRAIN_RUNS_ROOT


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabulate the finished RL baseline runs."
    )
    parser.add_argument("--root", default=None,
                        help="Directory of run directories. "
                             "Default: runs/train/baselines.")
    parser.add_argument("--split", nargs="+", default=None,
                        help="Splits to show. Default: every split present.")
    parser.add_argument("--baseline", default="max_pressure",
                        help="Which rule-based controller the delta is against.")
    parser.add_argument("--out", default=None, help="Also write the rows as JSON.")
    return parser.parse_args(argv)


def load(root) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/baseline.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"[skip] {path} is not readable yet")
    return rows


def main(argv=None) -> int:
    from pathlib import Path

    args = parse_args(argv)
    root = Path(args.root or TRAIN_RUNS_ROOT / "baselines")
    runs = load(root)
    if not runs:
        raise SystemExit(f"No finished runs under {root}")

    splits = args.split or list(dict.fromkeys(
        name for run in runs for name in run.get("eval", {})
    ))
    print(f"{len(runs)} runs from {root}\n")
    for split in splits:
        print(f"=== {split}")
        print(f"  {'algorithm':<14} {'reward':<16} {'mask':>5} {'travel*':>9} "
              f"{'vs mp median':>13} {'range':>22} {'wins':>8} {'clears':>7}")
        rows = []
        for run in runs:
            block = run.get("eval", {}).get(split)
            if not block or args.baseline not in block:
                continue
            entry = block[args.baseline]
            rows.append((entry["median_delta_travel_incl"], run, block, entry))
        for _, run, block, entry in sorted(rows):
            shortfall = block.get("throughput_shortfall", 0.0)
            flag = "" if block.get("comparable") else "  <- not comparable"
            print(f"  {run['algorithm']:<14} {run['reward']:<16} "
                  f"{'yes' if run['masked'] else 'no':>5} "
                  f"{block['policy']['avg_travel_time_incl_unfinished_s']:9.2f} "
                  f"{entry['median_delta_travel_incl']:+13.2f} "
                  f"{'[' + format(entry['best_delta_travel_incl'], '+.1f') + ', ' + format(entry['worst_delta_travel_incl'], '+.1f') + ']':>22} "
                  f"{entry['wins']:>4}/{block['episodes']:<3} "
                  f"{1 - shortfall:>6.0%}{flag}")
        print()

    print("travel* counts stranded vehicles at their time so far.")
    print("clears = vehicles completed as a fraction of the best baseline's; a cell")
    print("below 95% metered traffic away rather than controlling it, and its")
    print("travel time is not a control result.")

    if args.out:
        Path(args.out).write_text(json.dumps(runs, indent=2))
        print(f"\n[summary] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
