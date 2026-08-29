'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Simulate every rule-based baseline episode once, before the grid needs them.

Each cell of the baseline grid is scored against the same `max_pressure` and
`fixed_time` on the same scenarios and seeds, and an episode is sixteen seconds
of SUMO whoever is driving. Twelve cells over 91 scenarios would be two thousand
episodes; 182 distinct ones exist. `rl.baselines.controller_rollout` memoises
them, so this fills that cache up front -- embarrassingly parallel work that can
run while a cell trains, rather than serially inside the first cell that needs it.

Safe to run twice, and safe to run beside a training cell: a hit costs a file
read, and the write is atomic.
'''

from __future__ import annotations

import argparse
import time

from ...scenario.selection import SPLITS, load_selection
from ..baselines import controller_rollout


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute the rule-based baseline episodes the grid will need."
    )
    parser.add_argument("--split", nargs="+", default=None, choices=SPLITS,
                        help="Splits to cover. Default: all of them.")
    parser.add_argument("--controller", nargs="+",
                        default=["max_pressure", "fixed_time"])
    parser.add_argument("--seed", type=int, nargs="+", default=[7])
    parser.add_argument("--steps", type=int, default=None,
                        help="Simulated seconds per episode. Must match what the "
                             "cells will ask for, or they will miss the cache.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    selection = load_selection()
    scenarios = (
        [s for name in args.split for s in selection.split(name)]
        if args.split else list(selection.all())
    )
    jobs = [(spec, s, seed)
            for spec in args.controller for s in scenarios for seed in args.seed]

    print(f"[plan] {len(jobs)} episodes = {len(args.controller)} controllers "
          f"x {len(scenarios)} scenarios x {len(args.seed)} seeds")
    if args.dry_run:
        return 0

    started = time.time()
    for index, (spec, scenario, seed) in enumerate(jobs, start=1):
        controller_rollout(spec, scenario.junction, scenario.plan,
                           scenario.demand, seed, args.steps)
        elapsed = time.time() - started
        print(f"  [{index:>4}/{len(jobs)}] {spec:<14} {scenario.key:<48} "
              f"{elapsed / 60:5.1f} min", flush=True)
    print(f"\n[summary] {len(jobs)} episodes cached in "
          f"{(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
