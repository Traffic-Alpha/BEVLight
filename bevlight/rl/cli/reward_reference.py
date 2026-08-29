'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: What max-pressure costs on every scenario, so a mixed-scenario curve can be read.

A generalising run rotates through scenarios whose reward scales are not
comparable. Measured over the 91 active ones, at full episode length:

    visible_queue    -1.2664 .. -0.0511    24.8x
    full_wait       -32.8508 .. -0.4045    81.2x
    pressure         -0.5640 .. -0.0511    11.0x

So an episode return of -6 and one of -97 can be the same policy doing equally
well, and a pooled mean over them measures which scenarios happened to finish
recently. This writes the per-scenario, per-decision cost of max-pressure to
`runs/reports/reward_reference.json`, which `rl baseline` divides by to report a
curve where 1.0 is parity with max-pressure everywhere.

One reference serves every reward and every algorithm: `CostProbe` accumulates
all the registry's candidates in a single pass, so the cost is one episode per
scenario rather than one per (scenario, reward).
'''

from __future__ import annotations

import argparse
import json

from ...env.rewards import CANDIDATES
from ...paths import REPORTS_ROOT
from ...scenario.selection import SPLITS, load_selection
from ..preflight import rollout


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure max-pressure's per-decision cost on every scenario."
    )
    parser.add_argument("--split", nargs="+", default=None, choices=SPLITS,
                        help="Splits to measure. Default: all of them.")
    parser.add_argument("--controller", default="max_pressure",
                        help="The controller the reference is taken from.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=None,
                        help="Simulated seconds per episode. Default: the scenario's own.")
    parser.add_argument("--out", default=None,
                        help="Where to write. Default: runs/reports/reward_reference.json.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    selection = load_selection()
    scenarios = (
        [s for name in args.split for s in selection.split(name)]
        if args.split else list(selection.all())
    )
    out = args.out or (REPORTS_ROOT / "reward_reference.json")

    print(f"[plan] {len(scenarios)} scenarios under {args.controller}, seed {args.seed}")
    print(f"[plan] -> {out}")
    if args.dry_run:
        return 0

    table = {}
    for scenario in scenarios:
        row = rollout(scenario.junction, scenario.plan, scenario.demand,
                      args.controller, args.seed, args.steps)
        table[scenario.key] = {
            "split": scenario.split,
            "controller": args.controller,
            "decisions": row["decisions"],
            "per_decision": {name: row[f"reward_{name}"] for name in CANDIDATES},
        }
        print(f"  {scenario.key:<52} "
              + "  ".join(f"{n}={row[f'reward_{n}']:+9.4f}"
                          for n in ("visible_queue", "full_wait"))
              + f"  ({row['decisions']}d)", flush=True)

    from pathlib import Path

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))

    print()
    for name in CANDIDATES:
        values = [t["per_decision"][name] for t in table.values()]
        lo, hi = min(values), max(values)
        print(f"  {name:<20} min={lo:+10.4f} max={hi:+10.4f} "
              f"spread={abs(lo) / max(1e-9, abs(hi)):6.1f}x")
    print(f"\n[summary] {len(table)} scenarios -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
