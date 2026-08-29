'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Roll one run's per-split closed-loop results into a single table.

Reads what `eval closed-loop` already wrote; runs no simulation and loads no
model.
'''

from __future__ import annotations

from ..closed_loop import MIN_HEADROOM_S, summarize_run


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from ...paths import TRAIN_RUNS_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--policy", required=True,
                        help="Controller name as it appears in the results, e.g. checkpoint_060.")
    parser.add_argument("--expert", default="max_pressure")
    parser.add_argument("--weak", default="fixed_time")
    args = parser.parse_args(argv)

    run_dir = TRAIN_RUNS_ROOT / args.run
    summary = summarize_run(run_dir, args.policy, args.expert, args.weak)
    if not summary:
        raise SystemExit(f"no closed_loop_<split>.json under {run_dir}")

    print(f"{'split':22s} {'n':>3s} {'medΔ':>7s} {'meanΔ':>7s} {'worst':>7s} "
          f"{'medGain':>8s} {'vsWeak':>8s} {'unfin':>6s}")
    for split, s in summary.items():
        gain = f"{s['median_gain_pct']:7.0f}%" if s["median_gain_pct"] is not None else "      -"
        print(f"{split:22s} {s['n']:3d} {s['median_delta_s']:+7.2f} {s['mean_delta_s']:+7.2f} "
              f"{s['worst_delta_s']:+7.2f} {gain} {s['mean_gain_vs_weak_s']:+7.1f}s {s['unfinished']:6d}")
    target = run_dir / "closed_loop_summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  gain% only where {args.expert} beats {args.weak} by >{MIN_HEADROOM_S}s")
    print(f"  -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
