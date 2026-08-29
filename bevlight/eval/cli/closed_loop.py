'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: The learned policy inside the SUMO loop, judged on control rather than agreement.

The command over `eval.closed_loop`. `eval summarize` reads what this writes.
'''

from __future__ import annotations

from pathlib import Path

from ..closed_loop import PolicyController, checkpoints_of


def parse_args(argv=None):
    import argparse

    from ...scenario.selection import SPLITS

    parser = argparse.ArgumentParser(
        description="Rank checkpoints by closed-loop control quality."
    )
    parser.add_argument("--run", default=None, help="Run directory under runs/train/.")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="Checkpoints to score. Default: every one in --run.")
    parser.add_argument("--baseline", nargs="+", default=["fixed_time", "max_pressure"],
                        help="Rule-based controllers to run alongside. The first is the reference.")
    parser.add_argument("--split", choices=list(SPLITS), default="train",
                        help="Which split to draw scenarios from. Default: train, so test scenarios stay untouched.")
    parser.add_argument("--junction", nargs="+", default=None)
    parser.add_argument("--demand", nargs="+", default=None)
    parser.add_argument("--seed", nargs="+", type=int, default=[7])
    parser.add_argument("--steps", type=int, default=None, help="Episode length. Default: the scenario's num_seconds.")
    parser.add_argument("--window", type=int, default=5, help="Frames the policy reads per decision.")
    parser.add_argument("--panda-sky", default="day")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import json
    import time

    from ...env import run_episode
    from ...paths import TRAIN_RUNS_ROOT
    from ..compare import build_controller, scenarios_for, tabulate

    args = parse_args(argv)
    run_dir = TRAIN_RUNS_ROOT / args.run if args.run else None
    checkpoints = checkpoints_of(run_dir, args.checkpoint) if run_dir else [
        Path(p) for p in (args.checkpoint or [])
    ]
    missing = [p for p in checkpoints if not p.is_file()]
    if missing:
        raise SystemExit(f"No such checkpoint: {missing[0]}")

    scenarios = scenarios_for(args.split, args.junction, args.demand)
    if not scenarios:
        raise SystemExit("No scenarios matched. Check --junction / --demand / --split.")

    # Baselines first: tabulate() reports every other controller against the
    # first one, and the reference is fixed time.
    labels = list(args.baseline) + [p.stem for p in checkpoints]
    episodes = len(scenarios) * len(args.seed) * len(labels)
    print(
        f"[plan] split={args.split} scenarios={len(scenarios)} seeds={args.seed} "
        f"controllers={labels} -> {episodes} episodes"
    )
    for scenario in scenarios:
        print(f"  - {scenario}  [{scenario.split}]")
    if not checkpoints:
        print("  note: no checkpoint given; only the baselines will run")
    if args.dry_run:
        return 0

    from ...data.features import FeatureExtractor
    from ..offline import load_checkpoint

    # The backbone and the checkpoints do not change between episodes. Loading
    # them once keeps the per-episode cost purely rendering and simulation.
    extractor = FeatureExtractor(device=args.device) if checkpoints else None
    models = {p.stem: load_checkpoint(p, device=args.device) for p in checkpoints}

    results = []
    started = time.perf_counter()
    for scenario in scenarios:
        for seed in args.seed:
            for label in labels:
                if label in args.baseline:
                    controller = build_controller(label)
                    extra = {}
                else:
                    path = next(p for p in checkpoints if p.stem == label)
                    controller = PolicyController(
                        path, scenario.junction, scenario.plan,
                        window=args.window, device=args.device,
                        extractor=extractor, model=models[label],
                    )
                    extra = {"render_panda_images": True, "panda_sky": args.panda_sky}

                result = run_episode(
                    junction=scenario.junction,
                    plan=scenario.plan,
                    demand=scenario.demand,
                    controller=controller,
                    seed=seed,
                    num_seconds=args.steps,
                    **extra,
                )
                record = {
                    "scenario": str(scenario),
                    "split": scenario.split,
                    "junction": scenario.junction,
                    "plan": scenario.plan,
                    "demand": scenario.demand,
                    "seed": seed,
                    "controller": label,
                    **result.metrics,
                }
                if isinstance(controller, PolicyController):
                    record["frames_encoded"] = controller.encoded
                results.append(record)
                print(
                    f"    {scenario.junction}/{scenario.plan}/{scenario.demand} "
                    f"seed={seed} {label:18s} "
                    f"travel={record['avg_travel_time_s']:7.1f}s "
                    f"wait={record['avg_waiting_time_s']:7.1f}s "
                    f"queue={record['avg_queue_veh']:5.1f} "
                    f"done={record['throughput']:4d} stuck={record['unfinished']:3d}",
                    flush=True,
                )

    tabulate(results, labels)
    print(f"\n[time] {episodes} episodes in {time.perf_counter() - started:.0f}s")

    out = Path(args.out) if args.out else (
        (run_dir or Path.cwd()) / f"closed_loop_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"[summary] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
