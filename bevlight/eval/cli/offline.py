'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Score a trained checkpoint on cached features, before any simulation runs.

The command over `eval.offline`. Checkpoint resolution is the only thing here
that is not argument parsing: a bare name is looked up inside `--run`, the same
way `eval closed-loop` resolves one, so the two commands accept the same string.
'''

from __future__ import annotations

from pathlib import Path

from ..offline import checkpoints_of, evaluate, load_checkpoint, print_report


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Score trained checkpoints on cached features."
    )
    parser.add_argument("--run", default=None,
                        help="Run directory under runs/train/. Scores every checkpoint in it.")
    parser.add_argument("--checkpoint", nargs="+", default=None, help="Specific checkpoints.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name. Default: the one recorded in the run's config.")
    parser.add_argument("--holdout", nargs="+", default=None,
                        help="Episodes to score on. Default: the run's own validation split.")
    parser.add_argument("--on", default="valid", choices=["valid", "train", "all"],
                        help="Which half of the split to score.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import json

    from ...data.dataset import DecisionDataset
    from ...paths import TRAIN_RUNS_ROOT

    args = parse_args(argv)
    run_dir = TRAIN_RUNS_ROOT / args.run if args.run else None

    # A bare name is resolved against --run, the way eval_closed_loop does it.
    if args.checkpoint:
        paths = [
            run_dir / name if run_dir and not Path(name).is_absolute() and not Path(name).exists()
            else Path(name)
            for name in args.checkpoint
        ]
    else:
        paths = checkpoints_of(run_dir) if run_dir else []
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"No such checkpoint: {missing[0]}")
    if not paths:
        raise SystemExit("Nothing to score. Pass --run or --checkpoint.")

    dataset_name = args.dataset
    holdout = args.holdout
    if run_dir and (run_dir / "config.json").is_file():
        recorded = json.loads((run_dir / "config.json").read_text())
        dataset_name = dataset_name or recorded.get("dataset")
        holdout = holdout or recorded.get("holdout")
    if not dataset_name:
        raise SystemExit("No dataset recorded in the run; pass --dataset.")

    dataset = DecisionDataset(dataset_name)
    if args.on == "all" or not holdout:
        scored = dataset
        scope = "every sample"
    else:
        train_set, valid_set = dataset.split_by_episode(holdout)
        scored = valid_set if args.on == "valid" else train_set
        scope = f"{args.on} split ({len(holdout)} episodes held out)"

    print(f"[plan] dataset={dataset_name} scoring={len(scored)} decisions on {scope}")
    for path in paths:
        print(f"  - {path.name}")
    if args.dry_run:
        return 0
    if not len(scored):
        raise SystemExit(f"No decisions in the {args.on} split.")

    results = {}
    for path in paths:
        model = load_checkpoint(path, device=args.device)
        report = evaluate(model, scored, batch_size=args.batch_size, device=args.device)
        print_report(report, title=path.stem)
        results[path.stem] = report

    print(f"\n{'checkpoint':<20} {'acc':>7} {'switch':>8} {'top2':>7} {'macroF1':>8} {'queueMAE':>9}")
    for name, report in results.items():
        overall = report["overall"]
        switch = overall["accuracy_on_switch"]
        print(f"{name:<20} {overall['accuracy']:7.3f} "
              f"{'   n/a' if switch is None else f'{switch:8.3f}'} "
              f"{overall['top2']:7.3f} {overall['macro_f1']!s:>8} "
              f"{overall.get('queue_mae', float('nan')):9.3f}")
    reference = next(iter(results.values()))["references"]
    print("reference            " + "  ".join(f"{k}={v:.3f}" for k, v in reference.items()))
    print("\nOffline numbers shortlist checkpoints; they never select one. "
          "Rank on closed-loop control metrics.")

    out = Path(args.out) if args.out else (
        (run_dir or Path.cwd()) / "offline_eval.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[summary] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
