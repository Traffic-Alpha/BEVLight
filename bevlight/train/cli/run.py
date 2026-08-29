'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: CLI over the training loop.

The held-out episodes are named explicitly rather than sampled, because the split
has to be by episode: consecutive frames are a second apart and nearly identical,
so a random split would put near-duplicates on both sides and report a score that
means nothing.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import argparse

from ...ablation.registry import describe, resolve
from ...data.dataset import DecisionDataset
from ...model.bevlight import BEVLightConfig
from ...paths import TRAIN_RUNS_ROOT
from ..loop import TrainConfig, action_distribution, train


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behaviour-clone the expert.")
    parser.add_argument("--dataset", default="pinganli_pilot")
    parser.add_argument("--run", default=None, help="Run name under runs/train/. Default: the dataset name.")
    parser.add_argument("--holdout", nargs="+", default=None,
                        help="Episodes to hold out. Default: the last one.")
    parser.add_argument("--holdout-demand", nargs="+", default=None,
                        help="Hold out episodes of these demands (combined with --holdout-junction).")
    parser.add_argument("--holdout-junction", nargs="+", default=None,
                        help="Hold out episodes of these junctions (combined with --holdout-demand).")
    parser.add_argument("--ablation", default=None,
                        help="Named variant from bevlight.ablation. Overrides the "
                             "flags it owns; --list-ablations prints the table.")
    parser.add_argument("--list-ablations", action="store_true",
                        help="Print the ablation table and exit.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--no-temporal", action="store_true", help="Single-frame ablation.")
    parser.add_argument("--pooling", default="attention", choices=["attention", "deepsets"])
    parser.add_argument("--queue-weight", type=float, default=0.2)
    parser.add_argument("--occupancy-weight", type=float, default=0.2)
    parser.add_argument("--soft-weight", type=float, default=0.0,
                        help="Weight on the expert's per-candidate scores. 0 keeps pure one-hot.")
    parser.add_argument("--soft-temperature", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile. A step is launch-bound, so this costs ~1.6x.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_holdout(dataset, args) -> list[str]:
    """Which episodes the validation half is made of.

    Named explicitly, or by (junction, demand), never sampled: the split has to
    be by episode, and it also has to be the same one next run so two runs stay
    comparable. This is a *monitor* — checkpoints are selected on closed-loop
    control metrics, not on the loss it reports.
    """
    if args.holdout:
        return args.holdout
    if not (args.holdout_demand or args.holdout_junction):
        return [dataset.episode_names[-1]]

    episodes = dataset.index["episodes"]
    return [
        name for name in dataset.episode_names
        if (not args.holdout_demand or episodes[name]["demand"] in args.holdout_demand)
        and (not args.holdout_junction or episodes[name]["junction"] in args.holdout_junction)
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_ablations:
        print("ablations:")
        print(describe())
        return 0

    ablation = resolve(args.ablation) if args.ablation else None
    if ablation is not None and ablation.rebuild_features:
        raise SystemExit(
            f"'{ablation.name}' acts when the feature cache is built, not at "
            f"training time. Build a dataset with it first, then train on that "
            f"dataset without --ablation; training here would silently report "
            f"the un-ablated model."
        )

    dataset = DecisionDataset(
        args.dataset, **(ablation.data if ablation else {})
    )
    if not len(dataset):
        raise SystemExit(f"No decision samples in '{args.dataset}'. Render some frames first.")

    holdout = resolve_holdout(dataset, args)
    train_set, valid_set = dataset.split_by_episode(holdout)
    if not len(train_set) or not len(valid_set):
        raise SystemExit(
            f"Holdout leaves train={len(train_set)} valid={len(valid_set)}; "
            f"check --holdout / --holdout-demand / --holdout-junction."
        )

    print(
        f"[plan] dataset={args.dataset} samples={len(dataset)} "
        f"train={len(train_set)} valid={len(valid_set)}"
    )
    print(f"        held out: {holdout}")
    stats = action_distribution(train_set.samples)
    print(
        f"        expert keeps the current phase {stats['keep_current_phase']:.0%} "
        f"of the time; per-phase counts {stats['per_phase']}"
    )
    if stats["keep_current_phase"] > 0.7:
        print("        warning: behaviour cloning may collapse onto 'always keep'")
    if args.dry_run:
        return 0

    # The ablation owns the flags it names; anything it is silent about still
    # comes from the command line. The name goes in the run directory and in the
    # run's meta, so a finished run says which row of the table it is.
    name = args.run or args.dataset
    run_dir = TRAIN_RUNS_ROOT / (f"{name}__{ablation.name}" if ablation else name)
    model_kwargs = {
        "model_dim": args.model_dim,
        "pooling": args.pooling,
        "use_temporal": not args.no_temporal,
    }
    train_kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "queue_weight": args.queue_weight,
        "occupancy_weight": args.occupancy_weight,
        "soft_weight": args.soft_weight,
        "soft_temperature": args.soft_temperature,
        "compile_model": not args.no_compile,
    }
    if ablation:
        # Last word, rather than a second value for the same keyword: several
        # rows set a flag that also has its own command-line switch, and a run
        # named after a row has to be that row.
        model_kwargs.update(ablation.model)
        train_kwargs.update(ablation.train)
    model_config = BEVLightConfig(**model_kwargs)
    train_config = TrainConfig(**train_kwargs)
    result = train(
        train_set,
        valid_set,
        model_config,
        train_config,
        run_dir=run_dir,
        device=args.device,
        meta={"dataset": args.dataset, "holdout": holdout,
              "ablation": ablation.name if ablation else None,
              "ablation_why": ablation.why if ablation else None},
    )
    best = min(result["history"], key=lambda h: h["valid"]["ce"])
    print(
        f"\n[summary] {result['elapsed_s']}s; best valid ce={best['valid']['ce']:.4f} "
        f"at epoch {best['epoch']} (acc {best['valid']['accuracy']:.3f})"
    )
    print("          checkpoints are candidates only; select on closed-loop metrics")
    print(f"          -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
