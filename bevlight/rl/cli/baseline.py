'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Train one off-the-shelf RL algorithm on a junction, and score it against max-pressure.

The command over `rl.baselines`. One row of the baseline table per invocation:
an algorithm, a reward, a scenario, and the paired comparison against the
rule-based controllers on the same seeds.

Paired, not averaged separately. The demand realisation is the largest source of
variance in these numbers, so the difference is taken per seed before averaging;
without that a 1-3% gap sits under a noise floor measured at 1.2 s.
'''

from __future__ import annotations

import argparse
import json
import time

from ...env.rewards import REWARDS
from ...paths import TRAIN_RUNS_ROOT
from ...scenario.selection import SPLITS, load_selection
from .._internal.rollout import rollout_controller
from ..baselines import (
    ALGORITHMS,
    build,
    comparability,
    progress_callback,
    resolve,
    rollout,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an off-the-shelf RL baseline and score it against the rule-based controllers."
    )
    parser.add_argument("--algo", required=True, choices=sorted(ALGORITHMS),
                        help="Which published implementation to run.")
    parser.add_argument("--reward", default="visible_queue", choices=sorted(REWARDS),
                        help="Reward name from the registry every arm reads.")
    parser.add_argument("--train-split", default="train", choices=SPLITS,
                        help="Scenarios the workers rotate through while training.")
    parser.add_argument("--eval-split", nargs="+", default=None, choices=SPLITS,
                        help="Splits to score on. Default: the training split plus "
                             "every held-out one, which is the comparison that matters.")
    parser.add_argument("--junction", nargs="+", default=None,
                        help="Restrict the training scenarios to these junctions.")
    parser.add_argument("--observe", default="window", choices=("window", "full_lane"),
                        help="How far down the approach the policy may look.")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Training decisions, summed over workers.")
    parser.add_argument("--episode-steps", type=int, default=None,
                        help="Simulated seconds per episode. Default: the scenario's own.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=5000,
                        help="Record a history entry every N decisions.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[7],
                        help="Seeds the policy and the baselines are both scored on.")
    parser.add_argument("--eval-scenarios", type=int, default=None,
                        help="Score on at most this many scenarios per split. "
                             "Default: all of them.")
    parser.add_argument("--baseline", nargs="+", default=["max_pressure", "fixed_time"],
                        help="Rule-based controllers to pair against.")
    parser.add_argument("--run", default=None,
                        help="Run directory name. Default: <algo>_<reward>_<train split>.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the scenario counts, train nothing.")
    return parser.parse_args(argv)


def mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


#: What a row reports. `avg_travel_time_incl_unfinished_s` leads because average
#: travel time over *completed* trips is the one metric a degenerate policy can
#: win: hold one phase and the approach it serves gets a clear road while the
#: other three never finish and so never enter the average.
METRICS = ("avg_travel_time_incl_unfinished_s", "avg_travel_time_s",
           "avg_waiting_time_s", "avg_queue_veh", "throughput", "unfinished",
           "switch_rate")


def score(model, algorithm, scenarios, args) -> dict:
    """The policy and every rule-based baseline, over the same scenarios and seeds.

    Paired: the difference is taken per (scenario, seed) before averaging,
    because the traffic realisation is the largest source of variance here and
    averaging the two arms separately would bury a real difference under it.
    """
    pairs = [(s, seed) for s in scenarios for seed in args.eval_seeds]
    policy_rows = [
        rollout(model, algorithm, s.junction, s.plan, s.demand, seed,
                args.reward, args.episode_steps, args.observe)
        for s, seed in pairs
    ]
    row = {"scenarios": len(scenarios), "episodes": len(pairs),
           "policy": {k: mean([r[k] for r in policy_rows]) for k in METRICS}}
    for spec in args.baseline:
        reference = [
            rollout_controller(spec, s.junction, s.plan, s.demand, seed,
                               args.reward, args.episode_steps)
            for s, seed in pairs
        ]
        deltas = [p["avg_travel_time_incl_unfinished_s"]
                  - b["avg_travel_time_incl_unfinished_s"]
                  for p, b in zip(policy_rows, reference)]
        row[spec] = {k: mean([r[k] for r in reference]) for k in METRICS} | {
            "paired_delta_travel_incl": round(mean(deltas), 3),
            "wins": sum(1 for d in deltas if d < 0),
        }
    ok, shortfall = comparability(row["policy"],
                                  {spec: row[spec] for spec in args.baseline})
    row["throughput_shortfall"] = shortfall
    row["comparable"] = ok
    return row


def report(name: str, row: dict, args) -> None:
    print(f"\n=== {name}: {row['scenarios']} scenarios, {row['episodes']} episodes")
    print(f"{'controller':<24} {'travel*':>8} {'travel':>8} {'wait':>8} "
          f"{'queue':>7} {'done':>6} {'stuck':>6} {'switch':>7}")
    rows = [(f"{args.algo}/{args.reward}", row["policy"])]
    rows += [(spec, row[spec]) for spec in args.baseline]
    for label, values in rows:
        print(f"{label:<24} {values['avg_travel_time_incl_unfinished_s']:8.2f} "
              f"{values['avg_travel_time_s']:8.2f} {values['avg_waiting_time_s']:8.2f} "
              f"{values['avg_queue_veh']:7.2f} {values['throughput']:6.1f} "
              f"{values['unfinished']:6.1f} {values['switch_rate']:7.2f}")
    for spec in args.baseline:
        print(f"  vs {spec:<20} {row[spec]['paired_delta_travel_incl']:+7.2f} s   "
              f"wins {row[spec]['wins']}/{row['episodes']}")
    if not row["comparable"]:
        print(f"  [warning] cleared {row['throughput_shortfall']:.0%} fewer vehicles "
              f"than the best baseline; travel time is not comparable here.")


def main(argv=None) -> int:
    args = parse_args(argv)
    algorithm = resolve(args.algo)
    selection = load_selection()

    training = list(selection.split(args.train_split))
    if args.junction:
        training = [s for s in training if s.junction in set(args.junction)]
    if not training:
        raise SystemExit("No training scenarios matched.")
    eval_splits = args.eval_split or list(SPLITS)
    evaluation = {
        name: (list(selection.split(name))[: args.eval_scenarios]
               if args.eval_scenarios else list(selection.split(name)))
        for name in eval_splits
    }

    name = args.run or f"{args.algo}_{args.reward}_{args.train_split}"
    run_dir = TRAIN_RUNS_ROOT / "baselines" / name

    print(f"[plan] {args.algo} reward={args.reward} observe={args.observe}")
    print(f"[plan] train on {len(training)} scenarios from '{args.train_split}' "
          f"({len({s.junction for s in training})} junctions)")
    for split, rows in evaluation.items():
        print(f"[plan] score on {len(rows):>2} scenarios from '{split}'")
    print(f"[plan] {args.steps} decisions, {args.num_envs} envs, "
          f"action masking {'on' if algorithm.masked else 'off'}")
    print(f"[plan] -> {run_dir}")
    if args.dry_run:
        return 0

    from ...env.wrapper import make_vec_env

    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    envs = make_vec_env(
        training, num_envs=args.num_envs, seed=args.seed,
        monitor_dir=str(monitor_dir),
        render=False, allow_any_scenario=True, reward=args.reward,
        observe=args.observe, num_seconds=args.episode_steps,
    )
    started = time.time()
    try:
        model = build(algorithm, envs, seed=args.seed)
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=progress_callback(run_dir, args.log_every, started))
        model.save(run_dir / "model")
    finally:
        envs.close()
    trained_s = time.time() - started
    print(f"[train] {args.steps} decisions in {trained_s / 60:.1f} min "
          f"-> {run_dir / 'model.zip'}")

    result = {
        "algorithm": args.algo, "reward": args.reward, "observe": args.observe,
        "masked": algorithm.masked, "steps": args.steps, "seed": args.seed,
        "train_split": args.train_split, "train_scenarios": len(training),
        "train_junctions": sorted({s.junction for s in training}),
        "train_minutes": round(trained_s / 60, 2), "eval": {},
    }
    for split, rows in evaluation.items():
        print(f"\n[score] {split}: {len(rows)} scenarios x {len(args.eval_seeds)} seeds")
        result["eval"][split] = score(model, algorithm, rows, args)
        report(split, result["eval"][split], args)
    print("\ntravel* counts stranded vehicles at their time so far; travel does not.")

    out = run_dir / "baseline.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[summary] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
