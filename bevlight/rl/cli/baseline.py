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
from ..baselines import (
    ALGORITHMS,
    build,
    comparability,
    controller_rollout,
    progress_callback,
    resolve,
    resume,
    reward_reference,
    rollout,
)


def sample(scenarios: list, count: int | None) -> list:
    """An evenly spaced subset, or all of them.

    Taking the first N is a trap and was one: the splits are ordered by
    junction, junctions differ in phase count, and the first eight of `train`
    are all three-phase while `cross_plan_test` is entirely four-phase. A run
    truncated that way reported "generalises across demand, fails across plan"
    when what it had measured was "three phases work, four do not" -- the
    sampling had separated the two groups by the very thing under test.

    Evenly spaced keeps whatever mix the split has. It is still a subset, and
    any reported number should use all of them.
    """
    if not count or count >= len(scenarios):
        return list(scenarios)
    step = len(scenarios) / count
    return [scenarios[int(i * step)] for i in range(count)]


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
    parser.add_argument("--normalize-reward", action="store_true",
                        help="Divide each scenario's reward by max-pressure's cost "
                             "there, so scenarios weigh equally in the objective. "
                             "Measured by `bevlight rl reward-reference`; without "
                             "that table this is refused rather than silently skipped.")
    parser.add_argument("--observe", default="window", choices=("window", "full_lane"),
                        help="How far down the approach the policy may look.")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Training decisions, summed over workers.")
    parser.add_argument("--episode-steps", type=int, default=None,
                        help="Simulated seconds per episode. Default: the scenario's own.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=2000,
                        help="Record a history entry every N decisions. "
                             "Dense on purpose: this is what a figure gets drawn from.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[7],
                        help="Seeds the policy and the baselines are both scored on.")
    parser.add_argument("--eval-scenarios", type=int, default=None,
                        help="Score on an evenly spaced sample of this many "
                             "scenarios per split. Default, and what any "
                             "reported number should use: all of them.")
    parser.add_argument("--baseline", nargs="+", default=["max_pressure", "fixed_time"],
                        help="Rule-based controllers to pair against.")
    parser.add_argument("--resume", default=None,
                        help="Continue from a saved model.zip instead of starting "
                             "over. --steps is then the *additional* budget, and "
                             "the step counter and exploration schedule carry on.")
    parser.add_argument("--run", default=None,
                        help="Run directory name. Default: <algo>_<reward>_<train split>.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the scenario counts, train nothing.")
    return parser.parse_args(argv)


def mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


#: What the printed table shows. `avg_travel_time_incl_unfinished_s` leads
#: because average travel time over *completed* trips is the one metric a
#: degenerate policy can win: hold one phase and the approach it serves gets a
#: clear road while the other three never finish and so never enter the average.
#:
#: This is the display set, not the record. Every row keeps the whole of
#: `summary()` -- `queue_saturated_lane_seconds` says where the BEV window
#: actually bit, `phase_counts` says whether a policy is cycling or camping, and
#: neither can be recovered once dropped.
METRICS = ("avg_travel_time_incl_unfinished_s", "avg_travel_time_s",
           "avg_waiting_time_s", "avg_queue_veh", "throughput", "unfinished",
           "switch_rate")


def median(values) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def score(model, algorithm, scenarios, args) -> dict:
    """Every scenario kept, and the split summarised from it.

    The per-scenario rows are the record; the aggregate is a view of them. A
    mean over a split hides the thing worth knowing -- which junction the policy
    fails at, whether one scenario carries the whole difference, what the range
    is -- and `9-results.md` reports medians and ranges over scenarios, which
    cannot be recovered from a mean after the fact.

    Paired per (scenario, seed): the traffic realisation is the largest source
    of variance here, so the difference is taken before anything is averaged.
    """
    rows = []
    for scenario in scenarios:
        for seed in args.eval_seeds:
            policy = rollout(model, algorithm, scenario.junction, scenario.plan,
                             scenario.demand, seed, args.reward,
                             args.episode_steps, args.observe)
            entry = {
                "scenario": scenario.key, "junction": scenario.junction,
                "plan": scenario.plan, "demand": scenario.demand,
                "split": scenario.split, "seed": seed,
                "policy": policy,
            }
            for spec in args.baseline:
                reference = controller_rollout(spec, scenario.junction,
                                               scenario.plan, scenario.demand,
                                               seed, args.episode_steps)
                entry[spec] = reference
                entry[f"delta_{spec}"] = round(
                    policy["avg_travel_time_incl_unfinished_s"]
                    - reference["avg_travel_time_incl_unfinished_s"], 3
                )
            rows.append(entry)
            print(f"    {entry['scenario']:<48} "
                  + "  ".join(f"{spec} {entry['delta_' + spec]:+7.2f}"
                              for spec in args.baseline), flush=True)

    summary = {"scenarios": len(scenarios), "episodes": len(rows), "rows": rows,
               "policy": {k: mean([r["policy"][k] for r in rows])
                          for k in METRICS}}
    for spec in args.baseline:
        deltas = [r[f"delta_{spec}"] for r in rows]
        summary[spec] = {k: mean([r[spec][k] for r in rows]) for k in METRICS} | {
            "paired_delta_travel_incl": round(mean(deltas), 3),
            "median_delta_travel_incl": round(median(deltas), 3),
            "worst_delta_travel_incl": round(max(deltas), 3),
            "best_delta_travel_incl": round(min(deltas), 3),
            "wins": sum(1 for d in deltas if d < 0),
        }
    ok, shortfall = comparability(summary["policy"],
                                  {spec: summary[spec] for spec in args.baseline})
    summary["throughput_shortfall"] = shortfall
    summary["comparable"] = ok
    return summary


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
        entry = row[spec]
        print(f"  vs {spec:<20} mean {entry['paired_delta_travel_incl']:+7.2f} s  "
              f"median {entry['median_delta_travel_incl']:+7.2f} s  "
              f"[{entry['best_delta_travel_incl']:+.2f}, "
              f"{entry['worst_delta_travel_incl']:+.2f}]  "
              f"wins {entry['wins']}/{row['episodes']}")
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
    evaluation = {name: sample(list(selection.split(name)), args.eval_scenarios)
                  for name in eval_splits}

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
    scale = None
    if args.normalize_reward:
        scale = reward_reference(args.reward)
        missing = [s.key for s in training if s.key not in scale]
        if missing:
            raise SystemExit(
                f"--normalize-reward needs a reference for every training "
                f"scenario; {len(missing)} are missing (e.g. {missing[0]}). "
                f"Run `bevlight rl reward-reference` first."
            )
        print(f"[plan] rewards divided by max-pressure's cost per scenario "
              f"({len(scale)} measured)")
    envs = make_vec_env(
        training, num_envs=args.num_envs, seed=args.seed,
        monitor_dir=str(monitor_dir), reward_scale=scale,
        render=False, allow_any_scenario=True, reward=args.reward,
        observe=args.observe, num_seconds=args.episode_steps,
    )
    started = time.time()
    try:
        if args.resume:
            print(f"[train] continuing from {args.resume}")
            model = resume(algorithm, args.resume, envs, seed=args.seed)
        else:
            model = build(algorithm, envs, seed=args.seed)
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    reset_num_timesteps=not args.resume,
                    callback=progress_callback(run_dir, args.log_every, started,
                                              args.reward))
        model.save(run_dir / "model")
    finally:
        envs.close()
    trained_s = time.time() - started
    print(f"[train] {args.steps} decisions in {trained_s / 60:.1f} min "
          f"-> {run_dir / 'model.zip'}")

    result = {
        "algorithm": args.algo, "reward": args.reward, "observe": args.observe,
        "masked": algorithm.masked,
        "normalize_reward": bool(args.normalize_reward), "steps": args.steps, "seed": args.seed,
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
