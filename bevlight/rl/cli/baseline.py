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
        description="Train an off-the-shelf RL baseline and score it against max-pressure."
    )
    parser.add_argument("--algo", required=True, choices=sorted(ALGORITHMS),
                        help="Which published implementation to run.")
    parser.add_argument("--reward", default="visible_queue", choices=sorted(REWARDS),
                        help="Reward name from the registry both arms read.")
    parser.add_argument("--junction", required=True)
    parser.add_argument("--plan", default="normal")
    parser.add_argument("--demand", default="high_density")
    parser.add_argument("--observe", default="window", choices=("window", "full_lane"),
                        help="How far down the approach the policy may look.")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Training decisions. The SAC arm used 60000.")
    parser.add_argument("--episode-steps", type=int, default=None,
                        help="Simulated seconds per episode. Default: the scenario's own.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=5000,
                        help="Record a history entry every N decisions.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[7],
                        help="Seeds the trained policy and the baselines are both scored on.")
    parser.add_argument("--baseline", nargs="+", default=["max_pressure", "fixed_time"],
                        help="Rule-based controllers to pair against.")
    parser.add_argument("--run", default=None,
                        help="Run directory name. Default: <algo>_<reward>_<junction>.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the run directory, train nothing.")
    return parser.parse_args(argv)


def locate(junction: str, plan: str, demand: str):
    """The scenario as the active selection lists it, or an unlisted one.

    Which split a scenario belongs to is part of what a baseline row means -- a
    number measured on a held-out junction says something different from one
    measured on a training junction -- so it is read from
    `configs/scenario_selection.json` rather than assumed. A combination outside
    the active 64 is allowed, and says so.
    """
    from ...scenario.selection import Scenario, load_selection

    for candidate in load_selection().all():
        if (candidate.junction, candidate.plan, candidate.demand) == (junction, plan, demand):
            return candidate
    return Scenario(junction=junction, plan=plan, demand=demand, split="unlisted")


def main(argv=None) -> int:
    args = parse_args(argv)
    algorithm = resolve(args.algo)
    name = args.run or f"{args.algo}_{args.reward}_{args.junction}"
    run_dir = TRAIN_RUNS_ROOT / "baselines" / name

    print(f"[plan] {args.algo} on {args.junction}/{args.plan}_{args.demand} "
          f"reward={args.reward} observe={args.observe}")
    print(f"[plan] {args.steps} decisions, {args.num_envs} envs, "
          f"action masking {'on' if algorithm.masked else 'off'}")
    print(f"[plan] -> {run_dir}")
    if args.dry_run:
        return 0

    from ...env.wrapper import make_vec_env

    run_dir.mkdir(parents=True, exist_ok=True)
    scenario = locate(args.junction, args.plan, args.demand)
    # Monitor-wrapped so SB3 fills `ep_info_buffer`; without it the callback has
    # no episode returns to report and the run is a black box for an hour.
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    envs = make_vec_env(
        [scenario], num_envs=args.num_envs, seed=args.seed,
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

    print(f"[score] seeds {args.eval_seeds}, paired against {args.baseline}")
    policy_rows = [
        rollout(model, algorithm, args.junction, args.plan, args.demand, seed,
                args.reward, args.episode_steps, args.observe)
        for seed in args.eval_seeds
    ]
    references = {
        spec: [rollout_controller(spec, args.junction, args.plan, args.demand,
                                  seed, args.reward, args.episode_steps)
               for seed in args.eval_seeds]
        for spec in args.baseline
    }

    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    # `avg_travel_time_incl_unfinished_s` leads, and `stuck` is reported beside
    # it, because average travel time over *completed* trips is the one metric a
    # degenerate policy can win. Holding one phase forever gives the approach it
    # serves a clear road and short trips, while the other three never finish and
    # so never enter the average -- which is exactly what a barely-trained DQN
    # does here. `metrics.py` counts the stranded at their time-so-far for this
    # reason; the comparison has to read that column or it will report the
    # failure as a win.
    metrics = ("avg_travel_time_incl_unfinished_s", "avg_travel_time_s",
               "avg_waiting_time_s", "avg_queue_veh", "throughput", "unfinished",
               "switch_rate")
    result = {
        "algorithm": args.algo, "reward": args.reward, "observe": args.observe,
        "masked": algorithm.masked, "steps": args.steps, "seed": args.seed,
        "scenario": scenario.key, "split": scenario.split,
        "train_minutes": round(trained_s / 60, 2),
        "policy": {k: mean([r[k] for r in policy_rows]) for k in metrics},
    }
    for spec, rows in references.items():
        paired = {
            key: [p[key] - b[key] for p, b in zip(policy_rows, rows)]
            for key in ("avg_travel_time_incl_unfinished_s", "avg_travel_time_s")
        }
        result[spec] = {k: mean([r[k] for r in rows]) for k in metrics} | {
            "paired_delta_travel_incl": round(mean(paired["avg_travel_time_incl_unfinished_s"]), 3),
            "paired_delta_travel": round(mean(paired["avg_travel_time_s"]), 3),
            "per_seed_delta_incl": [round(d, 3) for d in paired["avg_travel_time_incl_unfinished_s"]],
            "wins": sum(1 for d in paired["avg_travel_time_incl_unfinished_s"] if d < 0),
        }

    header = (f"{'controller':<24} {'travel*':>8} {'travel':>8} {'wait':>8} "
              f"{'queue':>7} {'done':>6} {'stuck':>6} {'switch':>7}")
    print("\n" + header)
    rows = [(f"{args.algo}/{args.reward}", result["policy"])]
    rows += [(spec, result[spec]) for spec in references]
    for label, row in rows:
        print(f"{label:<24} {row['avg_travel_time_incl_unfinished_s']:8.2f} "
              f"{row['avg_travel_time_s']:8.2f} {row['avg_waiting_time_s']:8.2f} "
              f"{row['avg_queue_veh']:7.2f} {row['throughput']:6.0f} "
              f"{row['unfinished']:6.0f} {row['switch_rate']:7.2f}")
    print("travel* counts stranded vehicles at their time so far; travel does not.")

    print("\npaired difference in travel* (negative = the learner is ahead)")
    for spec in references:
        row = result[spec]
        print(f"  vs {spec:<20} {row['paired_delta_travel_incl']:+7.2f} s   "
              f"wins {row['wins']}/{len(args.eval_seeds)}")

    is_comparable, shortfall = comparability(
        result["policy"], {spec: result[spec] for spec in references}
    )
    result["throughput_shortfall"] = shortfall
    result["comparable"] = is_comparable
    if not is_comparable:
        print(f"\n[warning] the learner cleared {shortfall:.0%} fewer vehicles than the "
              f"best baseline.\n[warning] travel time is an average over trips that "
              f"finished, so a policy that\n[warning] strands traffic scores well on it. "
              f"This row is not comparable.")

    out = run_dir / "baseline.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[summary] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
