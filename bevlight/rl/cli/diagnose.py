'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Read a teacher run's history and say what state it is in.

A training curve is not self-explanatory: a flat return can be a converged policy
or a dead one, and rising entropy can be healthy exploration or a temperature
that has run away. Each check below pairs a number with the failure it would be
evidence of, so the output is a verdict rather than a plot to squint at.

The failure modes are the ones this problem actually has:

  * **degenerate switching.** "Always keep the current phase" and "switch every
    decision" are both reachable local optima that look fine in the return long
    before the queues show it. Max-pressure sits near 0.55 here; a policy at 0.00
    or 1.00 has stopped controlling.
  * **entropy at either wall.** At the top (log K) the policy is still uniform and
    nothing has been learned; at the bottom it has committed and stopped
    exploring. Which wall matters, because the fixes are opposite.
  * **critic divergence.** Q should approach the discounted return of the reward,
    which is bounded and negative here. A Q that grows without bound, or stays
    positive when every reward is negative, means the critic is not fitting.
  * **reward-metric divergence.** The return is the objective; travel time is the
    result. If the return improves while travel time does not, the reward is
    being optimised and control is not — which is the whole reason the preflight
    exists, caught here a second time in case it slipped through.
@LastEditTime: 2026-08-28
'''

from __future__ import annotations

import json
from pathlib import Path


def trend(values: list[float]) -> float:
    """Least-squares slope per entry. Reported as change over the whole run."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values)) / denominator


def diagnose(history: list[dict], num_phases: int | None = None) -> list[tuple[str, str, str]]:
    """-> [(check, reading, verdict)], most load-bearing first."""
    import math

    if not history:
        return [("history", "empty", "the run has not reached its first evaluation")]

    latest = history[-1]
    checks: list[tuple[str, str, str]] = []

    paired = [h["eval"]["max_pressure"]["paired_delta_travel"] for h in history]
    wins = latest["eval"]["max_pressure"]["wins"]
    seeds = len(latest["eval"]["max_pressure"]["per_seed_delta"])
    best = min(paired)
    checks.append((
        "vs max-pressure",
        f"latest {paired[-1]:+.2f}s, best {best:+.2f}s, {wins}/{seeds} seeds ahead",
        "AHEAD of the expert" if paired[-1] < 0 else
        "behind the expert; the gate has not been passed",
    ))
    checks.append((
        "  its direction",
        f"slope {trend(paired):+.3f}s per eval over {len(paired)} evals",
        "still improving" if trend(paired) < -0.01 else
        "flat or worsening — more steps will not help by themselves",
    ))

    returns = [h["return"] for h in history if h["return"] is not None]
    if returns:
        checks.append((
            "return",
            f"{returns[-1]:+.3f}, slope {trend(returns):+.4f}",
            "rising" if trend(returns) > 0 else "not rising — the objective is not being optimised",
        ))
        travel = [h["eval"]["policy"]["avg_travel_time_s"] for h in history]
        agree = (trend(returns) > 0) == (trend(travel) < 0)
        checks.append((
            "  reward vs metric",
            f"return slope {trend(returns):+.4f}, travel slope {trend(travel):+.4f}",
            "they agree" if agree else
            "THEY DISAGREE — the reward is improving while control is not",
        ))

    switch = latest["eval"]["policy"]["switch_rate"]
    checks.append((
        "switch rate",
        f"{switch:.2f} (max-pressure is near 0.55)",
        "degenerate: the policy has stopped switching" if switch < 0.05 else
        "degenerate: the policy switches at every decision" if switch > 0.98 else
        "in a plausible range",
    ))

    entropy = [h["entropy"] for h in history]
    ceiling = math.log(num_phases) if num_phases else None
    reading = f"{entropy[-1]:.3f}, slope {trend(entropy):+.4f}"
    if ceiling:
        reading += f" (uniform would be {ceiling:.3f})"
    checks.append((
        "entropy",
        reading,
        "collapsed — the policy has stopped exploring" if entropy[-1] < 0.02 else
        "still uniform — nothing has been committed to yet"
        if ceiling and entropy[-1] > 0.95 * ceiling else
        "between the walls",
    ))
    alpha = [h["alpha"] for h in history]
    checks.append((
        "  temperature",
        f"{alpha[-1]:.4f}, slope {trend(alpha):+.4f}",
        "running away — the entropy target is not being met" if alpha[-1] > 5 else
        "collapsed to zero; the entropy bonus is off" if alpha[-1] < 1e-3 else
        "being tuned normally",
    ))

    q = [h["q_mean"] for h in history]
    td = [h["td_error"] for h in history]
    checks.append((
        "critic",
        f"Q={q[-1]:+.3f} (slope {trend(q):+.4f}), TD error {td[-1]:.3f} (slope {trend(td):+.4f})",
        "Q is positive while every reward is negative — the critic is not fitting"
        if q[-1] > 0 else
        "TD error is growing — the critic is falling behind the policy"
        if trend(td) > 0.01 else "fitting",
    ))

    checks.append((
        "throughput",
        f"{latest['steps']} steps, {latest['episodes']} episodes, "
        f"{latest.get('steps_per_s', 0)}/s, {latest['elapsed_s'] / 60:.0f} min",
        "",
    ))
    return checks


def report(run_dir: Path, num_phases: int | None = None) -> None:
    history = json.loads((run_dir / "history.json").read_text())
    print(f"\n=== {run_dir.name}: {len(history)} evaluations ===\n")
    for check, reading, verdict in diagnose(history, num_phases):
        print(f"{check:20s} {reading:58s} {verdict}")

    print("\n--- paired travel time against max-pressure, per evaluation ---")
    print(f"{'steps':>8s} {'policy':>9s} {'MP':>9s} {'paired':>9s} {'seeds':>7s} "
          f"{'return':>9s} {'H':>7s} {'alpha':>7s} {'Q':>9s} {'switch':>7s}")
    for entry in history:
        evaluation = entry["eval"]
        print(f"{entry['steps']:>8d} {evaluation['policy']['avg_travel_time_s']:>9.2f} "
              f"{evaluation['max_pressure']['travel']:>9.2f} "
              f"{evaluation['max_pressure']['paired_delta_travel']:>+9.2f} "
              f"{evaluation['max_pressure']['wins']:>4d}/{len(evaluation['max_pressure']['per_seed_delta'])} "
              f"{entry['return'] if entry['return'] is not None else 0:>9.3f} "
              f"{entry['entropy']:>7.3f} {entry['alpha']:>7.3f} {entry['q_mean']:>+9.3f} "
              f"{evaluation['policy']['switch_rate']:>7.2f}")


def compare(runs: dict[str, Path]) -> None:
    """Two configurations, evaluation by evaluation, on the one number that decides.

    Printed side by side rather than as two tables, because the question is never
    "is this run good" — it is "did the change help", and that is a difference.
    """
    histories = {name: json.loads((path / "history.json").read_text())
                 for name, path in runs.items() if (path / "history.json").exists()}
    if not histories:
        print("no history yet")
        return

    names = list(histories)
    depth = max(len(h) for h in histories.values())
    print("\n=== paired travel time against max-pressure (negative = ahead) ===\n")
    header = f"{'eval':>5s}{'steps':>9s}"
    for name in names:
        header += f"{name[:22]:>24s}"
    print(header)
    print("-" * len(header))
    for i in range(depth):
        steps = next((h[i]["steps"] for h in histories.values() if len(h) > i), 0)
        line = f"{i + 1:>5d}{steps:>9d}"
        for name in names:
            history = histories[name]
            if len(history) > i:
                entry = history[i]
                line += (f"{entry['eval']['max_pressure']['paired_delta_travel']:>+11.2f}s"
                         f"  Q={entry['q_mean']:>+7.2f}")
            else:
                line += f"{'':>24s}"
        print(line)

    print()
    for name in names:
        values = [e["eval"]["max_pressure"]["paired_delta_travel"] for e in histories[name]]
        print(f"{name:24s} best {min(values):+.2f}s   latest {values[-1]:+.2f}s   "
              f"slope {trend(values):+.3f}s/eval")


def main(argv=None) -> int:
    import argparse

    from ...utils.paths import TRAIN_RUNS_ROOT

    parser = argparse.ArgumentParser(description="Diagnose a teacher run.")
    parser.add_argument("--run", default="gate1_ymt_high")
    parser.add_argument("--num-phases", type=int, default=None,
                        help="Candidate count, to place the entropy against its ceiling.")
    parser.add_argument("--against", nargs="+", default=None,
                        help="Other runs to put beside this one, evaluation by evaluation.")
    args = parser.parse_args(argv)
    report(TRAIN_RUNS_ROOT / args.run, args.num_phases)
    if args.against:
        compare({name: TRAIN_RUNS_ROOT / name
                 for name in [args.run, *args.against]})
    return 0
