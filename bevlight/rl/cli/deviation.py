'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Is there a single decision where deviating from max-pressure helps?

The question this project keeps running into is whether max-pressure has any
headroom above it at all. A learner failing to beat it is weak evidence, because
a learner can fail for its own reasons. This is direct: take max-pressure's
trajectory, change exactly one decision, let max-pressure carry on from there,
and measure the whole episode.

    Q(i, a) = the episode's total travel time when decision i is forced to a
              and every other decision is max-pressure's own

That is exactly the quantity a one-step rollout needs, and here it is computed
by *simulation from the start* rather than by forking a running one. Forking was
tried first and abandoned: `loadState` leaves TraCI subscriptions and tshub's
Python-side bookkeeping behind, and after restoring all of it the round trip
still moved a 300 s rollout's cost by ~3%. Branch costs differ by about the same
amount, so the search chose on noise and came out *worse* than the policy it was
improving on — which a one-step rollout cannot do, and which is therefore proof
the fork was wrong rather than that max-pressure is optimal.

Replaying from the start costs more simulation and buys two things worth more
than the saving. It is exact — the route files list every vehicle explicitly, so
an episode is a deterministic function of its action sequence and nothing has to
be restored. And every one of the T x (K-1) deviations is an independent
episode, so the whole scan is embarrassingly parallel across cores, which is
where this machine has room.

**Reading the result.** If no single deviation improves the episode, max-pressure
is a local optimum in a strong sense: not "no learner found anything", but "there
is nothing to find one decision at a time". If some do, the same scan run
repeatedly — fix the best deviation, scan again — is a constructive improvement
and its total gain is a lower bound on the real headroom.
@LastEditTime: 2026-08-28
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    """One episode's outcome, and the decisions that produced it."""

    actions: tuple[int, ...]
    travel: float
    waiting: float
    queue: float
    throughput: int
    switches: float


def num_phases(junction: str, plan: str) -> int:
    """How many candidate phases this (junction, plan) actually offers."""
    from ...expert.base import SignalPlan
    from ...scenario.lane_mask import load_lane_mask

    return SignalPlan.from_lane_mask(load_lane_mask(junction, plan)).num_phases


def run_episode_with(junction: str, plan: str, demand: str, seed: int,
                     steps: int | None, overrides: dict[int, int]) -> Episode:
    """Max-pressure, except at the decision indices named in `overrides`.

    Max-pressure keeps acting on the state it actually finds after a deviation,
    rather than replaying what it would have done — that is what makes this the
    value of the deviation *under the base policy*, and not the value of a
    pre-recorded sequence.
    """
    from ...env.gym_env import JunctionEnv
    from ...eval.compare import build_controller

    env = JunctionEnv(junction, plan, demand, seed=seed, num_seconds=steps,
                      render=False, allow_any_scenario=True)
    base = build_controller("max_pressure")
    base.reset(env.signal_plan)
    actions: list[int] = []
    try:
        _, _, done, _ = env.reset()
        index = 0
        while not done:
            choice = env.signal_plan.phases.index(
                base.act(env._pending, env.signal_plan)
            )
            action = overrides.get(index, choice)
            actions.append(action)
            _, _, done, _ = env.step(action)
            index += 1
        summary = env.summary()
    finally:
        env.close()
    return Episode(
        actions=tuple(actions),
        travel=summary["avg_travel_time_s"], waiting=summary["avg_waiting_time_s"],
        queue=summary["avg_queue_veh"], throughput=summary["throughput"],
        switches=summary["switch_rate"],
    )


def _evaluate(job: tuple) -> tuple:
    """One deviation, in its own process. Top level so it can be pickled."""
    junction, plan, demand, seed, steps, overrides, index, action = job
    from tshub.utils.init_log import set_logger

    from ...utils.paths import LOG_ROOT

    set_logger(str(LOG_ROOT / junction), terminal_log_level="ERROR")
    episode = run_episode_with(junction, plan, demand, seed, steps, overrides)
    return index, action, episode.travel, episode.throughput


def scan(junction: str, plan: str, demand: str, seed: int = 7,
         steps: int | None = None, workers: int = 16, rounds: int = 1,
         overrides: dict | None = None) -> dict:
    """Try every single-decision deviation from max-pressure, in parallel.

    With `rounds` above one this becomes greedy improvement: the best deviation
    found is kept and the scan repeats from there, so the total gain is a
    constructive lower bound on the headroom rather than a one-shot probe.
    """
    import multiprocessing as mp
    import time

    overrides = dict(overrides or {})
    context = mp.get_context("spawn")
    history = []

    for round_index in range(rounds):
        started = time.time()
        current = run_episode_with(junction, plan, demand, seed, steps, overrides)
        # From the signal plan, never from the actions taken: a base policy that
        # happens never to serve one phase would otherwise hide that phase from
        # the scan, and "no deviation helps" would be a statement about
        # max-pressure's habits rather than about the junction.
        candidates = num_phases(junction, plan)
        jobs = [
            (junction, plan, demand, seed, steps, {**overrides, index: action},
             index, action)
            for index in range(len(current.actions))
            for action in range(candidates)
            if action != current.actions[index]
        ]
        print(f"  round {round_index + 1}: {len(current.actions)} decisions, "
              f"{len(jobs)} deviations, travel now {current.travel:.2f}s", flush=True)

        with context.Pool(processes=workers) as pool:
            results = pool.map(_evaluate, jobs)

        improvements = sorted(
            ((current.travel - travel, index, action, travel, throughput)
             for index, action, travel, throughput in results),
            reverse=True,
        )
        best_gain, best_index, best_action, best_travel, _ = improvements[0]
        helped = [row for row in improvements if row[0] > 1e-9]
        entry = {
            "round": round_index + 1,
            "travel_before": round(current.travel, 3),
            "decisions": len(current.actions),
            "deviations_tried": len(jobs),
            "deviations_that_helped": len(helped),
            "best_gain_s": round(best_gain, 3),
            "best": {"decision": best_index, "action": best_action,
                     "travel_after": round(best_travel, 3)},
            "elapsed_s": round(time.time() - started, 1),
        }
        history.append(entry)
        print(f"    {len(helped)} of {len(jobs)} deviations improved the episode; "
              f"best is decision {best_index} -> action {best_action} "
              f"for {best_gain:+.3f}s  ({entry['elapsed_s']}s)", flush=True)

        if best_gain <= 1e-9:
            print("    no single deviation helps; max-pressure is a local optimum here.",
                  flush=True)
            break
        overrides[best_index] = best_action

    final = run_episode_with(junction, plan, demand, seed, steps, overrides)
    baseline = run_episode_with(junction, plan, demand, seed, steps, {})
    return {
        "scenario": f"{junction}/{plan}_{demand}", "seed": seed,
        "rounds": history, "overrides": {str(k): v for k, v in overrides.items()},
        "max_pressure_travel_s": round(baseline.travel, 3),
        "improved_travel_s": round(final.travel, 3),
        "total_gain_s": round(baseline.travel - final.travel, 3),
        "relative_gain": round(
            (baseline.travel - final.travel) / max(baseline.travel, 1e-9), 5),
        "throughput": {"max_pressure": baseline.throughput, "improved": final.throughput},
    }


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from ...utils.paths import REPORTS_ROOT

    parser = argparse.ArgumentParser(
        description="Does deviating from max-pressure at any single decision help?")
    parser.add_argument("--junction", default="Hongkong_YMT")
    parser.add_argument("--plan", default="normal")
    parser.add_argument("--demand", default="high_density")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel episodes. Every deviation is independent.")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Above 1, keep the best deviation and scan again.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    print(f"[plan] {args.junction}/{args.plan}_{args.demand} seed={args.seed}  "
          f"workers={args.workers}  rounds={args.rounds}  render=off")
    print("[plan] every deviation is a full episode from the start: exact, and "
          "nothing is restored")
    if args.dry_run:
        return 0

    result = scan(args.junction, args.plan, args.demand, args.seed, args.steps,
                  args.workers, args.rounds)
    print(f"\n=== {result['scenario']} ===")
    print(f"  max-pressure      {result['max_pressure_travel_s']:.2f}s")
    print(f"  after {len(result['rounds'])} improvement round(s)  "
          f"{result['improved_travel_s']:.2f}s")
    print(f"  headroom found    {result['total_gain_s']:+.2f}s "
          f"({result['relative_gain'] * 100:+.2f}%)")
    out = Path(args.out) if args.out else (
        REPORTS_ROOT / f"deviation_scan_{args.junction}_{args.plan}_{args.demand}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[summary] -> {out}")
    return 0
