'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Does the reward rank controllers the way the control metrics do?

A reward is a hypothesis: *optimising this produces good control*. Training is an
expensive way to test it. This tests it in minutes, by running controllers whose
ordering is already known and asking whether the reward reproduces that ordering.

The check that matters is a rank correlation against **travel time**, not a margin
between two controllers. Max-pressure is itself a queue heuristic, so a
queue-shaped reward is guaranteed to prefer it over fixed-time — which says
nothing about whether the reward tracks the metric the results table reports. Two
controllers cannot distinguish "correct" from "correct by construction"; several,
spanning random to tuned, can.

Several candidate rewards are measured in the same rollout, because the rollout is
the cost and the arithmetic is free. That turns "which reward should the teacher
optimise" from a sequence of training runs into one command:

    visible_queue    what `JunctionEnv._reward` returns today: mean queued
                     vehicles per incoming lane, counted inside the BEV window
    full_queue       the same over the whole lane. The control tables use the
                     full queue precisely because the visible one saturates at
                     the image edge — if that saturation is real here, a policy
                     can be rewarded for pushing queues out of frame
    full_wait        accumulated waiting per incoming lane. Closer to travel time
                     in shape, and unlike queue it keeps counting a vehicle that
                     has been stopped for a long time
    visible_occupancy  queue normalised by lane length rather than counted

`visible_queue` is reproduced from the environment's own reward stream and the two
are required to agree, so a candidate that disagrees with the live reward is a bug
in this file rather than a finding.
@LastEditTime: 2026-08-28
'''

from __future__ import annotations

from ..env.rewards import CANDIDATES, REWARDS, RewardContext

# What the reward is being asked to agree with. Travel time is the headline, and
# the one a queue-shaped reward is least automatically aligned with.
TARGETS = (("avg_travel_time_s", "lower"), ("avg_waiting_time_s", "lower"),
           ("avg_queue_veh", "lower"), ("throughput", "higher"))

DEFAULT_CONTROLLERS = ("random", "fixed_time:20", "fixed_time:30", "fixed_time:45",
                       "max_pressure", "max_pressure:occ")


class CostProbe:
    """Accumulates every candidate cost alongside the environment's own reward.

    Wraps `JunctionEnv._tick`, which is the one place a simulated second passes,
    so the accumulation cannot drift out of step with the interval the reward is
    computed over.
    """

    def __init__(self, env):
        self.env = env
        self._tick = env._tick
        env._tick = self._wrapped_tick
        self.saturated_lane_seconds = 0
        self._reset()

    def _reset(self) -> None:
        self.totals = dict.fromkeys(CANDIDATES, 0.0)
        self.seconds = 0

    def _wrapped_tick(self, obs, phase):
        done = self._tick(obs, phase)
        self._accumulate()
        return done

    def _accumulate(self) -> None:
        """One simulated second of every candidate, from the shared registry.

        The environment pays exactly one of these; measuring them all costs the
        arithmetic and the rollout is what is expensive, so "which reward should
        the teacher optimise" is one command rather than a sequence of runs.
        """
        env = self.env
        visible = env.observer(env.states)
        context = RewardContext(
            visible=visible,
            states=env.states,
            incoming_lanes=env.metrics.incoming_lanes,
            current_phase=env.current_phase,
            first_phase=env.signal_plan.phases[0],
        )
        for name in CANDIDATES:
            self.totals[name] += REWARDS[name](context)
        self.saturated_lane_seconds += sum(
            1 for s in context.visible_incoming if s.queue_saturated
        )
        self.seconds += 1

    def take(self) -> dict:
        """The interval's rewards, then start the next one — as `_reward` does."""
        if not self.seconds:
            rewards = dict.fromkeys(CANDIDATES, 0.0)
        else:
            rewards = {name: -total / self.seconds for name, total in self.totals.items()}
        self._reset()
        return rewards


def rollout(junction: str, plan: str, demand: str, controller_spec: str,
            seed: int, steps: int | None) -> dict:
    """One episode in the environment the teacher trains in, with no renderer."""
    from ..env.gym_env import JunctionEnv
    from ..eval.compare import build_controller

    env = JunctionEnv(junction, plan, demand, seed=seed, num_seconds=steps,
                      render=False, allow_any_scenario=True)
    controller = build_controller(controller_spec)
    controller.reset(env.signal_plan)
    probe = CostProbe(env)

    returns = dict.fromkeys(CANDIDATES, 0.0)
    live_return, decisions = 0.0, 0
    try:
        _, _, done, _ = env.reset()
        probe.take()                       # the pre-first-decision interval, as reset does
        while not done:
            action = env.signal_plan.phases.index(
                controller.act(env._pending, env.signal_plan)
            )
            _, reward, done, _ = env.step(action)
            interval = probe.take()
            for name in CANDIDATES:
                returns[name] += interval[name]
            live_return += reward
            decisions += 1
        summary = env.summary()
    finally:
        env.close()

    return {
        "controller": controller_spec,
        "seed": seed,
        "decisions": decisions,
        # Per decision rather than summed: episodes can end early on drain, and a
        # sum would then reward a controller for the episode being short.
        **{f"reward_{name}": returns[name] / max(1, decisions) for name in CANDIDATES},
        "reward_live": live_return / max(1, decisions),
        # The environment's own reward must be the `visible_queue` candidate. If
        # it is not, this file is measuring something the learner never sees.
        "reward_live_matches_visible_queue": abs(
            live_return - returns["visible_queue"]
        ) < 1e-6,
        "saturated_lane_seconds": probe.saturated_lane_seconds,
        **{k: summary[k] for k, _ in TARGETS},
        "avg_visible_queue_veh": summary["avg_visible_queue_veh"],
        "steps_run": summary["steps_run"],
    }


def ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not manufacture a correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. None when either side is constant and rho is undefined."""
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def alignment(rewards: list[float], targets: list[float], direction: str) -> float | None:
    """Rank agreement, signed so +1 always means "the reward is right".

    A reward is high when control is good; travel time is low when control is
    good. Flipping the sign here rather than at the reading end keeps every
    number in the report pointing the same way.
    """
    rho = spearman(rewards, targets)
    if rho is None:
        return None
    return rho if direction == "higher" else -rho


def analyse(rows: list[dict], controllers: list[str]) -> dict:
    """Per-scenario alignment for every candidate reward, then pooled."""
    scenarios = []
    for row in rows:
        if row["scenario"] not in scenarios:
            scenarios.append(row["scenario"])

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    per_scenario = []
    for scenario in scenarios:
        here = [r for r in rows if r["scenario"] == scenario]
        # Seeds are averaged first: the ordering being tested is between
        # controllers, and seed noise inside a controller is not part of it.
        points = {}
        for spec in controllers:
            seeds = [r for r in here if r["controller"] == spec]
            if seeds:
                points[spec] = {
                    key: mean([r[key] for r in seeds])
                    for key in list(seeds[0]) if isinstance(seeds[0][key], (int, float))
                }
        entry = {"scenario": scenario, "controllers": points, "alignment": {}}
        for candidate in CANDIDATES:
            rewards = [points[s][f"reward_{candidate}"] for s in points]
            entry["alignment"][candidate] = {
                target: alignment(rewards, [points[s][target] for s in points], direction)
                for target, direction in TARGETS
            }
        entry["saturated_lane_seconds"] = sum(r["saturated_lane_seconds"] for r in here)
        entry["visible_vs_full_queue"] = {
            spec: (points[spec]["avg_queue_veh"] - points[spec]["avg_visible_queue_veh"])
            / max(points[spec]["avg_queue_veh"], 1e-9)
            for spec in points
        }
        per_scenario.append(entry)

    pooled = {}
    for candidate in CANDIDATES:
        pooled[candidate] = {}
        for target, _ in TARGETS:
            values = [e["alignment"][candidate][target] for e in per_scenario
                      if e["alignment"][candidate][target] is not None]
            pooled[candidate][target] = {
                "mean": round(mean(values), 4),
                "min": round(min(values), 4) if values else None,
                "scenarios": len(values),
                "well_ordered": sum(1 for v in values if v >= 0.8),
            }
    return {"per_scenario": per_scenario, "pooled": pooled,
            "faithful": all(r["reward_live_matches_visible_queue"] for r in rows)}


def tabulate(report: dict, controllers: list[str]) -> None:
    pooled, per_scenario = report["pooled"], report["per_scenario"]

    if not report["faithful"]:
        print("\n[FAIL] the environment's reward stream does not match the "
              "`visible_queue` candidate — the probe is measuring the wrong thing.\n")

    print("\n\n=== reward alignment with the control metrics ===")
    print("+1 = the reward orders controllers exactly as the metric does, "
          "0 = no relation, -1 = backwards.\n")
    head = f"{'candidate reward':22s}"
    head += "".join(f"{t.replace('avg_','').replace('_s',''):>22s}" for t, _ in TARGETS)
    print(head)
    print("-" * len(head))
    for candidate in CANDIDATES:
        line = f"{candidate:22s}"
        for target, _ in TARGETS:
            cell = pooled[candidate][target]
            line += f"{cell['mean']:>+13.3f} ({cell['well_ordered']}/{cell['scenarios']})"
        print(line)
    print("\n(x/y) = scenarios where alignment reached 0.8, out of those it could be computed on.")

    print("\n\n=== per scenario, alignment with travel time ===\n")
    width = max(len(e["scenario"]) for e in per_scenario) + 2
    head = f"{'scenario':{width}s}" + "".join(f"{c:>20s}" for c in CANDIDATES)
    print(head)
    print("-" * len(head))
    for entry in per_scenario:
        line = f"{entry['scenario']:{width}s}"
        for candidate in CANDIDATES:
            value = entry["alignment"][candidate]["avg_travel_time_s"]
            line += f"{value:>+20.3f}" if value is not None else f"{'n/a':>20s}"
        print(line)

    print("\n\n=== does the BEV window hide queue? ===")
    print("If the visible queue saturates, a policy rewarded on it can be paid to "
          "push queues out of frame.\n")
    head = f"{'scenario':{width}s}{'saturated lane-s':>18s}{'worst visible/full gap':>26s}"
    print(head)
    print("-" * len(head))
    for entry in per_scenario:
        gaps = entry["visible_vs_full_queue"]
        worst_spec = max(gaps, key=lambda s: gaps[s]) if gaps else None
        worst = gaps[worst_spec] if worst_spec else 0.0
        print(f"{entry['scenario']:{width}s}{entry['saturated_lane_seconds']:>18d}"
              f"{worst * 100:>21.1f}%  {worst_spec or ''}")

    print("\n\n=== controller ordering, one scenario at a time ===\n")
    for entry in per_scenario:
        print(f"  {entry['scenario']}")
        rows = sorted(entry["controllers"].items(),
                      key=lambda kv: kv[1]["avg_travel_time_s"])
        for spec, values in rows:
            print(f"    {spec:20s} travel={values['avg_travel_time_s']:7.2f} "
                  f"wait={values['avg_waiting_time_s']:7.2f} "
                  f"queue={values['avg_queue_veh']:6.2f} "
                  f"thr={values['throughput']:6.0f}  |  "
                  + "  ".join(f"{c}={values[f'reward_{c}']:+8.4f}" for c in CANDIDATES))
        print()


def run(args) -> dict:
    from ..eval.compare import scenarios_for

    scenarios = scenarios_for(args.split, args.junction, args.demand)
    if not scenarios:
        raise SystemExit("No scenarios matched. Check --junction / --demand / --split.")

    jobs = len(scenarios) * len(args.controller) * len(args.seed)
    print(f"[plan] split={args.split}  scenarios={len(scenarios)}  "
          f"controllers={len(args.controller)}  seeds={len(args.seed)}  "
          f"episodes={jobs}  render=off")
    for scenario in scenarios:
        print(f"  - {scenario}  [{scenario.split}]")
    print(f"  controllers: {', '.join(args.controller)}")
    if args.dry_run:
        return {}

    rows, done = [], 0
    for scenario in scenarios:
        for seed in args.seed:
            for spec in args.controller:
                record = rollout(scenario.junction, scenario.plan, scenario.demand,
                                 spec, seed, args.steps)
                record.update(scenario=str(scenario), split=scenario.split,
                              junction=scenario.junction, plan=scenario.plan,
                              demand=scenario.demand)
                rows.append(record)
                done += 1
                print(f"  [{done:3d}/{jobs}] {scenario!s:46s} seed={seed} "
                      f"{spec:18s} travel={record['avg_travel_time_s']:7.2f} "
                      f"R_visible_queue={record['reward_visible_queue']:+8.4f}",
                      flush=True)
    return {"controllers": list(args.controller), "seeds": list(args.seed),
            "results": rows, "report": analyse(rows, list(args.controller))}
