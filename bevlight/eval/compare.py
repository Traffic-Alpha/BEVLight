'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Run several controllers over the same scenarios and tabulate.

Scenario membership always comes from `configs/scenario_selection.json` via
`bevlight.scenario.selection`, and the split is printed with the results. That
matters more than it looks: the two test demands and the four held-out plans
exist to measure generalization, and a number quietly produced on them while
tuning an expert is a number that can no longer be reported.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

from ..scenario.selection import load_selection

# Metrics worth showing side by side, and which direction is better.
REPORT = [
    ("avg_travel_time_s", "travel", "lower"),
    ("avg_waiting_time_s", "wait", "lower"),
    ("avg_queue_veh", "queue", "lower"),
    ("throughput", "done", "higher"),
    ("unfinished", "stuck", "lower"),
    ("switch_rate", "switch", None),
]


def build_controller(spec: str):
    """"max_pressure", "fixed_time:40" -> a controller instance."""
    from ..expert import CONTROLLERS

    name, _, argument = spec.partition(":")
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller '{name}'. Available: {list(CONTROLLERS)}")
    cls = CONTROLLERS[name]
    if not argument:
        return cls()
    if name == "fixed_time":
        return cls(green_duration=float(argument))
    if name == "max_pressure":
        return cls(use_occupancy=argument.lower() in ("occ", "occupancy", "true"))
    if name == "random":
        return cls(seed=int(argument))
    return cls()


def scenarios_for(split: str, junctions: list[str] | None, demands: list[str] | None):
    selection = load_selection()
    chosen = selection.split(split)
    if junctions:
        chosen = tuple(s for s in chosen if s.junction in junctions)
    if demands:
        chosen = tuple(s for s in chosen if s.demand in demands)
    return chosen


def run(args) -> dict:
    from ..env import run_episode

    scenarios = scenarios_for(args.split, args.junction, args.demand)
    if not scenarios:
        raise SystemExit("No scenarios matched. Check --junction / --demand / --split.")

    print(
        f"[plan] split={args.split}  scenarios={len(scenarios)}  "
        f"controllers={args.controller}  seeds={args.seed}  "
        f"steps={args.steps or 'from config'}  decision_interval=10s"
    )
    for scenario in scenarios:
        print(f"  - {scenario}  [{scenario.split}]")

    if args.dry_run:
        return {}

    results = []
    for scenario in scenarios:
        for seed in args.seed:
            for spec in args.controller:
                controller = build_controller(spec)
                result = run_episode(
                    junction=scenario.junction,
                    plan=scenario.plan,
                    demand=scenario.demand,
                    controller=controller,
                    seed=seed,
                    num_seconds=args.steps,
                )
                record = {
                    "scenario": str(scenario),
                    "split": scenario.split,
                    "junction": scenario.junction,
                    "plan": scenario.plan,
                    "demand": scenario.demand,
                    "seed": seed,
                    "controller": spec,
                    **result.metrics,
                }
                results.append(record)
                print(
                    f"    {scenario.junction}/{scenario.plan}/{scenario.demand} "
                    f"seed={seed} {spec:18s} "
                    f"travel={record['avg_travel_time_s']:7.1f}s "
                    f"wait={record['avg_waiting_time_s']:7.1f}s "
                    f"queue={record['avg_queue_veh']:5.1f} "
                    f"done={record['throughput']:4d} stuck={record['unfinished']:3d}"
                )
    return {"results": results}


def tabulate(results: list[dict], controllers: list[str]) -> None:
    """Per-scenario table plus the headline comparison against the first controller."""
    scenarios = []
    for record in results:
        key = (record["scenario"], record["split"])
        if key not in scenarios:
            scenarios.append(key)

    width = max(len(s) for s, _ in scenarios) + 2
    print("\n\n=== per scenario (mean over seeds) ===\n")
    header = f"{'scenario':{width}s} {'split':6s} {'controller':18s}"
    header += "".join(f"{label:>9s}" for _, label, _ in REPORT)
    print(header)
    print("-" * len(header))

    means: dict[tuple, dict] = {}
    for scenario, split in scenarios:
        for spec in controllers:
            rows = [r for r in results if r["scenario"] == scenario and r["controller"] == spec]
            if not rows:
                continue
            avg = {
                field: sum(r[field] for r in rows) / len(rows)
                for field, _, _ in REPORT
            }
            means[(scenario, spec)] = avg
            line = f"{scenario:{width}s} {split:6s} {spec:18s}"
            line += "".join(f"{avg[field]:9.1f}" for field, _, _ in REPORT)
            print(line)
        print()

    if len(controllers) < 2:
        return

    baseline = controllers[0]
    print(f"\n=== improvement over {baseline} (negative = better for travel/wait/queue) ===\n")
    head = f"{'scenario':{width}s} {'controller':18s}"
    head += "".join(f"{label:>10s}" for _, label, direction in REPORT if direction)
    print(head)
    print("-" * len(head))
    for scenario, _ in scenarios:
        base = means.get((scenario, baseline))
        if base is None:
            continue
        for spec in controllers[1:]:
            avg = means.get((scenario, spec))
            if avg is None:
                continue
            line = f"{scenario:{width}s} {spec:18s}"
            for field, _, direction in REPORT:
                if not direction:
                    continue
                before, after = base[field], avg[field]
                delta = (after - before) / before * 100 if before else 0.0
                line += f"{delta:9.1f}%"
            print(line)
