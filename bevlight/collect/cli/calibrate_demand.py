'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Tune traffic demand to a measured queue, not to a formula.

Demand has to land in a band, and both edges are hard limits:

  * Too light and most lane-seconds are empty. The current scenarios sit here:
    79.8% zeros, only 3.7% of samples reaching a queue of 3.
  * Too heavy and the queue runs past what the BEV window shows (60 m, ~8
    vehicles), so the label saturates and the extra congestion buys nothing. On
    two junctions it is worse than useless: France_Massy and Hongkong_YMT have
    signal-controlled approaches only 59 m and 46 m long, which store 6-8
    vehicles in total, so a heavy demand spills back off the end of the lane.

The other ten junctions store 22-27 vehicles per approach, three times the
visible ceiling, so there the window binds long before gridlock does.

That makes the sensible target a *queue distribution*, per junction, rather than
a global saturation multiplier: p90 queue of roughly 4-7 vehicles on the
controlled lanes, and P(queue >= visible capacity) kept small.

**A queue cannot be predicted from road capacity, because the signal decides the
capacity.** A lane's saturation flow (~1800 veh/h) is a free-flow property; what
a movement actually discharges is `green share x saturation flow`, and the green
share is chosen by the controller at run time. An adaptive controller moves green
towards whichever movement is loaded, so its effective capacity is a function of
the algorithm, not of the road. Fixed-time hands out a rigid 1/K share whatever
the demand, and therefore queues more at the same flow. So a demand figure
derived from a capacity formula means very little, and this module measures
instead of computing.

That matters here because **one set of routes is shared by every method in the
comparison tables**. A scale calibrated to sit inside the window under the expert
can still saturate under the weakest baseline, and a saturated label is a
truncated one. Calibrate under the controller that generates the data (the
expert), then re-check the headroom under the worst controller that will be
evaluated on the same routes.

Scanning is done with SUMO's own `--scale`, injected through a temporary
sumocfg, so nothing has to be regenerated to find the right factor. Only once a
factor is chosen does `generate_routes.py` need editing.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# The band worth aiming for, in vehicles of queue on a controlled lane.
TARGET_P90_MIN = 4.0
TARGET_P90_MAX = 7.0


def scaled_sumocfg(sumo_cfg: Path, scale: float, out_dir: Path) -> Path:
    """Copy a sumocfg with `<scale>` added, keeping every path resolvable.

    SUMO resolves relative paths against the config's own directory, so the copy
    has to sit next to the original rather than in a temp directory elsewhere.
    """
    tree = ET.parse(sumo_cfg)
    root = tree.getroot()
    processing = root.find("processing")
    if processing is None:
        processing = ET.SubElement(root, "processing")
    element = processing.find("scale")
    if element is None:
        element = ET.SubElement(processing, "scale")
    element.set("value", f"{scale:g}")

    out = out_dir / f".calibrate_scale_{scale:g}.sumocfg"
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    return out


def measure(junction: str, plan: str, demand: str, scale: float, seed: int = 7,
            steps: int = 700, controller_spec: str = "max_pressure") -> dict:
    """Run one scaled episode under a controller and describe its queues.

    The controller is a parameter because the queue is as much a property of the
    signal algorithm as of the demand.
    """
    from ..observation import ObservationExtractor
    from ...eval.compare import build_controller
    from ...expert import SignalPlan
    from ...scenario.lane_mask import load_lane_mask
    from ...scenario.loader import load_junction_config
    from ...utils.paths import LOG_ROOT
    from tshub.tshub_env.tshub_env import TshubEnvironment
    from tshub.utils.init_log import set_logger

    set_logger(str(LOG_ROOT / junction), terminal_log_level="ERROR")
    cfg = load_junction_config(junction, f"{plan}_{demand}")
    sumo_cfg = Path(cfg["sumo_cfg"])

    mask = load_lane_mask(junction, plan)
    signal_plan = SignalPlan.from_lane_mask(mask)
    controlled = [
        lane
        for phase in signal_plan.phases
        for movement in signal_plan.movements_of(phase)
        for lane in signal_plan.movement_in_lanes[movement]
    ]
    visible = {lane: mask.visible_length_m(lane) for lane in controlled}
    capacity = {lane: visible[lane] / 7.5 for lane in controlled}

    scaled = scaled_sumocfg(sumo_cfg, scale, sumo_cfg.parent)
    extractor = ObservationExtractor(mask, mask.tls_id)
    controller = build_controller(controller_spec)
    controller.reset(signal_plan)

    env = TshubEnvironment(
        sumo_cfg=str(scaled),
        is_map_builder_initialized=False,
        is_vehicle_builder_initialized=True,
        is_aircraft_builder_initialized=False,
        is_traffic_light_builder_initialized=True,
        is_person_builder_initialized=False,
        tls_ids=[cfg["tls_id"]],
        tls_action_type="choose_next_phase",
        delta_time=10,
        use_gui=False,
        is_libsumo=False,
        num_seconds=steps + 20,
        sumo_seed=str(seed),
    )

    queues: list[list[float]] = []
    spillback = 0
    try:
        states = env.reset()
        action = signal_plan.phases[0]
        for _ in range(steps):
            obs = extractor(states)
            if obs.can_act:
                action = controller.act(obs, signal_plan)
            states, _, _, done = env.step({"vehicle": {}, "tls": {cfg["tls_id"]: action}})
            row = [obs.lanes[lane].queued for lane in controlled]
            queues.append(row)
            spillback += sum(
                1 for lane in controlled if obs.lanes[lane].queue_saturated
            )
            if done:
                break
    finally:
        try:
            env._close_simulation()
        except SystemExit:
            pass
        scaled.unlink(missing_ok=True)

    values = np.asarray(queues, dtype=float).ravel()
    per_lane_capacity = float(np.mean(list(capacity.values())))
    return {
        "scale": scale,
        "controller": controller_spec,
        "zeros_pct": round(100 * float((values == 0).mean()), 1),
        "mean": round(float(values.mean()), 2),
        "p90": round(float(np.percentile(values, 90)), 1),
        "p99": round(float(np.percentile(values, 99)), 1),
        "max": int(values.max()),
        "ge3_pct": round(100 * float((values >= 3).mean()), 1),
        "over_window_pct": round(100 * float((values >= per_lane_capacity).mean()), 2),
        "visible_capacity_veh": round(per_lane_capacity, 1),
        "lane_seconds": int(values.size),
    }


def verdict(row: dict) -> str:
    if row["p90"] < TARGET_P90_MIN:
        return "too light"
    if row["p90"] > TARGET_P90_MAX or row["over_window_pct"] > 5.0:
        return "too heavy"
    return "GOOD"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the demand scale that lands queues in the useful band.")
    parser.add_argument("--junction", required=True)
    parser.add_argument("--plan", default="normal")
    parser.add_argument("--demand", default="high_density")
    parser.add_argument("--scale", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--controller", nargs="+", default=["max_pressure"],
                        help="Controllers to measure under. The queue depends on the signal algorithm, "
                             "and one set of routes is shared by every method, so check the weakest too.")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(
        f"[plan] {args.junction}/{args.plan}/{args.demand} "
        f"scales={args.scale} steps={args.steps}\n"
        f"        target: p90 queue in [{TARGET_P90_MIN:.0f}, {TARGET_P90_MAX:.0f}] vehicles, "
        f"and rarely past the BEV window"
    )
    rows = []
    header = (
        f"{'controller':14s} {'scale':>6s} {'zeros%':>7s} {'mean':>6s} {'p90':>5s} {'p99':>5s} "
        f"{'max':>4s} {'>=3veh%':>8s} {'past window%':>13s} {'verdict':>10s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for spec in args.controller:
        for scale in args.scale:
            row = measure(args.junction, args.plan, args.demand, scale, args.seed,
                          args.steps, controller_spec=spec)
            rows.append(row)
            print(
                f"{spec:14s} {row['scale']:6.2f} {row['zeros_pct']:7.1f} {row['mean']:6.2f} "
                f"{row['p90']:5.1f} {row['p99']:5.1f} {row['max']:4d} {row['ge3_pct']:8.1f} "
                f"{row['over_window_pct']:13.2f} {verdict(row):>10s}"
            )
        print()
    print(f"\n  BEV window holds ~{rows[0]['visible_capacity_veh']:.1f} queued vehicles on this junction")

    # A scale is only usable if it stays in band for *every* controller that will
    # be run on these routes, not just the expert that generated the data.
    specs = list(dict.fromkeys(r["controller"] for r in rows))
    usable = [
        scale
        for scale in args.scale
        if all(
            verdict(r) == "GOOD"
            for r in rows
            if r["scale"] == scale
        )
    ]
    if usable:
        print(f"  -> scale {usable[0]:g} stays in band under all of {specs}")
    elif len(specs) > 1:
        print(f"  -> no scale is in band for every controller in {specs}; "
              f"pick for the weakest one, or widen --scale")
    else:
        print("  -> no scale in range; widen --scale")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
