'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: The learned policy inside the SUMO loop, judged on control rather than agreement.

This is the only place a checkpoint is ranked. Action accuracy says how often the
model agrees with max-pressure; it does not say whether the junction clears, and
the two come apart exactly where it matters — the expert keeps the current phase
most of the time, so a model that only ever keeps it scores well and controls
badly.

Rendering is Panda, not Blender, and that is a hard constraint rather than a
preference: Blender is ~3.4 s/frame, so one 1000 s episode would cost half an
hour per checkpoint. Panda is ~0.2 s/frame and renders straight from the live
`states` object. `panda_day` is in the training set for exactly this reason — the
policy meets the same renderer here that it saw during behaviour cloning.

Nothing in this module computes a control number. Metrics come from
`bevlight.eval.metrics` through `run_episode`, the same code path the expert and
the baselines are measured on, so the three sit in one table honestly.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..env import DECISION_INTERVAL_S
from ..data.collate import collate, junction_structure
from ..expert.base import BaseController


class PolicyController(BaseController):
    """A trained checkpoint driving a junction from pixels.

    Holds the junction's static wiring and a rolling window of per-lane feature
    vectors. `observe` is fed a rendered frame each simulated second the policy
    asked for; `act` turns the window into a phase.
    """

    name = "policy"

    def __init__(self, checkpoint, junction: str, plan: str, window: int = 5,
                 device: str | None = None, decision_interval: int = DECISION_INTERVAL_S,
                 extractor=None, model=None):
        super().__init__()
        from ..data.features import FeatureExtractor
        from ..eval.offline import load_checkpoint
        from ..scenario.lane_mask import load_lane_mask

        self.checkpoint = Path(checkpoint)
        self.junction = junction
        # Not `self.plan`: BaseController.reset binds that to the SignalPlan
        # object, and the lane mask is loaded by the plan's name.
        self.plan_name = plan
        self.window = window
        self.decision_interval = decision_interval

        # A ranking run builds one controller per (scenario, seed, checkpoint);
        # the frozen backbone is the same object every time, so it is loaded once
        # and handed in rather than re-instantiated per episode.
        self.model = model or load_checkpoint(self.checkpoint, device=device)
        self.device = next(self.model.parameters()).device
        self.extractor = extractor or FeatureExtractor(device=str(self.device))

        mask = load_lane_mask(junction, self.plan_name)
        self.structure = junction_structure(mask)
        self.max_lanes = self.structure["lane_valid"].shape[0]
        self.num_lanes = len(self.structure["lane_order"])

        self.frames: deque = deque(maxlen=window)
        self.last_decision_at: float | None = None
        self.encoded = 0

    @property
    def obs_spec(self):
        """Window scope, pixels. What a drone could actually supply."""
        from ..env.obs_spec import ObsMode, ObsScope, ObsSpec

        return ObsSpec(scope=ObsScope.WINDOW, mode=ObsMode.FEATURES,
                       frames=self.window)

    def reset(self, plan) -> None:
        super().reset(plan)
        self.frames.clear()
        self.last_decision_at = None
        self.encoded = 0

    def needs_frame(self, second: int) -> bool:
        """Whether this simulated second has to be rendered.

        Only the `window` seconds leading up to a decision are ever read, and
        decisions come every `decision_interval` seconds, so roughly half the
        frames can be skipped — and rendering is the entire cost of an episode.
        Until the first decision has been seen the cadence is unknown, so
        everything is rendered; if a decision ever arrives late, the estimate
        falls behind and this returns True again rather than guessing.
        """
        if self.last_decision_at is None:
            return True
        return second > self.last_decision_at + self.decision_interval - self.window

    def observe(self, frame_index: int, images: dict) -> None:
        """One rendered frame -> one per-lane feature vector on the window."""
        rgb = images.get("rgb")
        if rgb is None:
            return
        self.frames.append(
            self.extractor.encode_array(rgb, self.junction, self.plan_name)
        )
        self.encoded += 1

    def _batch(self, obs, plan) -> dict:
        """The window, the wiring and the current phase, shaped like a training sample."""
        if not self.frames:
            raise RuntimeError(
                "PolicyController.act before any frame arrived; run_episode must "
                "render Panda images for a pixel policy"
            )
        # Before the window has filled, repeat the earliest frame. The first
        # decision of an episode would otherwise be taken on zeros.
        vectors = list(self.frames)
        vectors = [vectors[0]] * (self.window - len(vectors)) + vectors

        embed = vectors[0].shape[-1]
        lane_features = np.zeros((self.window, self.max_lanes, embed), dtype=np.float32)
        for step, vector in enumerate(vectors):
            lane_features[step, : vector.shape[0]] = vector

        sample = {
            **{k: v for k, v in self.structure.items()
               if k not in ("lane_order", "phase_order")},
            "lane_features": lane_features,
            "current_phase": plan.phases.index(obs.phase_index),
            "time_in_phase": float(obs.time_in_phase),
        }
        return {k: v.to(self.device) for k, v in collate([sample]).items()}

    @torch.no_grad()
    def act(self, obs, plan) -> int:
        batch = self._batch(obs, plan)
        position = int(self.model.act(batch)[0])
        self.last_decision_at = float(obs.time)
        self.last_action = plan.phases[position]
        return self.last_action


def run_closed_loop(
    checkpoint,
    junction: str,
    plan: str,
    demand: str,
    seed: int = 7,
    window: int = 5,
    device: str | None = None,
    panda_sky: str = "day",
    num_seconds: int | None = None,
    **kwargs,
) -> dict:
    """One episode under a learned policy. Returns the same summary as any controller."""
    from ..env import run_episode

    controller = PolicyController(
        checkpoint, junction, plan, window=window, device=device
    )
    result = run_episode(
        junction=junction,
        plan=plan,
        demand=demand,
        controller=controller,
        seed=seed,
        num_seconds=num_seconds,
        # No episode_dir: the frames are for the policy's eyes, not for disk.
        render_panda_images=True,
        panda_sky=panda_sky,
        **kwargs,
    )
    summary = dict(result.metrics)
    summary["frames_encoded"] = controller.encoded
    summary["checkpoint"] = str(Path(checkpoint).name)
    return summary


def checkpoints_of(run_dir: Path, names=None) -> list:
    """Which checkpoints of a run to score, in epoch order."""
    if names:
        return [run_dir / n if not Path(n).is_absolute() else Path(n) for n in names]
    return sorted(run_dir.glob("checkpoint_*.pt"))


def parse_args(argv=None):
    import argparse

    from ..scenario.selection import SPLITS

    parser = argparse.ArgumentParser(
        description="Rank checkpoints by closed-loop control quality."
    )
    parser.add_argument("--run", default=None, help="Run directory under runs/train/.")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="Checkpoints to score. Default: every one in --run.")
    parser.add_argument("--baseline", nargs="+", default=["fixed_time", "max_pressure"],
                        help="Rule-based controllers to run alongside. The first is the reference.")
    parser.add_argument("--split", choices=list(SPLITS), default="train",
                        help="Which split to draw scenarios from. Default: train, so test scenarios stay untouched.")
    parser.add_argument("--junction", nargs="+", default=None)
    parser.add_argument("--demand", nargs="+", default=None)
    parser.add_argument("--seed", nargs="+", type=int, default=[7])
    parser.add_argument("--steps", type=int, default=None, help="Episode length. Default: the scenario's num_seconds.")
    parser.add_argument("--window", type=int, default=5, help="Frames the policy reads per decision.")
    parser.add_argument("--panda-sky", default="day")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import json
    import time

    from ..env import run_episode
    from ..paths import TRAIN_RUNS_ROOT
    from .compare import build_controller, scenarios_for, tabulate

    args = parse_args(argv)
    run_dir = TRAIN_RUNS_ROOT / args.run if args.run else None
    checkpoints = checkpoints_of(run_dir, args.checkpoint) if run_dir else [
        Path(p) for p in (args.checkpoint or [])
    ]
    missing = [p for p in checkpoints if not p.is_file()]
    if missing:
        raise SystemExit(f"No such checkpoint: {missing[0]}")

    scenarios = scenarios_for(args.split, args.junction, args.demand)
    if not scenarios:
        raise SystemExit("No scenarios matched. Check --junction / --demand / --split.")

    # Baselines first: tabulate() reports every other controller against the
    # first one, and the reference is fixed time.
    labels = list(args.baseline) + [p.stem for p in checkpoints]
    episodes = len(scenarios) * len(args.seed) * len(labels)
    print(
        f"[plan] split={args.split} scenarios={len(scenarios)} seeds={args.seed} "
        f"controllers={labels} -> {episodes} episodes"
    )
    for scenario in scenarios:
        print(f"  - {scenario}  [{scenario.split}]")
    if not checkpoints:
        print("  note: no checkpoint given; only the baselines will run")
    if args.dry_run:
        return 0

    from ..data.features import FeatureExtractor
    from .offline import load_checkpoint

    # The backbone and the checkpoints do not change between episodes. Loading
    # them once keeps the per-episode cost purely rendering and simulation.
    extractor = FeatureExtractor(device=args.device) if checkpoints else None
    models = {p.stem: load_checkpoint(p, device=args.device) for p in checkpoints}

    results = []
    started = time.perf_counter()
    for scenario in scenarios:
        for seed in args.seed:
            for label in labels:
                if label in args.baseline:
                    controller = build_controller(label)
                    extra = {}
                else:
                    path = next(p for p in checkpoints if p.stem == label)
                    controller = PolicyController(
                        path, scenario.junction, scenario.plan,
                        window=args.window, device=args.device,
                        extractor=extractor, model=models[label],
                    )
                    extra = {"render_panda_images": True, "panda_sky": args.panda_sky}

                result = run_episode(
                    junction=scenario.junction,
                    plan=scenario.plan,
                    demand=scenario.demand,
                    controller=controller,
                    seed=seed,
                    num_seconds=args.steps,
                    **extra,
                )
                record = {
                    "scenario": str(scenario),
                    "split": scenario.split,
                    "junction": scenario.junction,
                    "plan": scenario.plan,
                    "demand": scenario.demand,
                    "seed": seed,
                    "controller": label,
                    **result.metrics,
                }
                if isinstance(controller, PolicyController):
                    record["frames_encoded"] = controller.encoded
                results.append(record)
                print(
                    f"    {scenario.junction}/{scenario.plan}/{scenario.demand} "
                    f"seed={seed} {label:18s} "
                    f"travel={record['avg_travel_time_s']:7.1f}s "
                    f"wait={record['avg_waiting_time_s']:7.1f}s "
                    f"queue={record['avg_queue_veh']:5.1f} "
                    f"done={record['throughput']:4d} stuck={record['unfinished']:3d}",
                    flush=True,
                )

    tabulate(results, labels)
    print(f"\n[time] {episodes} episodes in {time.perf_counter() - started:.0f}s")

    out = Path(args.out) if args.out else (
        (run_dir or Path.cwd()) / f"closed_loop_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"[summary] -> {out}")
    return 0


# --- Summary over splits -----------------------------------------------------

SPLIT_ORDER = ("train", "cross_demand_test", "cross_plan_test", "cross_structure_test")

#: Below this, `(fixed - policy) / (fixed - expert)` divides by a small number and
#: turns noise into a large percentage. One scenario where the expert gained
#: 1.5 s reported 47% off an 0.8 s difference.
MIN_HEADROOM_S = 3.0


def summarize_run(run_dir, policy: str, expert: str = "max_pressure",
                  weak: str = "fixed_time") -> dict:
    """Per-split medians of the policy's travel time against the two baselines."""
    import json
    import statistics as st
    from pathlib import Path

    run_dir = Path(run_dir)
    out = {}
    for split in SPLIT_ORDER:
        path = run_dir / f"closed_loop_{split}.json"
        if not path.is_file():
            continue
        scenarios: dict = {}
        for record in json.loads(path.read_text())["results"]:
            key = (record["junction"], record["plan"], record["demand"])
            scenarios.setdefault(key, {})[record["controller"]] = record
        rows = [v for v in scenarios.values() if {policy, expert, weak} <= set(v)]
        if not rows:
            continue
        delta, gain, versus_weak, unfinished = [], [], [], 0
        for row in rows:
            e, w, p = (row[expert]["avg_travel_time_s"], row[weak]["avg_travel_time_s"],
                       row[policy]["avg_travel_time_s"])
            delta.append(p - e)
            versus_weak.append(w - p)
            if w - e > MIN_HEADROOM_S:
                gain.append(100.0 * (w - p) / (w - e))
            unfinished += row[policy]["unfinished"]
        out[split] = {
            "n": len(rows),
            "median_delta_s": round(st.median(delta), 3),
            "mean_delta_s": round(sum(delta) / len(delta), 3),
            "worst_delta_s": round(max(delta), 3),
            "median_gain_pct": round(st.median(gain), 1) if gain else None,
            "mean_gain_vs_weak_s": round(sum(versus_weak) / len(versus_weak), 2),
            "unfinished": unfinished,
        }
    return out


def summarize_main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from ..paths import TRAIN_RUNS_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--policy", required=True,
                        help="Controller name as it appears in the results, e.g. checkpoint_060.")
    parser.add_argument("--expert", default="max_pressure")
    parser.add_argument("--weak", default="fixed_time")
    args = parser.parse_args(argv)

    run_dir = TRAIN_RUNS_ROOT / args.run
    summary = summarize_run(run_dir, args.policy, args.expert, args.weak)
    if not summary:
        raise SystemExit(f"no closed_loop_<split>.json under {run_dir}")

    print(f"{'split':22s} {'n':>3s} {'medΔ':>7s} {'meanΔ':>7s} {'worst':>7s} "
          f"{'medGain':>8s} {'vsWeak':>8s} {'unfin':>6s}")
    for split, s in summary.items():
        gain = f"{s['median_gain_pct']:7.0f}%" if s["median_gain_pct"] is not None else "      -"
        print(f"{split:22s} {s['n']:3d} {s['median_delta_s']:+7.2f} {s['mean_delta_s']:+7.2f} "
              f"{s['worst_delta_s']:+7.2f} {gain} {s['mean_gain_vs_weak_s']:+7.1f}s {s['unfinished']:6d}")
    target = run_dir / "closed_loop_summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  gain% only where {args.expert} beats {args.weak} by >{MIN_HEADROOM_S}s")
    print(f"  -> {target}")
    return 0
