'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Batch expert collection over a scenario split.

Scenarios come from the active manifest, and the split is stated on every run,
because collecting from a test split would quietly contaminate the very numbers
the split exists to produce.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ...scenario.layout import EpisodeKey, episode_dir
from ...scenario.selection import SPLITS, load_selection

# Short controller tags, so an episode directory name stays readable.
CONTROLLER_TAGS = {"max_pressure": "mp", "fixed_time": "ft"}


def episode_complete(path: Path) -> bool:
    return (
        (path / "episode.json").is_file()
        and (path / "blender_selected.json").is_file()
        and (path / "images" / "panda_day" / "rgb").is_dir()
        and (path / "images" / "panda_day" / "seg").is_dir()
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect expert episodes.")
    parser.add_argument("--controller", default="max_pressure", help="Expert to drive the episodes.")
    parser.add_argument("--split", choices=list(SPLITS), default="train",
                        help="Which split to collect. Default: train.")
    parser.add_argument("--junction", nargs="+", default=None, help="Restrict to these junctions.")
    parser.add_argument("--demand", nargs="+", default=None, help="Restrict to these demand patterns.")
    parser.add_argument("--seed", nargs="+", type=int, default=[7], help="SUMO seeds, one episode each.")
    parser.add_argument("--steps", type=int, default=None, help="Episode length. Default: the scenario's num_seconds.")
    parser.add_argument("--no-render-frames", action="store_true",
                        help="Write labels and decisions only, skipping the render trajectory.")
    parser.add_argument("--no-panda-images", action="store_true",
                        help="Do not render collected frames with Panda3D.")
    parser.add_argument("--panda-clean", action="store_true",
                        help="Remove existing panda_day images before rendering.")
    parser.add_argument("--panda-preset", default="auto",
                        help="Panda3D render preset. 'auto' matches the solved junction size.")
    parser.add_argument("--panda-backend", default="pandagl", help="Panda3D rendering backend.")
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="Print rollout progress every N simulation seconds. 0 disables progress logs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes already on disk.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned episodes without simulating.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from ...env import run_episode
    from ..episode_schema import load_episode
    from ..frame_selection import select_frames, write_blender_manifest
    from ...eval.compare import build_controller

    selection = load_selection()
    scenarios = selection.split(args.split)
    if args.junction:
        scenarios = tuple(s for s in scenarios if s.junction in args.junction)
    if args.demand:
        scenarios = tuple(s for s in scenarios if s.demand in args.demand)
    if not scenarios:
        raise SystemExit("No scenarios matched. Check --junction / --demand / --split.")

    tag = CONTROLLER_TAGS.get(args.controller.split(":")[0], args.controller.split(":")[0])
    jobs = [
        (scenario, seed, EpisodeKey(scenario.junction, scenario.plan, scenario.demand, seed, tag))
        for scenario in scenarios
        for seed in args.seed
    ]

    print(
        f"[plan] split={args.split} controller={args.controller} "
        f"episodes={len(jobs)} seeds={args.seed} steps={args.steps or 'from config'}",
        flush=True,
    )
    for scenario, seed, key in jobs:
        print(f"  - {scenario}  seed={seed}  [{scenario.split}] -> {episode_dir(key)}", flush=True)

    if args.dry_run:
        return 0

    failures = []
    for job_index, (scenario, seed, key) in enumerate(jobs, start=1):
        target = episode_dir(key)
        if args.skip_existing and episode_complete(target):
            print(f"[skip] {job_index}/{len(jobs)} {key}: already collected with Panda RGB/SEG", flush=True)
            payload = load_episode(target)
            write_blender_manifest(target, select_frames(payload))
            continue
        try:
            print(
                f"[start] {job_index}/{len(jobs)} {key}: "
                f"target={target} panda={not args.no_panda_images and not args.no_render_frames}",
                flush=True,
            )
            if not args.no_render_frames and target.exists():
                print(f"[clean] {key}: removing incomplete/old episode dir {target}", flush=True)
                shutil.rmtree(target)
            result = run_episode(
                junction=scenario.junction,
                plan=scenario.plan,
                demand=scenario.demand,
                controller=build_controller(args.controller),
                seed=seed,
                num_seconds=args.steps,
                episode_dir=None if args.no_render_frames else target,
                render_panda_images=not args.no_panda_images and not args.no_render_frames,
                panda_clean=args.panda_clean,
                panda_preset=args.panda_preset,
                panda_backend=args.panda_backend,
                progress_interval=args.progress_interval,
            )
            metrics = result.metrics
            print(
                f"[done] {key}: {metrics['decisions']} decisions "
                f"(keep {metrics['keep_rate']:.0%}), travel={metrics['avg_travel_time_s']}s, "
                f"queue={metrics['avg_queue_veh']}, done={metrics['throughput']} -> {target}",
                flush=True,
            )
            if not args.no_render_frames:
                payload = load_episode(target)
                selected = write_blender_manifest(target, select_frames(payload))
                print(f"[blender] selected manifest -> {selected}", flush=True)
        except Exception as exc:
            failures.append((key, exc))
            print(f"[fail] {key}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print("[summary] failures:", file=sys.stderr, flush=True)
        for key, exc in failures:
            print(f"  - {key}: {exc}", file=sys.stderr, flush=True)
        return 1

    print(f"[summary] collected {len(jobs) - len(failures)} episode(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
