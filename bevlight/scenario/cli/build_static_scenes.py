'''
@Author: WANG Maonan
@Date: 2026-08-19 19:06:35
@Description: Build static 3D scenes for BEVLight junction scenarios.

Pipeline:
  1) SUMO net.xml -> scenarios/<junction>/3d_assets/scene.json
  2) Blender build_scene.py -> scenarios/<junction>/3d_assets/*.glb
  3) Blender build_blend.py -> scenarios/<junction>/3d_assets/scene.blend

Examples:
  # Build one canonical static scene for each of the 12 junctions.
  python scenarios/build_static_scene.py

  # Only export glb files, skip .blend assembly.
  python scenarios/build_static_scene.py --skip-blend

  # Build selected junctions.
  python scenarios/build_static_scene.py --junction Beijing_Beihuan Hongkong_YMT
@LastEditTime: 2026-08-19
@LastEditors: WANG Maonan
'''

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ...cli.tshub import (
    PROBE_SCENE,
    blender_scene_scripts as get_blender_scripts,
    configure_tshub_import,
    find_blender,
    resolve_tshub_root as _resolve_tshub_root,
)
from ...paths import SCENARIOS_ROOT, SCENE_ASSETS_DIR_NAME


def resolve_tshub_root(cli_value: str | None) -> Path | None:
    """TransSimHub checkout that actually carries the scene-building scripts."""
    return _resolve_tshub_root(cli_value, PROBE_SCENE)


REQUIRED_SCENE_FILES = ("map.glb", "ground.glb", "road_lines.glb", "lane_lines.glb")
GENERATED_GLB_FILES = (*REQUIRED_SCENE_FILES, "buildings.glb", "vegetation.glb", "props.glb")
CANONICAL_NET_PLAN = "normal"
REAL_OSM_POLY_JUNCTIONS = {
    "France_Massy",
    "Hongkong_YMT",
}


@dataclass(frozen=True)
class SceneJob:
    junction: str
    scenario_dir: Path
    net_file: Path
    poly_file: Path | None
    output_dir: Path
    scene_json: Path
    scene_blend: Path


def available_junctions() -> list[str]:
    return sorted(
        path.name
        for path in SCENARIOS_ROOT.iterdir()
        if path.is_dir() and (path / "config.py").is_file()
    )


def find_building_poly_file(junction: str, scenario_dir: Path) -> Path | None:
    if junction not in REAL_OSM_POLY_JUNCTIONS:
        return None

    candidates = sorted((scenario_dir / "add").glob("*.poly.xml"))
    return candidates[0] if candidates else None


def make_jobs(junctions: Iterable[str], net_plan: str) -> list[SceneJob]:
    jobs = []
    known = set(available_junctions())

    for junction in junctions:
        if junction not in known:
            raise ValueError(f"Unknown junction '{junction}'. Available junctions: {sorted(known)}")

        scenario_dir = SCENARIOS_ROOT / junction
        poly_file = find_building_poly_file(junction, scenario_dir)
        net_file = scenario_dir / "networks" / f"{net_plan}.net.xml"
        if not net_file.exists():
            raise FileNotFoundError(f"Missing net file: {net_file}")

        output_dir = scenario_dir / SCENE_ASSETS_DIR_NAME
        jobs.append(
            SceneJob(
                junction=junction,
                scenario_dir=scenario_dir,
                net_file=net_file,
                poly_file=poly_file,
                output_dir=output_dir,
                scene_json=output_dir / "scene.json",
                scene_blend=output_dir / "scene.blend",
            )
        )

    return jobs


def remove_old_glbs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_GLB_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def required_outputs_exist(job: SceneJob, skip_blend: bool) -> bool:
    required = [job.output_dir / name for name in REQUIRED_SCENE_FILES]
    if not skip_blend:
        required.append(job.scene_blend)
    return all(path.exists() for path in required)


def run_blender(cmd: list[str], done_token: str, verbose: bool = False) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if verbose:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
    else:
        for line in result.stdout.splitlines():
            if line.startswith(("[assembly]", "BUILD_DONE", "BLEND_DONE")):
                print(line)

    if result.returncode != 0 or done_token not in result.stdout:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"Blender command failed: {' '.join(cmd[:4])}")


def build_one_scene(
    job: SceneJob,
    blender: Path,
    build_scene_script: Path,
    build_blend_script: Path,
    skip_blend: bool = False,
    skip_existing: bool = False,
    style: str = "day",
    samples: int = 64,
    resolution: str = "1022x1022",
    engine: str = "CYCLES",
    verbose: bool = False,
) -> Path:
    if skip_existing and required_outputs_exist(job, skip_blend=skip_blend):
        print(f"[skip] {job.junction}: existing outputs found")
        return job.output_dir

    from tshub.tshub_env3d.scene import export_scene_geometry

    job.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] {job.junction}")
    print(f"        net  : {job.net_file}")
    if job.poly_file:
        print(f"        poly : {job.poly_file}")
    print(f"        out  : {job.output_dir}")

    export_scene_geometry(
        str(job.net_file),
        str(job.scene_json),
        buildings_poly=str(job.poly_file) if job.poly_file else None,
        ground_margin=60,
    )

    remove_old_glbs(job.output_dir)
    run_blender(
        [
            str(blender),
            "--background",
            "--python",
            str(build_scene_script),
            "--",
            str(job.scene_json),
            str(job.output_dir),
        ],
        "BUILD_DONE",
        verbose=verbose,
    )

    missing = [name for name in REQUIRED_SCENE_FILES if not (job.output_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"{job.junction} did not produce required files: {', '.join(missing)}"
        )

    if not skip_blend:
        run_blender(
            [
                str(blender),
                "--background",
                "--python",
                str(build_blend_script),
                "--",
                str(job.output_dir),
                str(job.scene_blend),
                "--style",
                style,
                "--samples",
                str(samples),
                "--resolution",
                resolution,
                "--engine",
                engine,
            ],
            "BLEND_DONE",
            verbose=verbose,
        )

    return job.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BEVLight static 3D scenes.")
    parser.add_argument(
        "--junction",
        nargs="+",
        default=None,
        help="Junction names to build. Default: all junctions under scenarios/.",
    )
    parser.add_argument("--net-plan", default=CANONICAL_NET_PLAN, help="Network plan file to use for geometry. Default: normal.")
    parser.add_argument(
        "--tshub-root",
        default=None,
        help="Path to TransSimHub root. Defaults to TSHUB_ROOT or /home/wmn/code/TransSimHub.",
    )
    parser.add_argument("--skip-blend", action="store_true", help="Only build glb files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs whose outputs already exist.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue building remaining jobs after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without building.")
    parser.add_argument("--style", default="day", help="Blender light style for scene.blend.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples for scene.blend.")
    parser.add_argument("--resolution", default="1022x1022", help="Blend render resolution, e.g. 1022x1022.")
    parser.add_argument("--engine", default="CYCLES", help="Blender render engine.")
    parser.add_argument("--verbose", action="store_true", help="Print full Blender stdout/stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    junctions = args.junction or available_junctions()
    jobs = make_jobs(junctions, args.net_plan)

    print(f"[plan] jobs={len(jobs)} junctions={len(set(job.junction for job in jobs))} net_plan={args.net_plan}")
    for job in jobs:
        print(f"  - {job.junction} -> {job.output_dir}")

    if args.dry_run:
        return 0

    tshub_root = resolve_tshub_root(args.tshub_root)
    configure_tshub_import(tshub_root)
    build_scene_script, build_blend_script = get_blender_scripts(tshub_root)
    blender = find_blender()

    failures = []
    for job in jobs:
        try:
            build_one_scene(
                job=job,
                blender=blender,
                build_scene_script=build_scene_script,
                build_blend_script=build_blend_script,
                skip_blend=args.skip_blend,
                skip_existing=args.skip_existing,
                style=args.style,
                samples=args.samples,
                resolution=args.resolution,
                engine=args.engine,
                verbose=args.verbose,
            )
        except Exception as exc:
            failures.append((job, exc))
            print(f"[fail] {job.junction}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for job, exc in failures:
            print(f"  - {job.junction}: {exc}", file=sys.stderr)
        return 1

    print(f"[summary] built {len(jobs)} static scene job(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    raise SystemExit(main())
