"""Render the reference BEV frames described in `bev_reference`."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from ...utils.paths import PROJECT_ROOT, SCENARIOS_ROOT
from .._internal.bev_reference import BEV_REFERENCE_ROOT, REFERENCE_SECOND, reference_dir


def all_junctions() -> list[str]:
    return sorted(
        d.name for d in SCENARIOS_ROOT.iterdir()
        if (d / "lane_mask" / "lane_mask.json").is_file()
    )


def _demand_for(junction: str, plan: str) -> str:
    """A busy demand this junction actually has."""
    from ..loader import load_junction_config

    for demand in ("high_density", "increasing_demand", "medium_density"):
        try:
            load_junction_config(junction, f"{plan}_{demand}")
        except Exception:
            continue
        return demand
    raise SystemExit(f"{junction}: no usable demand for plan {plan}")


def render_panda(junction: str, plan: str, seconds: int, out_dir: Path) -> Path:
    """Simulate to a busy second, save the Panda frame, export the scene."""
    import cv2
    from tshub.tshub_env3d.core import build_frame

    from ...env.render import image_modality, make_panda_renderer, make_render_exporter
    from ...env.sumo import build_environment
    from ..bev_camera import bev_ortho_size, bev_resolution

    demand = _demand_for(junction, plan)
    env, config = build_environment(junction, f"{plan}_{demand}", 7, seconds + 60)
    states = env.reset()
    export_dir = out_dir / "export"
    shutil.rmtree(export_dir, ignore_errors=True)
    exporter = make_render_exporter(junction, plan, states, config["tls_id"], export_dir)
    renderer, _ = make_panda_renderer(
        junction=junction, plan=plan, states=states, tls_id=config["tls_id"],
        preset="auto", backend="pandagl", sky="day",
    )
    try:
        for _ in range(seconds):
            states, _, _, _ = env.step({"vehicle": {}, "tls": {config["tls_id"]: 0}})
        frame = build_frame(states)
        exporter.add_frame(frame)
        image = None
        for cameras in renderer.sync(frame).values():
            for kind, pixels in cameras.items():
                if image_modality(kind) == "rgb":
                    image = pixels
        if image is None:
            raise SystemExit(f"{junction}: Panda produced no RGB frame")
        cv2.imwrite(str(out_dir / "panda_day.png"), image[:, :, ::-1])
        manifest = exporter.close()
    finally:
        renderer.destroy()
        env._close_simulation()

    width, height = bev_resolution(junction, plan)
    (out_dir / "meta.json").write_text(json.dumps({
        "plan": plan, "demand": demand, "second": seconds,
        "ortho_size": bev_ortho_size(junction, plan),
        "resolution": [width, height],
    }, indent=2) + "\n")
    return Path(manifest)


def render_blender(junction: str, manifest: Path, out_dir: Path) -> bool:
    """Render the exported frame with Cycles and keep the BEV camera's image."""
    import cv2

    command = [
        sys.executable, str(PROJECT_ROOT / "tools" / "render_blender.py"),
        "--manifest", str(manifest), "--style", "day",
        "--passes", "rgb", "--workers", "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:])
        return False
    hits = sorted((out_dir / "export").rglob("junction_bev_rgb/*.png"))
    if not hits:
        return False
    image = cv2.imread(str(hits[-1]))
    cv2.imwrite(str(out_dir / "blender_day.png"), image)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junction", nargs="+", default=None)
    parser.add_argument("--plan", default="easy")
    parser.add_argument("--seconds", type=int, default=REFERENCE_SECOND)
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument("--keep-export", action="store_true",
                        help="Keep the exported scene (~30MB/junction) for debugging.")
    args = parser.parse_args(argv)

    junctions = args.junction or all_junctions()
    failed = []
    for junction in junctions:
        from ..bev_camera import bev_ortho_size
        out_dir = reference_dir(junction, bev_ortho_size(junction, args.plan))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{junction}] simulating {args.seconds}s ...", flush=True)
        try:
            manifest = render_panda(junction, args.plan, args.seconds, out_dir)
            print(f"[{junction}] panda -> panda_day.png", flush=True)
            if not args.skip_blender:
                if render_blender(junction, manifest, out_dir):
                    print(f"[{junction}] blender -> blender_day.png", flush=True)
                else:
                    print(f"[{junction}] blender FAILED", flush=True)
                    failed.append(f"{junction}/blender")
            if not args.keep_export:
                shutil.rmtree(out_dir / "export", ignore_errors=True)
        except Exception as error:  # one junction must not sink the rest
            print(f"[{junction}] FAILED: {error}", flush=True)
            traceback.print_exc()
            failed.append(junction)

    print(f"[summary] reference frames -> {BEV_REFERENCE_ROOT}")
    if failed:
        print(f"[summary] failed: {', '.join(failed)}")
        return 1
    return 0
