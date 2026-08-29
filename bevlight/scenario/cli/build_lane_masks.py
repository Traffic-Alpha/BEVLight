'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Build one static lane-id mask per junction BEV camera.

The junction BEV camera is orthographic and top-down, and the road surface is
flat, so the world -> pixel mapping is a fixed affine transform (see
`scenarios/bev_camera.py`). The lane layout never changes during an episode,
which is why the mask only has to be generated **once** per junction and can
then be reused by every rendered frame.

Pipeline per junction:
  1) SUMO reset (traffic-light builder only) -> BEV camera center, the lanes of
     the junction's in/out roads, and their movements
  2) sumolib lane shapes of those lanes -> lane polygons in world coordinates
  3) rasterize polygons -> uint16 label image, pixel value = mask id
  4) write mask png(s) + lane_mask.json linking mask id <-> SUMO lane id

Outputs are written under each junction:

    scenarios/<junction>/lane_mask/
      lane_mask_1022x1022.png   uint16, 0 = background, k = lanes[k-1]
      lane_mask_720x720.png
      lane_mask.json            mask id <-> lane id, camera, movements, phases
      preview.png               colored sanity-check image

Examples:
  # Build masks for all junctions.
  python scenarios/build_lane_masks.py

  # One junction, plus an overlay against an existing Panda BEV render.
  python scenarios/build_lane_masks.py --junction Beijing_Beihuan --overlay

  # Also label the internal (inside-junction) lanes.
  python scenarios/build_lane_masks.py --include-internal

@LastEditTime: 2026-08-20
@LastEditors: WANG Maonan
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...cli.tshub import configure_tshub_import, resolve_tshub_root
from ...cli.viz import write_preview
from ...paths import (
    LANE_MASK_CHECK_ROOT as OVERLAY_ROOT,
    LANE_MASK_DIR_NAME as MASK_DIR_NAME,
    PROJECT_ROOT,
    SCENARIOS_ROOT,
)
from .._internal.lane_geometry import collect_lanes, read_tls_state
from .._internal.mask_overlay import (
    RENDER_VARIANTS,
    find_bev_render,
    write_contact_sheet,
    write_overlay,
)

DEFAULT_RESOLUTIONS = ("1022x1022", "720x720")
META_NAME = "lane_mask.json"
PREVIEW_NAME = "preview.png"

# Drawing order: larger rank is drawn last and therefore wins overlapping pixels.
ROLE_DRAW_RANK = {"internal": 0, "outgoing": 1, "incoming": 2}


def parse_resolution(value: str) -> tuple[int, int]:
    parts = str(value).lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must look like WIDTHxHEIGHT, got: {value}")
    width, height = (int(v) for v in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got: {value}")
    return width, height


def available_junctions() -> list[str]:
    from ..loader import AVAILABLE_JUNCTIONS

    return list(AVAILABLE_JUNCTIONS)












def rasterize(records: list[dict], camera, resolution: tuple[int, int]):
    """Rasterize lane polygons into a uint16 label image."""
    import cv2
    import numpy as np

    width, height = resolution
    mask = np.zeros((height, width), dtype=np.uint16)
    for record in sorted(records, key=lambda rec: ROLE_DRAW_RANK[rec["role"]]):
        pixels = camera.world_to_pixel(record["polygon_world"], resolution)
        contour = np.round(pixels).astype(np.int32)
        cv2.fillPoly(mask, [contour], int(record["mask_id"]))
    return mask








# Suffix -> the roles that view paints. An empty suffix keeps the combined
# overlay at its usual name, which is what the contact sheet tiles.
OVERLAY_ROLE_VIEWS = (
    ("", None),
    ("_in", ("incoming",)),
    ("_out", ("outgoing",)),
)

CONTACT_SHEET_NAME = "lane_mask_overlays.png"




def plan_signal_record(tls_info: dict) -> dict:
    """The parts of the traffic-light state that depend on the signal plan."""
    return {
        "movement_ids": tls_info.get("movement_ids", []),
        "movement_directions": tls_info.get("movement_directions", {}),
        "movement_lane_ids": tls_info.get("movement_lane_ids", {}),
        "phase2movements": tls_info.get("phase2movements", {}),
        "num_phases": len(tls_info.get("phase2movements", {})),
    }


def geometry_fingerprint(records: list[dict], camera) -> tuple:
    """What must match for two plans to be able to share one mask image."""
    return (
        tuple(sorted((r["lane_id"], r["role"], r["width"], tuple(map(tuple, r["polygon_world"]))) for r in records)),
        (round(camera.center[0], 3), round(camera.center[1], 3)),
        round(camera.ortho_size, 2),
    )


def build_one_plan(
    junction: str,
    plan: str,
    demand: str,
    extra_resolutions: list[tuple[int, int]],
    include_internal: bool,
    overlay: bool,
    seed: int,
    out_dir: Path,
) -> dict:
    """Solve the window, rasterize the mask and write the images for one plan."""
    import cv2
    import numpy as np
    import sumolib

    from ..bev_camera import (
        BEV_HEIGHT_MARGIN_M,
        JAM_SPACING_M,
        TARGET_VISIBLE_APPROACH_M,
        BevCamera,
        resolution_for_ortho,
        solve_ortho_size,
        visible_approach_lengths,
    )
    from ..loader import load_junction_config

    env_name = f"{plan}_{demand}"
    cfg = load_junction_config(junction, env_name)
    tls_info, bev_rig = read_tls_state(junction, env_name, seed=seed)
    center = (float(bev_rig["position"][0]), float(bev_rig["position"][1]))

    net_file = Path(cfg["net_file"])
    net = sumolib.net.readNet(str(net_file), withInternal=True)
    records = collect_lanes(net, tls_info, include_internal=include_internal)
    if not records:
        raise RuntimeError(f"{junction}/{plan}: no lanes found in {net_file}")

    # Size the window from the incoming approaches: they are what has to hold a queue.
    approach_polygons = {
        r["lane_id"]: (r["polygon_world"], r["width"])
        for r in records
        if r["role"] == "incoming"
    }
    ortho = solve_ortho_size(center, approach_polygons)
    primary = resolution_for_ortho(ortho)
    camera = BevCamera(center=center, height=ortho + BEV_HEIGHT_MARGIN_M, ortho_size=ortho)

    visible = visible_approach_lengths(center, approach_polygons, ortho)
    visible_values = sorted(visible.values())
    resolutions = [(primary, primary)] + [r for r in extra_resolutions if r != (primary, primary)]

    masks = []
    for index, resolution in enumerate(resolutions):
        mask = rasterize(records, camera, resolution)
        counts = np.bincount(mask.reshape(-1), minlength=len(records) + 1)
        name = f"lane_mask_{plan}_{resolution[0]}x{resolution[1]}.png"
        cv2.imwrite(str(out_dir / name), mask)
        for record in records:
            record.setdefault("visible_pixels", {})[f"{resolution[0]}x{resolution[1]}"] = int(
                counts[record["mask_id"]]
            )
        masks.append(
            {
                "file": name,
                "width": resolution[0],
                "height": resolution[1],
                "scale_px_per_m": round(camera.scale(resolution), 6),
                "labeled_pixels": int((mask > 0).sum()),
            }
        )
        if index == 0:
            write_preview(out_dir / f"preview_{plan}.png", mask, records)
            # Always write the mask-over-render check beside the mask itself.
            # The colour-only preview cannot show alignment: it has no ground to
            # be aligned against. Only this one can be judged by eye, and it is
            # the judgement that has repeatedly caught what the numbers missed.
            if index == 0:
                for variant in RENDER_VARIANTS:
                    render = find_bev_render(junction, resolution, variant, plan)
                    short = variant.split("_")[0]
                    target = out_dir / f"overlay_{plan}_{short}.png"
                    if render is None:
                        target.unlink(missing_ok=True)
                        print(f"        overlay [{short}] skipped: no {variant} frame "
                              f"at {resolution[0]}x{resolution[1]} "
                              f"(run bevlight scenario render-reference --junction {junction})")
                        continue
                    # Combined, then each role alone. Outgoing lanes decide
                    # whether a movement's exit is blocked, and painted together
                    # with the incoming ones they are easy to mistake for them.
                    for suffix, roles in OVERLAY_ROLE_VIEWS:
                        write_overlay(
                            target.with_name(f"{target.stem}{suffix}.png"),
                            mask, records, render, roles,
                        )
                    print(f"        overlay [{short}] -> {target.name} (+in, +out)")

            if overlay:
                render = find_bev_render(junction, resolution)
                if render is not None:
                    write_overlay(OVERLAY_ROOT / f"{junction}_{plan}.png", mask, records, render)
                else:
                    print(
                        f"        overlay skipped for {plan}: "
                        f"no {resolution[0]}x{resolution[1]} BEV render"
                    )

    primary_key = f"{primary}x{primary}"
    invisible = [rec["lane_id"] for rec in records if rec["visible_pixels"][primary_key] == 0]

    return {
        "plan": plan,
        "env": env_name,
        "net_file": str(net_file.relative_to(PROJECT_ROOT)),
        "camera": camera.to_dict(),
        "resolution": [primary, primary],
        "projection": {
            "u": "(x - center_x) * scale_px_per_m + width / 2",
            "v": "(center_y - y) * scale_px_per_m + height / 2",
            "world_bounds": [round(v, 3) for v in camera.world_bounds((primary, primary))],
            "note": "Orthographic top-down camera over flat ground; +y world is up in the image.",
        },
        "window": {
            "target_visible_approach_m": TARGET_VISIBLE_APPROACH_M,
            "jam_spacing_m": JAM_SPACING_M,
            "visible_approach_m": {k: round(v, 2) for k, v in visible.items()},
            "visible_approach_min_m": round(visible_values[0], 2) if visible_values else 0.0,
            "visible_approach_median_m": round(
                visible_values[len(visible_values) // 2], 2
            ) if visible_values else 0.0,
            "queue_capacity_vehicles": round(
                (visible_values[len(visible_values) // 2] if visible_values else 0.0) / JAM_SPACING_M, 2
            ),
        },
        "masks": masks,
        "lane_count": len(records),
        "lanes_without_pixels": invisible,
        "tls": {
            "in_roads": tls_info.get("in_roads", []),
            "out_roads": tls_info.get("out_roads", []),
            "in_road_stop_line": tls_info.get("in_road_stop_line", {}),
            **plan_signal_record(tls_info),
        },
        "lanes": records,
        "_fingerprint": geometry_fingerprint(records, camera),
    }


def build_one_junction(
    junction: str,
    demand: str,
    extra_resolutions: list[tuple[int, int]],
    include_internal: bool,
    overlay: bool,
    seed: int = 7,
    plans: list[str] | None = None,
) -> dict:
    """Build one mask per signal plan.

    The phase composition differs between `easy` and `normal`, so a single mask
    carrying one `phase2movements` would silently mislabel the other plan — and
    at Chengdu_Guanghua even the lane geometry and the camera centre differ.
    Each plan therefore gets a complete, self-contained record.
    """
    from ..loader import available_plans, load_junction_config

    plans = plans or available_plans(junction)
    out_dir = SCENARIOS_ROOT / junction / MASK_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    per_plan = {}
    for plan in plans:
        per_plan[plan] = build_one_plan(
            junction, plan, demand, extra_resolutions, include_internal, overlay, seed, out_dir
        )

    fingerprints = {plan: rec.pop("_fingerprint") for plan, rec in per_plan.items()}
    shared = len(set(fingerprints.values())) == 1
    canonical = "normal" if "normal" in per_plan else plans[0]

    tls_id = load_junction_config(junction, f"{canonical}_{demand}")["tls_id"]
    meta = {
        "junction": junction,
        "tls_id": tls_id,
        "demand": demand,
        "generated_by": "bevlight/scenario/build_lane_masks.py",
        "include_internal": include_internal,
        "background_id": 0,
        "canonical_plan": canonical,
        "geometry_shared_across_plans": shared,
        "plans": per_plan,
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=False))

    for plan, rec in per_plan.items():
        window = rec["window"]
        print(
            f"[done] {junction}/{plan}: lanes={rec['lane_count']} K={rec['tls']['num_phases']} "
            f"ortho={rec['camera']['ortho_size']:.0f}m render={rec['resolution'][0]}px "
            f"visible={window['visible_approach_median_m']:.0f}m "
            f"(~{window['queue_capacity_vehicles']:.1f} veh)"
        )
        if rec["lanes_without_pixels"]:
            print(
                f"        {len(rec['lanes_without_pixels'])} lane(s) outside the window: "
                f"{rec['lanes_without_pixels'][:5]}"
            )
    if not shared:
        print(f"        note: {junction} plans do NOT share geometry; each plan has its own mask")
    return meta


def write_camera_table(metas: list[dict], path: Path) -> None:
    """Compact per-junction camera table for the renderers to read."""
    from ..bev_camera import PIXELS_PER_METER, TARGET_VISIBLE_APPROACH_M

    junctions = {}
    for meta in metas:
        junctions[meta["junction"]] = {
            "canonical_plan": meta["canonical_plan"],
            "geometry_shared_across_plans": meta["geometry_shared_across_plans"],
            "plans": {
                plan: {
                    "center": rec["camera"]["center"],
                    "height": rec["camera"]["height"],
                    "ortho_size": rec["camera"]["ortho_size"],
                    "resolution": rec["resolution"],
                    "visible_approach_median_m": rec["window"]["visible_approach_median_m"],
                    "queue_capacity_vehicles": rec["window"]["queue_capacity_vehicles"],
                }
                for plan, rec in meta["plans"].items()
            },
        }

    payload = {
        "generated_by": "bevlight/scenario/build_lane_masks.py",
        "pixels_per_meter": round(PIXELS_PER_METER, 6),
        "target_visible_approach_m": TARGET_VISIBLE_APPROACH_M,
        "junctions": junctions,
    }
    existing = json.loads(path.read_text()).get("junctions", {}) if path.is_file() else {}
    existing.update(junctions)
    payload["junctions"] = dict(sorted(existing.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static BEV lane-id masks.")
    parser.add_argument("--junction", nargs="+", default=None, help="Junctions to build. Default: all.")
    parser.add_argument("--demand", default=None, help="Demand used to start SUMO. Structure-only, so it does not affect the mask. Default: high_density.")
    parser.add_argument("--plan", nargs="+", default=None, help="Signal plans to build. Default: every plan the junction ships.")
    parser.add_argument(
        "--resolution",
        nargs="*",
        default=[],
        help="Extra mask resolutions on top of the solved one, e.g. 720x720.",
    )
    parser.add_argument("--include-internal", action="store_true", help="Also label internal (inside-junction) lanes.")
    parser.add_argument("--overlay", action="store_true", help="Blend the mask over an existing BEV render for checking.")
    parser.add_argument("--seed", type=int, default=7, help="SUMO seed for the one-off reset.")
    parser.add_argument("--tshub-root", default=None, help="TransSimHub root. Defaults to TSHUB_ROOT or /home/wmn/code/TransSimHub.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip junctions that already have a lane_mask.json.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a junction fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without building.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_tshub_import(resolve_tshub_root(args.tshub_root))

    from ..bev_camera import (
        BEV_CAMERA_TABLE,
        DEFAULT_BEV_DEMAND,
        JAM_SPACING_M,
        TARGET_VISIBLE_APPROACH_M,
    )
    from ..loader import available_plans

    demand = args.demand or DEFAULT_BEV_DEMAND
    junctions = args.junction or available_junctions()
    extra_resolutions = [parse_resolution(value) for value in args.resolution]

    print(
        f"[plan] junctions={len(junctions)} demand={demand} "
        f"target_visible={TARGET_VISIBLE_APPROACH_M:.0f}m "
        f"(~{TARGET_VISIBLE_APPROACH_M / JAM_SPACING_M:.1f} queued vehicles) "
        f"include_internal={args.include_internal}"
    )
    for junction in junctions:
        plans = args.plan or available_plans(junction)
        print(f"  - {junction} plans={plans} -> {SCENARIOS_ROOT / junction / MASK_DIR_NAME}")

    if args.dry_run:
        return 0

    failures = []
    metas = []
    for junction in junctions:
        meta_path = SCENARIOS_ROOT / junction / MASK_DIR_NAME / META_NAME
        if args.skip_existing and meta_path.exists():
            print(f"[skip] {junction}: {meta_path.name} already exists")
            continue
        try:
            metas.append(
                build_one_junction(
                    junction=junction,
                    demand=demand,
                    extra_resolutions=extra_resolutions,
                    include_internal=args.include_internal,
                    overlay=args.overlay,
                    seed=args.seed,
                    plans=args.plan,
                )
            )
        except Exception as exc:
            failures.append((junction, exc))
            print(f"[fail] {junction}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    # Every build refreshes the sheet, over every junction that has a mask —
    # not only the ones just rebuilt, so the picture is always complete.
    all_junctions = sorted(
        d.name for d in SCENARIOS_ROOT.iterdir()
        if (d / MASK_DIR_NAME / META_NAME).is_file()
    )
    sheet = write_contact_sheet(SCENARIOS_ROOT / CONTACT_SHEET_NAME, all_junctions)
    if sheet:
        missing = [f"{j}/{v}" for j in all_junctions for v in ("panda", "blender")
                   if not any((SCENARIOS_ROOT / j / MASK_DIR_NAME).glob(f"overlay_*_{v}.png"))]
        print(f"[summary] overlay sheet ({len(all_junctions)} junctions) -> {sheet}")
        if missing:
            print(f"[summary] no overlay yet for: {', '.join(missing)}")

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for junction, exc in failures:
            print(f"  - {junction}: {exc}", file=sys.stderr)
        return 1

    if metas:
        write_camera_table(metas, BEV_CAMERA_TABLE)
        print(f"[summary] camera table -> {BEV_CAMERA_TABLE}")
    print(f"[summary] built lane masks for {len(metas)} junction(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
