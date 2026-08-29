'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Split a junction BEV frame into per-lane views using the static lane mask.

For every junction this writes the untouched BEV frame plus one image per lane,
each keeping only that lane's pixels. The lane masks come from
`scenarios/build_lane_masks.py` and never change, so this step is pure image
work: no SUMO, no renderer.

Outputs:

    runs/reports/lane_views/<junction>/
      bev.png                        the source BEV frame, unmodified
      lanes/lane_01__<lane id>.png   only lane 1 kept, rest blacked out
      lanes/lane_02__<lane id>.png
      index.json                     file <-> SUMO lane id, role, phases
      contact_sheet.png              BEV + every lane view on one sheet

Examples:
  # All junctions, using the best available BEV render of each.
  conda run -n tshub python tools/export_lane_views.py

  # One junction, keep the surroundings dimmed instead of black.
  conda run -n tshub python tools/export_lane_views.py --junction Beijing_Beihuan --background dim

  # Crop each lane view to the lane itself.
  conda run -n tshub python tools/export_lane_views.py --junction Beijing_Beihuan --crop
@LastEditTime: 2026-08-20
@LastEditors: WANG Maonan
'''

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from ...paths import (
    EPISODES_ROOT,
    LANE_VIEWS_ROOT,
)

SOURCE_ORDER = ("blender", "panda")


def safe_name(lane_id: str) -> str:
    """SUMO lane ids contain '#', ':' and '.', so make them filename safe."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", lane_id)


def find_bev_frames(junction: str, source: str) -> list[Path]:
    """Rendered junction BEV frames for a junction, newest renderer first."""
    sources = SOURCE_ORDER if source == "auto" else (source,)
    frames: list[Path] = []
    for name in sources:
        variant = f"{name}_day"
        for root in sorted(EPISODES_ROOT.glob(f"{junction}__*/images/{variant}/rgb")):
            frames.extend(sorted(root.glob("*.png")))
        if frames:
            break
    return frames


def contact_sheet(bev, tiles: list[tuple[str, "object"]], columns: int = 6):
    """BEV plus every lane view on one labeled sheet."""
    import cv2
    import numpy as np

    tile_size = 200
    pad = 8
    label_h = 22
    cell_h = tile_size + label_h + pad
    cell_w = tile_size + pad

    def fit(image):
        canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        height, width = image.shape[:2]
        scale = min(tile_size / width, tile_size / height)
        resized = cv2.resize(
            image, (max(int(width * scale), 1), max(int(height * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
        y = (tile_size - resized.shape[0]) // 2
        x = (tile_size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    cells = [("BEV", fit(bev))] + [(label, fit(image)) for label, image in tiles]
    rows = (len(cells) + columns - 1) // columns
    sheet = np.full((rows * cell_h + pad, columns * cell_w + pad, 3), 30, dtype=np.uint8)
    for index, (label, tile) in enumerate(cells):
        row, col = divmod(index, columns)
        y0 = pad + row * cell_h
        x0 = pad + col * cell_w
        sheet[y0:y0 + tile_size, x0:x0 + tile_size] = tile
        cv2.putText(
            sheet, label[:28], (x0 + 2, y0 + tile_size + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA,
        )
    return sheet


def export_one_junction(
    junction: str,
    plan: str,
    source: str,
    frame_index: int,
    background: str,
    dim: float,
    crop: bool,
    image_override: Path | None,
    make_sheet: bool,
    clean: bool,
) -> dict:
    import cv2

    from ..lane_mask import load_lane_mask

    if image_override is not None:
        bev_path = image_override
    else:
        frames = find_bev_frames(junction, source)
        if not frames:
            raise FileNotFoundError(
                f"{junction}: no junction BEV render found under "
                f"{EPISODES_ROOT}/<episode>/images/(blender_day|panda_day)/rgb. "
                "Run `conda run -n tshub python tools/collect_episodes.py --junction ...` first."
            )
        if frame_index >= len(frames):
            raise IndexError(
                f"{junction}: frame {frame_index} requested but only {len(frames)} found"
            )
        bev_path = frames[frame_index]

    bev = cv2.imread(str(bev_path))
    if bev is None:
        raise FileNotFoundError(f"{junction}: cannot read BEV frame {bev_path}")
    resolution = (bev.shape[1], bev.shape[0])
    mask = load_lane_mask(junction, plan, resolution)

    out_dir = LANE_VIEWS_ROOT / f"{junction}__{plan}"
    lane_dir = out_dir / "lanes"
    if clean:
        shutil.rmtree(out_dir, ignore_errors=True)
    lane_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "bev.png"), bev)

    entries = []
    tiles = []
    skipped = []
    for record in mask.lanes:
        lane_id = record["lane_id"]
        if mask.lane_bbox(lane_id) is None:
            skipped.append(lane_id)
            continue
        view = mask.isolate(bev, lane_id, background=background, dim=dim, crop=crop)
        name = f"lane_{record['mask_id']:02d}__{safe_name(lane_id)}.png"
        cv2.imwrite(str(lane_dir / name), view)
        entries.append(
            {
                "file": f"lanes/{name}",
                "mask_id": record["mask_id"],
                "lane_id": lane_id,
                "edge_id": record["edge_id"],
                "lane_index": record["lane_index"],
                "role": record["role"],
                "directions": record["directions"],
                "phases": record["phases"],
                "pixels": int(mask.pixels_of(lane_id).sum()),
            }
        )
        tiles.append((f"{record['mask_id']:02d} {lane_id}", view))

    if make_sheet and tiles:
        cv2.imwrite(str(out_dir / "contact_sheet.png"), contact_sheet(bev, tiles))

    index = {
        "junction": junction,
        "tls_id": mask.tls_id,
        "source_frame": str(bev_path.relative_to(PROJECT_ROOT)),
        "resolution": list(resolution),
        "background": background,
        "crop": crop,
        "lane_mask": f"scenarios/{junction}/lane_mask/lane_mask_{resolution[0]}x{resolution[1]}.png",
        "lane_count": len(entries),
        "lanes_without_pixels": skipped,
        "lanes": entries,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    print(f"[done] {junction}: {len(entries)} lane view(s) from {bev_path.name} -> {out_dir}")
    if skipped:
        print(f"        {len(skipped)} lane(s) outside the BEV window, skipped")
    return index


def available_junctions() -> list[str]:
    from ..loader import AVAILABLE_JUNCTIONS

    return list(AVAILABLE_JUNCTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-lane BEV views from lane masks.")
    parser.add_argument("--junction", nargs="+", default=None, help="Junctions to export. Default: all.")
    parser.add_argument("--plan", default=None, help="Signal plan whose mask to apply. Default: the junction's canonical plan.")
    parser.add_argument("--source", choices=["auto", "blender", "panda"], default="auto",
                        help="Which rendered BEV frames to cut up. Default: blender, then panda.")
    parser.add_argument("--frame", type=int, default=0, help="Index into the sorted BEV frames. Default: 0.")
    parser.add_argument("--image", default=None, help="Use this BEV image instead of searching the render outputs.")
    parser.add_argument("--background", choices=["black", "dim", "white"], default="black",
                        help="What to do with everything outside the lane. Default: black.")
    parser.add_argument("--dim", type=float, default=0.2, help="Brightness kept outside the lane for --background dim.")
    parser.add_argument("--crop", action="store_true", help="Crop each lane view to the lane bounding box.")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Do not write the per-junction contact sheet.")
    parser.add_argument("--clean", action="store_true", help="Remove existing lane views for the junction first.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a junction fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without writing images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    junctions = args.junction or available_junctions()
    image_override = Path(args.image).resolve() if args.image else None
    if image_override is not None and len(junctions) != 1:
        raise SystemExit("--image needs exactly one --junction")

    from ..lane_mask import available_plans as mask_plans

    def plan_for(junction: str) -> str:
        plans = mask_plans(junction)
        if not plans:
            raise FileNotFoundError(
                f"{junction}: no lane mask built yet. "
                f"Run `conda run -n tshub python tools/build_lane_masks.py --junction {junction}`."
            )
        if args.plan:
            if args.plan not in plans:
                raise ValueError(f"{junction}: no mask for plan '{args.plan}'. Available: {plans}")
            return args.plan
        return "normal" if "normal" in plans else plans[0]

    print(
        f"[plan] junctions={len(junctions)} plan={args.plan or 'canonical'} "
        f"source={args.source} frame={args.frame} "
        f"background={args.background} crop={args.crop}"
    )
    for junction in junctions:
        print(f"  - {junction} -> {LANE_VIEWS_ROOT / f'{junction}__{plan_for(junction)}'}")

    if args.dry_run:
        return 0

    failures = []
    for junction in junctions:
        try:
            export_one_junction(
                junction=junction,
                plan=plan_for(junction),
                source=args.source,
                frame_index=args.frame,
                background=args.background,
                dim=args.dim,
                crop=args.crop,
                image_override=image_override,
                make_sheet=not args.no_contact_sheet,
                clean=args.clean,
            )
        except Exception as exc:
            failures.append((junction, exc))
            print(f"[fail] {junction}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for junction, exc in failures:
            print(f"  - {junction}: {exc}", file=sys.stderr)
        return 1

    print(f"[summary] exported lane views for {len(junctions)} junction(s) -> {LANE_VIEWS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
