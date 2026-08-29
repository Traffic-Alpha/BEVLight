'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: The mask painted over the frame it was built for, so a human can check it.

A lane mask is a uint8 image of lane ids: correct and incorrect ones look
identical. The only practical check is to put it on top of the render it is
supposed to align with and look, which is what these write -- per role, per
renderer, and a contact sheet across all twelve junctions.

Nothing here is part of building a mask. It is part of believing one.
'''

from __future__ import annotations

from pathlib import Path

from ...cli.viz import colorize
from ...paths import EPISODES_ROOT, LANE_MASK_DIR_NAME as MASK_DIR_NAME, SCENARIOS_ROOT

# One overlay per renderer. The two draw the same scene through their own
# cameras, so a mask can agree with one and not the other, and only checking
# both can tell those cases apart.
RENDER_VARIANTS = ("panda_day", "blender_day")


def find_bev_render(junction: str, resolution: tuple[int, int],
                    variant: str | None = None, plan: str | None = None) -> Path | None:
    """Find a rendered junction BEV frame matching the mask resolution.

    Reference frames first. They are the only renders that record the camera
    they were shot with, so they are the only ones that can be checked against
    the camera the mask was rasterised for. An episode frame carries no such
    record: matching resolutions say nothing about matching windows, and a
    stale one puts today's mask over yesterday's framing — which looks exactly
    like a broken mask and is not one.
    """
    import cv2

    from .bev_reference import reference_frame

    wanted = [variant] if variant else list(RENDER_VARIANTS)
    for name in wanted:
        path = reference_frame(junction, plan, name) if plan else None
        if path is None:
            continue
        image = cv2.imread(str(path))
        if image is not None and (image.shape[1], image.shape[0]) == resolution:
            return path
    roots = [
        (name, root)
        for name in wanted
        for root in sorted(EPISODES_ROOT.glob(f"{junction}__*/images/{name}/rgb"))
    ]
    for name, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.png")):
            image = cv2.imread(str(path))
            if image is not None and (image.shape[1], image.shape[0]) == resolution:
                print(f"        overlay [{name}] has no reference frame; falling back to an "
                      f"episode frame, whose camera is unrecorded and may be stale: {path}")
                return path
    return None


def write_overlay(
    path: Path, mask, records: list[dict], render: Path,
    roles: tuple[str, ...] | None = None,
) -> None:
    import cv2
    import numpy as np

    image = cv2.imread(str(render))
    color = colorize(mask, records, roles)
    blended = image.copy()
    hit = color.any(axis=2)
    blended[hit] = (0.42 * image[hit] + 0.58 * color[hit]).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)


def write_contact_sheet(path: Path, junctions: list, tile: int = 560) -> Path | None:
    """One image holding every junction's overlay, for checking them in one pass.

    Alignment is judged by eye and by nothing else, so the twelve checks belong
    on one sheet rather than in twelve directories — a systematic shift shows up
    as a pattern across junctions, which is exactly how the last one was caught.
    A junction with no rendered frame yet gets a labelled blank rather than being
    dropped, so its absence is visible instead of silent.
    """
    import cv2
    import numpy as np

    cells = []
    for junction, variant in [(j, v) for j in junctions for v in ("panda", "blender")]:
        overlays = sorted(
            (SCENARIOS_ROOT / junction / MASK_DIR_NAME).glob(f"overlay_*_{variant}.png")
        )
        if overlays:
            image = cv2.imread(str(overlays[0]))
            cell = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            plan = overlays[0].stem.split("_")[1]
            label = f"{junction}  {plan}  [{variant}]"
        else:
            cell = np.full((tile, tile, 3), 26, dtype=np.uint8)
            cv2.putText(cell, f"no {variant} frame", (tile // 2 - 110, tile // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
            label = f"{junction}  [{variant}]  (no render)"
        strip = np.full((34, tile, 3), 18, dtype=np.uint8)
        cv2.putText(strip, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (235, 235, 235), 1, cv2.LINE_AA)
        cells.append(np.vstack([strip, cell]))

    if not cells:
        return None
    columns = 4
    while len(cells) % columns:
        cells.append(np.full_like(cells[0], 18))
    rows = [np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path
