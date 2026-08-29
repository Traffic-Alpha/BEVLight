'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Painting a lane-id mask so a human can read it.

A lane mask is a uint8 image of mask ids, which is unreadable as pixels. These
turn it into something an eye can check: role by hue, lane by shade, id printed
on the lane it belongs to.

This lives in `utils` rather than beside the mask builder because two unrelated
callers need it -- the builder drawing its overlays, and the collection frame
check drawing a mask over a rendered frame -- and the builder is a one-off
command that nothing else should have to import to get a colour.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from pathlib import Path

ROLE_HUE_BAND = {
    # OpenCV hue is 0-180. Incoming lanes take the warm half of the wheel and
    # outgoing lanes the cool half, so an overlay says which role a lane plays
    # before you read a single id. Within a band the hue still advances by the
    # golden ratio: neighbouring lanes get neighbouring mask ids, and a linear
    # ramp would paint them in nearly the same color.
    "incoming": (0, 40),
    "outgoing": (95, 140),
    "internal": (150, 170),
}


def palette(records: list[dict]):
    """One BGR color per record, in the records' own order."""
    import cv2
    import numpy as np

    rows = np.zeros((max(len(records), 1), 1, 3), dtype=np.uint8)
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        role = record["role"]
        rank = seen.get(role, 0)
        seen[role] = rank + 1
        low, high = ROLE_HUE_BAND[role]
        rows[index, 0] = (
            low + (rank * 0.61803398875) % 1.0 * (high - low),
            235 if role == "internal" else 235 - (rank % 3) * 40,
            160 if role == "internal" else 255 - (rank % 4) * 35,
        )
    return cv2.cvtColor(rows, cv2.COLOR_HSV2BGR).reshape(-1, 3)


def colorize(mask, records: list[dict], roles: tuple[str, ...] | None = None):
    """Paint the mask. ``roles`` limits it to lanes playing those roles."""
    import numpy as np

    colors = palette(records)
    lut = np.zeros((len(records) + 1, 3), dtype=np.uint8)
    for index, record in enumerate(records):
        if roles is not None and record["role"] not in roles:
            continue
        lut[record["mask_id"]] = colors[index]
    return lut[np.clip(mask, 0, len(records))]


def write_preview(path: Path, mask, records: list[dict]) -> None:
    """The painted mask with every id printed on its own lane."""
    import cv2
    import numpy as np

    canvas = colorize(mask, records)
    for record in records:
        pixels = np.argwhere(mask == record["mask_id"])
        if pixels.size == 0:
            continue
        cy, cx = pixels.mean(axis=0)
        cv2.putText(
            canvas, str(record["mask_id"]), (int(cx) - 6, int(cy) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, str(record["mask_id"]), (int(cx) - 6, int(cy) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)
