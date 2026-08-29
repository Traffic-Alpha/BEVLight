"""One reference BEV frame per junction, kept so overlays can always be checked.

Ten of the twelve junctions are collected into episodes, so an overlay can be
drawn over a frame that already exists. The two cross-structure test junctions
are never collected — training must not touch them — and so have no frame of
their own. Without one, a third of the overlay checks silently print "skipped",
which is how a camera error stayed invisible once already.

These frames are that missing ground. They are cheap (one simulated episode,
one rendered frame per renderer) and they are regenerated from the same camera
solution the pipeline renders with, so a stale one is detectable rather than
merely wrong: `reference_frame` refuses any frame whose recorded ortho window
disagrees with the junction's current one.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...paths import BEV_REFERENCE_ROOT


#: Enough seconds for queues to build; a frame of empty road proves nothing.
REFERENCE_SECOND = 420


def reference_dir(junction: str, ortho_size: float) -> Path:
    """Frames live under the camera they were shot with, not the plan.

    Two plans at one junction often solve to the same window and can share a
    frame; when they do not, keying by plan would let one overwrite the other.
    """
    return BEV_REFERENCE_ROOT / junction / f"o{ortho_size:.0f}"


def reference_meta(junction: str, ortho_size: float) -> dict | None:
    path = reference_dir(junction, ortho_size) / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def reference_frame(junction: str, plan: str, variant: str) -> Path | None:
    """The stored frame for this variant, or None if there is none."""
    from ..bev_camera import bev_ortho_size

    ortho = bev_ortho_size(junction, plan)
    directory = reference_dir(junction, ortho)
    meta = reference_meta(junction, ortho)
    if meta is None or abs(float(meta["ortho_size"]) - ortho) > 0.01:
        # A frame whose recorded window disagrees with the current one would
        # show a misalignment that no longer exists, or hide one that does.
        return None
    path = directory / f"{variant}.png"
    return path if path.is_file() else None
