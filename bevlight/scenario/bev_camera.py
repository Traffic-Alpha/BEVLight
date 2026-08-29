'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Junction BEV camera geometry shared by rendering and lane masks.

The junction BEV sensor is an orthographic top-down camera placed above the
centroid of the incoming-road stop lines. Because the camera is orthographic and
the road surface is flat at z=0, the world -> pixel mapping is a plain affine
transform, which is what makes a one-off static lane mask possible:

    u = (x - cx) * scale + width  / 2
    v = (cy - y) * scale + height / 2
    scale = height_px / ortho_size

`ortho_size` is the *vertical* world extent covered by the camera, matching both
the Panda3D film height and the Blender `ortho_scale` with `sensor_fit=VERTICAL`.
The mapping was verified against rendered BEV frames: lane polygons rasterized
this way land exactly on the rendered lane edge lines.

Sizing the window is a two-sided problem, and both sides are handled here:

  * Too narrow and the queue runs out of the image. A queued vehicle occupies
    `JAM_SPACING_M` = 7.5 m (5.0 m body + SUMO's 2.5 m minGap), so exposing
    `TARGET_VISIBLE_APPROACH_M` = 60 m of approach shows about 8 queued
    vehicles per lane. Junction sizes differ a lot, so the window is solved
    per junction rather than fixed: at a common 90 m window the exposed
    approach ranges from 7 m (Songdo) to 46 m (Beihuan).

  * Too wide and the vehicles shrink. That is avoided by never changing the
    pixel scale: `PIXELS_PER_METER` is fixed, and the render resolution grows
    with the window instead. A car stays 20x57 px and a 3.2 m lane stays 2.6
    ViT patches wide at every junction, so widening the window costs render
    time and disk, never image detail — and every junction reaches the model
    at the same scale, which a single global window did not achieve.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..paths import BEV_CAMERA_TABLE

DEFAULT_BEV_ORTHO_SIZE = 90.0
DEFAULT_BEV_ENV = "normal_high_density"
DEFAULT_BEV_DEMAND = "high_density"

# --- window sizing ---------------------------------------------------------
# One queued vehicle: 5.0 m body (the scenarios' vType) + SUMO's default 2.5 m
# minGap. Everything about queue capacity is derived from this number.
JAM_SPACING_M = 7.5
# How much approach, measured back from the stop line, must stay inside the
# image. 60 m is about 8 queued vehicles; it is also close to the ceiling the
# networks allow, since Massy's and YMT's approaches are only ~98 m long.
TARGET_VISIBLE_APPROACH_M = 60.0
# Fixed image scale: this is what keeps vehicles the same size everywhere.
PIXELS_PER_METER = 1022.0 / 90.0
# Render resolutions are rounded to a multiple of the ViT patch size so the
# feature map has no partial patch at the edge.
PATCH_SIZE = 14
# The camera flies high enough to clear buildings; with an orthographic camera
# the altitude does not affect framing, only what it can see past.
BEV_HEIGHT_MARGIN_M = 60.0


def load_camera_table(path=None) -> dict:
    """Solved per-junction windows, or {} before `bevlight scenario build-lane-masks` runs."""
    table_path = BEV_CAMERA_TABLE if path is None else path
    if not table_path.is_file():
        return {}
    return json.loads(table_path.read_text()).get("junctions", {})


def camera_entry(junction: str, plan: str | None = None) -> dict | None:
    """The solved window for one junction, preferring a plan-specific entry."""
    entry = load_camera_table().get(junction)
    if entry is None:
        return None
    plans = entry.get("plans", {})
    if plan is not None and plan in plans:
        return plans[plan]
    # Geometry is shared across plans at every junction but Chengdu_Guanghua.
    return plans.get(entry.get("canonical_plan")) or next(iter(plans.values()), None)


def bev_ortho_size(junction: str, plan: str | None = None) -> float:
    """Vertical world extent (meters) covered by a junction BEV camera."""
    entry = camera_entry(junction, plan)
    return float(entry["ortho_size"]) if entry else float(DEFAULT_BEV_ORTHO_SIZE)


def bev_height(junction: str, plan: str | None = None) -> float:
    """Camera altitude. Orthographic, so this frames nothing; it only clears obstacles."""
    entry = camera_entry(junction, plan)
    if entry and "height" in entry:
        return float(entry["height"])
    return float(bev_ortho_size(junction, plan) + BEV_HEIGHT_MARGIN_M)


def bev_resolution(junction: str, plan: str | None = None) -> tuple[int, int]:
    """Render resolution that holds `PIXELS_PER_METER` at this junction."""
    entry = camera_entry(junction, plan)
    if entry:
        return int(entry["resolution"][0]), int(entry["resolution"][1])
    size = resolution_for_ortho(bev_ortho_size(junction, plan))
    return size, size


def resolution_for_ortho(ortho_size: float) -> int:
    """Square render size holding the fixed pixel scale, rounded to a whole patch."""
    raw = float(ortho_size) * PIXELS_PER_METER
    return max(PATCH_SIZE, int(round(raw / PATCH_SIZE)) * PATCH_SIZE)


def visible_approach_lengths(center, lane_polygons: dict, ortho_size: float) -> dict:
    """Metres of each lane left inside a candidate window.

    Uses the lane's own polygon area rather than a projection along an axis, so
    approaches that leave the junction diagonally are measured honestly.
    """
    from shapely.geometry import Polygon, box

    cx, cy = float(center[0]), float(center[1])
    half = float(ortho_size) / 2.0
    window = box(cx - half, cy - half, cx + half, cy + half)

    lengths = {}
    for lane_id, (polygon, width) in lane_polygons.items():
        if width <= 0:
            continue
        geom = Polygon(polygon)
        if not geom.is_valid:
            geom = geom.buffer(0)
        clipped = geom.intersection(window)
        lengths[lane_id] = (clipped.area / width) if clipped.area else 0.0
    return lengths


def solve_ortho_size(
    center,
    lane_polygons: dict,
    target: float = TARGET_VISIBLE_APPROACH_M,
    min_ortho: float = 60.0,
    max_ortho: float = 260.0,
    tolerance: float = 0.25,
) -> float:
    """Smallest window exposing `target` metres of the median incoming approach.

    Bisection on the median rather than the minimum: one short slip lane should
    not drag the whole window (and its render cost) wide. Returns `max_ortho`
    when the approaches physically end before the target is reachable.
    """
    import numpy as np

    def median_visible(ortho: float) -> float:
        lengths = visible_approach_lengths(center, lane_polygons, ortho)
        return float(np.median(list(lengths.values()))) if lengths else 0.0

    if median_visible(max_ortho) < target:
        return max_ortho

    low, high = min_ortho, max_ortho
    while high - low > tolerance:
        mid = (low + high) / 2.0
        if median_visible(mid) < target:
            low = mid
        else:
            high = mid
    return round(high, 2)


BEVLIGHT_PRESET = "BEVLIGHT"


def apply_junction_render_preset(junction: str, plan: str | None = None) -> str:
    """Register this junction's solved resolution as a Panda3D preset.

    tshub only accepts preset *names* from a fixed table, and BEVLight needs a
    different square size per junction, so the solved size is injected under one
    reserved name. Returns the name to pass as `preset`.
    """
    from tshub.tshub_env3d.renderers.panda.rendering_components import scene_sync

    width, height = bev_resolution(junction, plan)
    original = scene_sync.SceneSync.__init__

    if not getattr(scene_sync, "_bevlight_preset_patched", False):
        def __init__(self, *args, preset=BEVLIGHT_PRESET, **kwargs):
            if preset == BEVLIGHT_PRESET:
                # Borrow a valid name past the table check, then set the real size.
                original(self, *args, preset="720P_SQUARE", **kwargs)
                self.fig_width, self.fig_height = scene_sync._bevlight_preset_size
                return
            original(self, *args, preset=preset, **kwargs)

        scene_sync.SceneSync.__init__ = __init__
        scene_sync._bevlight_preset_patched = True

    scene_sync._bevlight_preset_size = (width, height)
    return BEVLIGHT_PRESET


def apply_junction_camera_overrides(junction: str, plan: str | None = None) -> None:
    """Patch the tshub BEV rigs in-place to this junction's solved window."""
    from dataclasses import replace
    from tshub.tshub_env3d.core.sensors import sensor_rig

    ortho = bev_ortho_size(junction, plan)
    for sensor_type in ("junction_bev_rgb", "junction_bev_seg"):
        rig = sensor_rig.CAMERA_RIGS.get(sensor_type)
        if rig is not None:
            sensor_rig.CAMERA_RIGS[sensor_type] = replace(rig, ortho_size=ortho)


@dataclass(frozen=True)
class BevCamera:
    """Orthographic top-down junction camera, in SUMO world coordinates."""

    center: tuple[float, float]
    height: float
    ortho_size: float

    def scale(self, resolution: Sequence[int]) -> float:
        """Pixels per meter for a given (width, height) render resolution."""
        return float(resolution[1]) / float(self.ortho_size)

    def world_to_pixel(self, points: Iterable[Sequence[float]], resolution: Sequence[int]):
        """Project world (x, y) points to (u, v) pixel coordinates.

        Returns a float ndarray of shape (N, 2); callers round it themselves so
        that polygon rasterization can use sub-pixel shifts if needed.
        """
        import numpy as np

        width, height = int(resolution[0]), int(resolution[1])
        scale = self.scale((width, height))
        pts = np.asarray(list(points), dtype=float)[:, :2]
        u = (pts[:, 0] - self.center[0]) * scale + width / 2.0
        v = (self.center[1] - pts[:, 1]) * scale + height / 2.0
        return np.stack([u, v], axis=1)

    def world_bounds(self, resolution: Sequence[int]) -> tuple[float, float, float, float]:
        """World-space (xmin, ymin, xmax, ymax) covered by the rendered image."""
        width, height = int(resolution[0]), int(resolution[1])
        scale = self.scale((width, height))
        half_w = width / (2.0 * scale)
        half_h = height / (2.0 * scale)
        cx, cy = self.center
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def to_dict(self) -> dict:
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "height": float(self.height),
            "ortho_size": float(self.ortho_size),
            "top_down": True,
        }


def resolve_bev_camera(junction: str, center: Sequence[float]) -> BevCamera:
    """Build the BEV camera for a junction from its computed center."""
    return BevCamera(
        center=(float(center[0]), float(center[1])),
        height=bev_height(junction),
        ortho_size=bev_ortho_size(junction),
    )
