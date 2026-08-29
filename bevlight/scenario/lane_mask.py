'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Read the static BEV lane masks built by `build_lane_masks.py`.

The mask is a uint16 label image aligned with the junction BEV render: pixel
value 0 is background and value `k` belongs to the lane with `mask_id == k`.
This module is the link between an image and the simulation, e.g.

    from bevlight.scenario.lane_mask import load_lane_mask
    mask = load_lane_mask("Beijing_Beihuan", "normal")
    mask.lane_id_of(512, 300)                 # pixel -> SUMO lane id
    mask.pixels_of("157863208#0.1590_1")      # SUMO lane id -> boolean pixels
    mask.incoming_lane_ids()                  # lanes controlled by the tls
    mask.phase2movements                      # this plan's candidate phases
@LastEditTime: 2026-08-20
@LastEditors: WANG Maonan
'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import LANE_MASK_DIR_NAME as MASK_DIR_NAME, SCENARIOS_ROOT

META_NAME = "lane_mask.json"


@dataclass
class LaneMask:
    """A junction lane-id mask plus its simulation-side metadata, for one plan.

    A mask is always scoped to a signal plan. The lane geometry is usually
    shared between `easy` and `normal`, but the phases each lane serves never
    are, so `plan` is part of the identity of a loaded mask rather than an
    afterthought.
    """

    junction: str
    tls_id: str
    plan: str
    labels: Any  # np.ndarray, uint16, shape (height, width)
    meta: dict          # the whole file
    plan_meta: dict     # meta["plans"][plan]

    @property
    def resolution(self) -> tuple[int, int]:
        height, width = self.labels.shape[:2]
        return width, height

    @property
    def lanes(self) -> list[dict]:
        return self.plan_meta["lanes"]

    @property
    def camera(self) -> dict:
        return self.plan_meta["camera"]

    @property
    def tls(self) -> dict:
        """Traffic-light structure under this plan."""
        return self.plan_meta["tls"]

    @property
    def phase2movements(self) -> dict[int, list[str]]:
        """Candidate phases of this plan, each as the set of movements it serves."""
        return {int(k): v for k, v in self.tls["phase2movements"].items()}

    @property
    def num_phases(self) -> int:
        return int(self.tls["num_phases"])

    @property
    def movement_ids(self) -> list[str]:
        return list(self.tls["movement_ids"])

    @property
    def visible_approach_m(self) -> dict[str, float]:
        """Metres of each incoming lane that the BEV window actually exposes."""
        return dict(self.plan_meta["window"]["visible_approach_m"])

    def visible_length_m(self, lane_id: str | None = None):
        """Metres of a lane inside the BEV window, for lanes of any role.

        `visible_approach_m` only covers the incoming lanes, because those are
        what the window is sized around. Outgoing lanes need the same number to
        judge spillback, so it is recomputed here from the stored world polygons
        — no SUMO, and the same clipping the window solver used.
        """
        cached = getattr(self, "_visible_length_cache", None)
        if cached is None:
            from .bev_camera import visible_approach_lengths

            polygons = {
                record["lane_id"]: (record["polygon_world"], record["width"])
                for record in self.lanes
            }
            cached = visible_approach_lengths(
                self.camera["center"], polygons, self.camera["ortho_size"]
            )
            object.__setattr__(self, "_visible_length_cache", cached)
        if lane_id is None:
            return dict(cached)
        return float(cached[lane_id])

    @property
    def queue_capacity_vehicles(self) -> float:
        """Roughly how many queued vehicles fit in the visible approach."""
        return float(self.plan_meta["window"]["queue_capacity_vehicles"])

    def phases_of(self, lane_id: str) -> list[int]:
        """Phases that give this lane green under the loaded plan."""
        return list(self.lane_record(lane_id)["phases"])

    def lane_record(self, lane_id: str) -> dict:
        for record in self.lanes:
            if record["lane_id"] == lane_id:
                return record
        raise KeyError(f"{self.junction}: unknown lane '{lane_id}'")

    def lane_id_of(self, row: int, col: int) -> str | None:
        """SUMO lane id at a pixel, or None for background."""
        mask_id = int(self.labels[row, col])
        return self.id_to_lane.get(mask_id)

    def pixels_of(self, lane_id: str):
        """Boolean pixel mask for one SUMO lane."""
        return self.labels == self.lane_to_id[lane_id]

    def lane_ids(self, role: str | None = None) -> list[str]:
        return [
            record["lane_id"]
            for record in self.lanes
            if role is None or record["role"] == role
        ]

    def incoming_lane_ids(self) -> list[str]:
        return self.lane_ids("incoming")

    def lane_bbox(self, lane_id: str) -> tuple[int, int, int, int] | None:
        """Pixel bounding box (x0, y0, x1, y1) of one lane, None when unseen."""
        import numpy as np

        rows, cols = np.nonzero(self.pixels_of(lane_id))
        if rows.size == 0:
            return None
        return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1

    def isolate(
        self,
        image,
        lane_id: str,
        background: str = "black",
        dim: float = 0.2,
        crop: bool = False,
        pad: int = 8,
    ):
        """Keep only one lane's pixels of a BEV image.

        Args:
            image: BEV frame with the same resolution as this mask.
            lane_id: SUMO lane id to keep.
            background: "black", "white", or "dim" (darken instead of erase).
            dim: brightness kept outside the lane when background is "dim".
            crop: crop the result to the lane bounding box.
            pad: padding in pixels around the crop.

        Returns:
            A new image; the input is not modified.
        """
        import numpy as np

        if (image.shape[1], image.shape[0]) != self.resolution:
            raise ValueError(
                f"{self.junction}: image is {image.shape[1]}x{image.shape[0]} but the "
                f"mask is {self.resolution[0]}x{self.resolution[1]}. Load the mask at "
                f"the image resolution."
            )

        keep = self.pixels_of(lane_id)
        if background == "dim":
            out = (image * dim).astype(image.dtype)
        elif background == "white":
            out = np.full_like(image, 255)
        elif background == "black":
            out = np.zeros_like(image)
        else:
            raise ValueError(f"Unknown background '{background}'")
        out[keep] = image[keep]

        if crop:
            box = self.lane_bbox(lane_id)
            if box is not None:
                x0, y0, x1, y1 = box
                height, width = image.shape[:2]
                out = out[
                    max(y0 - pad, 0):min(y1 + pad, height),
                    max(x0 - pad, 0):min(x1 + pad, width),
                ]
        return out

    @property
    def id_to_lane(self) -> dict[int, str]:
        return {record["mask_id"]: record["lane_id"] for record in self.lanes}

    @property
    def lane_to_id(self) -> dict[str, int]:
        return {record["lane_id"]: record["mask_id"] for record in self.lanes}

    def world_to_pixel(self, points):
        """Project world (x, y) points into this mask's pixel frame."""
        from .bev_camera import BevCamera

        camera = self.camera
        return BevCamera(
            center=tuple(camera["center"]),
            height=camera["height"],
            ortho_size=camera["ortho_size"],
        ).world_to_pixel(points, self.resolution)


def lane_mask_dir(junction: str, scenarios_root: Path | None = None) -> Path:
    return (scenarios_root or SCENARIOS_ROOT) / junction / MASK_DIR_NAME


def available_plans(junction: str, scenarios_root: Path | None = None) -> list[str]:
    """Plans that have a built mask for this junction."""
    meta_path = lane_mask_dir(junction, scenarios_root) / META_NAME
    if not meta_path.exists():
        return []
    return list(json.loads(meta_path.read_text())["plans"])


def load_lane_mask(
    junction: str,
    plan: str,
    resolution: str | tuple[int, int] | None = None,
    scenarios_root: Path | None = None,
) -> LaneMask:
    """Load a junction lane mask for one signal plan.

    Args:
        junction: Junction directory name, such as "Beijing_Beihuan".
        plan: Signal plan, "easy" or "normal". Required: the phases a lane
            serves differ between plans, so there is no safe default.
        resolution: "1274x1274" or (1274, 1274). Defaults to the solved
            resolution recorded first in `lane_mask.json`.
        scenarios_root: Override the scenarios directory.
    """
    import cv2

    mask_dir = lane_mask_dir(junction, scenarios_root)
    meta_path = mask_dir / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No lane mask for '{junction}'. Build it with "
            f"`conda run -n tshub bevlight scenario build-lane-masks --junction {junction}`."
        )
    meta = json.loads(meta_path.read_text())

    if plan not in meta["plans"]:
        raise ValueError(
            f"{junction}: no lane mask for plan '{plan}'. "
            f"Available plans: {list(meta['plans'])}"
        )
    plan_meta = meta["plans"][plan]

    if resolution is None:
        entry = plan_meta["masks"][0]
    else:
        if isinstance(resolution, str):
            width, height = (int(v) for v in resolution.lower().split("x"))
        else:
            width, height = int(resolution[0]), int(resolution[1])
        matches = [m for m in plan_meta["masks"] if m["width"] == width and m["height"] == height]
        if not matches:
            available = ["{width}x{height}".format(**m) for m in plan_meta["masks"]]
            raise ValueError(
                f"{junction}/{plan}: no {width}x{height} lane mask. Available: {available}"
            )
        entry = matches[0]

    labels = cv2.imread(str(mask_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(f"Cannot read lane mask image: {mask_dir / entry['file']}")

    return LaneMask(
        junction=junction,
        tls_id=meta["tls_id"],
        plan=plan,
        labels=labels,
        meta=meta,
        plan_meta=plan_meta,
    )
