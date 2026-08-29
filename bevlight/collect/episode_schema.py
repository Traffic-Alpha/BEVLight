'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: The trajectory file one rollout produces.

An episode is written once and read many times: by the renderers (which replay
it into several appearance variants), by the dataset builder, and by the
analysis. So it has to be self-describing and cheap to load.

Two things live side by side in an episode directory:

    episode.json        labels, expert decisions, and the static context
    frames/NNNN.json    tshub's render frames - vehicle poses and cameras

The render frames are tshub's own format, unchanged, because both the Blender
and the Panda replay paths already consume it. `episode.json` never duplicates
vehicle poses; it indexes them.

Per-lane truth is stored as parallel arrays in `lane_order`, not as one dict per
frame. A 1000-second episode has 1000 x N of these, and the array form loads
straight into a (T, N) tensor.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
EPISODE_FILE = "episode.json"

# Fields recorded per lane per simulation second, in this order.
LANE_FIELDS = ("vehicles", "queued", "queue_m", "occupancy")


@dataclass
class EpisodeStatic:
    """Everything that does not change during the episode."""

    junction: str
    plan: str
    demand: str
    seed: int
    controller: str
    tls_id: str
    env: str
    horizon_s: int
    decision_interval_s: int
    yellow_time_s: int
    camera: dict
    resolution: list
    lane_order: list
    lane_roles: dict
    visible_length_m: dict
    signal_plan: dict
    lane_mask_file: str


@dataclass
class Episode:
    """One rollout: static context, per-second truth, per-decision expert actions."""

    static: EpisodeStatic
    lane_truth: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    render_frames: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            **asdict(self.static),
            "lane_fields": list(LANE_FIELDS),
            "lane_truth": self.lane_truth,
            "decisions": self.decisions,
            "render_frames": self.render_frames,
            "metrics": self.metrics,
        }

    def write(self, episode_dir: Path) -> Path:
        episode_dir.mkdir(parents=True, exist_ok=True)
        path = episode_dir / EPISODE_FILE
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=False))
        return path


def load_episode(episode_dir: Path) -> dict:
    """Read an episode back. Returns the raw dict; callers slice what they need."""
    path = Path(episode_dir)
    if path.is_dir():
        path = path / EPISODE_FILE
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: episode schema v{payload.get('schema_version')}, "
            f"this code reads v{SCHEMA_VERSION}"
        )
    return payload


def lane_truth_array(payload: dict, field_name: str):
    """`(T, N)` array of one lane field, columns in `lane_order`."""
    import numpy as np

    return np.asarray(payload["lane_truth"][field_name], dtype=float)


def decision_window(payload: dict, decision_index: int, window: int = 5) -> list[int]:
    """Frame indices feeding one decision: the `window` frames ending at it.

    Clamped at the start of the episode, so the first decisions repeat the
    earliest frame rather than falling off the front.
    """
    decision = payload["decisions"][decision_index]
    end = decision["frame_index"]
    return [max(0, end - offset) for offset in range(window - 1, -1, -1)]
