'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Decide which frames of an episode Blender should render.

Panda images are rendered for every collected frame during collection. Blender is
much more expensive, so it renders only the frames that can feed a decision
sample, plus a small sample of empty windows.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WINDOW = 5
# Empty scenes are not useless: the model meets them at evaluation time and must
# not panic. A small share is kept so it sees them, without letting them
# dominate the action distribution.
DEFAULT_EMPTY_KEEP = 0.1


@dataclass
class Selection:
    """Which frames Blender should render, and why that is the number it is."""

    frame_indices: list[int]
    decisions_kept: list[int]
    decisions_empty: int
    decisions_total: int
    frames_total: int

    @property
    def frames_kept(self) -> int:
        return len(self.frame_indices)

    @property
    def saving(self) -> float:
        return 1.0 - self.frames_kept / self.frames_total if self.frames_total else 0.0

    def summary(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_kept": self.frames_kept,
            "render_saving": round(self.saving, 4),
            "decisions_total": self.decisions_total,
            "decisions_kept": len(self.decisions_kept),
            "decisions_empty_dropped": self.decisions_empty - (
                len(self.decisions_kept) - (self.decisions_total - self.decisions_empty)
            ),
        }


def select_frames(
    payload: dict,
    window: int = DEFAULT_WINDOW,
    empty_keep: float = DEFAULT_EMPTY_KEEP,
    seed: int = 7,
) -> Selection:
    """Frames covered by a decision window, minus most of the empty ones."""
    import numpy as np

    lane_order = payload["lane_order"]
    roles = payload["lane_roles"]
    incoming = [i for i, lane in enumerate(lane_order) if roles[lane] == "incoming"]
    vehicles = np.asarray(payload["lane_truth"]["vehicles"], dtype=float)
    per_frame = vehicles[:, incoming].sum(axis=1) if incoming else vehicles.sum(axis=1)

    rng = random.Random(seed)
    frames: set[int] = set()
    kept: list[int] = []
    empty = 0

    for index, decision in enumerate(payload["decisions"]):
        end = decision["frame_index"]
        span = [max(0, end - offset) for offset in range(window - 1, -1, -1)]
        if all(per_frame[f] == 0 for f in span):
            empty += 1
            if rng.random() >= empty_keep:
                continue
        kept.append(index)
        frames.update(span)

    return Selection(
        frame_indices=sorted(frames),
        decisions_kept=kept,
        decisions_empty=empty,
        decisions_total=len(payload["decisions"]),
        frames_total=len(per_frame),
    )


def write_blender_manifest(
    episode_dir: Path,
    selection: Selection,
    tag: str = "selected",
) -> Path:
    """A Blender manifest listing only selected frames.

    It is a single JSON file in the episode root. Panda renders every collected
    frame and does not read this manifest.
    """
    episode_dir = Path(episode_dir)
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    frame_files = manifest["frames"]

    out_path = episode_dir / f"blender_{tag}.json"
    manifest["frames"] = [frame_files[i] for i in selection.frame_indices]
    manifest["bevlight_selection"] = {
        **selection.summary(),
        "source_frame_indices": selection.frame_indices,
        "decision_indices": selection.decisions_kept,
    }
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path
