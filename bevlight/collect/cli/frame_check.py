"""Random collected frames with their lane mask drawn on top, for a human to look at.

Collection is the one hop where a camera error is still cheap to find. After it
the frames are rendered again by Blender and then compressed into a feature
cache, and by then a wrong window has been baked into everything downstream.

So this samples real training frames — not the reference frames the mask build
draws over, which are rendered separately and could agree with the mask while
the collected ones do not — and writes them with their mask on top. It reports
what it sampled and nothing else: there is no threshold here and no verdict,
because the failure this exists to catch is one that every statistic tried so
far has been blind to.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ...paths import EPISODES_ROOT, FRAME_CHECK_ROOT


def episode_meta(episode_dir: Path) -> dict | None:
    path = episode_dir / "episode.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def selected_stems(episode_dir: Path) -> set[str]:
    """Frame stems listed in this episode's Blender selection, if it has one."""
    path = episode_dir / "blender_selected.json"
    if not path.is_file():
        return set()
    frames = json.loads(path.read_text()).get("frames", [])
    return {Path(name).stem for name in frames}


def sample_frames(episode_dirs: list[Path], variant: str, per_episode: int,
                  seed: int) -> list[tuple[Path, dict, Path]]:
    """(episode_dir, meta, frame_path) triples, drawn without replacement."""
    rng = random.Random(seed)
    picks = []
    for episode_dir in episode_dirs:
        meta = episode_meta(episode_dir)
        if meta is None:
            continue
        root = episode_dir / "images" / variant / "rgb"
        frames = sorted(root.glob("*.png")) if root.is_dir() else []
        if not frames:
            continue
        # Draw from the frames that can feed a decision sample, which is the
        # distribution training actually sees. Sampling the episode uniformly
        # instead lands mostly in the tail, after routes stop departing and the
        # network drains: an empty junction hides the misalignment that a lane
        # full of vehicles makes obvious.
        selected = selected_stems(episode_dir)
        if selected:
            frames = [f for f in frames if f.stem in selected] or frames
        for frame in rng.sample(frames, min(per_episode, len(frames))):
            picks.append((episode_dir, meta, frame))
    return picks


def write_checks(picks, out_dir: Path) -> list[Path]:
    import cv2
    import numpy as np

    from ...cli.viz import colorize
    from ...scenario.lane_mask import load_lane_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for episode_dir, meta, frame in picks:
        image = cv2.imread(str(frame))
        if image is None:
            print(f"[skip] unreadable {frame}")
            continue
        mask = load_lane_mask(meta["junction"], meta["plan"])
        if mask.labels.shape != image.shape[:2]:
            print(f"[MISMATCH] {episode_dir.name}/{frame.name}: "
                  f"mask {mask.labels.shape} vs frame {image.shape[:2]}")
            continue
        color = colorize(mask.labels, mask.lanes)
        blended = image.copy()
        hit = color.any(axis=2)
        blended[hit] = (0.42 * image[hit] + 0.58 * color[hit]).astype(np.uint8)
        target = out_dir / f"{episode_dir.name}__{frame.stem}.png"
        cv2.imwrite(str(target), np.hstack([image, blended]))
        written.append(target)
        print(f"[ok] {target.name}")
    return written


def write_contact_sheet(paths: list[Path], target: Path, tile: int = 520) -> Path | None:
    import cv2
    import numpy as np

    if not paths:
        return None
    tiles = []
    for path in paths:
        image = cv2.imread(str(path))
        # Each file is raw | overlay side by side; the sheet shows the overlay.
        overlay = image[:, image.shape[1] // 2:]
        cell = cv2.resize(overlay, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(cell, path.stem.split("__")[0][:26], (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, path.stem.split("__")[0][:26], (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cell)
    columns = min(4, len(tiles))
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), np.vstack(rows))
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="panda_day")
    parser.add_argument("--per-episode", type=int, default=1)
    parser.add_argument("--junction", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    episode_dirs = sorted(d for d in EPISODES_ROOT.iterdir() if d.is_dir())
    if args.junction:
        wanted = set(args.junction)
        episode_dirs = [d for d in episode_dirs
                        if (episode_meta(d) or {}).get("junction") in wanted]
    if not episode_dirs:
        raise SystemExit(f"no episodes under {EPISODES_ROOT}")

    picks = sample_frames(episode_dirs, args.variant, args.per_episode, args.seed)
    if not picks:
        raise SystemExit(f"no {args.variant} frames in {len(episode_dirs)} episode(s)")

    out_dir = Path(args.out) if args.out else FRAME_CHECK_ROOT / args.variant
    written = write_checks(picks, out_dir)
    sheet = write_contact_sheet(written, out_dir / "contact_sheet.png")
    print(f"[summary] {len(written)} frame(s) from {len(episode_dirs)} episode(s) -> {out_dir}")
    if sheet:
        print(f"[summary] contact sheet -> {sheet}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
