'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Move TransSimHub's Blender output into the flat pass directories BEVLight reads.

The second command over `collect/cli/blender.py`, and the reason it is separate:
`collect blender` renders, which costs hours, and this only moves what a render
already produced. Re-running a render to fix a layout mistake is not a thing
anyone should have to do, so the move is its own command with its own --dry-run.

With no --conditions and no --style, it flattens whatever `images/blender_*`
directories are actually on disk, so the usual invocation needs to know nothing
about which conditions were rendered.
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...paths import EPISODES_ROOT
from .blender import (
    available_episode_manifests,
    flatten_blender_variant,
    output_dir_for,
    parse_conditions,
    selected_manifest_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move TSHub Blender outputs into flat BEVLight pass directories."
    )
    parser.add_argument("--junction", nargs="+", default=None, help="Junctions to flatten. Default: all.")
    parser.add_argument("--episode-dir", default=None, help="Flatten one episode directory.")
    parser.add_argument("--selection", default="selected", help="Selected manifest tag.")
    parser.add_argument("--conditions", default=None, help="Limit to style[:weather] conditions, comma separated. Default: auto-detect existing images/blender_* directories.")
    parser.add_argument("--style", default=None, help="Flatten one condition only, overriding --conditions.")
    parser.add_argument("--weather", default=None, help="Weather for --style. Default: clear.")
    parser.add_argument("--passes", default="rgb,seg", help="rgb[,seg][,depth].")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without moving files.")
    return parser.parse_args()


def pass_names(spec: str) -> list[str]:
    names = list(dict.fromkeys(part.strip() for part in spec.split(",") if part.strip()))
    if not names:
        raise SystemExit(f"Empty --passes: {spec!r}")
    return names


def existing_variant_dirs(manifest_path: Path) -> list[Path]:
    images_dir = Path(manifest_path).parent / "images"
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.glob("blender_*")
        if path.is_dir()
    )


def main() -> int:
    args = parse_args()
    modalities = pass_names(args.passes)
    if args.episode_dir:
        episode_dir = Path(args.episode_dir)
        jobs = [(episode_dir.name, selected_manifest_path(episode_dir, args.selection))]
    else:
        jobs = available_episode_manifests(args.junction, args.selection)

    if not jobs:
        print(f"[plan] no Blender manifests found under {EPISODES_ROOT}/*/blender_{args.selection}.json")
        return 0 if args.dry_run else 1

    if args.style or args.weather:
        conditions = [(args.style or "day", args.weather or "clear")]
    elif args.conditions:
        conditions = parse_conditions(args.conditions)
    else:
        conditions = None

    if conditions is None:
        flattened_jobs = [
            (name, manifest_path, variant_dir.name.removeprefix("blender_"), variant_dir)
            for name, manifest_path in jobs
            for variant_dir in existing_variant_dirs(manifest_path)
        ]
        labels = "auto"
    else:
        flattened_jobs = [
            (name, manifest_path, f"{style}:{weather}", output_dir_for(manifest_path, style, weather))
            for name, manifest_path in jobs
            for style, weather in conditions
        ]
        labels = ", ".join(f"{style}:{weather}" for style, weather in conditions)

    print(
        f"[plan] episodes={len(jobs)} conditions={labels} "
        f"jobs={len(flattened_jobs)} passes={','.join(modalities)}"
    )
    for name, _, label, out_dir in flattened_jobs:
        print(f"  - {name} [{label}]: {out_dir}")

    if args.dry_run:
        return 0

    failures = []
    for name, _, variant, out_dir in flattened_jobs:
        label = f"{name} [{variant}]"
        try:
            counts = flatten_blender_variant(out_dir, modalities)
            print(f"[done] {label}: {counts or 'nothing to move'} -> {out_dir}")
        except Exception as exc:
            failures.append((label, exc))
            print(f"[fail] {label}: {exc}", file=sys.stderr)

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for label, exc in failures:
            print(f"  - {label}: {exc}", file=sys.stderr)
        return 1

    print(f"[summary] flattened {len(flattened_jobs)} variant(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
