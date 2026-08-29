'''
@Author: WANG Maonan
@Date: 2026-08-19
@Description: Offline Blender rendering from collected BEVLight episodes.

This script does not run SUMO. It consumes the selected Blender manifests written
by `bevlight collect episodes`:

  data/episodes/<key>/blender_selected.json

and renders them with the corresponding:

  scenarios/<junction>/3d_assets/scene.blend

Frames are written beside the episode labels, so the two never drift apart.
The first render pass keeps TSHub's native camera-oriented directory shape:

  data/episodes/<key>/images/blender_<style>/<element>/<sensor>_<pass>/

Run `bevlight collect flatten` afterwards to move that into:

  data/episodes/<key>/images/blender_<style>/<pass>/

Episodes render concurrently, one Blender process each (`--workers`, default 4).
'''

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ...paths import (
    EPISODES_ROOT,
    episode_images_dir,
)
from ...cli.tshub import (
    PROBE_BLENDER_RENDER,
    blender_render_episode_script as render_episode_script,
    find_blender,
    resolve_tshub_root as _resolve_tshub_root,
)


# The appearance conditions every episode is rendered under. Each lands in its
# own variant directory (blender_day, blender_golden, blender_dusk_rain), and
# bevlight.data.sample_index picks up new variants by scanning, so adding one
# here needs no change on the dataset side.
#
# Chosen by rendering all 5 light styles x 3 weather presets on one frame and
# comparing them (runs/reports/weather_grid). Two things that measurement settled:
#
#   * `fog` is unusable. Its depth fog is start 8m / depth 130m, and the BEV
#     camera sits at 172m, so the whole frame is past saturation -- all five fog
#     renders came out byte-identical, a flat grey with 5 distinct pixel values.
#   * Global brightness is most of what the presets change. After matching mean
#     and standard deviation, every pair of the remaining 10 sits within 3-14 on
#     a 0-255 scale. `overcast` vs `day` is 78% brightness: 23.6 raw, 5.2 once
#     aligned, and that residual is concentrated 3.3x inside day's shadows.
#
# So the two levers that actually change image structure are sun elevation
# (shadow geometry) and rain (wet, reflective road). These three take one point
# on each: 48 degrees, 11 degrees, and 1.5 degrees with rain.
RENDER_CONDITIONS = (
    "day:clear",     # baseline: 48 deg sun, crisp shadows, accurate colour
    "golden:clear",  # 11 deg sun: long shadows across the lanes, warm cast
    "dusk:rain",     # 1.5 deg sun: dark, wet reflective road, visible rain
)


def parse_conditions(spec: str) -> list[tuple[str, str]]:
    """'day:clear,dusk:rain' -> [('day', 'clear'), ('dusk', 'rain')].

    A bare style means clear weather, so 'day,dusk:rain' also works.
    """
    conditions = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        style, _, weather = part.partition(":")
        style, weather = style.strip(), weather.strip() or "clear"
        if not style:
            raise ValueError(f"Invalid condition in --conditions: {part!r}")
        if (style, weather) not in conditions:
            conditions.append((style, weather))
    if not conditions:
        raise ValueError(f"Empty --conditions: {spec!r}")
    return conditions


def resolve_tshub_root(cli_value: str | None) -> Path | None:
    """TransSimHub checkout that actually carries the offline Blender renderer."""
    return _resolve_tshub_root(cli_value, PROBE_BLENDER_RENDER)


def output_dir_for(
    manifest_path: Path,
    style: str,
    weather: str = "clear",
) -> Path:
    """Where one Blender render lands inside the owning episode."""
    style_dir_name = style if weather == "clear" else f"{style}_{weather}"
    return episode_images_dir(Path(manifest_path).parent, f"blender_{style_dir_name}")


def selected_manifest_path(episode_dir: Path, selection: str = "selected") -> Path:
    return Path(episode_dir) / f"blender_{selection}.json"


def available_episode_manifests(
    junctions: list[str] | None = None,
    selection: str = "selected",
) -> list[tuple[str, Path]]:
    """Collected episode manifests ready for Blender."""
    if not EPISODES_ROOT.exists():
        return []
    wanted = set(junctions or [])
    jobs: list[tuple[str, Path]] = []
    for episode_dir in sorted(path for path in EPISODES_ROOT.iterdir() if path.is_dir()):
        if wanted and not any(episode_dir.name.startswith(f"{junction}__") for junction in wanted):
            continue
        manifest_path = selected_manifest_path(episode_dir, selection)
        if manifest_path.is_file():
            jobs.append((episode_dir.name, manifest_path))
    return sorted(
        jobs,
        key=lambda item: item[0],
    )


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing episode manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def materialize_manifest_for_tshub(manifest_path: Path, out_dir: Path) -> Path:
    """Write the selected manifest in the directory shape TSHub expects."""
    manifest_path = Path(manifest_path)
    episode_dir = manifest_path.parent
    manifest = load_manifest(manifest_path)
    manifest["frames"] = [
        str((episode_dir / frame).resolve()) if not Path(frame).is_absolute() else frame
        for frame in manifest["frames"]
    ]
    target = out_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2))
    return target


def scene_blend_from_manifest(manifest: dict) -> Path:
    scenario_glb_dir = Path(manifest["scenario_glb_dir"])
    scene_blend = scenario_glb_dir / "scene.blend"
    if not scene_blend.exists():
        raise FileNotFoundError(f"Missing scene.blend: {scene_blend}")
    return scene_blend


def resolution_from_manifest(manifest: dict) -> str | None:
    resolution = manifest.get("resolution")
    if not resolution:
        return None
    if len(resolution) != 2:
        raise ValueError(f"Invalid manifest resolution: {resolution}")
    width, height = (int(v) for v in resolution)
    return f"{width}x{height}"


def samples_from_manifest(manifest: dict) -> int | None:
    samples = manifest.get("samples")
    if samples is None:
        return None
    return int(samples)


def pass_groups(passes: str) -> list[str]:
    """'rgb,seg' -> ['rgb', 'seg'] (one Blender run each)."""
    names = list(dict.fromkeys(p.strip() for p in passes.split(",") if p.strip()))
    if not names:
        raise ValueError(f"Empty --passes: {passes!r}")
    return names


def flatten_blender_output(render_dir: Path, out_dir: Path, modality: str) -> None:
    """Move TSHub's camera-oriented output into BEVLight's pass directory."""
    ext = "exr" if modality == "depth" else "png"
    sources = sorted(render_dir.glob(f"*/junction_bev_{modality}/*.{ext}"))
    if not sources:
        return

    target_dir = out_dir / modality
    target_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for source in sources:
        if source.name in seen:
            raise RuntimeError(f"Multiple BEV cameras wrote frame {source.name} in {out_dir}")
        seen.add(source.name)
        target = target_dir / source.name
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))


def flatten_blender_variant(out_dir: Path, modalities: list[str] | None = None) -> dict:
    """Flatten one rendered Blender variant after TSHub has written it."""
    out_dir = Path(out_dir)
    counts = {}
    for modality in modalities or ["rgb", "seg", "depth"]:
        before = len(list((out_dir / modality).glob("*"))) if (out_dir / modality).is_dir() else 0
        flatten_blender_output(out_dir, out_dir, modality)
        after = len(list((out_dir / modality).glob("*"))) if (out_dir / modality).is_dir() else 0
        if after:
            counts[modality] = after - before

    for child in sorted(out_dir.iterdir()) if out_dir.is_dir() else []:
        if child.is_dir() and child.name not in {"rgb", "seg", "depth"}:
            shutil.rmtree(child, ignore_errors=True)
    return counts


def run_blender(
    cmd: list[str], done_token: str = "RENDER_DONE"
) -> tuple[str, float, list[str]]:
    """Run one Blender process and hand its interesting lines back to the caller.

    Nothing is printed here: episodes render concurrently, so their output has to
    be buffered per job and emitted in one block, or four Blender logs interleave
    line by line into something unreadable.
    """
    started = time.perf_counter()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    wall_elapsed_s = time.perf_counter() - started
    # Per-frame and per-model lines stay out of the terminal: across 45 episodes
    # they are ~12k and ~350 lines of repetition. The frame timings they carry
    # are not lost -- parse_render_stats reads them from the full stdout below.
    noise = ("[render] frame", "[assembly] model")
    log = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("[assembly]", "[weather]", done_token))
        and not line.startswith(noise)
    ]
    if result.returncode != 0 or done_token not in result.stdout:
        raise RuntimeError(
            f"Blender command failed: {' '.join(cmd[:4])}\n"
            f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{result.stderr[-3000:]}"
        )
    return result.stdout, wall_elapsed_s, log


def parse_render_stats(
    junction: str, passes: str, stdout: str, out_dir: Path, wall_elapsed_s: float
) -> dict:
    done_match = re.search(
        r"RENDER_DONE:\s+(\d+)\s+images in\s+([0-9.]+)s .*samples=([^,]+),\s+([0-9]+x[0-9]+)",
        stdout,
    )
    frame_matches = re.findall(
        r"\[render\] frame\s+(\d+):\s+(\d+)\s+images,\s+(\d+)\s+vehicles,\s+([0-9.]+)s",
        stdout,
    )
    if not done_match:
        return {
            "junction": junction,
            "passes": passes,
            "output_dir": str(out_dir),
            "wall_elapsed_s": wall_elapsed_s,
        }

    image_count = int(done_match.group(1))
    elapsed_s = float(done_match.group(2))
    frame_times = [float(match[3]) for match in frame_matches]
    stats = {
        "junction": junction,
        "passes": passes,
        "output_dir": str(out_dir),
        "images": image_count,
        "elapsed_s": elapsed_s,
        "seconds_per_image": elapsed_s / image_count if image_count else None,
        "wall_elapsed_s": wall_elapsed_s,
        "wall_seconds_per_image": wall_elapsed_s / image_count if image_count else None,
        "samples": done_match.group(3),
        "resolution": done_match.group(4),
    }
    if frame_times:
        stats["frames"] = len(frame_times)
        stats["seconds_per_frame"] = elapsed_s / len(frame_times)
        stats["mean_frame_elapsed_s"] = sum(frame_times) / len(frame_times)
        stats["min_frame_elapsed_s"] = min(frame_times)
        stats["max_frame_elapsed_s"] = max(frame_times)
        stats["mean_vehicles"] = sum(int(match[2]) for match in frame_matches) / len(frame_matches)
    return stats


def render_junction(
    junction: str,
    style: str,
    weather: str,
    frames: str | None,
    samples: int | None,
    resolution: str | None,
    passes: str,
    cameras: str | None,
    script: Path,
    blender: Path,
    manifest_path: Path,
) -> tuple[Path, list[dict], list[str]]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    scene_blend = scene_blend_from_manifest(manifest)
    effective_resolution = resolution or resolution_from_manifest(manifest)
    effective_samples = samples if samples is not None else samples_from_manifest(manifest)

    out_dir = output_dir_for(manifest_path, style, weather)

    log = [
        f"[render] {junction}: {manifest_path} -> {out_dir} "
        f"resolution={effective_resolution or 'blend'} samples={effective_samples or 'blend'}"
    ]

    # TSHub denoises on the GPU (scene_assembly.setup_denoiser). Blender's factory
    # default runs OpenImageDenoise on the CPU, which cost 2.10s of the 2.10s+
    # per frame here while the GPU sat idle; on the GPU the same denoiser at the
    # same quality gives 0.685s per frame. Pixels are unchanged (max 1/255), so
    # frames rendered before and after the switch stay comparable.
    #
    # One Blender run per pass: mixing rgb with seg/depth in a single run makes
    # every render see a different material_override state, so Cycles drops its
    # use_persistent_data scene cache each time. Measured on a 101-frame BEV
    # episode: rgb 140.6s + seg 48.3s split, vs 298.5s fused.
    stats = []
    with tempfile.TemporaryDirectory(prefix="bevlight_blender_") as tmp:
        tshub_episode_dir = Path(tmp)
        materialize_manifest_for_tshub(manifest_path, tshub_episode_dir)
        for group in pass_groups(passes):
            cmd = [
                str(blender),
                "--background",
                str(scene_blend),
                "--python",
                str(script),
                "--",
                str(tshub_episode_dir),
                str(out_dir),
                "--style",
                style,
                "--weather",
                weather,
                "--passes",
                group,
            ]
            if frames:
                cmd += ["--frames", frames]
            if effective_samples is not None:
                cmd += ["--samples", str(effective_samples)]
            if effective_resolution:
                cmd += ["--resolution", effective_resolution]
            cmd += ["--cameras", cameras or "junction_bev_rgb"]

            stdout, wall_elapsed_s, run_log = run_blender(cmd)
            log.extend(run_log)
            stats.append(parse_render_stats(junction, group, stdout, out_dir, wall_elapsed_s))
    return out_dir, stats, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render BEVLight Blender outputs from JSON episodes.")
    parser.add_argument("--junction", nargs="+", default=None, help="Junctions to render. Default: all collected episodes with a selected Blender manifest.")
    parser.add_argument("--episode-dir", default=None, help="Render one episode directory, e.g. data/episodes/<key>.")
    parser.add_argument("--manifest", default=None, help="Render this selected manifest JSON directly.")
    parser.add_argument("--selection", default="selected", help="Selected manifest tag, e.g. selected -> blender_selected.json.")
    parser.add_argument("--conditions", default=",".join(RENDER_CONDITIONS), help=f"Appearance conditions as style[:weather], comma separated. Default: {','.join(RENDER_CONDITIONS)}.")
    parser.add_argument("--style", default=None, help="Render one condition only, overriding --conditions.")
    parser.add_argument("--weather", default=None, help="Weather for --style. Default: clear.")
    parser.add_argument("--frames", default=None, help="Frame range, e.g. 0:10:2.")
    parser.add_argument("--samples", type=int, default=None, help="Override Cycles samples.")
    parser.add_argument("--resolution", default=None, help="Override resolution, e.g. 1022x1022.")
    parser.add_argument("--passes", default="rgb", help="rgb[,seg][,depth].")
    parser.add_argument("--cameras", default=None, help="Camera filter passed to render_episode.py.")
    parser.add_argument("--workers", type=int, default=4, help="Episodes to render concurrently. Each Blender process holds its own scene (~3.2GB VRAM), so 4 fits a 24GB card. Measured on a 4090: 0.79 s/frame at 1 worker, 0.41 at 2, 0.26 at 4.")
    parser.add_argument("--stats-file", default=None, help="Optional JSON file for render timing stats.")
    parser.add_argument("--tshub-root", default=None, help="TransSimHub root. Defaults to TSHUB_ROOT or /home/wmn/code/TransSimHub.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without running Blender.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest:
        manifest_path = Path(args.manifest)
        jobs = [(manifest_path.parent.name, manifest_path)]
    elif args.episode_dir:
        episode_dir = Path(args.episode_dir)
        jobs = [(episode_dir.name, selected_manifest_path(episode_dir, args.selection))]
    else:
        jobs = available_episode_manifests(args.junction, args.selection)
    if not jobs:
        print(f"[plan] no Blender manifests found under {EPISODES_ROOT}/*/blender_{args.selection}.json")
        return 0 if args.dry_run else 1

    # An explicit --style renders that one condition; otherwise the whole list.
    if args.style or args.weather:
        conditions = [(args.style or "day", args.weather or "clear")]
    else:
        conditions = parse_conditions(args.conditions)

    # One render job per (episode, condition). Conditions are independent: each
    # writes its own variant directory, so they can be interleaved freely.
    jobs = [
        (name, manifest_path, style, weather)
        for name, manifest_path in jobs
        for style, weather in conditions
    ]

    labels = ", ".join(f"{style}:{weather}" for style, weather in conditions)
    print(f"[plan] episodes={len(jobs) // len(conditions)} conditions={labels} jobs={len(jobs)}")
    for name, manifest_path, style, weather in jobs:
        effective_resolution = args.resolution
        effective_samples = args.samples
        if effective_resolution is None and manifest_path.is_file():
            manifest = load_manifest(manifest_path)
            effective_resolution = resolution_from_manifest(manifest)
            if effective_samples is None:
                effective_samples = samples_from_manifest(manifest)
        out_dir = output_dir_for(manifest_path, style, weather)
        print(
            f"  - {name} [{style}:{weather}]: -> {out_dir} "
            f"resolution={effective_resolution or 'blend'} samples={effective_samples or 'blend'}"
        )

    if args.dry_run:
        return 0

    script = render_episode_script(resolve_tshub_root(args.tshub_root))
    blender = find_blender()

    # One episode per worker. Each Blender process owns its own scene and renders
    # its frames strictly in order, so nothing about a single episode changes --
    # sync and render still alternate one frame at a time inside a process, and a
    # frame's content depends only on that frame's data. Verified: rendering
    # frames 30:42 in a fresh process matches rendering them as part of 0:60 to
    # within the GPU denoiser's own run-to-run noise (max 1/255 either way).
    #
    # Parallelism pays because what is left per frame is CPU work -- writing the
    # PNG and syncing vehicles -- while the GPU is busy only ~20% of the time.
    # Threads, not processes: each worker just blocks in subprocess.run.
    workers = max(1, min(args.workers, len(jobs)))
    print(f"[plan] workers={workers}")

    failures = []
    stats = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                render_junction,
                junction=name,
                style=style,
                weather=weather,
                frames=args.frames,
                samples=args.samples,
                resolution=args.resolution,
                passes=args.passes,
                cameras=args.cameras,
                script=script,
                blender=blender,
                manifest_path=manifest_path,
            ): f"{name} [{style}:{weather}]"
            for name, manifest_path, style, weather in jobs
        }
        for future in as_completed(futures):
            name = futures[future]
            done += 1
            try:
                _, junction_stats, log = future.result()
            except Exception as exc:
                failures.append((name, exc))
                print(f"[fail {done}/{len(jobs)}] {name}: {exc}", file=sys.stderr)
                continue
            stats.extend(junction_stats)
            print(f"[ok {done}/{len(jobs)}] {name}")
            for line in log:
                print(f"  {line}")

    stats_file = Path(args.stats_file) if args.stats_file else EPISODES_ROOT / "blender_render_stats.json"
    if stats:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps(stats, indent=2, sort_keys=True))
        print(f"render stats: {stats_file}")

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for junction, exc in failures:
            print(f"  - {junction}: {exc}", file=sys.stderr)
        return 1

    print(f"[summary] rendered {len(jobs)} job(s) over {len(conditions)} condition(s)")
    for style, weather in conditions:
        suffix = style if weather == "clear" else f"{style}_{weather}"
        print(f"blender outputs: {EPISODES_ROOT}/<episode>/images/blender_{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
