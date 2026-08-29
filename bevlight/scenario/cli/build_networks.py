'''
@Author: WANG Maonan
@Date: 2026-08-19 20:08:05
@Description: Build SUMO OSM polygons for BEVLight scenarios.

Default mode is poly-only: it regenerates
`scenarios/<junction>/add/map.poly.xml` from the scenario OSM file and the
existing SUMO network. It does not touch `networks/*.net.xml`.

Examples:
  conda run -n tshub bevlight scenario build-networks
  conda run -n tshub bevlight scenario build-networks --junction Beijing_Pinganli
  conda run -n tshub bevlight scenario build-networks --dry-run
@LastEditTime: 2026-08-19
@LastEditors: WANG Maonan
'''

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from ...cli.tshub import configure_tshub_import, resolve_tshub_root as _resolve_tshub_root
from ...paths import SCENARIOS_ROOT

DEFAULT_NET_PLAN = "normal"
PROBE_OSM_BUILD = Path("tshub/sumo_tools/osm_build.py")


def resolve_tshub_root(cli_value: str | None) -> Path | None:
    """TransSimHub checkout that actually carries the OSM build helpers."""
    return _resolve_tshub_root(cli_value, PROBE_OSM_BUILD)


def available_junctions() -> list[str]:
    return sorted(
        path.name
        for path in SCENARIOS_ROOT.iterdir()
        if path.is_dir() and (path / "config.py").is_file()
    )


def find_osm_file(scenario_dir: Path) -> Path:
    osm_files = sorted((scenario_dir / "add").glob("*.osm"))
    if not osm_files:
        raise FileNotFoundError(f"No OSM file found under {scenario_dir / 'add'}")
    if len(osm_files) > 1:
        names = ", ".join(path.name for path in osm_files)
        raise ValueError(f"Expected one OSM file under {scenario_dir / 'add'}, found: {names}")
    return osm_files[0]


def make_jobs(junctions: Iterable[str], net_plan: str) -> list[tuple[str, Path, Path, Path]]:
    known = set(available_junctions())
    jobs = []
    for junction in junctions:
        if junction not in known:
            raise ValueError(f"Unknown junction '{junction}'. Available junctions: {sorted(known)}")

        scenario_dir = SCENARIOS_ROOT / junction
        osm_file = find_osm_file(scenario_dir)
        net_file = scenario_dir / "networks" / f"{net_plan}.net.xml"
        if not net_file.exists():
            raise FileNotFoundError(f"Missing net file: {net_file}")
        poly_file = scenario_dir / "add" / "map.poly.xml"
        jobs.append((junction, osm_file, net_file, poly_file))
    return jobs


def build_poly_file(
    osm_file: Path,
    net_file: Path,
    poly_file: Path,
    poly_typemap: str | list[str] | None = None,
    keep_intermediate: bool = False,
) -> Path:
    import sumolib
    from tshub.sumo_tools.osm_build import (
        _merge_typemaps,
        _resolve_typemap_paths,
        enrich_poly_with_osm_tags,
        prepare_osm_for_polyconvert,
    )

    osm_file = osm_file.resolve()
    net_file = net_file.resolve()
    poly_file = poly_file.resolve()
    output_dir = poly_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    polyconvert = sumolib.checkBinary("polyconvert")
    patched_osm = output_dir / f"{poly_file.stem}.polyconvert.osm"
    patched_count = prepare_osm_for_polyconvert(osm_file, patched_osm)
    poly_osm_file = patched_osm if patched_count else osm_file

    resolved_typemaps = _resolve_typemap_paths(poly_typemap, "poly")
    merged_typemap = _merge_typemaps(
        resolved_typemaps,
        output_dir / f"{poly_file.stem}.merged.poly.typ.xml",
    )

    cmd = [
        polyconvert,
        "--type-file",
        merged_typemap,
        "--osm-files",
        str(poly_osm_file),
        "--discard",
        "true",
        "--osm.merge-relations",
        "1",
        "-n",
        str(net_file),
        "-o",
        str(poly_file),
    ]
    output = subprocess.check_output(cmd, cwd=output_dir, stderr=subprocess.STDOUT)
    output_text = output.decode(errors="replace")
    if "Error" in output_text:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output=output)

    enrich_poly_with_osm_tags(poly_file, osm_file)

    if not keep_intermediate:
        if patched_osm.exists():
            patched_osm.unlink()
        merged_path = Path(merged_typemap)
        if merged_path.parent == output_dir and merged_path.name.endswith(".merged.poly.typ.xml"):
            merged_path.unlink(missing_ok=True)

    return poly_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate scenario map.poly.xml files.")
    parser.add_argument(
        "--junction",
        nargs="+",
        default=None,
        help="Junction names to build. Default: all junctions under scenarios/.",
    )
    parser.add_argument("--net-plan", default=DEFAULT_NET_PLAN, help="Network plan used by polyconvert. Default: normal.")
    parser.add_argument("--poly-typemap", nargs="+", default=None, help="Poly typemap short names or files. Default: poly.")
    parser.add_argument("--tshub-root", default=None, help="Path to TransSimHub root. Defaults to TSHUB_ROOT or /home/wmn/code/TransSimHub.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep temporary polyconvert OSM/typemap files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without generating files.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after per-junction failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tshub_root = resolve_tshub_root(args.tshub_root)
    configure_tshub_import(tshub_root)

    if shutil.which("polyconvert") is None:
        print("[warn] polyconvert is not on PATH; relying on sumolib.checkBinary.")

    junctions = args.junction or available_junctions()
    jobs = make_jobs(junctions, args.net_plan)

    print(f"[plan] jobs={len(jobs)} net_plan={args.net_plan}")
    for junction, osm_file, net_file, poly_file in jobs:
        print(f"  - {junction}: {osm_file.name} + {net_file.name} -> {poly_file}")

    if args.dry_run:
        return 0

    failures = []
    for junction, osm_file, net_file, poly_file in jobs:
        try:
            print(f"[build] {junction}")
            build_poly_file(
                osm_file=osm_file,
                net_file=net_file,
                poly_file=poly_file,
                poly_typemap=args.poly_typemap,
                keep_intermediate=args.keep_intermediate,
            )
            print(f"[done] {junction}: {poly_file}")
        except Exception as exc:
            failures.append((junction, exc))
            print(f"[fail] {junction}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("[summary] failures:", file=sys.stderr)
        for junction, exc in failures:
            print(f"  - {junction}: {exc}", file=sys.stderr)
        return 1

    print(f"[summary] regenerated {len(jobs)} map.poly.xml file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
