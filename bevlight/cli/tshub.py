'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Locating TransSimHub and Blender.

BEVLight drives TransSimHub (SUMO + Panda3D + Blender scene building) from a
sibling checkout rather than an installed package, so every entry point needs
the same "find tshub, find blender" dance. It lives here once.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_TSHUB_ROOT = Path("/home/wmn/code/TransSimHub")
DEFAULT_BLENDER = Path("/home/wmn/blender/blender")

# Relative probes used to confirm a candidate directory really is TransSimHub.
PROBE_PACKAGE = Path("tshub")
PROBE_SCENE = Path("tshub/tshub_env3d/scene")
PROBE_BLENDER_RENDER = Path("tshub/tshub_env3d/renderers/blender/render_episode.py")


def resolve_tshub_root(
    cli_value: str | None = None,
    probe: Path | str = PROBE_PACKAGE,
) -> Path | None:
    """Find the TransSimHub checkout.

    Order: explicit CLI value, ``TSHUB_ROOT``, then the default location. A
    candidate only counts if ``probe`` exists inside it, so callers that need a
    specific tshub feature can prove it is there before returning.
    """
    candidates: list[Path] = []
    if cli_value:
        candidates.append(Path(cli_value))
    if os.environ.get("TSHUB_ROOT"):
        candidates.append(Path(os.environ["TSHUB_ROOT"]))
    candidates.append(DEFAULT_TSHUB_ROOT)

    for candidate in candidates:
        target = candidate / probe
        if target.is_dir() or target.is_file():
            return candidate
    return None


def configure_tshub_import(tshub_root: Path | None) -> None:
    """Put TransSimHub on ``sys.path`` so ``import tshub`` works."""
    if tshub_root is not None and str(tshub_root) not in sys.path:
        sys.path.insert(0, str(tshub_root))


def require_tshub_root(cli_value: str | None = None, probe: Path | str = PROBE_PACKAGE) -> Path:
    """Like :func:`resolve_tshub_root` but raises instead of returning None."""
    tshub_root = resolve_tshub_root(cli_value, probe)
    if tshub_root is None:
        raise FileNotFoundError(
            "Cannot locate TransSimHub. Set --tshub-root or TSHUB_ROOT to the "
            f"repository that contains {Path(probe)}."
        )
    return tshub_root


def blender_scene_scripts(tshub_root: Path | None) -> tuple[Path, Path]:
    """TransSimHub's ``build_scene.py`` and ``build_blend.py``."""
    if tshub_root is None:
        raise FileNotFoundError(
            "Cannot locate TransSimHub. Set --tshub-root or TSHUB_ROOT to the "
            "repository that contains tshub/tshub_env3d."
        )
    scene_dir = tshub_root / "tshub" / "tshub_env3d" / "scene" / "blender"
    build_scene_script = scene_dir / "build_scene.py"
    build_blend_script = scene_dir / "build_blend.py"
    missing = [path for path in (build_scene_script, build_blend_script) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Blender helper scripts: " + ", ".join(str(path) for path in missing)
        )
    return build_scene_script, build_blend_script


def blender_render_episode_script(tshub_root: Path | None) -> Path:
    """TransSimHub's offline Blender episode renderer."""
    if tshub_root is None:
        raise FileNotFoundError(
            "Cannot locate TransSimHub. Set --tshub-root or TSHUB_ROOT."
        )
    script = tshub_root / PROBE_BLENDER_RENDER
    if not script.exists():
        raise FileNotFoundError(f"Missing Blender render script: {script}")
    return script


def find_blender() -> Path:
    """Locate the Blender executable: ``BLENDER``, the default path, then PATH."""
    env_blender = os.environ.get("BLENDER")
    if env_blender:
        blender = Path(env_blender)
        if not blender.exists():
            raise FileNotFoundError(f"BLENDER points to a missing executable: {blender}")
        return blender

    if DEFAULT_BLENDER.exists():
        return DEFAULT_BLENDER

    blender = shutil.which("blender")
    if blender:
        return Path(blender)

    raise FileNotFoundError(
        "Blender is required. Install Blender or set BLENDER=/path/to/blender."
    )
