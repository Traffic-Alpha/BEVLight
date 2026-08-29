'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Single source of truth for every on-disk location BEVLight uses.

Every other module imports its paths from here instead of recomputing
`Path(__file__).resolve().parents[n]`, so moving a file never silently changes
where its outputs land.

The tree is split by **lifecycle**, not by kind, because that is the split
`.gitignore` has to make:

    scenarios/  configs/  assets/     tracked inputs; a human wrote them
    data/                             generated, expensive: hours of rendering
    runs/                             generated, cheap: rerun the command

`data/` and `runs/` are both gitignored, but they are not interchangeable.
Deleting `runs/` costs an afternoon of retraining; deleting `data/` costs a week
of Blender. Keeping them apart is what makes "safe to clean" a statement anyone
can act on without checking first.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from pathlib import Path

# bevlight/paths.py -> bevlight -> repo root. This is the one place in the tree
# that counts `parents[n]`, which is exactly why it is also the one place a move
# can break silently -- so it says so rather than resolving to somewhere wrong.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "pyproject.toml").is_file():  # pragma: no cover
    raise RuntimeError(
        f"paths.py resolved the repo root to {PROJECT_ROOT}, which holds no "
        "pyproject.toml. It has been moved; fix the parents[n] above."
    )

# ---------------------------------------------------------------- tracked ----

# Scenario data: networks, routes, add, 3d_assets, lane_mask, and the two
# per-junction scripts (config.py, generate_routes.py) that produced them.
SCENARIOS_ROOT = PROJECT_ROOT / "scenarios"

# Cross-scenario manifests the code reads. Small, hand-maintained, tracked --
# these are *inputs*, which is why they do not live beside the results computed
# from them.
CONFIG_ROOT = PROJECT_ROOT / "configs"
SCENARIO_SELECTION = CONFIG_ROOT / "scenario_selection.json"
BEV_CAMERA_TABLE = CONFIG_ROOT / "bev_cameras.json"

# Figures tracked in git, for the README and the docs.
ASSETS_ROOT = PROJECT_ROOT / "assets"

# ------------------------------------------------------- generated: data ----

# What the pipeline renders and assembles. An episode is one directory holding
# its trajectory, its labels and the images rendered from it, so a label and its
# pixels never drift apart.
DATA_ROOT = PROJECT_ROOT / "data"
EPISODES_ROOT = DATA_ROOT / "episodes"
SAMPLES_ROOT = DATA_ROOT / "samples"
BEV_REFERENCE_ROOT = DATA_ROOT / "bev_reference"

# Rendered frames of an episode, under its own directory.
EPISODE_IMAGES_DIR_NAME = "images"

# ------------------------------------------------------- generated: runs ----

# Everything a command produces about a model or an experiment. One root, one
# `.gitignore` line, and nothing in here is an input to anything else.
RUNS_ROOT = PROJECT_ROOT / "runs"

# Pretrained backbone weights, downloaded once. Everything stays inside the
# project: the extracted weights, and the Hugging Face staging cache they are
# pulled through, so nothing important lives in ~/.cache.
BACKBONE_ROOT = RUNS_ROOT / "backbones"
HF_CACHE_ROOT = RUNS_ROOT / "hf_cache"

# One directory per training run: config, history, checkpoints.
TRAIN_RUNS_ROOT = RUNS_ROOT / "train"

# Human-facing artifacts: figures, logs, overlay checks, result JSON. Nothing
# here is read back by the pipeline, so this is the part that is safe to clean.
REPORTS_ROOT = RUNS_ROOT / "reports"
FIGURES_ROOT = REPORTS_ROOT / "figures"
LOG_ROOT = REPORTS_ROOT / "logs"
LANE_MASK_CHECK_ROOT = REPORTS_ROOT / "lane_mask"
LANE_VIEWS_ROOT = REPORTS_ROOT / "lane_views"
FRAME_CHECK_ROOT = REPORTS_ROOT / "frame_check"

# ------------------------------------------------------------- per-junction --

SCENE_ASSETS_DIR_NAME = "3d_assets"
LANE_MASK_DIR_NAME = "lane_mask"


def scenario_dir(junction: str) -> Path:
    """Data directory of one junction."""
    return SCENARIOS_ROOT / junction


def scene_assets_dir(junction: str) -> Path:
    """Static 3D assets (GLB + scene.blend) of one junction."""
    return SCENARIOS_ROOT / junction / SCENE_ASSETS_DIR_NAME


def lane_mask_dir(junction: str) -> Path:
    """Static lane-id mask directory of one junction."""
    return SCENARIOS_ROOT / junction / LANE_MASK_DIR_NAME


def episode_images_dir(episode_dir: Path, variant: str | None = None) -> Path:
    """Where an episode's rendered frames live, optionally one variant of them.

    A variant names the *appearance* ("blender_day"), never the episode: the
    episode is already the directory this sits in.
    """
    root = Path(episode_dir) / EPISODE_IMAGES_DIR_NAME
    return root / variant if variant else root
