'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Naming rules for everything the pipeline writes under data/.

Collection, rendering, dataset assembly and evaluation all have to agree on
where an episode's trajectory and its rendered variants live. That agreement is
this file, and nothing else builds those paths by hand.

    data/episodes/<episode_key>/episode.json
    data/episodes/<episode_key>/images/<variant>/...
    data/samples/<dataset_name>/index.json

`episode_key` is `<junction>__<plan>__<demand>__seed<k>__<expert>`, so an
episode directory name states its full provenance without opening a file.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..utils.paths import EPISODES_ROOT, SAMPLES_ROOT, episode_images_dir

EPISODE_FILE = "episode.json"
FRAME_SUFFIX = ".png"
FRAME_DIGITS = 4
INDEX_FILE = "index.json"

_EPISODE_KEY_RE = re.compile(
    r"^(?P<junction>[^_]+(?:_[^_]+)*?)__(?P<plan>[^_]+)__(?P<demand>.+)__seed(?P<seed>\d+)__(?P<expert>[^_]+)$"
)


@dataclass(frozen=True)
class EpisodeKey:
    """Everything that makes one episode reproducible."""

    junction: str
    plan: str
    demand: str
    seed: int
    expert: str

    def __str__(self) -> str:
        return f"{self.junction}__{self.plan}__{self.demand}__seed{self.seed}__{self.expert}"

    @property
    def env_name(self) -> str:
        return f"{self.plan}_{self.demand}"

    @classmethod
    def parse(cls, key: str) -> "EpisodeKey":
        match = _EPISODE_KEY_RE.match(key)
        if match is None:
            raise ValueError(f"Not an episode key: {key!r}")
        return cls(
            junction=match["junction"],
            plan=match["plan"],
            demand=match["demand"],
            seed=int(match["seed"]),
            expert=match["expert"],
        )


def episode_dir(key: EpisodeKey | str) -> Path:
    return EPISODES_ROOT / str(key)


def episode_file(key: EpisodeKey | str) -> Path:
    return episode_dir(key) / EPISODE_FILE


def variant_dir(key: EpisodeKey | str, variant: str) -> Path:
    """Rendered frames of one appearance variant, e.g. "blender_day".

    Images sit inside the episode they were rendered from, so an episode is one
    self-contained directory: trajectory, labels and pixels together.
    """
    return episode_images_dir(episode_dir(key), variant)


def frame_file(
    key: EpisodeKey | str,
    variant: str,
    frame_index: int,
    modality: str = "rgb",
) -> Path:
    return variant_dir(key, variant) / modality / f"{frame_index:0{FRAME_DIGITS}d}{FRAME_SUFFIX}"


def variants_of(key: EpisodeKey | str) -> list[str]:
    """Appearance variants already rendered for an episode."""
    root = episode_images_dir(episode_dir(key))
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def dataset_dir(dataset_name: str) -> Path:
    return SAMPLES_ROOT / dataset_name


def dataset_index(dataset_name: str) -> Path:
    return dataset_dir(dataset_name) / INDEX_FILE


def available_episodes() -> list[EpisodeKey]:
    """Episodes that have a finished trajectory on disk."""
    if not EPISODES_ROOT.is_dir():
        return []
    keys = []
    for path in sorted(EPISODES_ROOT.iterdir()):
        if (path / EPISODE_FILE).is_file():
            keys.append(EpisodeKey.parse(path.name))
    return keys
