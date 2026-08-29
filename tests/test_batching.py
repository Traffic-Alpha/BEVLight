'''The fast batch path must be the slow one, exactly.

`BatchSource` exists only to remove per-sample Python work. If it ever produces
something different from `collate([dataset[i] ...])` it stops being an
optimisation and becomes an undetected change of experiment, so the two are
required to agree tensor for tensor.

The cases that would break first are the ones with structure: a batch spanning
junctions of different lane counts, and `queue_valid`, whose padded lanes are
stored unsaturated and would read as *valid* if the lane mask were not applied.
'''

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")

from bevlight.data.batching import BatchSource
from bevlight.data.collate import collate
from bevlight.data.dataset import DecisionDataset


def _datasets() -> list[str]:
    import numpy as np

    names = []
    for path in sorted((ROOT / "data" / "samples").glob("*")):
        cache = path / "lane_features.npz"
        if not cache.is_file():
            continue
        with np.load(cache, allow_pickle=False) as data:
            if "num_lanes" in data.files:
                names.append(path.name)
    return names

DATASETS = _datasets()
needs_data = pytest.mark.skipif(not DATASETS, reason="no built dataset on disk")


@pytest.fixture(scope="module")
def dataset():
    return DecisionDataset(DATASETS[0])


@needs_data
def test_device_batches_equal_per_sample_batches(dataset):
    """Same rows in, identical tensors out."""
    # Untrimmed, so the two paths are comparable tensor for tensor; trimming is
    # covered separately below.
    source = BatchSource(dataset, device="cpu", trim_lanes=False)
    meta = dataset.index["episodes"]

    # Pick rows spanning as many different lane counts as the dataset holds, so
    # padding is actually exercised rather than assumed.
    by_width = {}
    for position, decision in enumerate(dataset.samples):
        width = len(meta[decision["episode"]]["lane_order"])
        by_width.setdefault(width, []).append(position)
    rows = [p for group in by_width.values() for p in group[:4]]

    fast = source.batch(rows)
    slow = collate([dataset[p] for p in rows])

    assert set(fast) == set(slow)
    for key, expected in slow.items():
        assert torch.equal(fast[key], expected), f"{key} differs between batch paths"


@needs_data
def test_a_subset_batches_its_own_rows(dataset):
    """A split must gather the parent rows it actually holds, not the first N."""
    held = [dataset.episode_names[-1]]
    _, valid = dataset.split_by_episode(held)
    source = BatchSource(valid, device="cpu", trim_lanes=False)

    assert len(source) == len(valid)
    rows = list(range(min(8, len(valid))))
    fast = source.batch(rows)
    slow = collate([valid[p] for p in rows])
    for key, expected in slow.items():
        assert torch.equal(fast[key], expected), f"{key} differs on a subset"


@needs_data
def test_padded_lanes_are_never_counted_as_valid_queue_readings(dataset):
    """Padding is stored unsaturated; without the lane mask it would read as valid."""
    source = BatchSource(dataset, device="cpu", trim_lanes=False)
    meta = dataset.index["episodes"]
    rows = list(range(min(32, len(dataset))))
    batch = source.batch(rows)

    for row, position in enumerate(rows):
        real = len(meta[dataset.samples[position]["episode"]]["lane_order"])
        assert not batch["queue_valid"][row, real:].any()
        assert not batch["queue_target"][row, real:].any()
        assert not batch["lane_features"][row, :, real:].any()


@needs_data
def test_trimming_lane_padding_does_not_change_the_answer(dataset):
    """Padding to the batch's own width instead of the global ceiling is free.

    MAX_LANES is 48 because one test junction has 40 lanes, so most batches drag
    dead slots through every attention and pooling. Nothing in the model has a
    dimension that depends on N, so trimming must be a pure cost change — if it
    is not, the masks are leaking and this is the cheapest place to find out.
    """
    from bevlight.model.bevlight import BEVLight, BEVLightConfig

    torch.manual_seed(0)
    model = BEVLight(BEVLightConfig()).eval()

    rows = list(range(min(24, len(dataset))))
    trimmed = BatchSource(dataset, device="cpu", trim_lanes=True).batch(rows)
    full = BatchSource(dataset, device="cpu", trim_lanes=False).batch(rows)
    assert trimmed["lane_features"].shape[2] < full["lane_features"].shape[2]

    with torch.no_grad():
        a, b = model(trimmed), model(full)
    assert torch.allclose(a["logits"], b["logits"], atol=1e-5)
    width = trimmed["lane_features"].shape[2]
    assert torch.allclose(a["queue"], b["queue"][:, :width], atol=1e-5)
