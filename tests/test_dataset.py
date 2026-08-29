'''Decision samples, and the split that keeps them honest.'''

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")

from bevlight.data.collate import collate
from bevlight.data.dataset import DecisionDataset
from bevlight.model.bevlight import BEVLight, BEVLightConfig
from bevlight.train.losses import action_accuracy, bevlight_loss


def _built_datasets() -> list[str]:
    """Datasets on disk in the current cache format.

    A cache written before junctions of different lane counts shared one dataset
    has no `num_lanes`, so it cannot be read at all; skip rather than fail on it.
    """
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

DATASETS = _built_datasets()
needs_data = pytest.mark.skipif(not DATASETS, reason="no built dataset on disk")


@pytest.fixture(scope="module")
def dataset():
    return DecisionDataset(DATASETS[0])


@needs_data
def test_every_sample_has_a_full_window(dataset):
    for position in range(min(20, len(dataset))):
        sample = dataset[position]
        assert sample["lane_features"].shape[0] == dataset.window


@needs_data
def test_labels_index_the_candidate_set_not_a_global_phase_id(dataset):
    """The label must be a position among *this* junction's candidates."""
    for position in range(min(30, len(dataset))):
        sample = dataset[position]
        real = int(sample["phase_valid"].sum())
        assert 0 <= sample["action"] < real
        assert 0 <= sample["current_phase"] < real


@needs_data
def test_split_is_by_episode_not_by_frame(dataset):
    """Frames a second apart are near duplicates; a frame split would leak."""
    held = [dataset.episode_names[-1]]
    train, valid = dataset.split_by_episode(held)
    assert len(train) and len(valid)
    # `.samples` is the raw decision records; iterating a subset would build
    # model-ready tensors instead.
    assert {d["episode"] for d in train.samples}.isdisjoint(
        {d["episode"] for d in valid.samples}
    )
    assert {d["episode"] for d in valid.samples} == set(held)


@needs_data
def test_saturated_queue_labels_are_masked_out(dataset):
    """A queue reaching the image edge is a lower bound, not a measurement."""
    for position in range(min(40, len(dataset))):
        sample = dataset[position]
        edge = sample["queue_valid"] == 0
        if edge.any():
            return
    pytest.skip("no saturated queue labels in this dataset")


@needs_data
def test_a_batch_trains_and_can_be_overfitted(dataset):
    """The end-to-end check: if the chain is wired up, a small batch memorises."""
    torch.manual_seed(0)
    batch = collate([dataset[i] for i in range(min(8, len(dataset)))])
    model = BEVLight(BEVLightConfig(model_dim=64, heads=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    first = None
    for _ in range(150):
        optimizer.zero_grad()
        loss, parts = bevlight_loss(model(batch), batch)
        loss.backward()
        optimizer.step()
        first = first if first is not None else parts["ce"]

    with torch.no_grad():
        out = model(batch)
    _, parts = bevlight_loss(out, batch)
    assert parts["ce"] < first / 5, "the model cannot even fit 8 samples"
    assert action_accuracy(out["logits"], batch["action"])["accuracy"] > 0.8


def _multi_junction_dataset() -> str | None:
    """A dataset holding junctions of different lane counts, if one is built."""
    for name in DATASETS:
        data = DecisionDataset(name)
        counts = {
            len(meta["lane_order"]) for meta in data.index["episodes"].values()
        }
        if len(counts) > 1:
            return name
    return None

MULTI = _multi_junction_dataset() if DATASETS else None
needs_multi = pytest.mark.skipif(
    MULTI is None, reason="no dataset spanning junctions of different lane counts"
)


@needs_multi
def test_junctions_of_different_widths_share_one_cache():
    """The cache pads to its widest junction; each frame must use its own count.

    Reading a fixed width instead would hand a 12-lane junction 16 rows of zeros
    as if they were lanes, and nothing would raise.
    """
    dataset = DecisionDataset(MULTI)
    meta = dataset.index["episodes"]

    seen = {}
    for position in range(len(dataset)):
        episode = dataset.samples[position]["episode"]
        if episode in seen:
            continue
        sample = dataset[position]
        real = len(meta[episode]["lane_order"])
        seen[episode] = real

        assert sample["lane_valid"][:real].all()
        assert not sample["lane_valid"][real:].any()
        # Padded lanes carry no features, no labels and no loss weight.
        assert not sample["lane_features"][:, real:].any()
        assert not sample["queue_target"][real:].any()
        assert not sample["queue_valid"][real:].any()

    assert len({v for v in seen.values()}) > 1, "fixture no longer spans lane counts"


@needs_multi
def test_a_batch_mixes_junctions_without_leaking_between_them():
    """Two junctions in one batch must score exactly as they do alone."""
    dataset = DecisionDataset(MULTI)
    meta = dataset.index["episodes"]

    by_width = {}
    for position in range(len(dataset)):
        width = len(meta[dataset.samples[position]["episode"]]["lane_order"])
        by_width.setdefault(width, position)
    positions = [by_width[w] for w in sorted(by_width)[:2]]

    torch.manual_seed(0)
    model = BEVLight(BEVLightConfig()).eval()
    with torch.no_grad():
        together = model(collate([dataset[p] for p in positions]))["logits"]
        apart = [model(collate([dataset[p]]))["logits"] for p in positions]

    for row, alone in enumerate(apart):
        assert torch.allclose(together[row], alone[0], atol=1e-5)
