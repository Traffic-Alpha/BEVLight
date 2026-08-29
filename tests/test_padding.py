'''Padding must never change an answer.

N, R and K all vary by junction, so batching means padding. A mask that fails to
reach one attention, pooling, softmax or loss does not crash — it makes a
sample's output depend on which other samples shared its batch. That is close to
undetectable once training is under way, so it is pinned down here.

The test is the one the project plan calls for: pad the same sample two
different ways and require identical output.
'''

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bevlight.data.collate import collate, junction_structure
from bevlight.model.bevlight import BEVLight, BEVLightConfig
from bevlight.scenario.lane_mask import load_lane_mask

EMBED, FRAMES = 384, 5


def make_sample(junction: str, plan: str, max_lanes: int, max_movements: int,
                max_phases: int, seed: int = 0) -> dict:
    mask = load_lane_mask(junction, plan)
    structure = junction_structure(mask, max_lanes, max_movements, max_phases)
    real = int(structure["lane_valid"].sum())

    # Draw only the real lanes, at a shape that does not depend on the padding,
    # then place them. Drawing the full padded array instead would give the real
    # lanes *different* values at different padding sizes, and the test would be
    # comparing two different inputs.
    rng = np.random.default_rng(seed)
    features = np.empty((FRAMES, max_lanes, EMBED), dtype=np.float32)
    features[:, :real] = rng.standard_normal((FRAMES, real, EMBED)).astype(np.float32)
    # Padded lanes carry garbage on purpose: if a mask leaks, it will show.
    features[:, real:] = 1e3
    return {
        **{k: v for k, v in structure.items() if k not in ("lane_order", "phase_order")},
        "lane_features": features,
        "current_phase": 1,
        "time_in_phase": 12.0,
        "action": 0,
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BEVLight(BEVLightConfig()).eval()


@pytest.mark.parametrize(
    "junction,plan", [("Beijing_Pinganli", "easy"), ("Hongkong_YMT", "normal")]
)
@pytest.mark.needs_scenarios
def test_extra_padding_does_not_change_the_scores(model, junction, plan):
    """The core property: the same junction, padded two ways, scores the same."""
    tight = make_sample(junction, plan, 32, 12, 4)
    loose = make_sample(junction, plan, 48, 16, 6)

    with torch.no_grad():
        a = model(collate([tight]))["logits"]
        b = model(collate([loose]))["logits"]

    real = int(tight["phase_valid"].sum())
    assert torch.allclose(a[0, :real], b[0, :real], atol=1e-5), (
        f"{junction}/{plan}: padding changed the phase scores"
    )


@pytest.mark.needs_scenarios
def test_batching_with_a_bigger_junction_does_not_change_the_smaller(model):
    """A sample must not notice what else is in its batch."""
    small = make_sample("Hongkong_YMT", "normal", 48, 16, 6, seed=1)
    big = make_sample("SouthKorea_Songdo", "normal", 48, 16, 6, seed=2)

    with torch.no_grad():
        alone = model(collate([small]))["logits"][0]
        together = model(collate([small, big]))["logits"][0]

    real = int(small["phase_valid"].sum())
    assert torch.allclose(alone[:real], together[:real], atol=1e-5)


@pytest.mark.parametrize(
    "junction,plan", [("Beijing_Beihuan", "normal"), ("Beijing_Pinganli", "easy")]
)
@pytest.mark.needs_scenarios
def test_padded_phases_are_unselectable(model, junction, plan):
    sample = make_sample(junction, plan, 48, 16, 6)
    with torch.no_grad():
        logits = model(collate([sample]))["logits"][0]

    real = int(sample["phase_valid"].sum())
    assert torch.isfinite(logits[:real]).all()
    assert (logits[real:] < -1e8).all(), "a padded phase could still be chosen"
    assert int(logits.argmax()) < real
    # And they take no softmax mass from the real candidates.
    assert torch.softmax(logits, dim=-1)[real:].sum() < 1e-9


@pytest.mark.needs_scenarios
def test_variable_phase_count_shares_one_model(model):
    """Junctions with K=3 and K=4 run through the same weights."""
    three = make_sample("Hongkong_YMT", "normal", 48, 16, 6)
    four = make_sample("Beijing_Pinganli", "easy", 48, 16, 6)
    with torch.no_grad():
        out = model(collate([three, four]))
    assert int(three["phase_valid"].sum()) == 3
    assert int(four["phase_valid"].sum()) == 4
    assert out["logits"].shape == (2, 6)


@pytest.mark.needs_scenarios
def test_phase_pooling_is_permutation_invariant(model):
    """A phase is a set, so reordering its movements must change nothing."""
    sample = make_sample("Beijing_Pinganli", "easy", 48, 16, 6)
    shuffled = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    members = shuffled["phase_members"]
    valid = shuffled["phase_member_valid"]
    for k in range(members.shape[0]):
        count = int(valid[k].sum())
        if count > 1:
            members[k, :count] = members[k, :count][::-1]

    with torch.no_grad():
        a = model(collate([sample]))["logits"]
        b = model(collate([shuffled]))["logits"]
    assert torch.allclose(a, b, atol=1e-5), "phase aggregation is order-dependent"


@pytest.mark.needs_scenarios
def test_lane_padding_does_not_leak_into_the_queue_head(model):
    sample = make_sample("Beijing_Pinganli", "easy", 48, 16, 6)
    with torch.no_grad():
        out = model(collate([sample]))
    real = int(sample["lane_valid"].sum())
    assert torch.isfinite(out["queue"]).all()
    # Padded lanes were fed 1e3; a leak would show up as a huge prediction.
    assert out["queue"][0, :real].max() < 1e3
