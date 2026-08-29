'''The ablation table: that every row is runnable, and says what it is for.

A row that cannot be constructed is discovered on the day the table is due; a
row whose claim was never written gets read afterwards as evidence for whatever
its number suggests. Both are cheap to prevent here.
'''

from __future__ import annotations

import pytest

from bevlight.ablation import ABLATIONS, resolve
from bevlight.model.bevlight import BEVLight, BEVLightConfig
from bevlight.train.loop import TrainConfig


def test_the_table_has_a_reference_row():
    """Every other row is read as a difference, so the baseline must be in it."""
    assert "full" in ABLATIONS
    row = ABLATIONS["full"]
    assert not row.model and not row.train and not row.data


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_every_row_says_what_it_is_evidence_for(name):
    assert len(ABLATIONS[name].why.split()) >= 8, "a why has to be a claim, not a label"


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_every_override_names_a_real_setting(name):
    """A typo'd key would be silently ignored and report the un-ablated model."""
    row = ABLATIONS[name]
    model_fields = set(BEVLightConfig().__dataclass_fields__)
    train_fields = set(TrainConfig().__dataclass_fields__)

    assert set(row.model) <= model_fields, f"{name}: unknown BEVLightConfig keys"
    assert set(row.train) <= train_fields, f"{name}: unknown TrainConfig keys"
    assert set(row.data) <= {"window", "variant"}, f"{name}: unknown dataset keys"


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_every_row_builds_a_model(name):
    """Constructing it here costs milliseconds; discovering it later costs a run."""
    row = ABLATIONS[name]
    config = BEVLightConfig(**row.model)
    TrainConfig(**row.train)
    assert BEVLight(config) is not None


def test_the_rows_actually_differ_from_the_reference():
    """A row identical to `full` would print a difference of zero and mean nothing."""
    reference = BEVLightConfig()
    for name, row in ABLATIONS.items():
        if name == "full" or row.rebuild_features:
            continue
        changed = (BEVLightConfig(**row.model) != reference
                   or row.train or row.data)
        assert changed, f"{name} changes nothing"


def test_dropping_the_phase_context_removes_those_inputs():
    """The ablation has to actually reach the layer it names."""
    with_context = BEVLight(BEVLightConfig(use_phase_context=True))
    without = BEVLight(BEVLightConfig(use_phase_context=False))
    assert without.decision.use_phase_context is False
    assert sum(p.numel() for p in without.parameters()) < sum(
        p.numel() for p in with_context.parameters()
    )


def test_hard_pooling_gives_each_patch_to_one_lane():
    """The alternative `mask_pool`'s docstring argues against, made measurable."""
    import numpy as np

    from bevlight.model.mask_pool import lane_patch_weights

    labels = np.zeros((28, 28), dtype=np.uint16)
    labels[:, :14] = 1
    labels[:, 14:] = 2
    labels[0, 13] = 2                      # one disputed pixel on the boundary

    soft = lane_patch_weights(labels, [1, 2])
    hard = lane_patch_weights(labels, [1, 2], hard=True)

    assert 0.0 < float(soft[1, 0, 0]) < 1.0, "the boundary patch should be shared"
    assert bool((hard.sum(dim=0) <= 1.0).all()), "hard weights must be one-hot"
    assert float(hard[0, 0, 0]) == 1.0, "the patch goes to the lane covering most"


def test_an_unknown_ablation_is_refused_with_the_list():
    with pytest.raises(ValueError, match="Unknown ablation"):
        resolve("no_such_row")


def test_cache_time_ablations_are_marked():
    """`hard_pool` cannot be applied at training time, and must say so."""
    assert ABLATIONS["hard_pool"].rebuild_features
    assert not ABLATIONS["no_temporal"].rebuild_features
