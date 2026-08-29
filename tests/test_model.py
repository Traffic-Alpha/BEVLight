'''Backbone, MaskPool and the auxiliary heads.

The properties here are the ones the whole design leans on: pooling is per lane,
carries no parameters, and does not care how many lanes there are.
'''

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bevlight.model.heads import LaneHead, QueueHead, masked_regression_loss
from bevlight.model.mask_pool import MaskPool, lane_patch_weights


def test_patch_weights_are_coverage_fractions():
    labels = np.zeros((28, 28), dtype=np.uint16)
    labels[:14, :14] = 1          # exactly one whole patch
    labels[14:21, 14:28] = 2      # half of one patch, half of another
    weights = lane_patch_weights(labels, [1, 2], patch_size=14)

    assert weights.shape == (2, 2, 2)
    assert weights[0, 0, 0] == pytest.approx(1.0)
    assert weights[0].sum() == pytest.approx(1.0)
    assert weights[1, 1, 1] == pytest.approx(0.5)


def test_patch_weights_reject_a_size_that_is_not_whole_patches():
    with pytest.raises(ValueError):
        lane_patch_weights(np.zeros((30, 30), dtype=np.uint16), [1], patch_size=14)


def test_mask_pool_averages_only_over_the_lane():
    features = torch.zeros(1, 3, 2, 2)
    features[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    features[0, :, 1, 1] = torch.tensor([9.0, 9.0, 9.0])
    weights = torch.zeros(1, 1, 2, 2)
    weights[0, 0, 0, 0] = 1.0                       # only the first cell belongs

    pooled = MaskPool()(features, weights)
    assert torch.allclose(pooled[0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_mask_pool_weights_partial_patches_proportionally():
    features = torch.zeros(1, 1, 1, 2)
    features[0, 0, 0, 0] = 0.0
    features[0, 0, 0, 1] = 10.0
    weights = torch.tensor([[[[1.0, 1.0]]]])        # both cells fully in the lane
    assert MaskPool()(features, weights)[0, 0, 0] == pytest.approx(5.0)

    weights = torch.tensor([[[[3.0, 1.0]]]])        # three quarters from the first
    assert MaskPool()(features, weights)[0, 0, 0] == pytest.approx(2.5)


def test_mask_pool_has_no_parameters():
    """A parameter here would be a per-lane weight, and lane counts vary."""
    assert list(MaskPool().parameters()) == []


def test_mask_pool_handles_any_lane_count():
    features = torch.randn(2, 8, 5, 5)
    for lanes in (1, 6, 12, 40):
        weights = torch.rand(lanes, 5, 5)
        assert MaskPool()(features, weights).shape == (2, lanes, 8)


def test_an_empty_lane_does_not_produce_nan():
    """A lane outside the window has zero weight everywhere."""
    features = torch.randn(1, 4, 3, 3)
    weights = torch.zeros(1, 1, 3, 3)
    pooled = MaskPool()(features, weights)
    assert torch.isfinite(pooled).all()


def test_lane_head_is_shared_across_lanes():
    """The same vector must score the same wherever it sits."""
    head = LaneHead(16).eval()
    vector = torch.randn(16)
    batch = torch.randn(1, 5, 16)
    batch[0, 0] = vector
    batch[0, 3] = vector
    with torch.no_grad():
        out = head(batch)
    assert out[0, 0] == pytest.approx(float(out[0, 3]), abs=1e-6)


def test_queue_head_never_predicts_a_negative_queue():
    head = QueueHead(16)
    assert bool((head(torch.randn(4, 7, 16)) >= 0).all())


def test_masked_loss_ignores_invalid_lanes():
    prediction = torch.zeros(1, 4)
    target = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert float(masked_regression_loss(prediction, target, valid)) == pytest.approx(0.0)


def test_masked_loss_is_zero_when_nothing_is_valid():
    loss = masked_regression_loss(torch.zeros(1, 3), torch.ones(1, 3), torch.zeros(1, 3))
    assert float(loss) == pytest.approx(0.0)


def test_queue_head_survives_a_mostly_zero_target():
    """Regression test for a dead output ReLU.

    Queue labels are mostly zero. With a ReLU output the gradient drives every
    pre-activation negative, the head lands in the flat region, and from then on
    it emits exactly zero with no gradient to escape. The giveaway is a probe
    whose metrics match "predict zero" to the decimal.
    """
    torch.manual_seed(0)
    features = torch.randn(1500, 24)
    weights = torch.randn(24)
    target = torch.relu(features @ weights * 0.5 - 1.0)
    assert (target == 0).float().mean() > 0.5, "target should be sparse for this test"

    head = QueueHead(24, hidden=64)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(400):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(head(features.unsqueeze(0)).squeeze(0), target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = head(features.unsqueeze(0)).squeeze(0)
    assert prediction.std() > 0.1, "head collapsed to a constant"
    assert prediction.abs().mean() > 0.05, "head collapsed to zero"
    predict_zero = target.abs().mean()
    assert (prediction - target).abs().mean() < 0.6 * predict_zero
