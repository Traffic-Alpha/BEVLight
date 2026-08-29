'''The behaviour-cloning loss, and the label convention it depends on.'''

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bevlight.data.collate import collate, junction_structure
from bevlight.model.bevlight import BEVLight, BEVLightConfig
from bevlight.scenario.lane_mask import load_lane_mask
from bevlight.train.losses import (
    action_accuracy,
    bevlight_loss,
    masked_l1,
    phase_cross_entropy,
)


def make(junction: str, plan: str, action: int, seed: int = 0) -> dict:
    mask = load_lane_mask(junction, plan)
    structure = junction_structure(mask)
    rng = np.random.default_rng(seed)
    return {
        **{k: v for k, v in structure.items() if k not in ("lane_order", "phase_order")},
        "lane_features": rng.standard_normal((5, 48, 384)).astype(np.float32),
        "current_phase": 0,
        "time_in_phase": 12.0,
        "action": action,
        "queue_target": (rng.random(48) * 4).astype(np.float32),
        "occupancy_target": rng.random(48).astype(np.float32),
        "queue_valid": np.ones(48, dtype=np.float32),
    }


def test_a_label_pointing_at_a_padded_phase_is_rejected():
    """The failure this guards is silent: a misaligned label still trains."""
    batch = collate([make("Hongkong_YMT", "normal", action=5)])
    logits = torch.zeros(1, 6)
    with pytest.raises(ValueError, match="padded phase"):
        phase_cross_entropy(logits, batch["action"], batch["phase_valid"])


def test_padded_candidates_take_no_probability():
    torch.manual_seed(0)
    batch = collate([make("Hongkong_YMT", "normal", action=1)])
    model = BEVLight(BEVLightConfig()).eval()
    with torch.no_grad():
        logits = model(batch)["logits"]
    real = int(batch["phase_valid"][0].sum())
    assert torch.softmax(logits, dim=-1)[0, real:].sum() < 1e-9


def test_junctions_with_different_k_share_one_batch():
    """K=4 and K=3 together; each label is a position in its own candidate set."""
    batch = collate([
        make("Beijing_Pinganli", "easy", action=3),
        make("Hongkong_YMT", "normal", action=2, seed=1),
    ])
    torch.manual_seed(0)
    model = BEVLight(BEVLightConfig())
    loss, parts = bevlight_loss(model(batch), batch)
    assert torch.isfinite(loss)
    assert set(parts) >= {"ce", "queue", "occupancy", "total"}


def test_saturated_queue_labels_can_be_excluded():
    """A queue at the image edge is a lower bound, not a measurement."""
    sample = make("Beijing_Pinganli", "easy", action=0)
    sample["queue_valid"] = np.zeros(48, dtype=np.float32)
    batch = collate([sample])
    torch.manual_seed(0)
    outputs = BEVLight(BEVLightConfig())(batch)
    _, parts = bevlight_loss(outputs, batch)
    assert parts["queue"] == pytest.approx(0.0)


def test_masked_l1_ignores_invalid_lanes():
    prediction = torch.zeros(1, 4)
    target = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert float(masked_l1(prediction, target, valid)) == pytest.approx(0.0)


def test_auxiliary_weight_zero_reduces_to_pure_behaviour_cloning():
    batch = collate([make("Beijing_Pinganli", "easy", action=1)])
    torch.manual_seed(0)
    outputs = BEVLight(BEVLightConfig())(batch)
    total, parts = bevlight_loss(outputs, batch, queue_weight=0.0, occupancy_weight=0.0)
    assert float(total.detach()) == pytest.approx(parts["ce"], abs=1e-5)


def test_a_confident_correct_prediction_has_low_loss():
    logits = torch.tensor([[10.0, 0.0, 0.0, -1e9]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    right = phase_cross_entropy(logits, torch.tensor([0]), valid)
    wrong = phase_cross_entropy(logits, torch.tensor([1]), valid)
    assert float(right) < 0.01
    assert float(wrong) > float(right)


def test_action_accuracy_is_monitoring_only():
    logits = torch.tensor([[3.0, 1.0], [0.0, 5.0]])
    stats = action_accuracy(logits, torch.tensor([0, 1]))
    assert stats == {"accuracy": 1.0, "n": 2}


def test_the_soft_loss_is_a_real_number_when_phases_are_padded():
    """Every junction has fewer candidate phases than MAX_PHASES.

    A padded candidate carries target 0 against log-probability -inf, and
    `0 * -inf` is NaN. The gradient survives that — the infinity sits on the
    expert's scores, which are constants — so a NaN here does not stop a run
    from training correctly, and that is exactly what makes it dangerous: the
    loss it reports is NaN in every epoch of every run, and a soft target that
    really did diverge would look identical.
    """
    import torch

    from bevlight.train.losses import expert_score_cross_entropy

    logits = torch.randn(4, 6, requires_grad=True)
    score = torch.tensor([[3.0, 1.0, 2.0, 0.0, 0.0, 0.0]] * 4)
    phase_valid = torch.zeros(4, 6, dtype=torch.bool)
    phase_valid[:, :3] = True                      # K = 3, three padded slots
    informative = torch.ones(4)

    loss = expert_score_cross_entropy(logits, score, phase_valid, informative)
    assert torch.isfinite(loss), f"padded candidates made the loss {loss.item()}"

    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[~phase_valid].abs().max() == 0, "a padded candidate got gradient"


def test_the_soft_loss_actually_differs_from_the_one_hot_one():
    """Otherwise `--soft-weight` would be an expensive no-op.

    The scores say the runner-up was nearly as good; the argmax throws that
    away. If both losses produced the same gradient direction there would be
    nothing to ablate.
    """
    import torch

    from bevlight.train.losses import expert_score_cross_entropy, phase_cross_entropy

    base = torch.randn(8, 4)
    score = torch.tensor([[3.0, 2.8, 1.0, 0.2]] * 8)   # top two nearly tied
    phase_valid = torch.ones(8, 4, dtype=torch.bool)
    action = torch.zeros(8, dtype=torch.long)

    soft = base.clone().requires_grad_(True)
    expert_score_cross_entropy(soft, score, phase_valid, torch.ones(8)).backward()

    hard = base.clone().requires_grad_(True)
    phase_cross_entropy(hard, action, phase_valid).backward()

    cosine = torch.nn.functional.cosine_similarity(
        soft.grad.flatten(), hard.grad.flatten(), dim=0
    )
    assert cosine < 0.99, f"soft and one-hot gradients are the same direction ({cosine:.4f})"
