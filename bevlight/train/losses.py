'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Behaviour-cloning loss plus auxiliary lane-state grounding.

    L = CE(phase scores, expert choice) + l1 * queue + l2 * occupancy

The cross-entropy runs over **the candidate phases of this junction under this
plan**, and the label is the expert's choice *as a position in that candidate
set* — never a global phase id. This is the hinge the cross-plan experiment turns
on: phase 0 at one junction serves one straight movement, at another two
opposing straights, at a third three movements including a left. A label that
meant "phase 0" would be teaching a symbol with no meaning anywhere else.

Padded candidates are already at -inf from the decision layer, so they take no
softmax mass. Auxiliary losses are computed on real lanes only, and the queue
target is additionally masked where the queue reached the edge of the image:
that label is a lower bound, and training on it teaches the model to under-count
exactly when the junction is busiest.

Auxiliary weights stay small (0.1-0.3) and the targets are normalised first. The
job of these heads is to stop the visual features drifting into whatever happens
to satisfy a sparse, imbalanced action label; they are not the objective.
'''

from __future__ import annotations

import torch
import torch.nn.functional as F

# Queue is measured in vehicles; the BEV window holds about 8, so this maps the
# target to roughly [0, 1] and keeps the auxiliary weight comparable to the CE.
QUEUE_SCALE = 8.0


# Padded candidates are removed from every softmax by this fill.
NEG_INF = float("-inf")


def phase_cross_entropy(logits: torch.Tensor, action: torch.Tensor,
                        phase_valid: torch.Tensor) -> torch.Tensor:
    """CE over the current candidate set, label = position within it."""
    if (action >= phase_valid.shape[1]).any():
        raise ValueError("action indexes past the padded phase dimension")
    chosen_is_real = torch.gather(phase_valid, 1, action.unsqueeze(1)).squeeze(1)
    if not bool((chosen_is_real > 0).all()):
        raise ValueError("expert chose a padded phase; the label is misaligned")
    return F.cross_entropy(logits, action)


def expert_score_cross_entropy(logits: torch.Tensor, score: torch.Tensor,
                               phase_valid: torch.Tensor, informative: torch.Tensor,
                               temperature: float = 0.5) -> torch.Tensor:
    """Cross-entropy against what the expert scored every candidate, not just its pick.

    Max-pressure computes a pressure per phase and takes the argmax. A one-hot
    label keeps the argmax and discards the margins, which costs twice: the model
    is never told that second place was nearly as good, and where two candidates
    are scored equal it is trained to prefer whichever the tie-break happened to
    return. On the collected episodes 19% of the informative decisions have their
    top two within 20% of each other.

    Scores are normalised by their own maximum before the softmax, because
    pressure magnitudes differ by an order of magnitude between an empty junction
    and a congested one, and a fixed temperature would sharpen one and flatten the
    other. `informative` is zero where every candidate scored the same, and those
    samples contribute nothing here — a junction with no traffic is a real
    statement that the choice does not matter, not a preference to learn.
    """
    scale = score.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    scaled = (score / scale) / temperature
    target = torch.softmax(scaled.masked_fill(~phase_valid.bool(), NEG_INF), dim=1)

    log_probability = torch.log_softmax(
        logits.masked_fill(~phase_valid.bool(), NEG_INF), dim=1
    )
    # A padded candidate has target 0 and log-probability -inf, and `0 * -inf`
    # is NaN. The gradient survives it — the -inf sits on the target, which is a
    # constant and needs none — but the reported loss does not, and a metric
    # that is NaN in every run cannot show a soft target diverging. Drop those
    # terms explicitly rather than relying on which branch autograd walks.
    contribution = torch.where(
        phase_valid.bool(), target * log_probability, torch.zeros_like(target)
    )
    per_sample = -contribution.sum(dim=1)
    total = informative.sum()
    if total == 0:
        return logits.sum() * 0.0
    return (per_sample * informative).sum() / total


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
    total = valid.sum()
    if total == 0:
        return prediction.sum() * 0.0
    return ((prediction - target).abs() * valid).sum() / total


def bevlight_loss(
    outputs: dict,
    batch: dict,
    queue_weight: float = 0.2,
    occupancy_weight: float = 0.2,
    soft_weight: float = 0.0,
    soft_temperature: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Total loss and its parts, for logging."""
    losses = {}
    hard = phase_cross_entropy(outputs["logits"], batch["action"], batch["phase_valid"])
    losses["ce"] = float(hard.detach())
    total = hard

    # Mixed rather than swapped: the argmax is still the expert's actual decision,
    # and the scores only say how close the alternatives were.
    if soft_weight > 0 and "expert_score" in batch:
        soft = expert_score_cross_entropy(
            outputs["logits"], batch["expert_score"], batch["phase_valid"],
            batch["expert_score_valid"], soft_temperature,
        )
        total = (1.0 - soft_weight) * hard + soft_weight * soft
        losses["soft_ce"] = float(soft.detach())

    if "queue" in outputs and "queue_target" in batch:
        # Real incoming lanes, minus the ones whose queue ran off the image.
        valid = batch["incoming_valid"]
        if "queue_valid" in batch:
            valid = valid * batch["queue_valid"]
        queue = masked_l1(
            outputs["queue"] / QUEUE_SCALE, batch["queue_target"] / QUEUE_SCALE, valid
        )
        total = total + queue_weight * queue
        losses["queue"] = float(queue.detach())

    if "occupancy" in outputs and "occupancy_target" in batch:
        occupancy = masked_l1(
            outputs["occupancy"], batch["occupancy_target"], batch["lane_valid"]
        )
        total = total + occupancy_weight * occupancy
        losses["occupancy"] = float(occupancy.detach())

    losses["total"] = float(total.detach())
    return total, losses


def action_accuracy(logits: torch.Tensor, action: torch.Tensor) -> dict:
    """Monitoring only.

    Never used to select a checkpoint: it is inflated by how often the expert
    keeps the current phase, and agreeing with the expert is not the same as
    controlling well. Checkpoints are chosen on closed-loop metrics.
    """
    predicted = logits.argmax(dim=-1)
    return {
        "accuracy": float((predicted == action).float().mean()),
        "n": int(action.numel()),
    }
