'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: Score a trained checkpoint on cached features, before any simulation runs.

Offline metrics are seconds of compute on a small tensor, so they come first:
they say whether training produced anything, and they shortlist which checkpoints
are worth the ~3 minutes an episode costs in closed loop.

What they are not is a selection criterion. Agreeing with the expert is not the
same as controlling well, and overall action accuracy is inflated by how often the
expert simply keeps the current phase. Every number here is therefore reported
against the reference that would produce it for free:

  * always keep the current phase - what a collapsed head scores
  * this junction's majority phase - what memorising the prior scores
  * uniform over the candidate set - what guessing scores

and the headline is split into *switch* and *keep* decisions, because a head that
has learned only "keep" posts a good overall accuracy and 0% on the decisions
that actually change the signal.

The variant breakdown matters for a different reason. Closed-loop evaluation can
only afford Panda's renderer, so if `panda_day` scores far below the Blender
variants, the policy is about to be evaluated out of its own domain — better to
find that here than after hours of simulation.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..data.collate import collate
from ..model.bevlight import BEVLight, BEVLightConfig


def episodes_meta(dataset) -> dict:
    """The index's per-episode block, whether given a dataset or a split of one."""
    parent = getattr(dataset, "parent", dataset)
    return parent.index["episodes"]


def load_checkpoint(path, device=None) -> BEVLight:
    """A saved checkpoint, rebuilt with the config it was trained under."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=dev, weights_only=False)
    model = BEVLight(BEVLightConfig(**payload["config"])).to(dev)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, dataset, batch_size: int = 64, device=None) -> dict:
    """Run the model over every sample and return flat arrays, one row each.

    Predictions come from `model.act`, not a raw argmax: minimum green is part of
    the policy, so the offline number describes the same decision rule the closed
    loop will execute.
    """
    dev = next(model.parameters()).device
    meta = episodes_meta(dataset)

    chosen, argmax, expert, current, top2 = [], [], [], [], []
    occupancy_error = []
    # Lane-level pairs, kept flat with the row they came from, so a proper R2 can
    # be computed over any subset rather than over per-sample averages.
    lane_row, lane_pred, lane_true = [], [], []
    for start in range(0, len(dataset), batch_size):
        rows = list(range(start, min(start + batch_size, len(dataset))))
        batch = {k: v.to(dev) for k, v in collate([dataset[i] for i in rows]).items()}
        out = model(batch)
        logits = out["logits"]

        chosen.append(model.decision.act(
            logits, batch["current_phase"], batch["time_in_phase"]
        ).cpu().numpy())
        argmax.append(logits.argmax(dim=-1).cpu().numpy())
        expert.append(batch["action"].cpu().numpy())
        current.append(batch["current_phase"].cpu().numpy())

        # "Expert's phase among the model's two best" — a near miss is a
        # different failure from scoring the right movement last.
        ranked = logits.argsort(dim=-1, descending=True)[:, :2]
        top2.append((ranked == batch["action"].unsqueeze(1)).any(dim=1).cpu().numpy())

        if "queue" in out:
            # Real incoming lanes only, minus those whose queue ran off the image:
            # the same accounting the training loss and eval/probe.py use.
            valid = batch["incoming_valid"] * batch.get(
                "queue_valid", torch.ones_like(batch["incoming_valid"])
            )
            picked = valid.bool()
            sample_of = torch.arange(len(rows), device=dev).unsqueeze(1).expand_as(picked)
            lane_row.append((sample_of[picked] + start).cpu().numpy())
            lane_pred.append(out["queue"][picked].cpu().numpy())
            lane_true.append(batch["queue_target"][picked].cpu().numpy())
        if "occupancy" in out:
            valid = batch["lane_valid"]
            error = (out["occupancy"] - batch["occupancy_target"]).abs() * valid
            occupancy_error.append(
                (error.sum(dim=1) / valid.sum(dim=1).clamp(min=1)).cpu().numpy()
            )

    samples = dataset.samples
    return {
        "chosen": np.concatenate(chosen),
        "argmax": np.concatenate(argmax),
        "expert": np.concatenate(expert),
        "current": np.concatenate(current),
        "top2": np.concatenate(top2),
        "lane_row": np.concatenate(lane_row) if lane_row else np.zeros(0, dtype=np.int64),
        "lane_pred": np.concatenate(lane_pred) if lane_pred else np.zeros(0),
        "lane_true": np.concatenate(lane_true) if lane_true else np.zeros(0),
        "occupancy_mae": np.concatenate(occupancy_error) if occupancy_error else np.zeros(0),
        "episode": np.array([s["episode"] for s in samples]),
        "variant": np.array([s["variant"] for s in samples]),
        "junction": np.array([meta[s["episode"]]["junction"] for s in samples]),
        "plan": np.array([meta[s["episode"]]["plan"] for s in samples]),
        "demand": np.array([meta[s["episode"]]["demand"] for s in samples]),
        "group": np.array([
            f'{meta[s["episode"]]["junction"]}/{meta[s["episode"]]["plan"]}'
            for s in samples
        ]),
        "num_phases": np.array([s["num_phases"] for s in samples]),
    }


def agreement(prediction: dict, rows: np.ndarray | None = None) -> dict:
    """Action agreement, split into the two cases that mean different things."""
    if rows is None:
        rows = np.arange(prediction["expert"].size)
    if not rows.size:
        return {"n": 0}

    expert = prediction["expert"][rows]
    chosen = prediction["chosen"][rows]
    current = prediction["current"][rows]
    switches = expert != current

    def share(mask, values):
        return round(float(values[mask].mean()), 4) if mask.any() else None

    return {
        "n": int(rows.size),
        "accuracy": round(float((chosen == expert).mean()), 4),
        "accuracy_argmax": round(float((prediction["argmax"][rows] == expert).mean()), 4),
        "top2": round(float(prediction["top2"][rows].mean()), 4),
        "switch_share": round(float(switches.mean()), 4),
        "accuracy_on_switch": share(switches, chosen == expert),
        "accuracy_on_keep": share(~switches, chosen == expert),
        # The model's own switching rate. Matching the expert's says it is not
        # simply refusing to move, which accuracy alone cannot rule out.
        "predicted_switch_share": round(float((chosen != current).mean()), 4),
    }


def references(prediction: dict, rows: np.ndarray | None = None) -> dict:
    """What the same decisions score without reading anything."""
    if rows is None:
        rows = np.arange(prediction["expert"].size)
    if not rows.size:
        return {}

    expert = prediction["expert"][rows]
    current = prediction["current"][rows]
    group = prediction["group"][rows]

    majority = np.zeros_like(expert)
    for name in np.unique(group):
        mask = group == name
        majority[mask] = Counter(expert[mask].tolist()).most_common(1)[0][0]

    return {
        "always_keep_current": round(float((current == expert).mean()), 4),
        "group_majority_phase": round(float((majority == expert).mean()), 4),
        "uniform_random": round(float((1.0 / prediction["num_phases"][rows]).mean()), 4),
    }


def macro_f1(prediction: dict, rows: np.ndarray) -> float | None:
    """Mean per-junction macro-F1 over candidate positions.

    Averaged within a junction and then across them, because position 2 serves
    different movements at different junctions and pooling them says nothing. Its
    job here is narrow: expose a candidate the model never picks, which overall
    accuracy hides.
    """
    scores = []
    for name in np.unique(prediction["group"][rows]):
        mask = rows[prediction["group"][rows] == name]
        expert, chosen = prediction["expert"][mask], prediction["chosen"][mask]
        per_class = []
        for label in np.unique(expert):
            tp = float(((chosen == label) & (expert == label)).sum())
            fp = float(((chosen == label) & (expert != label)).sum())
            fn = float(((chosen != label) & (expert == label)).sum())
            per_class.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        if per_class:
            scores.append(sum(per_class) / len(per_class))
    return round(sum(scores) / len(scores), 4) if scores else None


def auxiliary(prediction: dict, rows: np.ndarray | None = None) -> dict:
    """Queue and occupancy heads, on the same decisions.

    Reported per lane rather than per decision, so the figure is comparable with
    the standalone probe in eval/probe.py.
    """
    if rows is None:
        rows = np.arange(prediction["expert"].size)
    result = {}

    if prediction["lane_row"].size:
        wanted = np.zeros(prediction["expert"].size, dtype=bool)
        wanted[rows] = True
        keep = wanted[prediction["lane_row"]]
        pred, true = prediction["lane_pred"][keep], prediction["lane_true"][keep]
        if pred.size:
            error = pred - true
            ss_tot = float(((true - true.mean()) ** 2).sum())
            nonempty = true > 0
            result["queue_mae"] = round(float(np.abs(error).mean()), 4)
            result["queue_r2"] = (
                round(1.0 - float((error ** 2).sum()) / ss_tot, 4) if ss_tot > 0 else None
            )
            result["queue_mae_nonempty"] = (
                round(float(np.abs(error[nonempty]).mean()), 4) if nonempty.any() else None
            )
            result["queue_lanes"] = int(pred.size)

    if prediction["occupancy_mae"].size:
        result["occupancy_mae"] = round(float(prediction["occupancy_mae"][rows].mean()), 4)
    return result


def evaluate(model, dataset, batch_size: int = 64, device=None) -> dict:
    """Every offline number for one checkpoint, overall and broken down."""
    prediction = predict(model, dataset, batch_size=batch_size, device=device)
    everything = np.arange(prediction["expert"].size)

    report = {
        "overall": {
            **agreement(prediction),
            "macro_f1": macro_f1(prediction, everything),
            **auxiliary(prediction),
        },
        "references": references(prediction),
        "by": {},
    }
    for field in ("variant", "junction", "plan", "demand"):
        block = {}
        for name in sorted(set(prediction[field].tolist())):
            rows = np.nonzero(prediction[field] == name)[0]
            block[name] = {
                **agreement(prediction, rows),
                **auxiliary(prediction, rows),
                **references(prediction, rows),
            }
        report["by"][field] = block
    return report


def print_report(report: dict, title: str = "") -> None:
    overall, refs = report["overall"], report["references"]
    print(f"\n=== {title} ===" if title else "")
    print(f"    decisions            {overall['n']}")
    print(f"    action accuracy      {overall['accuracy']:.3f}   "
          f"(argmax {overall['accuracy_argmax']:.3f}, expert in top-2 {overall['top2']:.3f})")
    switch = overall["accuracy_on_switch"]
    keep = overall["accuracy_on_keep"]
    print(f"      on switch          {switch if switch is None else f'{switch:.3f}'}   "
          f"({overall['switch_share']:.0%} of decisions)")
    print(f"      on keep            {keep if keep is None else f'{keep:.3f}'}")
    print(f"    macro-F1 (per junction) {overall['macro_f1']}")
    print(f"    switch rate          model {overall['predicted_switch_share']:.3f} "
          f"vs expert {overall['switch_share']:.3f}")
    if "queue_mae" in overall:
        print(f"    aux queue MAE        {overall['queue_mae']:.3f} vehicles "
              f"(R2 {overall['queue_r2']}, non-empty MAE {overall['queue_mae_nonempty']}, "
              f"{overall['queue_lanes']} lane readings)")
    if "occupancy_mae" in overall:
        print(f"    aux occupancy MAE    {overall['occupancy_mae']:.3f}")
    print("    reference            " + "  ".join(
        f"{name}={value:.3f}" for name, value in refs.items()
    ))

    for field in ("variant", "junction"):
        block = report["by"].get(field) or {}
        if len(block) < 2:
            continue
        print(f"    by {field}:")
        for name, stats in block.items():
            switch = stats.get("accuracy_on_switch")
            print(f"      {name:<32} n={stats['n']:<6} acc={stats['accuracy']:.3f} "
                  f"switch={switch if switch is None else f'{switch:.3f}'} "
                  f"keep_ref={stats['always_keep_current']:.3f}")


def checkpoints_of(run_dir: Path) -> list:
    return sorted(run_dir.glob("checkpoint_*.pt"))
