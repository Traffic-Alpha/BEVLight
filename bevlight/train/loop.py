'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Behaviour cloning over cached lane features.

Deliberately plain: the backbone is frozen and its output is cached, so an epoch
is a few matrix multiplies over a small tensor. The interesting decisions are
elsewhere.

Two of them are worth stating because they are easy to get wrong quietly:

  * **Checkpoints are not selected here.** Action accuracy is logged, but it is
    inflated by how often the expert keeps the current phase, and agreeing with
    the expert is not the same as controlling well. Every checkpoint is saved and
    scored later in closed loop; this loop only produces candidates.
  * **The action distribution is printed before training starts.** If "keep the
    current phase" dominates, behaviour cloning will happily collapse onto it and
    still post a good accuracy. Seeing the distribution first is what makes that
    diagnosable rather than mysterious.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..data.collate import collate
from ..model.bevlight import BEVLight, BEVLightConfig
from .losses import action_accuracy, bevlight_loss


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    queue_weight: float = 0.2
    occupancy_weight: float = 0.2
    # Weight on the expert's per-candidate scores rather than only its argmax.
    soft_weight: float = 0.0
    soft_temperature: float = 0.5
    # Every 5, not every 10: on train45 the best validation loss landed at epoch
    # 6 and no checkpoint existed within four epochs of it.
    checkpoint_every: int = 5
    seed: int = 0
    # The model is small enough that a step is dominated by kernel-launch
    # overhead rather than arithmetic: batch 32 and batch 256 cost almost the
    # same per step. Compiling fuses that overhead away without touching the
    # batch size, which is a hyperparameter and not a speed knob.
    compile_model: bool = True


def action_distribution(samples) -> dict:
    """What the expert actually did, before anything is trained on it."""
    keeps = 0
    chosen = Counter()
    for decision in samples:
        chosen[decision["action"]] += 1
        keeps += int(decision["action"] == decision["phase"])
    total = max(1, len(samples))
    return {
        "n": len(samples),
        "keep_current_phase": round(keeps / total, 3),
        "per_phase": {str(k): v for k, v in sorted(chosen.items())},
    }


def batches(dataset, batch_size: int, shuffle: bool, generator=None, source=None):
    """Batches of `dataset`, from a staged `BatchSource` when one is given.

    The fallback assembles each sample in NumPy and stacks them, which is what
    `tests/test_batching.py` pins the staged path against.
    """
    order = (
        torch.randperm(len(dataset), generator=generator).tolist()
        if shuffle else list(range(len(dataset)))
    )
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        if not chunk:
            continue
        yield source.batch(chunk) if source is not None else collate(
            [dataset[i] for i in chunk]
        )


def evaluate(model, dataset, config: TrainConfig, device, source=None) -> dict:
    model.eval()
    totals, correct, seen = [], 0, 0
    with torch.no_grad():
        for batch in batches(dataset, config.batch_size, shuffle=False, source=source):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            _, parts = bevlight_loss(
                outputs, batch, config.queue_weight, config.occupancy_weight,
                config.soft_weight, config.soft_temperature,
            )
            totals.append(parts)
            stats = action_accuracy(outputs["logits"], batch["action"])
            correct += stats["accuracy"] * stats["n"]
            seen += stats["n"]
    mean = {k: sum(t[k] for t in totals) / len(totals) for k in totals[0]} if totals else {}
    mean["accuracy"] = correct / max(1, seen)
    return mean


def train(
    train_set,
    valid_set,
    model_config: BEVLightConfig | None = None,
    config: TrainConfig | None = None,
    run_dir: Path | None = None,
    device: str | None = None,
    meta: dict | None = None,
) -> dict:
    config = config or TrainConfig()
    model_config = model_config or BEVLightConfig()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    # Stage both halves on the device once; a batch is then a few gathers rather
    # than 32 NumPy assemblies. Measured on train45: ~20 s/epoch -> ~8 s/epoch.
    from ..data.batching import BatchSource

    train_source = BatchSource(train_set, device=str(dev))
    valid_source = BatchSource(valid_set, device=str(dev))

    model = BEVLight(model_config).to(dev)
    # Training only. nn.TransformerEncoder takes a fused fastpath in eval mode
    # that Inductor cannot trace, and evaluation is a small share of the epoch
    # anyway, so the compiled wrapper is used for the step and the plain module
    # for everything that reads the model.
    compiled = model
    if config.compile_model and dev.type == "cuda":
        try:
            compiled = torch.compile(model)
        except Exception as error:  # a compile failure must not lose a run
            print(f"  [warn] torch.compile unavailable ({error}); running eager")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    def lr_at(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(1, config.warmup_epochs)
        span = max(1, config.epochs - config.warmup_epochs)
        progress = (epoch - config.warmup_epochs) / span
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        # `meta` records which dataset and which held-out episodes produced these
        # checkpoints, so eval_offline / eval_closed_loop can score them later
        # without being told again — and cannot silently score the wrong split.
        (run_dir / "config.json").write_text(
            json.dumps(
                {**(meta or {}), "train": asdict(config), "model": asdict(model_config)},
                indent=2,
            )
        )
        (run_dir / "action_stats.json").write_text(
            json.dumps(
                {
                    "train": action_distribution(train_set.samples),
                    "valid": action_distribution(valid_set.samples),
                },
                indent=2,
            )
        )

    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        seen = []
        for batch in batches(train_set, config.batch_size, True, generator, train_source):
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, parts = bevlight_loss(
                compiled(batch), batch, config.queue_weight, config.occupancy_weight,
                config.soft_weight, config.soft_temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            seen.append(parts)
        schedule.step()

        train_mean = {k: sum(s[k] for s in seen) / len(seen) for k in seen[0]}
        validation = evaluate(model, valid_set, config, dev, valid_source)
        history.append({"epoch": epoch, "train": train_mean, "valid": validation})
        print(
            f"  epoch {epoch:3d}  train ce={train_mean['ce']:.4f}  "
            f"valid ce={validation['ce']:.4f}  "
            f"valid acc={validation['accuracy']:.3f}  "
            f"queue={validation.get('queue', 0):.4f}"
        )

        # Every checkpoint is a candidate; the closed loop decides between them.
        if run_dir and (epoch + 1) % config.checkpoint_every == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": asdict(model_config)},
                run_dir / f"checkpoint_{epoch + 1:03d}.pt",
            )

    elapsed = time.perf_counter() - started
    if run_dir:
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "elapsed_s": round(elapsed, 1), "model": model}
