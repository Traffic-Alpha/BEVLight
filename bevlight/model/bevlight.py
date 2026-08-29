'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: The whole model, assembled.

    BEV pixels -> lane -> movement -> phase -> decision

Each hop is a separate module with one job, and the seams are deliberate: the
ablation table removes one at a time, and a layer that quietly did two things
would make those rows uninterpretable.

The model never sees a lane index, a direction label, or a phase number. Lane
identity arrives as a mask, movement identity as the pair of lanes it joins,
phase identity as the set of movements it serves. That is the entire basis for
transferring to a junction whose geometry and signal plan were never seen.

Nothing here has a dimension that depends on N, R or K. Padding is carried by
three masks that must reach every attention, every pooling, every softmax and
every loss — `tests/test_padding.py` checks that a sample's output does not
change when it is padded differently.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ._internal.decision import DecisionLayer
from ._internal.movement import MovementLayer
from ._internal.phase import PhaseLayer
from ._internal.temporal import TemporalFusion
from .heads import LaneHead, QueueHead
from .mask_pool import MaskPool


@dataclass
class BEVLightConfig:
    """Everything that changes between ablations."""

    embed_dim: int = 384          # backbone width
    model_dim: int = 256          # width of the trainable stack
    heads: int = 4
    temporal_layers: int = 1
    max_frames: int = 8
    pooling: str = "attention"
    lane_attention: bool = True
    movement_attention: bool = True
    use_temporal: bool = True
    min_green_s: float = 10.0
    use_phase_context: bool = True
    aux_queue: bool = True
    aux_occupancy: bool = True


class BEVLight(nn.Module):
    """Lane features in, phase scores and auxiliary lane states out.

    Takes pooled lane vectors rather than pixels: the backbone is frozen and
    MaskPool has no parameters, so those vectors are a fixed function of the
    image and are cached. `MaskPool` is kept here so the fine-tuning ablation can
    feed raw feature maps through the same object.
    """

    def __init__(self, config: BEVLightConfig | None = None):
        super().__init__()
        self.config = config or BEVLightConfig()
        c = self.config

        self.pool = MaskPool()
        self.project = nn.Sequential(
            nn.LayerNorm(c.embed_dim), nn.Linear(c.embed_dim, c.model_dim)
        )
        self.temporal = (
            TemporalFusion(c.model_dim, c.heads, c.temporal_layers, c.max_frames)
            if c.use_temporal else None
        )
        self.movement = MovementLayer(
            c.model_dim, c.heads,
            lane_attention=c.lane_attention,
            movement_attention=c.movement_attention,
        )
        self.phase = PhaseLayer(c.model_dim, c.heads, pooling=c.pooling)
        self.decision = DecisionLayer(c.model_dim, min_green_s=c.min_green_s,
                                      use_phase_context=c.use_phase_context)

        self.queue_head = QueueHead(c.model_dim) if c.aux_queue else None
        self.occupancy_head = LaneHead(c.model_dim) if c.aux_occupancy else None

    def encode_lanes(self, lane_features: torch.Tensor, lane_valid: torch.Tensor):
        """`(B, T, N, E)` cached vectors -> `(B, N, D)` state-plus-trend."""
        projected = self.project(lane_features)
        if self.temporal is None:
            return projected[:, -1] * lane_valid.unsqueeze(-1)
        return self.temporal(projected, lane_valid)

    def forward(self, batch: dict) -> dict:
        lanes = self.encode_lanes(batch["lane_features"], batch["lane_valid"])

        movements, competed = self.movement(
            lanes,
            batch["lane_valid"],
            batch["movement_in_index"],
            batch["movement_in_weight"],
            batch["movement_out_index"],
            batch["movement_out_weight"],
            batch["movement_valid"],
        )
        phases = self.phase(
            competed,
            batch["phase_members"],
            batch["phase_member_valid"],
            batch["phase_valid"],
        )
        logits = self.decision(
            phases, batch["phase_valid"], batch["current_phase"], batch["time_in_phase"]
        )

        out = {
            "logits": logits,
            "lane_features": lanes,
            "movement_features": movements,
            "phase_features": phases,
        }
        if self.queue_head is not None:
            out["queue"] = self.queue_head(lanes)
        if self.occupancy_head is not None:
            out["occupancy"] = torch.sigmoid(self.occupancy_head(lanes))
        return out

    @torch.no_grad()
    def act(self, batch: dict) -> torch.Tensor:
        out = self.forward(batch)
        return self.decision.act(
            out["logits"], batch["current_phase"], batch["time_in_phase"]
        )
