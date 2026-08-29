'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: The privileged teacher: the same model, reading numbers instead of pixels.

The teacher exists to answer one question the student cannot afford to ask —
*can anything beat max-pressure here* — without paying 2.6 s per sample to render
what it looks at. So it reads the per-lane state directly: queue, occupancy, and
the flag that says the queue reached the edge of the window and the count is a
lower bound.

It is deliberately not a new architecture. Everything above the lane encoder is
`BEVLight` unchanged — the same movement layer, the same set-based phase encoding,
the same candidate scoring. Three things follow from that, and they are the reason
for the choice:

  * it inherits variable N and K for free, so one teacher covers every junction
  * the distillation gap measures perception, not two different inductive biases
  * the student's decision layers can be initialised from the teacher's

What the teacher may see is fixed by what the image carries, not by what the
simulator can produce. Queue and occupancy come from `ObservationExtractor`, which
already clips to the BEV window, so they are the numbers a perfect reading of the
image would give. Vehicle identities, accumulated waiting, turn intentions and
anything beyond the window are not here, and must not be added: a teacher that
acts on them is teaching a lesson the student has no way to learn.

The auxiliary heads are off. Regressing the queue from the queue is free and
meaningless; on the student it is the whole grounding signal.
@LastEditTime: 2026-08-28
'''

from __future__ import annotations

import torch
import torch.nn as nn

from .bevlight import BEVLight, BEVLightConfig

# queue count, occupancy, and "this count is a lower bound".
LANE_STATE_DIM = 3
# Queue counts run to a few tens while occupancy is a fraction. Dividing by a
# constant rather than learning the scale keeps the encoder's first layer from
# having to spend capacity on units.
QUEUE_SCALE = 10.0


def teacher_config(model_dim: int = 128, embed_dim: int = 64, **overrides) -> BEVLightConfig:
    """`BEVLightConfig` for a structured-state policy."""
    return BEVLightConfig(
        embed_dim=embed_dim, model_dim=model_dim,
        aux_queue=False, aux_occupancy=False, **overrides,
    )


class LaneStateEncoder(nn.Module):
    """`(B, T, N, 3)` lane state -> `(B, T, N, embed_dim)`.

    Stands where the frozen vision backbone stands for the student: it turns
    whatever a lane looks like into a vector of the width the rest of the model
    expects. That is the only seam between the two.
    """

    def __init__(self, embed_dim: int = 64, in_dim: int = LANE_STATE_DIM):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor([1.0 / QUEUE_SCALE] + [1.0] * (in_dim - 1))
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, lane_state: torch.Tensor) -> torch.Tensor:
        return self.net(lane_state * self.scale)


class TeacherNet(nn.Module):
    """Lane state in, one score per candidate phase out.

    Used twice with different meanings: as the actor, whose scores are logits over
    the candidate phases, and as a critic, whose scores are Q-values. Nothing about
    the module changes between the two — the loss decides what the numbers mean.
    """

    def __init__(self, config: BEVLightConfig | None = None):
        super().__init__()
        self.config = config or teacher_config()
        self.encoder = LaneStateEncoder(self.config.embed_dim)
        self.core = BEVLight(self.config)

    def forward(self, batch: dict) -> torch.Tensor:
        """-> `(B, K)` scores, with padded candidates already at -inf."""
        lifted = self.encoder(batch["lane_state"])
        return self.core({**batch, "lane_features": lifted})["logits"]
