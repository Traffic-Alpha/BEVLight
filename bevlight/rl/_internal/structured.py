'''
@Author: WANG Maonan
@Date: 2026-08-30
@Description: The baselines, given the network the observation was always shaped for.

A flat network cannot generalise across junctions here, and the reason is not the
algorithm. Shown forty-eight lane queues and a phase index, it has to learn
"action 2 shortens lanes 5, 6 and 12" from experience -- an association true of
one junction under one signal plan and wrong the moment either changes. At
Hongkong_YMT action 0 releases six lanes under `normal` and three different ones
under `easy`. That is a memorised phase id, and it is what a cross-plan
evaluation collapses on.

The wiring is in the observation as gather indices, and this is what reads them:
lanes are encoded, pooled into the movements that own them, and pooled again
into the phases that serve those movements. A phase's representation is then
built out of the lanes that phase actually releases, so an unseen plan is a
different gather rather than an index the network has never been rewarded for.

`TeacherNet` already is that network -- `(B, K)` scores from `lane_state` and the
wiring, with padded candidates at -inf and one set of weights shared across
candidates. Nothing here re-implements it; what these classes do is let SB3 use
it as its Q-network and its actor, so the algorithm is the only thing that
varies between a row of the baseline table and the row above it.
'''

from __future__ import annotations

import torch
from torch import nn

#: What `TeacherNet` reads out of the observation, and the shape it wants it in.
#: The environment publishes `current_phase` and `time_in_phase` as `(B, 1)`
#: because a gymnasium Box needs a shape; the decision layer indexes with them
#: and wants `(B,)`.
SQUEEZED = ("current_phase", "time_in_phase")
INDEXED = ("current_phase", "movement_in_index", "movement_out_index",
           "phase_members")


def as_batch(observation: dict) -> dict:
    """SB3's observation dict, in the form the structured network reads.

    Gather indices have to be integers however the replay buffer stored them --
    SB3 keeps a Box's dtype, but a buffer that has been through normalisation or
    a float cast would silently produce a `gather` on floats.
    """
    batch = dict(observation)
    for key in SQUEEZED:
        if key in batch and batch[key].dim() > 1:
            batch[key] = batch[key].squeeze(-1)
    for key in INDEXED:
        if key in batch:
            batch[key] = batch[key].long()
    if "time_in_phase" in batch:
        batch["time_in_phase"] = batch["time_in_phase"].float()
    return batch


class StructuredTrunk(nn.Module):
    """`TeacherNet` plus a value head, so one module serves both algorithm families.

    DQN reads the scores as Q-values and never calls `value`; PPO reads them as
    logits and does. Sharing the trunk is not an efficiency: it is what makes
    "same network, different algorithm" true of the table.
    """

    def __init__(self, model_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        from ...model.teacher import TeacherNet, teacher_config

        self.net = TeacherNet(teacher_config(model_dim=model_dim,
                                             embed_dim=embed_dim))
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
        )

    def scores(self, observation: dict) -> torch.Tensor:
        """-> `(B, K)`, padded candidates at -inf."""
        return self.net(as_batch(observation))

    def value(self, observation: dict) -> torch.Tensor:
        """-> `(B, 1)`, pooled over the phases the junction actually has."""
        batch = as_batch(observation)
        lifted = self.net.encoder(batch["lane_state"])
        features = self.net.core({**batch, "lane_features": lifted})["phase_features"]
        valid = batch["phase_valid"].unsqueeze(-1)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.value_head(pooled)


def structured_dqn_policy():
    """`DQNPolicy` whose Q-network is the hierarchy rather than an MLP.

    Built on call so that importing this module costs no stable-baselines3.
    """
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.dqn.policies import DQNPolicy

    class StructuredQNetwork(BasePolicy):
        def __init__(self, observation_space, action_space, features_extractor,
                     features_dim, **_ignored):
            super().__init__(observation_space, action_space,
                             features_extractor=features_extractor,
                             normalize_images=False)
            self.trunk = StructuredTrunk()

        def forward(self, obs) -> torch.Tensor:
            # No `extract_features`: the trunk reads the observation's own keys,
            # and flattening them is exactly what removes the structure.
            return self.trunk.scores(obs)

        def _predict(self, observation, deterministic: bool = True) -> torch.Tensor:
            return self(observation).argmax(dim=1).reshape(-1)

    class StructuredDQNPolicy(DQNPolicy):
        def make_q_net(self):
            return StructuredQNetwork(**self._update_features_extractor(
                self.net_args, features_extractor=None)).to(self.device)

    return StructuredDQNPolicy


def structured_actor_critic_policy(masked: bool = False):
    """`ActorCriticPolicy` with the hierarchy as its actor and critic.

    The action distribution is built from the trunk's scores, which already
    carry -inf on padded candidates -- so an unavailable phase takes no
    probability whether or not the algorithm can mask.
    """
    if masked:
        from sb3_contrib.common.maskable.policies import (
            MaskableActorCriticPolicy as Base,
        )
    else:
        from stable_baselines3.common.policies import ActorCriticPolicy as Base

    class StructuredActorCritic(Base):
        def _build(self, lr_schedule) -> None:
            self.trunk = StructuredTrunk()
            self.action_net = nn.Identity()
            self.value_net = nn.Identity()
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

        def _build_mlp_extractor(self) -> None:
            # The trunk is the extractor; SB3's two-tower MLP would sit on a
            # flattened observation and undo the point of it.
            self.mlp_extractor = nn.Identity()

        def extract_features(self, obs, features_extractor=None):
            return obs

        def _distribution(self, obs):
            return self.action_dist.proba_distribution(
                action_logits=self.trunk.scores(obs)
            )

        def forward(self, obs, deterministic: bool = False, **kwargs):
            distribution = self._distribution(obs)
            if kwargs.get("action_masks") is not None:
                distribution.apply_masking(kwargs["action_masks"])
            actions = distribution.get_actions(deterministic=deterministic)
            return actions, self.trunk.value(obs), distribution.log_prob(actions)

        def evaluate_actions(self, obs, actions, action_masks=None):
            distribution = self._distribution(obs)
            if action_masks is not None:
                distribution.apply_masking(action_masks)
            return (self.trunk.value(obs), distribution.log_prob(actions),
                    distribution.entropy())

        def get_distribution(self, obs, action_masks=None):
            distribution = self._distribution(obs)
            if action_masks is not None:
                distribution.apply_masking(action_masks)
            return distribution

        def predict_values(self, obs):
            return self.trunk.value(obs)

        def _predict(self, observation, deterministic: bool = False, **kwargs):
            distribution = self._distribution(observation)
            if kwargs.get("action_masks") is not None:
                distribution.apply_masking(kwargs["action_masks"])
            return distribution.get_actions(deterministic=deterministic)

    return StructuredActorCritic
