'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Where a transition goes between being seen and being learned from.

Off-policy is the whole reason this is affordable -- a SUMO episode costs seconds
and PPO would throw each one away after a few epochs -- so the buffer is not
incidental to the method, it is the method's budget.

Two things here are shaped by that. Junction structure is stored once per
(junction, plan) and referenced by index, because at ~5 KB a transition it would
otherwise be four fifths of the buffer describing wiring that never changes. And
n-step returns are accumulated on the way in rather than reconstructed on the way
out, so a sample is a sample and the buffer holds no ordering assumptions.

Nothing in this file knows what SAC is.
'''

from __future__ import annotations

import numpy as np
import torch

# Everything about a junction that does not change during an episode. Stored once
# per (junction, plan) and referenced by index: at ~5 KB a transition it would
# otherwise be four fifths of the replay buffer, describing wiring that is the
# same in every row.
STRUCTURE_KEYS = (
    "lane_valid", "movement_in_index", "movement_in_weight",
    "movement_out_index", "movement_out_weight", "movement_valid",
    "phase_members", "phase_member_valid", "phase_valid",
)


class NStepAccumulator:
    """Turns one environment's single steps into n-step transitions.

    One per environment, because an n-step window must never span two episodes —
    and with auto-reset the boundary is invisible in the observation stream. On
    an episode end the whole queue is flushed, each remaining start with its own
    shorter horizon, so the last few decisions of an episode are not discarded.

    The bootstrap discount travels with the transition rather than being assumed:
    at an episode end a window may be shorter than n, and using gamma^n there
    would discount a value that is only m steps away.
    """

    def __init__(self, n: int, gamma: float):
        from collections import deque

        self.n, self.gamma = int(n), float(gamma)
        self.pending: deque = deque()

    def push(self, observation, action, reward, next_observation, terminal, done):
        """-> the n-step transitions this step completed, possibly none."""
        self.pending.append((observation, action, float(reward)))
        if done:
            emitted = self._drain(len(self.pending), next_observation, terminal)
            self.pending.clear()
            return emitted
        if len(self.pending) >= self.n:
            emitted = self._drain(1, next_observation, False)
            self.pending.popleft()
            return emitted
        return []

    def _drain(self, count: int, final_observation, terminal: bool) -> list:
        entries = list(self.pending)
        out = []
        for start in range(count):
            total, discount = 0.0, 1.0
            for _, _, reward in entries[start:]:
                total += discount * reward
                discount *= self.gamma
            observation, action, _ = entries[start]
            # `discount` is now gamma ** (steps in this window) — the factor the
            # bootstrapped value must be multiplied by.
            out.append((observation, action, total, final_observation,
                        terminal, discount))
        return out


class ReplayBuffer:
    """Transitions, with the junction wiring factored out.

    `lane_state` is kept in float16: it holds queue counts, an occupancy fraction
    and a flag, none of which carry seven significant digits, and it is otherwise
    four fifths of the memory.
    """

    def __init__(self, capacity: int, window: int, max_lanes: int, lane_dim: int):
        self.capacity = int(capacity)
        self.size, self.cursor = 0, 0
        shape = (self.capacity, window, max_lanes, lane_dim)
        self.lane_state = np.zeros(shape, dtype=np.float16)
        self.next_lane_state = np.zeros(shape, dtype=np.float16)
        self.current_phase = np.zeros(self.capacity, dtype=np.int64)
        self.next_current_phase = np.zeros(self.capacity, dtype=np.int64)
        self.time_in_phase = np.zeros(self.capacity, dtype=np.float32)
        self.next_time_in_phase = np.zeros(self.capacity, dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        # True only when the network really drained. A horizon cut leaves traffic
        # on the road, and its value has to be bootstrapped rather than zeroed.
        self.terminal = np.zeros(self.capacity, dtype=np.float32)
        # gamma ** (steps in this transition's window). Not a constant: a window
        # cut short by the end of an episode bootstraps from closer in.
        self.discount = np.zeros(self.capacity, dtype=np.float32)
        self.structure_id = np.zeros(self.capacity, dtype=np.int64)
        self._structure_index: dict = {}
        self._structures: list = []
        self._stacked: dict | None = None

    def structure_slot(self, key, observation: dict) -> int:
        if key not in self._structure_index:
            self._structure_index[key] = len(self._structures)
            self._structures.append({k: observation[k] for k in STRUCTURE_KEYS})
            self._stacked = None
        return self._structure_index[key]

    def stack_structures(self) -> dict:
        """Cached: rebuilt only when a junction the buffer has not seen arrives."""
        if self._stacked is None:
            self._stacked = {k: np.stack([s[k] for s in self._structures])
                             for k in STRUCTURE_KEYS}
        return self._stacked

    def add(self, key, observation, action, reward, next_observation, terminal,
            discount) -> None:
        i = self.cursor
        self.lane_state[i] = observation["lane_state"]
        self.next_lane_state[i] = next_observation["lane_state"]
        self.current_phase[i] = observation["current_phase"]
        self.next_current_phase[i] = next_observation["current_phase"]
        self.time_in_phase[i] = observation["time_in_phase"]
        self.next_time_in_phase[i] = next_observation["time_in_phase"]
        self.action[i] = action
        self.reward[i] = reward
        self.terminal[i] = float(terminal)
        self.discount[i] = float(discount)
        self.structure_id[i] = self.structure_slot(key, observation)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device) -> dict:
        index = np.random.randint(0, self.size, size=batch_size)
        structures = self.stack_structures()
        ids = self.structure_id[index]

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(array, dtype=dtype, device=device)

        batch = {k: tensor(structures[k][ids],
                           torch.int64 if structures[k].dtype == np.int64 else torch.float32)
                 for k in STRUCTURE_KEYS}
        batch.update(
            lane_state=tensor(self.lane_state[index].astype(np.float32)),
            next_lane_state=tensor(self.next_lane_state[index].astype(np.float32)),
            current_phase=tensor(self.current_phase[index], torch.int64),
            next_current_phase=tensor(self.next_current_phase[index], torch.int64),
            time_in_phase=tensor(self.time_in_phase[index]),
            next_time_in_phase=tensor(self.next_time_in_phase[index]),
            action=tensor(self.action[index], torch.int64),
            reward=tensor(self.reward[index]),
            terminal=tensor(self.terminal[index]),
            discount=tensor(self.discount[index]),
        )
        return batch


def to_batch(observations: list[dict], device) -> dict:
    """Stack environment observations into what `TeacherNet` consumes."""
    batch = {}
    for key in STRUCTURE_KEYS:
        stacked = np.stack([o[key] for o in observations])
        dtype = torch.int64 if stacked.dtype == np.int64 else torch.float32
        batch[key] = torch.as_tensor(stacked, dtype=dtype, device=device)
    batch["lane_state"] = torch.as_tensor(
        np.stack([o["lane_state"] for o in observations]), dtype=torch.float32, device=device
    )
    batch["current_phase"] = torch.as_tensor(
        [o["current_phase"] for o in observations], dtype=torch.int64, device=device
    )
    batch["time_in_phase"] = torch.as_tensor(
        [o["time_in_phase"] for o in observations], dtype=torch.float32, device=device
    )
    return batch


def next_batch(batch: dict) -> dict:
    """The same wiring, the next state. The junction did not change mid-episode."""
    return {**batch,
            "lane_state": batch["next_lane_state"],
            "current_phase": batch["next_current_phase"],
            "time_in_phase": batch["next_time_in_phase"]}
