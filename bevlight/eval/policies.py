'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: One entry point from a name to something that can drive a junction.

A results table is only a comparison if every row was produced the same way.
Before this, three kinds of policy reached the metrics by three routes: the
rule-based controllers through `run_episode`, a behaviour-cloned checkpoint
through `closed_loop.PolicyController`, and a reinforcement-learning teacher
through its own rollout loop inside `rl/sac.py` -- which is why the teacher has
never appeared in a table beside max-pressure. It was not that the numbers
disagreed; it was that nothing could put them in the same column.

Everything here returns a `Controller`, so `run_episode` and `eval.compare`
score all three through the path the reported results were measured on.

    max_pressure            a rule-based baseline (any name in CONTROLLERS)
    fixed_time:30           with its argument
    checkpoint:<path>       a behaviour-cloned BEVLight, reading pixels
    teacher:<path>          a privileged SAC teacher, reading the same window
                            as numbers

Each declares its `obs_spec`, so a comparison that mixes observation scopes can
be caught rather than published.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from ..expert.base import BaseController


class TeacherController(BaseController):
    """A trained SAC teacher driving a junction through the controller interface.

    The teacher reads per-lane numbers over the same rolling window the vision
    student reads pixels over -- a teacher that saw one second where the student
    sees five would be a different agent solving a different problem, and the gap
    between them would be read as a distillation loss it is not. So the window is
    kept here, exactly as `closed_loop.PolicyController` keeps its feature window.

    The numbers come from the same `ObservationExtractor` output the expert and
    the labels use, clipped to the BEV window. Nothing the image cannot carry
    reaches this: no vehicle identities, no accumulated waiting, nothing past the
    window. A teacher acting on those would be teaching a lesson the student has
    no way to learn.
    """

    name = "teacher"

    def __init__(self, checkpoint, junction: str, plan: str, window: int = 5,
                 device: str | None = None, model=None):
        super().__init__()
        import torch

        from ..data.collate import junction_structure
        from ..scenario.lane_mask import load_lane_mask

        self.checkpoint = Path(checkpoint)
        self.junction = junction
        self.plan_name = plan
        self.window = window
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.mask = load_lane_mask(junction, plan)
        self.structure = junction_structure(self.mask)
        self.lane_order = self.structure["lane_order"]

        self.model = model if model is not None else self._load(self.checkpoint)
        self.model.to(self.device).eval()

        self.states: deque = deque(maxlen=window)

    @staticmethod
    def _load(checkpoint: Path):
        """The actor out of an SAC agent's state dict.

        A run is trained with `torch.compile`, which saves every parameter under
        an `_orig_mod.` prefix. The prefix belongs to the compilation, not to the
        model, so it is stripped rather than reproduced -- loading into a
        compiled wrapper here would pay a warm-up cost per episode to run a few
        hundred forward passes.
        """
        import torch

        from ..model.teacher import TeacherNet, teacher_config

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = state["actor"] if "actor" in state else state
        weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}

        config = state.get("config", {})
        net = TeacherNet(teacher_config(
            config.get("model_dim", 128), config.get("embed_dim", 64)
        ))
        net.load_state_dict(weights)
        return net

    @property
    def obs_spec(self):
        """Window scope, numbers rather than pixels. Privileged only in cost."""
        from ..env.obs_spec import ObsMode, ObsScope, ObsSpec

        return ObsSpec(scope=ObsScope.WINDOW, mode=ObsMode.STATE,
                       frames=self.window)

    def reset(self, plan) -> None:
        super().reset(plan)
        self.states.clear()

    def _lane_numbers(self, obs) -> np.ndarray:
        """This second's per-lane state, in the exact encoding the env produces.

        Three fields, and the third is `queue_valid` -- 0 when the queue reached
        the edge of the window and the count is a lower bound rather than a
        reading. Not `queue_saturated`, which is its inverse: the teacher was
        trained against `JunctionEnv._lane_targets`, so this has to be that
        function and not a plausible-looking restatement of it.
        """
        # Padded to the wiring's lane dimension, not to the junction's own lane
        # count: `lane_valid` is MAX_LANES wide and the two are multiplied.
        width = self.structure["lane_valid"].shape[0]
        lanes = np.zeros((width, 3), dtype=np.float32)
        view = getattr(obs, "lane_view", obs)
        for i, lane_id in enumerate(self.lane_order[:width]):
            state = view.lanes[lane_id]
            lanes[i] = (float(state.queued), float(state.occupancy),
                        0.0 if state.queue_saturated else 1.0)
        return lanes

    def observe_state(self, second: int, obs) -> None:
        """One entry per simulated second, as the environment pushes it.

        `act` fires only at decision points, so building the window there would
        cover five *decisions* -- fifty seconds -- and the policy would be
        reading a different signal than the one it was trained on.
        """
        self.states.append(self._lane_numbers(obs))

    def _window(self) -> np.ndarray:
        """Repeat the earliest entry until the window is full, as the env does."""
        values = list(self.states)
        padded = [values[0]] * (self.window - len(values)) + values
        return np.stack(padded)

    def act(self, obs, plan) -> int:
        import torch

        if not self.states:      # no per-second hook ran; act on this instant
            self.states.append(self._lane_numbers(obs))
        batch = {
            k: torch.as_tensor(np.stack([v]), device=self.device)
            for k, v in self.structure.items()
            if k not in ("lane_order", "phase_order")
        }
        batch["lane_state"] = torch.as_tensor(
            np.stack([self._window()]), dtype=torch.float32, device=self.device
        )
        batch["current_phase"] = torch.as_tensor(
            [plan.phases.index(obs.phase_index)], dtype=torch.int64, device=self.device
        )
        batch["time_in_phase"] = torch.as_tensor(
            [float(obs.time_in_phase)], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            scores = self.model(batch)
            valid = batch["phase_valid"].bool()
            position = int(scores.masked_fill(~valid, -1e9).argmax(dim=-1)[0])
        self.last_action = plan.phases[position]
        return self.last_action


def build_policy(spec: str, junction: str | None = None, plan: str | None = None,
                 **kwargs):
    """A name -> a `Controller`. The one place that mapping lives.

    `checkpoint:` and `teacher:` need the junction and plan, because both load
    that junction's lane mask to know the wiring they are scoring on. A
    rule-based controller does not: it is handed the plan at `reset`.
    """
    kind, _, argument = spec.partition(":")

    if kind == "checkpoint":
        from .closed_loop import PolicyController

        if junction is None or plan is None:
            raise ValueError("checkpoint: policies need junction= and plan=")
        return PolicyController(argument, junction, plan, **kwargs)

    if kind == "teacher":
        if junction is None or plan is None:
            raise ValueError("teacher: policies need junction= and plan=")
        return TeacherController(argument, junction, plan, **kwargs)

    from .compare import build_controller

    return build_controller(spec)
