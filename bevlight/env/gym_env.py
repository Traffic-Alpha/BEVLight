'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: The junction as a step-based environment, for RL and for evaluation alike.

`episode.py` drives a controller through a loop it owns; this exposes the same
loop body so a learner can own it instead. They are deliberately the same world:
the environment, the observation extractor, the renderer and the metrics are the
objects `run_episode` uses, not re-implementations of them. `tests/test_gym_env.py`
requires a rollout here to reproduce `run_episode` metric for metric under the
same controller — if that ever fails, a policy is being trained against something
other than what it will be scored on.

One step is one *decision*, not one simulated second. tshub only grants
`can_perform_action` every `delta_time` seconds, so the seconds in between are
advanced internally, holding the chosen phase, and only the frames a decision
window actually reads are rendered.

The observation is what the model consumes — pooled per-lane vectors, padded, with
the junction's wiring and the three masks. Whether the backbone runs inside the
environment or in the learner is a deployment choice, not a semantic one: pass an
extractor to pool here, or leave it out to receive the raw frames and pool a whole
batch of environments at once. The backbone is frozen either way, so the numbers
are identical; only the throughput differs.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from collections import deque

import numpy as np

from .render import image_modality, make_panda_renderer, render_sensor_config
from .sumo import DECISION_INTERVAL_S, YELLOW_TIME_S, build_environment


class JunctionEnv:
    """One (junction, plan, demand) as a `reset()` / `step(action)` environment.

    Plain Python rather than a `gymnasium.Env` subclass, so the library is not a
    hard dependency of the project; `wrapper.py` puts the gymnasium face on it.
    """

    def __init__(
        self,
        junction: str,
        plan: str,
        demand: str,
        seed: int = 7,
        window: int = 5,
        num_seconds: int | None = None,
        render: bool = True,
        extractor=None,
        panda_sky: str = "day",
        panda_preset: str = "auto",
        panda_backend: str = "pandagl",
        idle_timeout: int = 0,
        allow_any_scenario: bool = False,
        log_level: str = "ERROR",
        reward: str = "visible_queue",
        observe: str = "window",
        obs_spec=None,
    ):
        from ..collect.observation import ObservationExtractor
        from ..expert.base import SignalPlan
        from ..scenario.lane_mask import load_lane_mask
        from ..scenario.loader import load_junction_config

        if not allow_any_scenario:
            require_trainable(junction, plan, demand)

        self.junction, self.plan, self.demand, self.seed = junction, plan, demand, seed
        self.render = render
        self.extractor = extractor
        # Routes stop departing around 600 s of a 1000 s episode, so more than a
        # third of the decisions are taken on an empty network: at YMT the tail
        # carries 3 units of queue out of 5942, for 38% of the rendering cost.
        # Ending once the network has actually drained skips the dead time while
        # keeping the jam-clearing phase, which is real control work. Off by
        # default, because evaluation must keep the full fixed horizon.
        self.idle_timeout = idle_timeout
        self._idle = 0
        self._seen_traffic = False
        # What the reward counts, and why it may read more than the policy
        # does, is argued in `env/rewards.py` -- including why
        # `probe_constant_phase` is a diagnostic and never a control objective.
        from .rewards import REWARDS

        if reward not in REWARDS:
            raise ValueError(
                f"Unknown reward '{reward}'. Available: {sorted(REWARDS)}"
            )
        self.reward_kind = reward
        self.panda = dict(sky=panda_sky, preset=panda_preset, backend=panda_backend)

        # tshub logs one INFO line per simulated second. `run_episode` quiets it;
        # this did not, so anything driving the environment directly — a learner
        # above all — buried its own output under a million lines. It costs no
        # measurable time (14.25 s vs 14.35 s on a 1000 s episode); it costs
        # legibility, which is what a long run is debugged with.
        self.log_level = log_level

        self.env_name = f"{plan}_{demand}"
        self.mask = load_lane_mask(junction, plan)
        self.signal_plan = SignalPlan.from_lane_mask(self.mask)
        self.observer = ObservationExtractor(self.mask, self.mask.tls_id)
        # What the *policy* reads, as a published value -- see `env/obs_spec.py`
        # for what each scope means and which of them is deployable.
        #
        # `state_observer` is deliberately a second extractor rather than a wider
        # first one. The reward, the control metrics and every rule-based
        # baseline read `self.observer`, so widening that one would change the
        # reward and turn max-pressure into the stronger full-lane variant at the
        # same time -- three changes at once, and a comparison that attributes
        # nothing.
        self.obs_spec = self._resolve_spec(obs_spec, observe, window, render, extractor)
        self.observe = self.obs_spec.scope.value
        self.window = self.obs_spec.frames
        self.state_observer = (
            self.observer if self.obs_spec.deployable
            else ObservationExtractor(self.mask, self.mask.tls_id, full_lane=True)
        )
        self.horizon = num_seconds or int(
            load_junction_config(junction, self.env_name)["num_seconds"]
        )

        self._env = None
        self._renderer = None
        self.frames: deque = deque(maxlen=self.window)
        # The same `window` seconds the frames cover, as per-lane numbers. A
        # structured-state policy that saw one second where the vision model sees
        # five would be a different agent solving a different problem, and the gap
        # between them would be read as a distillation loss it is not.
        self.lane_states: deque = deque(maxlen=self.window)
        self.second = 0
        self.phase_started_at = 0.0
        self.current_phase = self.signal_plan.phases[0]
        self.metrics = None
        self.states = None
        self.decision_interval = DECISION_INTERVAL_S
        self._last_decision_at = None

    # ------------------------------------------------------------------ setup

    def _open(self):
        from ..eval.metrics import EpisodeMetrics
        from ..paths import LOG_ROOT
        from tshub.utils.init_log import set_logger

        set_logger(str(LOG_ROOT / self.junction), terminal_log_level=self.log_level)
        self.close()
        self._env, cfg = build_environment(
            self.junction, self.env_name, self.seed, self.horizon
        )
        self.tls_id = cfg["tls_id"]
        self.states = self._env.reset()

        self.metrics = EpisodeMetrics()
        self.metrics.incoming_lanes = set(self.mask.incoming_lane_ids())
        # Set here rather than only in reset(), so opening an environment never
        # leaves it half-built for anything that steps it directly.
        self._idle, self._seen_traffic, self.drained = 0, False, False
        self._last_decision_at = None
        self._reset_interval()

        if self.render:
            self._renderer, _ = make_panda_renderer(
                junction=self.junction, plan=self.plan, states=self.states,
                tls_id=self.tls_id, preset=self.panda["preset"],
                backend=self.panda["backend"], sky=self.panda["sky"],
            )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.destroy()
            self._renderer = None
        if self._env is not None:
            try:
                self._env._close_simulation()
            except SystemExit:
                pass
            self._env = None

    # ------------------------------------------------------------- the loop

    def _capture(self) -> None:
        """Render this second and push it onto the decision window."""
        if self._renderer is None:
            return
        from tshub.tshub_env3d.core import build_frame

        sensor = self._renderer.sync(build_frame(self.states))
        images = {
            image_modality(kind): image
            for cameras in sensor.values()
            for kind, image in cameras.items()
        }
        rgb = images.get("rgb")
        if rgb is None:
            return
        self.frames.append(
            self.extractor.encode_array(rgb, self.junction, self.plan)
            if self.extractor is not None else rgb
        )

    def _capture_state(self, obs) -> None:
        """Push this second's per-lane state onto the window.

        Pushed at the same point in the second that `_capture` renders a frame,
        so the two windows cover the same seconds by construction rather than by
        two rules that agree today.
        """
        self._ensure_structure()
        targets = self._lane_targets(obs)
        self.lane_states.append(
            np.stack([targets["queue_target"], targets["occupancy_target"],
                      targets["queue_valid"]], axis=-1)
        )

    def _window(self, values: list) -> np.ndarray:
        """Stack a window, repeating the earliest entry until it is full.

        Zero-padding would tell the policy the junction was empty before the
        episode began; the first decision of an episode is a real decision.
        """
        padded = [values[0]] * (self.window - len(values)) + list(values)
        return np.stack(padded)

    def _observation(self, obs) -> dict:
        """The window, the wiring and the current phase — one training sample."""
        self._ensure_structure()
        base = {
            **{k: v for k, v in self._structure.items()
               if k not in ("lane_order", "phase_order")},
            "current_phase": self.signal_plan.phases.index(obs.phase_index),
            "time_in_phase": float(obs.time_in_phase),
            # The grounding targets, free from the simulator at every step. Keeping
            # the auxiliary losses alive during RL is what stops the trunk drifting
            # into a representation that no longer reads the junction.
            **self._lane_targets(obs),
            # The window a structured-state policy acts on. Present in both modes:
            # the vision model ignores it, the teacher consumes it, and having one
            # environment produce both is what makes them the same world.
            "lane_state": self._window(list(self.lane_states) or [
                np.zeros((self._structure["lane_valid"].shape[0], 3), dtype=np.float32)
            ]),
        }
        # No renderer means a structured-state environment — legitimate for
        # driving a rule-based controller, and for the equivalence test.
        if not self.frames:
            return base

        stacked = self._window(list(self.frames))
        if self.extractor is not None:
            lanes = self._structure["lane_valid"].shape[0]
            padded = np.zeros(
                (self.window, lanes, stacked.shape[-1]), dtype=np.float32
            )
            padded[:, : stacked.shape[1]] = stacked
            stacked = padded

        key = "lane_features" if self.extractor is not None else "frames"
        return {**base, key: stacked}

    @staticmethod
    def _resolve_spec(obs_spec, observe, window, render, extractor):
        """The published contract for what this environment shows a policy.

        Assembled from the older keyword arguments when one is not passed, so
        every existing caller keeps working and gets a spec anyway -- the point
        is that `env.obs_spec` is always there to hand to a policy, an
        evaluation or a run's config, not that callers must build one.

        Passing a spec and a contradicting keyword is refused rather than
        silently resolved: which of the two won would be invisible in the
        results, and the scope in particular is the difference between a
        deployable measurement and a control experiment.
        """
        from .obs_spec import ObsMode, ObsScope, ObsSpec

        mode = (ObsMode.STATE if not render
                else ObsMode.FEATURES if extractor is not None
                else ObsMode.FRAMES)
        if obs_spec is None:
            return ObsSpec(scope=ObsScope(observe), mode=mode, frames=window)

        spec = obs_spec if isinstance(obs_spec, ObsSpec) else ObsSpec(**obs_spec)
        conflicts = []
        if observe != "window" and ObsScope(observe) is not spec.scope:
            conflicts.append(f"observe={observe!r} vs spec.scope={spec.scope.value!r}")
        if window != 5 and window != spec.frames:
            conflicts.append(f"window={window} vs spec.frames={spec.frames}")
        if mode is not spec.mode:
            conflicts.append(
                f"render={render}/extractor={extractor is not None} implies "
                f"{mode.value!r} vs spec.mode={spec.mode.value!r}"
            )
        if conflicts:
            raise ValueError(
                "obs_spec contradicts the keyword arguments: " + "; ".join(conflicts)
            )
        return spec

    def _ensure_structure(self) -> None:
        from ..data.collate import junction_structure

        if not hasattr(self, "_structure"):
            self._structure = junction_structure(self.mask)

    def _lane_targets(self, obs) -> dict:
        """Per-lane queue and occupancy truth, padded like everything else."""
        obs = getattr(obs, "lane_view", obs)
        lanes = self._structure["lane_order"]
        width = self._structure["lane_valid"].shape[0]
        queue = np.zeros(width, dtype=np.float32)
        occupancy = np.zeros(width, dtype=np.float32)
        valid = np.zeros(width, dtype=np.float32)
        for i, lane_id in enumerate(lanes[:width]):
            state = obs.lanes[lane_id]
            queue[i] = state.queued
            occupancy[i] = state.occupancy
            # A queue at the edge of the image is a lower bound, not a reading —
            # the same exclusion the cached labels use.
            valid[i] = 0.0 if state.queue_saturated else 1.0
        return {"queue_target": queue, "occupancy_target": occupancy,
                "queue_valid": valid}

    def _look(self):
        """Structured state of the junction right now.

        Carries the policy's view with it rather than re-extracting it later, so
        the lane numbers a decision is taken on always belong to the same second
        as the phase and the clock it is taken at.
        """
        obs = self.observer(self.states)
        obs.time = float(self.second)
        obs.time_in_phase = float(self.second) - self.phase_started_at
        obs.lane_view = (obs if self.state_observer is self.observer
                         else self.state_observer(self.states))
        return obs

    def reset(self) -> tuple:
        """Open the simulation and run to the first second a decision is allowed."""
        self._open()
        self.frames.clear()
        self.lane_states.clear()
        self.second = 0
        self.phase_started_at = 0.0
        self.current_phase = self.signal_plan.phases[0]
        self._pending = None
        self._idle, self._seen_traffic, self.drained = 0, False, False
        self._last_decision_at = None
        self._reset_interval()
        observation, _, done, info = self._to_next_decision()
        return observation, 0.0, done, info

    def step(self, action: int) -> tuple:
        """Take `action` at the pending decision, then run to the next one.

        `action` is a position in this junction's candidate list, which is what
        the model scores — never a global phase id.
        """
        obs = self._pending
        if obs is None:
            raise RuntimeError("step() before reset(), or the episode is over")

        phase = self.signal_plan.phases[action]
        self.metrics.record_decision(obs.phase_index, phase)
        if phase != obs.phase_index:
            self.phase_started_at = float(self.second)
        self.current_phase = phase
        self._pending = None

        # This second's step carries the new action, exactly as the callback loop
        # does: look, act, step, record — in that order.
        if self._tick(obs, phase):
            return self._terminal(obs)
        return self._to_next_decision()

    def _tick(self, obs, phase: int) -> bool:
        """Advance one simulated second holding `phase`. True if the episode ended."""
        self.states, _, _, done = self._env.step(
            {"vehicle": {}, "tls": {self.tls_id: phase}}
        )
        self.metrics.update(float(self.second), self.states, obs)
        self._queue_integral += self._queue_cost()
        self._interval_seconds += 1

        # Drain detection, counted only after traffic has actually appeared so the
        # quiet seconds before the first departure do not end the episode.
        if self.states["vehicle"]:
            self._seen_traffic, self._idle = True, 0
        elif self._seen_traffic:
            self._idle += 1
        if self.idle_timeout and self._idle >= self.idle_timeout:
            self.drained = True
            return True
        return bool(done)

    def _to_next_decision(self) -> tuple:
        """Run until tshub grants an action, holding the current phase."""
        while self.second < self.horizon:
            self.second += 1
            obs = self._look()
            self._capture_state(obs)
            if self.render and self._is_in_window(obs):
                self._capture()
            if obs.can_act:
                self._pending = obs
                self._last_decision_at = float(self.second)
                reward = self._reward()
                self._reset_interval()
                return self._observation(obs), reward, False, self._info(obs)
            if self._tick(obs, self.current_phase):
                return self._terminal(obs)
        # The horizon ran out. The state has moved since the last look, so the
        # terminal observation is taken fresh rather than reusing a stale window.
        obs = self._look()
        self._capture_state(obs)
        return self._terminal(obs)

    def _terminal(self, obs) -> tuple:
        self.metrics.finalize(float(self.second))
        self._pending = None
        return self._observation(obs), self._reward(), True, self._info(obs)

    def _is_in_window(self, obs) -> bool:
        """Whether a decision window will read this second.

        Only the `window` seconds before a decision are ever looked at, and
        rendering is the entire cost of a step — but *which* seconds those are
        cannot be read off a modulo. A phase change adds yellow time, so the
        cadence drifts: decisions land at 1, 11, 21, ..., 51, 64, 77, 87. Under
        `second % interval` the decision at 64 would be handed frames 56-59 and
        64, an eight-second window with a hole in it, where the model was trained
        on five consecutive seconds.

        So the next decision is predicted from the last one actually seen. Until
        one has been, everything is rendered; if a decision arrives later than
        predicted the estimate falls behind and this stays true, which costs
        frames rather than correctness.
        """
        if not self.frames or self._last_decision_at is None:
            return True
        if obs.can_act:
            return True
        next_decision = self._last_decision_at + self.decision_interval
        return self.second > next_decision - self.window

    # ------------------------------------------------------------- reward

    def _queue_cost(self) -> float:
        """This second's cost under the configured reward.

        Built from `env.rewards`, which `bevlight.rl.preflight` also measures
        its candidates with -- so the reward the learner is paid and the reward
        the preflight ranks cannot drift apart.
        """
        from .rewards import REWARDS, RewardContext

        return REWARDS[self.reward_kind](RewardContext(
            visible=self.observer(self.states),
            states=self.states,
            incoming_lanes=self.metrics.incoming_lanes,
            current_phase=self.current_phase,
            first_phase=self.signal_plan.phases[0],
        ))

    def _reward(self) -> float:
        """Minus the mean queue held over the interval just elapsed.

        The *integral*, not the change. A difference in queue between the two ends
        of an interval telescopes away over an episode that starts and finishes
        empty — every controller scores zero and the reward says nothing about how
        long the queues persisted, which is the whole of control quality. This is
        what a reward preflight catches, and did — see docs/10-rl-posttraining.md.

        Per incoming lane, so a 28-lane junction and a 12-lane one produce
        gradients of the same scale rather than the larger junction dominating.
        """
        if not self._interval_seconds:
            return 0.0
        return -float(self._queue_integral / self._interval_seconds)

    def _reset_interval(self) -> None:
        self._queue_integral = 0.0
        self._interval_seconds = 0

    def _info(self, obs) -> dict:
        return {
            # A time-limit cut and a drained network are different endings, and
            # GAE must not confuse them: after draining the state really is worth
            # nothing, but at a horizon cut there is still traffic to clear, so
            # the value has to be bootstrapped rather than zeroed.
            "drained": bool(getattr(self, "drained", False)),
            "truncated": self.second >= self.horizon and not getattr(self, "drained", False),
            "second": self.second,
            "phase": obs.phase_index,
            "time_in_phase": obs.time_in_phase,
            "num_phases": self.signal_plan.num_phases,
            "junction": self.junction,
            "plan": self.plan,
            "demand": self.demand,
        }

    def summary(self) -> dict:
        """Control metrics for the episode so far — the same ones eval reports.

        Including the three fields `run_episode` appends. Without them a summary
        from here and a summary from there are not the same dictionary, and the
        difference shows up as a `KeyError` in whatever tries to tabulate both.
        """
        summary = self.metrics.summary()
        summary["steps_run"] = self.second
        summary["horizon_s"] = self.horizon
        summary["drained_at_s"] = self.second if getattr(self, "drained", False) else None
        return summary


def require_trainable(junction: str, plan: str, demand: str) -> None:
    """Refuse scenarios that RL must never touch.

    The held-out plans and the two unseen junctions carry the cross-plan and
    cross-structure results. An environment makes training on them *convenient*,
    which is exactly why crossing that line has to be an explicit act rather than
    an available default.
    """
    from ..scenario.selection import load_selection

    allowed = {(s.junction, s.plan, s.demand) for s in load_selection().split("train")}
    if (junction, plan, demand) not in allowed:
        raise SystemExit(
            f"{junction}/{plan}/{demand} is not in the train split. Training on it "
            f"would spend a test-set result. Pass allow_any_scenario=True only to "
            f"*evaluate* on it."
        )
