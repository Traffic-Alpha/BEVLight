'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: One episode under one controller — the loop everything drives.

Expert collection, baseline comparison and closed-loop policy evaluation are the
same run with a different controller and a different set of outputs, so they are
one function rather than three. That is not tidiness: control metrics are only
comparable between a policy and max-pressure if the two were measured by the
same code on the same clock.

A controller is anything with `act(obs, plan)`. Two optional hooks feed a
controller whose input is a *window* rather than an instant, because `act` is
called only at decision points and a window built there would cover decisions
instead of seconds:

    observe(frame_index, images)  each rendered second, for a pixel policy
    observe_state(second, obs)    each simulated second, for a policy reading
                                  per-lane numbers

A pixel policy may also declare `needs_frame(second)` to skip the seconds no
decision window covers — rendering is the entire cost of an episode with a
renderer attached.

With `episode_dir` the run is also recorded to disk; without it nothing is
written and Panda renders straight into the controller's hands.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .render import (image_modality, make_panda_renderer, make_render_exporter,
                     save_panda_image)
from .sumo import (DECISION_INTERVAL_S, IDLE_TIMEOUT_S, PANDA_VARIANT,
                   YELLOW_TIME_S, build_environment, mask_dir_of)


@dataclass
class EpisodeResult:
    """What one rollout produced."""

    junction: str
    plan: str
    demand: str
    seed: int
    controller: str
    steps: int
    metrics: dict
    decisions: list = field(default_factory=list)
    lane_states: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.junction}__{self.plan}__{self.demand}__seed{self.seed}__{self.controller}"



def run_episode(
    junction: str,
    plan: str,
    demand: str,
    controller,
    seed: int = 7,
    num_seconds: int | None = None,
    record_lane_states: bool = False,
    log_dir: Path | None = None,
    episode_dir: Path | None = None,
    idle_timeout: int = IDLE_TIMEOUT_S,
    render_panda_images: bool = False,
    panda_variant: str = PANDA_VARIANT,
    panda_clean: bool = False,
    panda_preset: str = "auto",
    panda_backend: str = "pandagl",
    panda_sky: str = "day",
    progress_interval: int = 100,
) -> EpisodeResult:
    """Drive one episode and return its control metrics (and optionally the trace).

    With `episode_dir`, also writes the trajectory: per-lane truth, expert
    decisions, and tshub render frames, all indexed by the same frame numbers.
    """
    from ..collect.episode_schema import LANE_FIELDS, Episode, EpisodeStatic
    from ..collect.observation import ObservationExtractor
    from ..eval.metrics import EpisodeMetrics
    from ..expert.base import SignalPlan
    from ..scenario.lane_mask import load_lane_mask
    from ..utils.paths import LOG_ROOT, PROJECT_ROOT, episode_images_dir
    from tshub.tshub_env3d.core import build_frame
    from tshub.utils.init_log import set_logger

    set_logger(str(log_dir or (LOG_ROOT / junction)), terminal_log_level="ERROR")

    env_name = f"{plan}_{demand}"
    mask = load_lane_mask(junction, plan)
    signal_plan = SignalPlan.from_lane_mask(mask)
    extractor = ObservationExtractor(mask, mask.tls_id)

    from ..scenario.loader import load_junction_config

    horizon = num_seconds or int(load_junction_config(junction, env_name)["num_seconds"])
    env, cfg = build_environment(junction, env_name, seed, horizon)
    tls_id = cfg["tls_id"]

    metrics = EpisodeMetrics()
    # Full-lane queue for the control metrics, not just the part the camera sees.
    metrics.incoming_lanes = set(mask.incoming_lane_ids())
    controller.reset(signal_plan)

    decisions: list = []
    lane_states: list = []
    lane_order = [record["lane_id"] for record in mask.lanes]
    truth: dict[str, list] = {field_name: [] for field_name in LANE_FIELDS}
    phase_started_at = 0.0
    action = signal_plan.phases[0]
    exporter = None
    panda_renderer = None
    panda_out_dir = episode_images_dir(episode_dir, panda_variant) if episode_dir else None
    panda_images = 0
    panda_preset_used = None
    frame_index = -1
    idle_seconds = 0
    traffic_started = False
    stopped_early_at = None

    try:
        print(
            f"[rollout] {junction}/{plan}_{demand} seed={seed}: "
            f"reset SUMO horizon={horizon}s",
            flush=True,
        )
        states = env.reset()
        print(
            f"[rollout] {junction}/{plan}_{demand} seed={seed}: "
            f"SUMO ready vehicles={len(states.get('vehicle', {}))}",
            flush=True,
        )
        if episode_dir is not None:
            if render_panda_images and panda_clean and panda_out_dir and panda_out_dir.exists():
                import shutil

                shutil.rmtree(panda_out_dir)
                print(f"[panda] cleaned {panda_out_dir}", flush=True)
            print(f"[export] writing frames/manifest under {episode_dir}", flush=True)
            exporter = make_render_exporter(junction, plan, states, tls_id, episode_dir)
        # Panda does not need an episode directory. Closed-loop control renders
        # for the controller's eyes only and writes nothing to disk.
        if render_panda_images:
            print(
                f"[panda] initializing renderer preset={panda_preset} "
                f"backend={panda_backend} sky={panda_sky}",
                flush=True,
            )
            panda_renderer, panda_preset_used = make_panda_renderer(
                junction=junction,
                plan=plan,
                states=states,
                tls_id=tls_id,
                preset=panda_preset,
                backend=panda_backend,
                sky=panda_sky,
            )
            print(
                f"[panda] ready variant={panda_variant} preset={panda_preset_used} "
                f"out={panda_out_dir or 'in memory'}",
                flush=True,
            )
        for second in range(1, horizon + 1):
            obs = extractor(states)
            obs.time = float(second)
            obs.time_in_phase = float(second) - phase_started_at

            # One render frame per simulation second, exported before the action
            # so that frame N shows the state the decision at N was taken on.
            if exporter is not None or panda_renderer is not None:
                frame_index += 1
                # A pixel policy only reads the frames its window covers, so it
                # can decline the rest — half of them, at a 5-frame window and a
                # 10 s decision interval. Rendering is the whole cost here.
                wants_pixels = panda_renderer is not None and (
                    not hasattr(controller, "needs_frame")
                    or controller.needs_frame(second)
                )
                frame = build_frame(states) if (exporter is not None or wants_pixels) else None
                if exporter is not None:
                    exporter.add_frame(frame)
                if wants_pixels:
                    images = {}
                    sensor_data = panda_renderer.sync(frame)
                    for element_id, cameras in sensor_data.items():
                        for sensor_type, image in cameras.items():
                            modality = image_modality(sensor_type)
                            images[modality] = image
                            if panda_out_dir is not None:
                                save_panda_image(
                                    panda_out_dir / modality / f"{frame_index:04d}.png",
                                    image,
                                )
                                panda_images += 1
                    # `act` takes structured state only; a controller that reads
                    # pixels gets them here rather than by widening that signature
                    # for every rule-based baseline that will never use them.
                    if hasattr(controller, "observe"):
                        controller.observe(frame_index, images)
            # The structured-state mirror of `observe` above. A policy whose
            # input is a window of per-lane numbers needs one entry per
            # simulated second, not one per decision -- `act` is called only at
            # decision points, which would silently stretch a 5-second window
            # into a 50-second one and change what the policy is.
            if hasattr(controller, "observe_state"):
                controller.observe_state(second, obs)

            if exporter is not None:
                for field_name in LANE_FIELDS:
                    truth[field_name].append(
                        [getattr(obs.lanes[lane_id], field_name) for lane_id in lane_order]
                    )

            if obs.can_act:
                previous = obs.phase_index
                action = controller.act(obs, signal_plan)
                if action not in signal_plan.phases:
                    raise ValueError(
                        f"{controller.name} returned phase {action}, "
                        f"not one of {signal_plan.phases}"
                    )
                metrics.record_decision(previous, action)
                if action != previous:
                    phase_started_at = float(second)
                record = {
                    "t": second,
                    "frame_index": frame_index,
                    "phase": previous,
                    "time_in_phase": round(obs.time_in_phase, 1),
                    "action": action,
                    "num_phases": signal_plan.num_phases,
                }
                if hasattr(controller, "pressures"):
                    record["phase_pressure"] = {
                        str(phase): round(value, 3)
                        for phase, value in controller.pressures(obs, signal_plan).items()
                    }
                decisions.append(record)

            states, _, _, done = env.step({"vehicle": {}, "tls": {tls_id: action}})
            metrics.update(float(second), states, obs)

            # Drain detection. Only after traffic has actually appeared, so the
            # quiet seconds before the first departure do not end the episode.
            if states["vehicle"]:
                traffic_started = True
                idle_seconds = 0
            elif traffic_started:
                idle_seconds += 1
            if record_lane_states:
                lane_states.append(
                    {
                        "t": second,
                        "lanes": {
                            lane_id: {
                                "vehicles": st.vehicles,
                                "queued": st.queued,
                                "queue_m": st.queue_m,
                                "occupancy": st.occupancy,
                            }
                            for lane_id, st in obs.lanes.items()
                        },
                    }
                )
            if idle_timeout and traffic_started and idle_seconds >= idle_timeout:
                stopped_early_at = second
                break
            if done:
                break
            if progress_interval and second % progress_interval == 0:
                print(
                    f"[progress] {junction}/{plan}_{demand} seed={seed}: "
                    f"t={second}/{horizon} frames={frame_index + 1} "
                    f"panda_images={panda_images} decisions={len(decisions)} "
                    f"vehicles={len(states.get('vehicle', {}))}",
                    flush=True,
                )
    finally:
        if panda_renderer is not None:
            print(
                f"[panda] closing renderer images={panda_images} out={panda_out_dir}",
                flush=True,
            )
            panda_renderer.destroy()
        try:
            env._close_simulation()
        except SystemExit:
            pass

    steps_run = stopped_early_at or horizon
    metrics.finalize(float(steps_run))
    summary = metrics.summary()
    summary["steps_run"] = steps_run
    summary["horizon_s"] = horizon
    summary["drained_at_s"] = stopped_early_at
    if render_panda_images and panda_out_dir is not None:
        summary["panda_images"] = panda_images
        summary["panda_variant"] = panda_variant
        summary["panda_dir"] = str(panda_out_dir.relative_to(PROJECT_ROOT))
        summary["panda_preset"] = panda_preset_used or panda_preset

    if exporter is not None and episode_dir is not None:
        from ._internal.recorder import write_episode

        write_episode(
            episode_dir,
            exporter,
            junction=junction,
            plan=plan,
            demand=demand,
            seed=seed,
            controller_name=controller.name,
            tls_id=tls_id,
            env_name=env_name,
            steps_run=steps_run,
            decision_interval_s=DECISION_INTERVAL_S,
            yellow_time_s=YELLOW_TIME_S,
            mask=mask,
            lane_order=lane_order,
            signal_plan=signal_plan,
            truth=truth,
            decisions=decisions,
            summary=summary,
        )

    return EpisodeResult(
        junction=junction,
        plan=plan,
        demand=demand,
        seed=seed,
        controller=controller.name,
        steps=steps_run,
        metrics=summary,
        decisions=decisions,
        lane_states=lane_states,
    )
