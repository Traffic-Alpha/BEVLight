'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: The SUMO environment, and the timing that defines a decision.

`delta_time` is both the decision interval and the minimum green: tshub only
grants `can_perform_action` every `delta_time` seconds, and a phase change adds
`yellow_time` on top before the new green starts. Every controller in the project
— expert, baseline and learned policy alike — is bound by the same two numbers,
which is what makes their control metrics comparable.

The action space is `choose_next_phase` throughout: the controller names the
phase to run next, which is the space the network scores. tshub's own default is
`next_or_not`, so the action type is passed explicitly here rather than inherited.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from pathlib import Path

DECISION_INTERVAL_S = 10      # also the minimum green
YELLOW_TIME_S = 3
# The routes stop departing around 600s; `num_seconds` is 1000 only so a jam has
# room to clear, which leaves a long empty tail. Simulation is cheap, so the
# episode is still run in full and everything is recorded. Blender later drops
# much of the tail through a selected-frame manifest; Panda images are rendered
# for every collected frame.
IDLE_TIMEOUT_S = 0
PANDA_VARIANT = "panda_day"



def mask_dir_of(junction: str) -> Path:
    from ..paths import lane_mask_dir

    return lane_mask_dir(junction)



def build_environment(junction: str, env_name: str, seed: int, num_seconds: int,
                      delta_time: int = DECISION_INTERVAL_S,
                      yellow_time: int = YELLOW_TIME_S):
    """A SUMO environment wired for phase-choice control, vehicles on, no renderer."""
    from ..scenario.loader import load_junction_config
    from tshub.tshub_env.tshub_env import TshubEnvironment

    cfg = load_junction_config(junction, env_name)
    env = TshubEnvironment(
        sumo_cfg=cfg["sumo_cfg"],
        is_map_builder_initialized=False,
        is_vehicle_builder_initialized=True,
        is_aircraft_builder_initialized=False,
        is_traffic_light_builder_initialized=True,
        is_person_builder_initialized=False,
        tls_ids=[cfg["tls_id"]],
        # Explicit: tshub defaults to next_or_not, which cannot express
        # "skip an empty phase" and does not match the model's output space.
        tls_action_type="choose_next_phase",
        delta_time=delta_time,
        use_gui=False,
        is_libsumo=False,   # traci: one SUMO process per episode, no leaked global state
        num_seconds=num_seconds + delta_time + yellow_time,
        sumo_seed=str(seed),
    )
    return env, cfg
