'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: Write one episode's trajectory to disk.

Collection and closed-loop evaluation drive the same loop, but only collection
keeps the result: an episode directory holds the per-lane truth, the expert's
decisions and the Blender frame manifest, all indexed by the same frame numbers.
Evaluation renders for the policy's eyes and writes nothing.

Keeping this out of `env/episode.py` is what keeps the dependency pointing one
way. The loop knows how to run a junction; it does not know what an episode file
looks like, so `env` never has to import the schema that `collect` owns.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

import json
from pathlib import Path


def write_episode(
    episode_dir: Path,
    exporter,
    *,
    junction: str,
    plan: str,
    demand: str,
    seed: int,
    controller_name: str,
    tls_id: str,
    env_name: str,
    steps_run: int,
    decision_interval_s: int,
    yellow_time_s: int,
    mask,
    lane_order: list,
    signal_plan,
    truth: dict,
    decisions: list,
    summary: dict,
) -> Path:
    """Close the frame exporter and write `episode.json` beside its images."""
    from ...collect.episode_schema import Episode, EpisodeStatic
    from ...paths import PROJECT_ROOT
    from ..sumo import mask_dir_of

    # close() returns the manifest *path*; the frame list lives inside it.
    manifest = json.loads(Path(exporter.close()).read_text())
    Episode(
        static=EpisodeStatic(
            junction=junction,
            plan=plan,
            demand=demand,
            seed=seed,
            controller=controller_name,
            tls_id=tls_id,
            env=env_name,
            horizon_s=steps_run,
            decision_interval_s=decision_interval_s,
            yellow_time_s=yellow_time_s,
            camera=mask.camera,
            resolution=list(mask.resolution),
            lane_order=lane_order,
            lane_roles={r["lane_id"]: r["role"] for r in mask.lanes},
            visible_length_m={
                lane_id: round(value, 3)
                for lane_id, value in mask.visible_length_m().items()
            },
            signal_plan={
                "phases": list(signal_plan.phases),
                "phase2movements": {
                    str(k): list(v) for k, v in signal_plan.phase2movements.items()
                },
                "movement_in_lanes": {
                    k: list(v) for k, v in signal_plan.movement_in_lanes.items()
                },
                "movement_out_lanes": {
                    k: list(v) for k, v in signal_plan.movement_out_lanes.items()
                },
            },
            lane_mask_file=str(
                (mask_dir_of(junction) / "lane_mask.json").relative_to(PROJECT_ROOT)
            ),
        ),
        lane_truth=truth,
        decisions=decisions,
        metrics=summary,
        render_frames=manifest.get("frames", []),
    ).write(episode_dir)
    return episode_dir / "episode.json"
