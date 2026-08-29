'''
@Author: WANG Maonan
@Date: 2026-08-25
@Description: The two renderers, wired to a junction's solved BEV camera.

Panda runs inside the simulation loop and is fast enough to drive control
(~0.2 s/frame); Blender runs offline from an exported manifest and is not
(~3.4 s/frame). Closed-loop evaluation therefore renders with Panda, which is
why `panda_day` is one of the appearance variants the model trains on.

Exporting a Blender frame involves no renderer at all — it is bookkeeping over
vehicle poses, which is exactly what lets simulation and rendering stay decoupled.
@LastEditTime: 2026-08-25
'''

from __future__ import annotations

from pathlib import Path

from .sumo import PANDA_VARIANT

def render_sensor_config(junction: str, plan: str, tls_id: str) -> dict:
    from ..scenario.bev_camera import bev_height, bev_ortho_size

    return {
        "tls": {
            tls_id: {
                "sensor_types": ["junction_bev_rgb", "junction_bev_seg"],
                "junction_bev_height": bev_height(junction, plan),
                # The window this junction's lane mask was solved for. Without
                # it every junction renders tshub's default 90 m while its mask
                # is rasterised for 112-157 m — a scale error, not a crop, that
                # draws every lane nearer the centre than it appears.
                "junction_bev_ortho_size": bev_ortho_size(junction, plan),
            }
        }
    }


def make_render_exporter(junction: str, plan: str, states: dict, tls_id: str, episode_dir: Path):
    """tshub's episode exporter, wired to this junction's solved BEV camera.

    No renderer is involved: exporting a frame is bookkeeping over vehicle poses,
    which is exactly why simulation can stay decoupled from rendering.
    """
    from ..scenario.bev_camera import bev_resolution
    from ..utils.paths import scene_assets_dir
    from tshub.tshub_env3d.core import build_tls_rigs
    from tshub.tshub_env3d.core.export import BlenderEpisodeExporter

    sensor_config = render_sensor_config(junction, plan, tls_id)
    exporter = BlenderEpisodeExporter(
        episode_dir=str(episode_dir),
        scenario_glb_dir=str(scene_assets_dir(junction)),
        sensor_config=sensor_config,
        resolution=bev_resolution(junction, plan),
        samples=8,
        style="day",
    )
    exporter.reset(tls_rigs=build_tls_rigs(states, sensor_config))
    return exporter


def image_modality(sensor_type: str) -> str:
    """Single-junction BEV datasets only need the pass name on disk."""
    if sensor_type.endswith("_rgb"):
        return "rgb"
    if sensor_type.endswith("_seg"):
        return "seg"
    if sensor_type.endswith("_depth"):
        return "depth"
    raise ValueError(f"Unknown image sensor type: {sensor_type}")


def save_panda_image(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if len(image.shape) == 2:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image[:, :, ::-1])


def make_panda_renderer(
    junction: str,
    plan: str,
    states: dict,
    tls_id: str,
    preset: str,
    backend: str,
    sky: str,
):
    """Create TSHub's Panda renderer for the live SUMO state stream."""
    import os

    from ..scenario.bev_camera import apply_junction_render_preset
    from ..utils.paths import scene_assets_dir
    from tshub.tshub_env3d.core import SceneStatic, build_tls_rigs
    from tshub.tshub_env3d.renderers import create_renderer

    sensor_config = render_sensor_config(junction, plan, tls_id)
    render_preset = apply_junction_render_preset(junction, plan) if preset == "auto" else preset
    tls_rigs = build_tls_rigs(states, sensor_config)
    renderer = create_renderer(
        "panda",
        simid=f"bevlight-collect-{junction}-{os.getpid()}",
        scenario_glb_dir=str(scene_assets_dir(junction)),
        sensor_config=sensor_config,
        preset=render_preset,
        resolution=0.5,
        render_mode="offscreen",
        rendering_backend=backend,
        sky=sky,
    )
    renderer.reset(
        SceneStatic(
            scenario_glb_dir=str(scene_assets_dir(junction)),
            sensor_config=sensor_config,
            tls_rigs=tls_rigs,
        )
    )
    return renderer, render_preset
