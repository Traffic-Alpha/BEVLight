'''Static junction facts, and the builders that produce them.

Public -- read side, cheap, no SUMO:
    loader          scenarios/<junction>/config.py -> SUMO paths + signal params
    bev_camera      orthographic BEV camera geometry, world <-> pixel
    lane_mask       the lane-id mask that links BEV pixels back to SUMO lanes
    selection       configs/scenario_selection.json -> train/test splits
    layout          episode/sample directory naming, shared with collect

_internal/ -- only this package:
    bev_reference   one BEV frame per camera window, kept for overlay checks

cli/ -- build side, runs SUMO / Blender, one-off per junction:
    build_networks      OSM -> SUMO network and polygons
    build_static_scene  SUMO network -> GLB assets + scene.blend
    build_lane_masks    SUMO network -> lane_mask.png + lane_mask.json,
                        plus the mask-over-render overlays for every junction
    render_reference    the BEV frames those overlays are drawn over
    lane_views          a BEV frame + a mask -> one image per lane (inspection)
'''
from .lane_mask import LaneMask, load_lane_mask
from .loader import (
    AVAILABLE_JUNCTIONS,
    load_event_config,
    load_junction_config,
)
from .selection import Scenario, load_selection

__all__ = [
    "AVAILABLE_JUNCTIONS",
    "LaneMask",
    "Scenario",
    "load_event_config",
    "load_junction_config",
    "load_lane_mask",
    "load_selection",
]
