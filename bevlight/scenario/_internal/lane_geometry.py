'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: What SUMO knows about a junction's lanes, read out once.

Building a lane mask needs three things SUMO will only tell you while a
simulation is running: which lanes a traffic light actually controls, which
phase each of them belongs to, and where each one is on the ground. So this
starts a headless SUMO, asks, and shuts it down -- the answer is static for a
(junction, plan), which is why it is worth paying for once and rasterising the
result rather than querying per frame.

Internal lanes -- the short connectors inside the junction box -- are included
deliberately. They are what a vehicle occupies while crossing, and a mask that
omits them has a hole in the middle of every intersection.
'''

from __future__ import annotations

from ...paths import LOG_ROOT


def read_tls_state(junction: str, env_name: str, seed: int = 7) -> tuple[dict, dict]:
    """Reset SUMO once and return (tls_info, bev_rig) for the junction.

    Only the traffic-light builder is initialized: no vehicles, no rendering.
    The BEV rig is computed by the same tshub helper the renderers use, so the
    mask camera center is identical to the one used at render time.
    """
    from tshub.tshub_env.tshub_env import TshubEnvironment
    from tshub.tshub_env3d.core import build_tls_rigs
    from tshub.utils.init_log import set_logger

    from ..bev_camera import bev_height
    from ..loader import load_junction_config

    cfg = load_junction_config(junction, env_name)
    tls_id = cfg["tls_id"]
    set_logger(str(LOG_ROOT / junction),
               terminal_log_level="ERROR")

    env = TshubEnvironment(
        sumo_cfg=cfg["sumo_cfg"],
        is_map_builder_initialized=False,
        is_vehicle_builder_initialized=False,
        is_aircraft_builder_initialized=False,
        is_traffic_light_builder_initialized=True,
        is_person_builder_initialized=False,
        tls_ids=[tls_id],
        use_gui=False,
        is_libsumo=False,  # traci: a separate SUMO process per junction, no global state to leak
        num_seconds=10,
        sumo_seed=str(seed),
    )
    try:
        states = env.reset()
        sensor_config = {
            "tls": {
                tls_id: {
                    "sensor_types": ["junction_bev_rgb"],
                    "junction_bev_height": bev_height(junction),
                }
            }
        }
        tls_rigs = build_tls_rigs(states, sensor_config)
        bev_rig = tls_rigs.get(f"{tls_id}_bev")
        if bev_rig is None:
            raise RuntimeError(f"{junction}: tshub did not produce a junction BEV rig")
        return states["tls"][tls_id], bev_rig
    finally:
        try:
            env._close_simulation()
        except SystemExit:
            pass


def lane_polygon(lane):
    """Lane center line + width -> world-space outline, or None when degenerate."""
    from shapely.geometry import LineString

    shape = [(float(p[0]), float(p[1])) for p in lane.getShape()]
    if len(shape) < 2:
        return None
    geom = LineString(shape).buffer(
        lane.getWidth() / 2.0, cap_style=2, join_style=2, mitre_limit=2.0
    )
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    coords = [(round(float(x), 3), round(float(y), 3)) for x, y, *_ in geom.exterior.coords]
    return coords if len(coords) >= 3 else None


def road_lane_ids(tls_info: dict, roads: list[str]) -> list[str]:
    """Lane ids of the given roads, as resolved by the tshub traffic-light builder."""
    return [lane_id for road in roads for lane_id in tls_info["roads_lanes"][road]]


def internal_lane_ids(net, in_lane_ids: list[str]) -> list[str]:
    """Internal lanes of this junction, reached from its incoming lanes.

    Following `via` connections keeps the walk inside the junction: starting from
    the outgoing lanes instead would wander into the neighbouring junctions.
    """
    internal: list[str] = []
    frontier = list(in_lane_ids)
    while frontier:
        for connection in net.getLane(frontier.pop()).getOutgoing():
            via = connection.getViaLaneID()
            if via and via not in internal:
                internal.append(via)
                frontier.append(via)
    return internal


def collect_lanes(net, tls_info: dict, include_internal: bool) -> list[dict]:
    """Build one record per junction lane, with its polygon and simulation role.

    Only the lanes of this traffic light's in/out roads are labeled: these are
    single-junction scenarios, and the rest of the network file is never in play.
    """
    in_lane_ids = road_lane_ids(tls_info, tls_info["in_roads"])
    out_lane_ids = road_lane_ids(tls_info, tls_info["out_roads"])
    roles = dict.fromkeys(in_lane_ids, "incoming")
    roles.update(dict.fromkeys(out_lane_ids, "outgoing"))
    if include_internal:
        roles.update(dict.fromkeys(internal_lane_ids(net, in_lane_ids), "internal"))

    lane_movements: dict[str, list[str]] = {}
    for movement_id, lane_ids in tls_info.get("movement_lane_ids", {}).items():
        for lane_id in lane_ids:
            lane_movements.setdefault(lane_id, []).append(movement_id)

    movement_phases: dict[str, list[int]] = {}
    for phase_index, movement_ids in tls_info.get("phase2movements", {}).items():
        for movement_id in movement_ids:
            movement_phases.setdefault(movement_id, []).append(int(phase_index))

    directions = tls_info.get("movement_directions", {})
    from_to = tls_info.get("fromEdge_toEdge", {})

    records = []
    for lane_id, role in sorted(roles.items()):
        lane = net.getLane(lane_id)
        polygon = lane_polygon(lane)
        if polygon is None:
            continue

        movements = sorted(lane_movements.get(lane_id, []))
        # `phases` is plan-specific: the same lane serves different phases under
        # `easy` and `normal`. It is recorded per plan by the caller.
        phases = sorted({p for m in movements for p in movement_phases.get(m, [])})
        records.append(
            {
                "lane_id": lane_id,
                "edge_id": lane.getEdge().getID(),
                "lane_index": int(lane.getIndex()),
                "role": role,
                "width": round(float(lane.getWidth()), 3),
                "length": round(float(lane.getLength()), 3),
                "speed_limit": round(float(lane.getSpeed()), 3),
                "allows_passenger": bool(lane.allows("passenger")),
                "movements": movements,
                "directions": [directions.get(m) for m in movements],
                "phases": phases,
                "to_lanes": sorted(
                    {from_to[m][3] for m in movements if m in from_to and len(from_to[m]) > 3}
                ),
                "polygon_world": polygon,
            }
        )

    records.sort(key=lambda rec: (rec["role"] == "internal", rec["edge_id"], rec["lane_index"]))
    for mask_id, record in enumerate(records, start=1):
        record["mask_id"] = mask_id
    return records
