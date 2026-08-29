'''Per-lane counting must respect what the camera can see.

A queue label computed over the whole SUMO lane would describe vehicles that are
not in the image, and an expert reading it would make decisions the vision model
cannot explain. These lock down the clipping.
'''

from __future__ import annotations

import pytest

from bevlight.collect.observation import HALTING_SPEED_MS, ObservationExtractor
from bevlight.scenario.lane_mask import load_lane_mask
from bevlight.scenario.loader import AVAILABLE_JUNCTIONS


@pytest.fixture(scope="module")
def mask():
    return load_lane_mask("Beijing_Beihuan", "normal")


def fake_states(tls_id, vehicles):
    return {
        "tls": {tls_id: {"this_phase_index": 0, "can_perform_action": True}},
        "vehicle": {f"v{i}": v for i, v in enumerate(vehicles)},
        "sim_step": 100.0,
    }


def vehicle(lane_id, lane_position, speed=0.0, length=5.0):
    return {
        "lane_id": lane_id,
        "lane_position": lane_position,
        "speed": speed,
        "length": length,
    }


def test_vehicles_beyond_the_window_are_not_counted(mask):
    """The whole point: a queue tail outside the image must not reach the label."""
    extractor = ObservationExtractor(mask, mask.tls_id)
    lane = mask.incoming_lane_ids()[0]
    lane_length = mask.lane_record(lane)["length"]
    visible = mask.visible_length_m(lane)

    inside = vehicle(lane, lane_length - (visible - 5.0))
    outside = vehicle(lane, lane_length - (visible + 20.0))

    obs = extractor(fake_states(mask.tls_id, [inside, outside]))
    assert obs.lanes[lane].queued == 1, "the vehicle past the window edge leaked in"
    assert obs.lanes[lane].vehicles == 1


def test_incoming_distance_is_measured_back_from_the_stop_line(mask):
    extractor = ObservationExtractor(mask, mask.tls_id)
    lane = mask.incoming_lane_ids()[0]
    lane_length = mask.lane_record(lane)["length"]
    # At the stop line the vehicle is at the far end of the lane, distance 0.
    assert extractor.distance_from_junction(lane, lane_length) == pytest.approx(0.0)
    assert extractor.distance_from_junction(lane, lane_length - 30.0) == pytest.approx(30.0)


def test_outgoing_distance_is_measured_forward_from_the_junction(mask):
    extractor = ObservationExtractor(mask, mask.tls_id)
    lane = mask.lane_ids("outgoing")[0]
    assert extractor.distance_from_junction(lane, 0.0) == pytest.approx(0.0)
    assert extractor.distance_from_junction(lane, 30.0) == pytest.approx(30.0)


def test_only_halting_vehicles_count_as_queued(mask):
    extractor = ObservationExtractor(mask, mask.tls_id)
    lane = mask.incoming_lane_ids()[0]
    length = mask.lane_record(lane)["length"]
    obs = extractor(
        fake_states(
            mask.tls_id,
            [
                vehicle(lane, length - 10.0, speed=0.0),
                vehicle(lane, length - 20.0, speed=HALTING_SPEED_MS + 5.0),
            ],
        )
    )
    assert obs.lanes[lane].vehicles == 2
    assert obs.lanes[lane].queued == 1


def test_queue_saturation_is_flagged(mask):
    """When the queue reaches the image edge its true length is unknown."""
    extractor = ObservationExtractor(mask, mask.tls_id)
    lane = mask.incoming_lane_ids()[0]
    length = mask.lane_record(lane)["length"]
    visible = mask.visible_length_m(lane)
    obs = extractor(fake_states(mask.tls_id, [vehicle(lane, length - (visible - 0.2))]))
    assert obs.lanes[lane].queue_saturated
    assert lane in obs.saturated_lanes()


def test_lanes_outside_this_junction_are_ignored(mask):
    extractor = ObservationExtractor(mask, mask.tls_id)
    obs = extractor(fake_states(mask.tls_id, [vehicle("some_other_edge_0", 10.0)]))
    assert all(state.vehicles == 0 for state in obs.lanes.values())


@pytest.mark.parametrize("junction", AVAILABLE_JUNCTIONS)
def test_visible_stretch_is_contiguous_from_the_junction(junction):
    """`visible_length_m` divides clipped area by width.

    That is only the distance from the stop line if the lane leaves the window
    once and never comes back. If a lane clipped in two pieces, the label would
    silently include vehicles behind a gap the camera cannot see.
    """
    from shapely.geometry import Polygon, box

    mask = load_lane_mask(junction, "normal")
    cx, cy = mask.camera["center"]
    half = mask.camera["ortho_size"] / 2.0
    window = box(cx - half, cy - half, cx + half, cy + half)

    for record in mask.lanes:
        geom = Polygon(record["polygon_world"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        clipped = geom.intersection(window)
        if clipped.is_empty:
            continue
        pieces = len(clipped.geoms) if clipped.geom_type == "MultiPolygon" else 1
        assert pieces == 1, f"{junction}/{record['lane_id']} is visible in {pieces} disjoint pieces"
