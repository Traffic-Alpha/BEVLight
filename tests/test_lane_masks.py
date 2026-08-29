'''The lane mask is the link between a rendered pixel and a SUMO lane.

These check the two properties every later stage relies on: that a mask is
scoped to one signal plan, and that the BEV window is sized consistently.
'''

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from bevlight.scenario import bev_camera
from bevlight.scenario.lane_mask import available_plans, load_lane_mask
from bevlight.scenario.loader import AVAILABLE_JUNCTIONS

JUNCTIONS = list(AVAILABLE_JUNCTIONS)
PAIRS = [(j, p) for j in JUNCTIONS for p in ("easy", "normal")]


@pytest.mark.parametrize("junction", JUNCTIONS)
def test_every_junction_has_a_mask_for_every_plan(junction):
    assert sorted(available_plans(junction)) == ["easy", "normal"]


@pytest.mark.parametrize("junction,plan", PAIRS)
def test_phase_count_matches_the_network(junction, plan):
    """The bug this guards: a mask built from one plan mislabels the other."""
    mask = load_lane_mask(junction, plan)
    root = ET.parse(ROOT / "scenarios" / junction / "networks" / f"{plan}.net.xml").getroot()
    logic = next(t for t in root.findall("tlLogic") if t.get("id") == mask.tls_id)
    green_phases = [p for p in logic.findall("phase") if "y" not in p.get("state").lower()]
    assert mask.num_phases == len(green_phases)


def test_plans_can_differ_in_phase_count():
    """Hongkong_YMT is the junction that makes variable K real: 4 vs 3."""
    assert load_lane_mask("Hongkong_YMT", "easy").num_phases == 4
    assert load_lane_mask("Hongkong_YMT", "normal").num_phases == 3


@pytest.mark.parametrize("junction,plan", PAIRS)
def test_phase_membership_agrees_between_lanes_and_tls(junction, plan):
    mask = load_lane_mask(junction, plan)
    for phase, movements in mask.phase2movements.items():
        for movement in movements:
            for lane_id in mask.tls["movement_lane_ids"].get(movement, []):
                assert phase in mask.phases_of(lane_id), (junction, plan, phase, lane_id)


@pytest.mark.parametrize("junction,plan", PAIRS)
def test_pixel_scale_is_the_same_everywhere(junction, plan):
    """Every junction must reach the model at one scale.

    Songdo used to render at 7.3 px/m against everyone else's 11.36, which put
    the cross-structure test junction at 64% of the training scale.
    """
    mask = load_lane_mask(junction, plan)
    width, _ = mask.resolution
    scale = width / mask.camera["ortho_size"]
    assert scale == pytest.approx(bev_camera.PIXELS_PER_METER, rel=0.01)


@pytest.mark.parametrize("junction,plan", PAIRS)
def test_window_exposes_the_target_queue(junction, plan):
    mask = load_lane_mask(junction, plan)
    target = bev_camera.TARGET_VISIBLE_APPROACH_M
    assert mask.plan_meta["window"]["visible_approach_median_m"] >= target - 1.0
    assert mask.queue_capacity_vehicles >= target / bev_camera.JAM_SPACING_M - 0.2


@pytest.mark.parametrize("junction,plan", PAIRS)
def test_render_size_is_a_whole_number_of_vit_patches(junction, plan):
    width, height = load_lane_mask(junction, plan).resolution
    assert width == height
    assert width % bev_camera.PATCH_SIZE == 0


@pytest.mark.parametrize("junction,plan", PAIRS)
def test_mask_image_matches_its_declared_lanes(junction, plan):
    mask = load_lane_mask(junction, plan)
    assert mask.labels.shape[:2] == tuple(reversed(mask.resolution))
    assert not mask.plan_meta["lanes_without_pixels"], "a lane fell outside the window"
    for record in mask.lanes:
        assert int(mask.pixels_of(record["lane_id"]).sum()) > 0


def test_guanghua_is_flagged_as_not_sharing_geometry():
    """Its two plans build different junction shapes, so they cannot share a mask."""
    meta = json.loads(
        (ROOT / "scenarios" / "Chengdu_Guanghua" / "lane_mask" / "lane_mask.json").read_text()
    )
    assert meta["geometry_shared_across_plans"] is False
    easy = load_lane_mask("Chengdu_Guanghua", "easy").camera["center"]
    normal = load_lane_mask("Chengdu_Guanghua", "normal").camera["center"]
    assert easy != normal


@pytest.mark.parametrize("junction", [j for j in JUNCTIONS if j != "Chengdu_Guanghua"])
def test_other_junctions_do_share_geometry(junction):
    meta = json.loads(
        (ROOT / "scenarios" / junction / "lane_mask" / "lane_mask.json").read_text()
    )
    assert meta["geometry_shared_across_plans"] is True


def test_loading_without_a_plan_is_impossible():
    with pytest.raises(TypeError):
        load_lane_mask("Beijing_Beihuan")
    with pytest.raises(ValueError):
        load_lane_mask("Beijing_Beihuan", "nonexistent_plan")


@pytest.mark.slow
@pytest.mark.parametrize("junction,plan,demand", [
    ("Beijing_Pinganli", "easy", "high_density"),
    ("SouthKorea_Songdo", "easy", "increasing_demand"),
])
def test_the_mask_raster_agrees_with_its_own_projection(junction, plan, demand):
    """The raster and the projection that produced it stay consistent.

    **This does not check that the mask lines up with the rendered image**, and
    it must not be read as if it did. The mask is rasterised by pushing lane
    polygons through `world_to_pixel`; this pushes vehicles through the same
    function and looks them up in that raster. Both sides use one camera, so the
    test is closed on itself and passes whatever that camera is. It caught a
    rasterisation or lane-indexing fault and nothing else — it was briefly, and
    wrongly, taken as proof of alignment.

    Alignment against the render is a separate question with a separate answer:
    `overlay_{plan}.png`, written beside the mask on every build, and looked at.
    Measured against vehicle positions recovered from the segmentation pass
    rather than from the projection, the two cameras disagree by roughly 3 m plus
    4% of the radius at Beijing_Pinganli.
    """
    import json

    import numpy as np
    import traci

    from bevlight.scenario.bev_camera import BevCamera
    from bevlight.scenario.lane_mask import load_lane_mask
    from bevlight.paths import BEV_CAMERA_TABLE, SCENARIOS_ROOT

    mask = load_lane_mask(junction, plan)
    labels = mask.labels
    resolution = (labels.shape[1], labels.shape[0])
    entry = json.loads(BEV_CAMERA_TABLE.read_text())
    solved = entry["junctions"][junction]["plans"][plan]
    camera = BevCamera(center=tuple(solved["center"]), height=solved["height"],
                       ortho_size=solved["ortho_size"])
    lane_of = {record["mask_id"]: record["lane_id"] for record in mask.lanes}
    masked_lanes = set(lane_of.values())

    traci.start(["sumo", "-c", str(SCENARIOS_ROOT / junction / f"{plan}_{demand}.sumocfg"),
                 "--no-warnings", "--no-step-log"])
    agree = disagree = 0
    try:
        for _ in range(200):
            traci.simulationStep()
            for vehicle in traci.vehicle.getIDList():
                lane = traci.vehicle.getLaneID(vehicle)
                if lane not in masked_lanes:
                    continue
                u, v = camera.world_to_pixel(
                    [traci.vehicle.getPosition(vehicle)], resolution
                )[0]
                column, row = int(round(u)), int(round(v))
                if not (0 <= column < resolution[0] and 0 <= row < resolution[1]):
                    continue          # beyond the window; nothing to check here
                if lane_of.get(int(labels[row, column])) == lane:
                    agree += 1
                else:
                    disagree += 1
    finally:
        traci.close()

    assert agree > 500, f"only {agree} vehicles landed inside the window; test is vacuous"
    assert disagree == 0, (
        f"{disagree} of {agree + disagree} vehicles projected onto a lane mask "
        f"other than the one SUMO puts them on"
    )


def test_role_views_paint_only_their_own_role():
    """An `_in` overlay must not carry a single outgoing pixel.

    The role views exist so incoming and outgoing lanes can be judged apart;
    a leak between them would make the separate views pointless while still
    looking plausible.
    """
    import numpy as np

    from bevlight.cli.viz import ROLE_HUE_BAND, colorize

    records = [
        {"mask_id": 1, "role": "incoming"},
        {"mask_id": 2, "role": "outgoing"},
        {"mask_id": 3, "role": "incoming"},
    ]
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint16)

    incoming = colorize(mask, records, ("incoming",))
    outgoing = colorize(mask, records, ("outgoing",))

    assert incoming[1, 0].sum() == 0, "outgoing lane painted into the incoming view"
    assert outgoing[0, 1].sum() == 0 and outgoing[1, 1].sum() == 0
    assert incoming[0, 1].sum() > 0 and incoming[1, 1].sum() > 0
    assert outgoing[1, 0].sum() > 0
    # Background stays background in both.
    assert incoming[0, 0].sum() == 0 and outgoing[0, 0].sum() == 0
    assert ROLE_HUE_BAND["incoming"][1] < ROLE_HUE_BAND["outgoing"][0]


def test_a_reference_frame_shot_under_a_different_camera_is_refused(tmp_path, monkeypatch):
    """The stale-frame guard, which is the whole reason meta.json exists.

    Drawing today's mask over a frame shot through yesterday's camera is how a
    misalignment gets attributed to the mask instead of the camera.
    """
    import json

    from bevlight.scenario._internal import bev_reference

    monkeypatch.setattr(bev_reference, "BEV_REFERENCE_ROOT", tmp_path)
    monkeypatch.setattr(
        "bevlight.scenario.bev_camera.bev_ortho_size", lambda junction, plan=None: 120.0
    )

    directory = bev_reference.reference_dir("J", 120.0)
    directory.mkdir(parents=True)
    (directory / "panda_day.png").write_bytes(b"")

    (directory / "meta.json").write_text(json.dumps({"ortho_size": 120.0}))
    assert bev_reference.reference_frame("J", "easy", "panda_day") is not None

    (directory / "meta.json").write_text(json.dumps({"ortho_size": 90.0}))
    assert bev_reference.reference_frame("J", "easy", "panda_day") is None
