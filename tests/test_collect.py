'''Episode files and render-frame selection.'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from bevlight.collect.episode_schema import LANE_FIELDS, decision_window, load_episode
from bevlight.collect.frame_selection import select_frames, write_blender_manifest

EPISODES = sorted((ROOT / "data" / "episodes").glob("*/episode.json"))
needs_episodes = pytest.mark.skipif(not EPISODES, reason="no collected episodes on disk")


@pytest.fixture(scope="module")
def payload():
    return load_episode(EPISODES[0].parent)


@needs_episodes
def test_lane_truth_is_rectangular_and_matches_lane_order(payload):
    width = len(payload["lane_order"])
    horizon = len(payload["lane_truth"]["vehicles"])
    for field in LANE_FIELDS:
        rows = payload["lane_truth"][field]
        assert len(rows) == horizon
        assert all(len(row) == width for row in rows)


@needs_episodes
def test_queued_never_exceeds_vehicles_present(payload):
    for queued, present in zip(payload["lane_truth"]["queued"], payload["lane_truth"]["vehicles"]):
        assert all(q <= v for q, v in zip(queued, present))


@needs_episodes
def test_queue_never_exceeds_the_visible_stretch(payload):
    visible = [payload["visible_length_m"][lane] for lane in payload["lane_order"]]
    for row in payload["lane_truth"]["queue_m"]:
        assert all(q <= v + 1e-6 for q, v in zip(row, visible))


@needs_episodes
def test_every_decision_names_a_valid_phase_and_frame(payload):
    phases = set(payload["signal_plan"]["phases"])
    horizon = len(payload["lane_truth"]["vehicles"])
    for decision in payload["decisions"]:
        assert decision["action"] in phases
        assert decision["phase"] in phases
        assert 0 <= decision["frame_index"] < horizon
        assert decision["num_phases"] == len(phases)


@needs_episodes
def test_decisions_are_spaced_by_the_decision_interval(payload):
    times = [d["t"] for d in payload["decisions"]]
    gaps = {b - a for a, b in zip(times, times[1:])}
    interval = payload["decision_interval_s"]
    yellow = payload["yellow_time_s"]
    # Keeping a phase costs delta_time; switching adds the yellow interval.
    assert gaps <= {interval, interval + yellow}


@needs_episodes
def test_decision_window_is_clamped_at_the_episode_start(payload):
    window = decision_window(payload, 0, window=5)
    assert len(window) == 5
    assert all(index >= 0 for index in window)
    assert window == sorted(window)


@needs_episodes
def test_render_frame_count_matches_the_frames_on_disk(payload):
    episode_dir = EPISODES[0].parent
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    assert len(manifest["frames"]) == len(payload["lane_truth"]["vehicles"])


@needs_episodes
def test_selection_cuts_frames_and_drops_empty_decisions(payload):
    selection = select_frames(payload, window=5, empty_keep=0.0)
    assert selection.frames_kept < selection.frames_total
    assert len(selection.decisions_kept) == selection.decisions_total - selection.decisions_empty

    # Every kept decision must have traffic somewhere in its window.
    import numpy as np

    roles = payload["lane_roles"]
    incoming = [i for i, l in enumerate(payload["lane_order"]) if roles[l] == "incoming"]
    vehicles = np.asarray(payload["lane_truth"]["vehicles"], dtype=float)[:, incoming].sum(axis=1)
    for index in selection.decisions_kept:
        span = decision_window(payload, index, window=5)
        assert vehicles[span].sum() > 0


@needs_episodes
def test_selection_can_keep_a_sample_of_empty_scenes(payload):
    """The model meets empty junctions at evaluation time and must not panic."""
    none_kept = select_frames(payload, empty_keep=0.0)
    all_kept = select_frames(payload, empty_keep=1.0)
    assert len(all_kept.decisions_kept) > len(none_kept.decisions_kept)
    assert len(all_kept.decisions_kept) == all_kept.decisions_total


@needs_episodes
def test_derived_manifest_points_at_the_original_frames(tmp_path, payload):
    episode_dir = EPISODES[0].parent
    selection = select_frames(payload, empty_keep=0.0)
    out_path = write_blender_manifest(episode_dir, selection, tag="pytest")
    manifest = json.loads(out_path.read_text())
    assert len(manifest["frames"]) == selection.frames_kept
    for relative in manifest["frames"][:20]:
        assert (episode_dir / relative).resolve().is_file()


def test_frame_check_samples_only_frames_training_will_see(tmp_path):
    """A frame with no vehicles cannot show a misalignment.

    Routes stop departing well before the horizon, so a uniform sample lands
    mostly in the drained tail. The Blender selection is exactly the set of
    frames that can feed a decision sample, so that is what gets checked.
    """
    import json

    from bevlight.collect.cli.frame_check import sample_frames

    episode = tmp_path / "J__easy__low__seed7__mp"
    frames = episode / "images" / "panda_day" / "rgb"
    frames.mkdir(parents=True)
    for index in range(50):
        (frames / f"{index:04d}.png").write_bytes(b"")
    (episode / "episode.json").write_text(json.dumps({"junction": "J", "plan": "easy"}))
    (episode / "blender_selected.json").write_text(json.dumps(
        {"frames": [f"frames/{i:04d}.json" for i in range(10, 20)]}
    ))

    picks = sample_frames([episode], "panda_day", per_episode=8, seed=0)
    assert len(picks) == 8
    assert all(10 <= int(frame.stem) < 20 for _, _, frame in picks)


def test_frame_check_falls_back_when_an_episode_has_no_selection(tmp_path):
    import json

    from bevlight.collect.cli.frame_check import sample_frames

    episode = tmp_path / "J__easy__low__seed7__mp"
    frames = episode / "images" / "panda_day" / "rgb"
    frames.mkdir(parents=True)
    for index in range(50):
        (frames / f"{index:04d}.png").write_bytes(b"")
    (episode / "episode.json").write_text(json.dumps({"junction": "J", "plan": "easy"}))

    assert len(sample_frames([episode], "panda_day", per_episode=8, seed=0)) == 8


def test_frame_check_reports_a_variant_it_cannot_find(tmp_path):
    import json

    from bevlight.collect.cli.frame_check import sample_frames

    episode = tmp_path / "J__easy__low__seed7__mp"
    (episode / "images" / "panda_day" / "rgb").mkdir(parents=True)
    (episode / "episode.json").write_text(json.dumps({"junction": "J", "plan": "easy"}))

    assert sample_frames([episode], "blender_day", per_episode=3, seed=0) == []
