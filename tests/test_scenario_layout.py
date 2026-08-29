'''Split manifest and on-disk naming: the two things every later stage assumes.'''

from __future__ import annotations

from pathlib import Path

import pytest

from bevlight.scenario.layout import EpisodeKey, episode_file, frame_file, variant_dir
from bevlight.scenario.loader import AVAILABLE_JUNCTIONS, load_junction_config
from bevlight.scenario.selection import load_selection


def test_active_set_matches_manifest_totals():
    sel = load_selection()
    assert len(sel.train) == 45
    assert len(sel.cross_plan_test) == 8
    assert len(sel.cross_structure_test) == 8
    # The manifest counts the splits it declares; 45 + 8 + 8.
    declared = len(sel.train) + len(sel.cross_plan_test) + len(sel.cross_structure_test)
    assert declared == 61


def test_cross_demand_is_derived_from_train_and_the_held_out_demands():
    """Tier one is the trained geometry under traffic it never saw.

    Derived rather than listed, so it must stay exactly the trained (junction,
    plan) pairs — a drift here would silently evaluate tier one on a junction the
    model was never trained for, which is tier three wearing tier one's label.
    """
    sel = load_selection()
    trained = {(s.junction, s.plan) for s in sel.train}
    derived = {(s.junction, s.plan) for s in sel.cross_demand_test}
    assert derived == trained

    train_demands = {s.demand for s in sel.train}
    test_demands = {s.demand for s in sel.cross_demand_test}
    assert not (train_demands & test_demands), "tier one must use unseen demands"
    assert len(sel.cross_demand_test) == len(trained) * len(test_demands)


def test_beihuan_contributes_one_plan_only():
    """Its two plans are the same signal plan, so the second would be a duplicate."""
    plans = {s.plan for s in load_selection().of_junction("Beijing_Beihuan")}
    assert plans == {"normal"}


@pytest.mark.needs_scenarios
def test_every_active_scenario_resolves_to_a_real_sumocfg():
    for scenario in load_selection().all():
        cfg = load_junction_config(scenario.junction, scenario.env_name)
        assert Path(cfg["sumo_cfg"]).is_file(), scenario
        assert Path(cfg["net_file"]).is_file(), scenario


def test_cross_plan_test_plan_is_never_the_trained_plan():
    """The held-out plan is the whole point of the cross-plan split."""
    sel = load_selection()
    trained = {(s.junction, s.plan) for s in sel.train}
    for scenario in sel.cross_plan_test:
        assert (scenario.junction, scenario.plan) not in trained, scenario


def test_cross_structure_junctions_are_absent_from_training():
    sel = load_selection()
    trained = set(sel.junctions("train"))
    for junction in sel.junctions("cross_structure_test"):
        assert junction not in trained, junction


def test_test_demands_are_held_out_from_training():
    sel = load_selection()
    train_demands = {s.demand for s in sel.train}
    test_demands = {s.demand for s in sel.cross_plan_test + sel.cross_structure_test}
    assert train_demands.isdisjoint(test_demands)


def test_selection_junctions_all_exist_on_disk():
    for junction in load_selection().junctions():
        assert junction in AVAILABLE_JUNCTIONS


@pytest.mark.parametrize(
    "junction,plan,demand",
    [
        ("Beijing_Beihuan", "easy", "low_density"),
        # Junction and demand names both contain underscores; the key must survive.
        ("Chengdu_Chenghannanlu", "normal", "fluctuating_commuter"),
        ("SouthKorea_Songdo", "normal", "random_perturbation"),
    ],
)
def test_episode_key_roundtrip(junction, plan, demand):
    key = EpisodeKey(junction, plan, demand, seed=7, expert="mp")
    assert EpisodeKey.parse(str(key)) == key


def test_episode_key_rejects_garbage():
    with pytest.raises(ValueError):
        EpisodeKey.parse("not-an-episode-key")


def test_layout_paths_are_nested_under_the_episode_key():
    key = EpisodeKey("Beijing_Beihuan", "easy", "low_density", seed=7, expert="mp")
    assert episode_file(key).parent.name == str(key)
    assert variant_dir(key, "panda_day").parent.parent.name == str(key)
    assert frame_file(key, "panda_day", 42).parent.name == "rgb"
    assert frame_file(key, "panda_day", 42).name == "0042.png"
