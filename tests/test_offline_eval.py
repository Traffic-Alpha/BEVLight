'''Offline metrics must expose the collapse they exist to catch.

Overall action accuracy is inflated by how often the expert keeps the current
phase, so a model that has learned only "keep" posts a respectable number while
getting every phase change wrong. These tests pin the two properties that make
that visible: the switch/keep split recombines into the overall figure, and an
always-keep predictor scores exactly the keep rate and zero on switches.
'''

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from bevlight.eval.offline import agreement, macro_f1, references


def make_prediction(expert, current, chosen) -> dict:
    """The flat arrays `predict()` produces, without running a model."""
    expert = np.array(expert)
    current = np.array(current)
    chosen = np.array(chosen)
    return {
        "expert": expert,
        "current": current,
        "chosen": chosen,
        "argmax": chosen,
        "top2": chosen == expert,
        "group": np.array(["J/easy"] * expert.size),
        "num_phases": np.full(expert.size, 4),
        "lane_row": np.zeros(0, dtype=np.int64),
        "lane_pred": np.zeros(0),
        "lane_true": np.zeros(0),
        "occupancy_mae": np.zeros(0),
    }


def test_switch_and_keep_recombine_into_the_overall_accuracy():
    expert = [0, 1, 2, 3, 0, 1]
    current = [0, 0, 2, 2, 0, 3]
    chosen = [0, 1, 2, 0, 1, 1]
    stats = agreement(make_prediction(expert, current, chosen))

    switches = np.array(expert) != np.array(current)
    total = (
        stats["accuracy_on_switch"] * switches.sum()
        + stats["accuracy_on_keep"] * (~switches).sum()
    ) / len(expert)
    assert total == pytest.approx(stats["accuracy"])


def test_always_keep_scores_the_keep_rate_and_nothing_on_switches():
    """The degenerate model this metric exists to expose."""
    expert = [0, 1, 2, 2, 0, 1]
    current = [0, 0, 2, 1, 0, 3]
    prediction = make_prediction(expert, current, current)  # never switches

    stats = agreement(prediction)
    refs = references(prediction)

    assert stats["accuracy_on_switch"] == 0.0
    assert stats["predicted_switch_share"] == 0.0
    # Its accuracy is exactly the free reference — no information was added.
    assert stats["accuracy"] == pytest.approx(refs["always_keep_current"])
    assert stats["accuracy"] == pytest.approx(1 - stats["switch_share"])


def test_macro_f1_falls_when_a_candidate_is_never_chosen():
    """Accuracy can stay high while one phase is never served; F1 cannot."""
    expert = [0, 0, 0, 0, 0, 0, 1, 2]
    current = [3, 3, 3, 3, 3, 3, 3, 3]
    dominant = macro_f1(make_prediction(expert, current, [0] * 8), np.arange(8))
    balanced = macro_f1(
        make_prediction(expert, current, [0, 0, 0, 0, 0, 0, 1, 2]), np.arange(8)
    )
    assert dominant < balanced == 1.0


def test_references_are_computed_per_junction_not_pooled():
    """Two junctions with different majority phases must not average into one."""
    prediction = make_prediction([0, 0, 1, 1], [3, 3, 3, 3], [0, 0, 1, 1])
    prediction["group"] = np.array(["A/easy", "A/easy", "B/easy", "B/easy"])
    # Each junction's own majority is right every time; pooling them would not be.
    assert references(prediction)["group_majority_phase"] == 1.0


def test_the_summary_refuses_a_denominator_too_small_to_divide_by(tmp_path):
    """`gain%` is `(weak - policy) / (weak - expert)`.

    Where the expert barely beats the weak baseline that denominator is small,
    and an ordinary sub-second difference comes out as a large percentage. One
    real scenario reported 47% off an 0.8 s gap because the expert had gained
    only 1.5 s. Those scenarios are excluded rather than reported.
    """
    import json

    from bevlight.eval.closed_loop import MIN_HEADROOM_S, summarize_run

    def record(junction, demand, controller, travel):
        return {"junction": junction, "plan": "easy", "demand": demand,
                "controller": controller, "avg_travel_time_s": travel, "unfinished": 0}

    results = []
    # Wide headroom: expert gains 20 s, policy gives back 2 s -> 90%.
    for c, t in (("max_pressure", 40.0), ("fixed_time", 60.0), ("policy", 42.0)):
        results.append(record("Wide", "low", c, t))
    # Narrow headroom: expert gains 1 s, policy gives back 0.5 s -> would be 50%.
    for c, t in (("max_pressure", 40.0), ("fixed_time", 41.0), ("policy", 40.5)):
        results.append(record("Narrow", "low", c, t))

    (tmp_path / "closed_loop_train.json").write_text(json.dumps({"results": results}))
    summary = summarize_run(tmp_path, "policy")["train"]

    assert summary["n"] == 2, "both scenarios count toward the delta"
    assert summary["median_delta_s"] == 1.25
    # Only the wide-headroom scenario reaches the gain median.
    assert summary["median_gain_pct"] == 90.0
    assert MIN_HEADROOM_S > 41.0 - 40.0


def test_the_summary_sums_unfinished_vehicles_across_scenarios():
    """A vehicle that never completes is invisible in mean travel time.

    It is excluded from the mean it would have raised, so a controller that
    strands traffic can look *faster*. The count has to be carried separately.
    """
    import json
    import tempfile

    from bevlight.eval.closed_loop import summarize_run

    with tempfile.TemporaryDirectory() as d:
        results = []
        for junction, unfinished in (("A", 2), ("B", 1)):
            for c, t, u in (("max_pressure", 40.0, 0), ("fixed_time", 60.0, 0),
                            ("policy", 39.0, unfinished)):
                results.append({"junction": junction, "plan": "easy", "demand": "low",
                                "controller": c, "avg_travel_time_s": t, "unfinished": u})
        Path(d, "closed_loop_train.json").write_text(json.dumps({"results": results}))
        assert summarize_run(Path(d), "policy")["train"]["unfinished"] == 3
