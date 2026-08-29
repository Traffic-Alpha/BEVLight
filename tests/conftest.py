'''Why a test skips, decided in one place.

Two things this repository's tests need are not in the repository. SUMO comes
from the `tshub` conda environment and announces its absence by calling
`sys.exit` while being imported, which pytest reports as a failure inside
somebody else's file. The generated per-junction artefacts -- `networks/*.net.xml`,
`lane_mask/lane_mask.json`, the `*.sumocfg` set -- are built by
`bevlight scenario ...` commands and are not tracked, so on a fresh clone the
tests that read them raise FileNotFoundError.

Neither is a failure. A test that cannot run should say so and say why, because
the alternative is a wall of red that hides the tests that did fail. So:

    @pytest.mark.slow             needs a SUMO simulation; skipped without SUMO_HOME
    @pytest.mark.needs_scenarios  reads built scenario data; skipped without it

Marking is deliberate rather than inferred. A test that grows a dependency on
scenario data and forgets the marker fails loudly on a fresh clone, which is the
right way round: the marker is a claim about the test, and a wrong claim should
be visible.
'''

from __future__ import annotations

import os

import pytest

from bevlight.paths import SCENARIOS_ROOT

BUILD_HINT = (
    "no built scenario data under scenarios/ -- run `bevlight scenario "
    "build-networks` and `bevlight scenario build-lane-masks`"
)
SUMO_HINT = (
    "SUMO_HOME is not set -- these need the tshub conda environment "
    "(`conda run -n tshub python -m pytest`)"
)


def scenario_data_present() -> bool:
    """Whether the artefacts the domain tests read were ever built here.

    One `net.xml` is enough to tell the two situations apart: a checkout that
    has been built in, and a fresh clone. It is not a completeness check --
    a half-built tree should fail rather than quietly skip.
    """
    return any(SCENARIOS_ROOT.glob("*/networks/*.net.xml"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_scenarios: reads generated per-junction scenario data",
    )


def pytest_collection_modifyitems(config, items):
    reasons = []
    if "SUMO_HOME" not in os.environ:
        reasons.append(("slow", pytest.mark.skip(reason=SUMO_HINT)))
    if not scenario_data_present():
        reasons.append(("needs_scenarios", pytest.mark.skip(reason=BUILD_HINT)))
    for item in items:
        for keyword, mark in reasons:
            if keyword in item.keywords:
                item.add_marker(mark)
