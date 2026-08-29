#!/usr/bin/env python
'''Build SUMO networks and OSM polygons for the junction scenarios.

Thin CLI over `bevlight.scenario.build_networks`; all logic lives there.
Equivalent to: python -m bevlight.scenario.build_networks
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.scenario.cli.build_networks import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
