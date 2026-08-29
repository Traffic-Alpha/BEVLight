#!/usr/bin/env python
'''Split a rendered BEV frame into one image per lane.

Thin CLI over `bevlight.scenario.lane_views`; all logic lives there.
Equivalent to: python -m bevlight.scenario.lane_views
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.scenario.cli.lane_views import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
