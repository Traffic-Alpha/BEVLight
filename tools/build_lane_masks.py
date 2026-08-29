#!/usr/bin/env python
'''Build the static BEV lane-id mask for each junction.

Thin CLI over `bevlight.scenario.build_lane_masks`; all logic lives there.
Equivalent to: python -m bevlight.scenario.build_lane_masks
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.scenario.cli.build_lane_masks import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
