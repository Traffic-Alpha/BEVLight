#!/usr/bin/env python
'''Render one reference BEV frame per junction, for lane-mask overlay checks.

Thin CLI over `bevlight.scenario.render_reference`; all logic lives there.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.scenario.cli.render_reference import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
