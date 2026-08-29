#!/usr/bin/env python
'''Read a teacher run's history and say what state the training is in.

Pairs each number with the failure it would be evidence of, so a flat curve is
reported as "converged" or "dead" rather than left as a plot to squint at.

Equivalent to: python -m bevlight.rl.cli.diagnose
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.rl.cli.diagnose import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
