#!/usr/bin/env python
'''Compare controllers on the same scenarios, seeds and decision interval.

The point is to confirm the expert is worth imitating before spending any
rendering time on it: if max-pressure cannot beat fixed-time here, nothing
downstream can be trusted.

Only *training* scenarios are used by default. The held-out demands
(increasing_demand, random_perturbation) and the held-out plans belong to the
test splits and must not be looked at while tuning anything, so asking for them
takes an explicit --split.

Equivalent to: python -m bevlight.eval.compare
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.eval.compare import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
