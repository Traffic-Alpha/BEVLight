#!/usr/bin/env python
'''Rank checkpoints by closed-loop control quality, against the rule-based baselines.

Equivalent to: python -m bevlight.eval.closed_loop
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.eval.closed_loop import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
