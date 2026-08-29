#!/usr/bin/env python
'''Ask whether BEV + lane mask can recover per-lane queue length.

Equivalent to: python -m bevlight.eval.probe
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.eval.cli.probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
