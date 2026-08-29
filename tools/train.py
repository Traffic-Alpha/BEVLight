#!/usr/bin/env python
'''Train the phase-scoring model by behaviour cloning on cached features.

Equivalent to: python -m bevlight.train.run
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.train.cli.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
