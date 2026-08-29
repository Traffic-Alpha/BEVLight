#!/usr/bin/env python
'''Score trained checkpoints on cached features, before any simulation runs.

Equivalent to: python -m bevlight.eval.offline
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.eval.offline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
