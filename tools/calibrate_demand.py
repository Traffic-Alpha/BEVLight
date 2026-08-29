#!/usr/bin/env python
'''Find the demand scale whose queues land in the useful band.

Equivalent to: python -m bevlight.collect.calibrate_demand
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.collect.cli.calibrate_demand import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
