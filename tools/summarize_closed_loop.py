#!/usr/bin/env python
'''Roll the per-split closed-loop results of a run into one table.

Thin CLI over `bevlight.eval.closed_loop.summarize`; all logic lives there.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.eval.closed_loop import summarize_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(summarize_main())
