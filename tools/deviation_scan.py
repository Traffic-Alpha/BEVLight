#!/usr/bin/env python
'''Does changing any single decision improve on max-pressure?

Runs one full episode per deviation, from the start, in parallel across cores.
Exact: the route files list every vehicle, so an episode is a deterministic
function of its action sequence and nothing has to be saved or restored.

If nothing helps, max-pressure is a local optimum one decision at a time — a
statement about the problem, not about any method that failed to beat it.

Equivalent to: python -m bevlight.rl.cli.deviation
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.rl.cli.deviation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
