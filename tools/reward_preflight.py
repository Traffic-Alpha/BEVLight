#!/usr/bin/env python
'''Check the RL reward before spending a training run on it.

Runs controllers whose ordering is already known — random, three fixed-time
timings, two max-pressure variants — on structured state with no renderer, and
asks whether each candidate reward reproduces that ordering. Minutes, not hours.

The question is not "does the reward prefer max-pressure to fixed-time". A
queue-shaped reward must, because max-pressure is a queue heuristic; that answer
is a tautology. The question is whether the reward ranks *travel time*, which is
what the results table reports.

Equivalent to: python -m bevlight.rl.preflight
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.rl.preflight import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
