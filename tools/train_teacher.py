#!/usr/bin/env python
'''Train the privileged teacher on structured state, with no renderer.

This is the gate before distillation, not the method: it asks whether a learned
controller can beat max-pressure on this scenario at all, and by how much. A
1000-second episode costs seconds of SUMO here against 22 minutes per update
through the rendering loop, so the answer arrives in hours.

The teacher observes only what the BEV window shows — per-lane queue, occupancy
and the flag that says the queue ran off the edge of the image — over the same
five-second window the vision model reads. That restriction is what makes the
result transferable: a teacher that acted on anything else would be teaching a
lesson the student has no way to learn.

Equivalent to: python -m bevlight.rl.sac
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.rl.sac import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
