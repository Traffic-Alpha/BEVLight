#!/usr/bin/env python
'''Build the static 3D assets (GLB + scene.blend) for each junction.

Thin CLI over `bevlight.scenario.build_static_scene`; all logic lives there.
Equivalent to: python -m bevlight.scenario.build_static_scene
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.scenario.cli.build_static_scene import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
