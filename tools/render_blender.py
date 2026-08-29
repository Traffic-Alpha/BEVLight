#!/usr/bin/env python
'''Render collected episode manifests with Blender Cycles.

Thin CLI over `bevlight.collect.blender`; all logic lives there.
Equivalent to: python -m bevlight.collect.blender
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.collect.cli.blender import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
