#!/usr/bin/env python
'''Overlay lane masks on random collected frames, to be checked by eye.

Thin CLI over `bevlight.collect.frame_check`; all logic lives there.
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.collect.cli.frame_check import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
