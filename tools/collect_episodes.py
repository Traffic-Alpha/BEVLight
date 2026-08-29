#!/usr/bin/env python
'''Run expert episodes and write their trajectories under data/episodes/.

No rendering happens here. One trajectory is replayed later into as many
appearance variants as needed, all sharing this episode's labels.

Equivalent to: python -m bevlight.collect.collect
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.collect.cli.collect import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
