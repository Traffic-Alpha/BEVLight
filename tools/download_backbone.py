#!/usr/bin/env python
'''Download backbone weights into runs/backbones/ for offline use.

Equivalent to: python -m bevlight.model.download
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.model.cli.download import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
