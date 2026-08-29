'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Put finished ablation runs beside each other.

Thin CLI over `bevlight.ablation.cli.summarize`; all logic lives there.
Equivalent to: python -m bevlight.ablation.cli.summarize
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevlight.ablation.cli.summarize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
