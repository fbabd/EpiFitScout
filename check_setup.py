#!/usr/bin/env python3
"""Check that EpiFitScout is set up and working.

    python check_setup.py            # are MASTER and the database in place?
    python check_setup.py --search   # also run a real search end to end

Equivalent to ``epifitscout check``. Kept at the top level because it is the
first thing a new clone runs, and it works before the package is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epifitscout.tools.setup_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
