#!/usr/bin/env python3
"""Download and verify upstream sources (OW-001). Thin wrapper around `python -m occamworm.sources`.

Examples:
    python scripts/acquire_sources.py --milestone M0 --dry-run
    python scripts/acquire_sources.py --milestone M0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from occamworm.sources.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry-run" in args
    selection = [a for a in args if a != "--dry-run"]
    code = main(["fetch", *selection, *(["--dry-run"] if dry else [])])
    if code == 0 and not dry:
        code = main(["verify", *selection])
    sys.exit(code)
