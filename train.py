"""Compatibility wrapper for the clean public package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_attnres_project.legacy_engine import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
