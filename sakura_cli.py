"""Run the community CLI directly from this checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sakura" / "src"))
from sakura.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
