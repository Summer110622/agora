"""Run from the repository root; no upstream Agora dependencies are imported."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "sakura" / "src"))

from sakura.inference_cli import main

if __name__ == "__main__":
    main()
