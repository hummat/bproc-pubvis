import blenderproc as bproc  # noqa: I001,F401  # required first import for BlenderProc CLI
import sys
from pathlib import Path

# Ensure the package is importable when BlenderProc runs from another CWD.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bproc_pubvis.main import Config, main, run  # noqa: E402  # re-export for compatibility

__all__ = ["Config", "main", "run"]

if __name__ == "__main__":
    main()
