"""Allow `python -m src.recorder ...`."""
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
