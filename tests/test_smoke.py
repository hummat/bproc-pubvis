import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import constants  # type: ignore[import]


def test_color_enum_has_white() -> None:
    assert hasattr(constants.Color, "WHITE")
