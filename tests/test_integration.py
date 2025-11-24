from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Opt-in: require explicit env flag; assumes BlenderProc is correctly configured
pytestmark = pytest.mark.skipif(
    not os.getenv("BPROC_INTEGRATION"),
    reason="Set BPROC_INTEGRATION=1 to run BlenderProc integration tests.",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _run_main_with_external_bpy(
    tmp_path: Path, extra_args: list[str]
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["USE_EXTERNAL_BPY_MODULE"] = "1"
    env["BPROC_TEMP_DIR"] = str(tmp_path)

    cmd = [
        sys.executable,
        str(MAIN_PATH),
        "--data",
        "suzanne",
        "--resolution",
        "64",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        check=True,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        env=env,
    )


def test_integration_static_mesh_render(tmp_path: Path) -> None:
    out_path = tmp_path / "mesh.png"
    _run_main_with_external_bpy(tmp_path, ["--save", str(out_path)])

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_integration_depth_to_pointcloud(tmp_path: Path) -> None:
    out_path = tmp_path / "depth.png"
    _run_main_with_external_bpy(
        tmp_path,
        ["--depth", "ray_trace", "--save", str(out_path)],
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
