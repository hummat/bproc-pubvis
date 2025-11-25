from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Opt-in: require explicit env flag; assumes BlenderProc is correctly configured
pytestmark = pytest.mark.skipif(
    not os.getenv("BPROC_INTEGRATION"),
    reason="Set BPROC_INTEGRATION=1 to run BlenderProc integration tests.",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
BLENDERPROC_BIN = shutil.which(os.getenv("BLENDERPROC_BIN", "blenderproc"))
EXAMPLES: list[tuple[str, str, list[str]]] = [
    ("mesh", "png", []),
    ("pcd", "png", ["--pcd", "True", "--light", "very_bright"]),
    ("mesh_depth", "png", ["--depth", "ray_trace", "--pcd", "1024", "--keep-mesh", "--point-size", "0.01"]),
    ("depth", "png", ["--depth", "ray_trace", "--pcd", "True", "--point-size", "0.01"]),
    ("mesh_color", "png", ["--color", "bright_blue"]),
    ("pcd_color", "png", ["--pcd", "True", "--color", "cool"]),
    ("bg_color", "png", ["--bg-color", "pale_turquoise"]),
    ("backdrop", "png", ["--transparent", "False"]),
    ("backdrop_colored", "png", ["--transparent", "False", "--bg-color", "pale_red"]),
    ("hdri", "png", ["--transparent", "False"]),  # backdrop path injected at runtime
    ("very_dark", "png", ["--light", "very_dark"]),
    ("dark", "png", ["--light", "dark"]),
    ("medium", "png", ["--light", "medium"]),
    ("shadow_soft", "png", ["--shadow", "soft"]),
    ("shadow_hard", "png", ["--shadow", "hard"]),
    ("noshadow", "png", ["--shadow", "off"]),
    ("smooth", "png", ["--shade", "smooth"]),
    ("auto-smooth", "png", ["--shade", "auto"]),
    ("wireframe", "png", ["--wireframe", "True"]),
    ("wireframe_mesh", "png", ["--wireframe", "True", "--keep-mesh"]),
    ("wireframe_mesh_white", "png", ["--wireframe", "white", "--keep-mesh"]),
    ("wireframe_mesh_color", "png", ["--wireframe", "red", "--keep-mesh"]),
    ("outline", "png", ["--outline", "True"]),
    ("gravity", "png", ["--gravity"]),
    ("turn", "gif", ["--animate", "turn"]),
    ("tumble", "gif", ["--animate", "tumble"]),
]


@pytest.fixture(autouse=True)
def _require_blenderproc_bin() -> None:
    if os.getenv("BPROC_INTEGRATION") and BLENDERPROC_BIN is None:  # pragma: no cover - skip logic
        pytest.skip("blenderproc binary not found in PATH; set BLENDERPROC_BIN or adjust PATH.")


def _run_main_with_blenderproc(
    tmp_path: Path, extra_args: list[str], resolution: int = 64
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BPROC_TEMP_DIR"] = str(tmp_path)

    cmd = [
        BLENDERPROC_BIN or "blenderproc",
        "run",
        "--temp-dir",
        str(tmp_path),
        str(MAIN_PATH),
        "--data",
        "suzanne",
        "--resolution",
        str(resolution),
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
    _run_main_with_blenderproc(tmp_path, ["--save", str(out_path)])

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_integration_depth_to_pointcloud(tmp_path: Path) -> None:
    out_path = tmp_path / "depth.png"
    _run_main_with_blenderproc(
        tmp_path,
        ["--depth", "ray_trace", "--save", str(out_path)],
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.fixture
def readme_gallery_config() -> dict[str, object]:
    output_dir = Path(os.getenv("BPROC_EXAMPLES_OUT", PROJECT_ROOT / "examples"))
    output_dir.mkdir(parents=True, exist_ok=True)

    haven_dir = Path(os.getenv("BPROC_HAVEN_DIR", ""))

    return {
        "output_dir": output_dir,
        "resolution": int(os.getenv("BPROC_README_RES", "512")),
        "frames": int(os.getenv("BPROC_README_FRAMES", "72")),
        "haven_dir": haven_dir,
        "have_hdri": (haven_dir / "hdris").exists(),
    }


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e[0])
@pytest.mark.skipif(
    not os.getenv("BPROC_EXAMPLES"),
    reason="Set BPROC_EXAMPLES=1 to rebuild the README gallery.",
)
def test_integration_readme_gallery(
    example: tuple[str, str, list[str]], tmp_path: Path, readme_gallery_config: dict[str, object]
) -> None:
    """Rebuild each README asset individually (opt-in via BPROC_EXAMPLES)."""

    name, ext, args = example
    output_dir: Path = readme_gallery_config["output_dir"]  # type: ignore[assignment]
    resolution: int = readme_gallery_config["resolution"]  # type: ignore[assignment]
    frames: int = readme_gallery_config["frames"]  # type: ignore[assignment]
    haven_dir: Path = readme_gallery_config["haven_dir"]  # type: ignore[assignment]
    have_hdri: bool = readme_gallery_config["have_hdri"]  # type: ignore[assignment]

    if name == "hdri":
        if not have_hdri:
            pytest.skip("HDRI directory not found; set BPROC_HAVEN_DIR to enable HDRI example.")
        args = [*args, "--backdrop", str(haven_dir)]

    out_path = output_dir / f"{name}.{ext}"
    full_args = [*args, "--save", str(out_path)]
    if name in {"turn", "tumble"}:
        full_args.extend(["--frames", str(frames)])
        resolution = resolution // 2  # Animations rendered at half-res to save space/time

    _run_main_with_blenderproc(tmp_path, full_args, resolution=resolution)

    assert out_path.exists(), f"{name} output missing"
    assert out_path.stat().st_size > 0, f"{name} output empty"
