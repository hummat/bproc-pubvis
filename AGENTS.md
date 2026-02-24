# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Conventions

Read relevant `docs/agent/` files before proceeding:
- `workflow.md` — **read before creating issues, PRs, or branches** (templates, labels, naming)
- `releases.md` — **read before releasing** (changelog, versioning, conventional commits)

---

## Project Overview

**bproc-pubvis** generates publication-ready 3D visualizations of meshes and point clouds using BlenderProc/Blender. Primary invocation: `uv run blenderproc run main.py --data <mesh_or_primitive> [options...]`

## Development Commands

Requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Setup
make deps                              # uv sync --group dev + git hooks

# Run application
uv run blenderproc run main.py --data path/to/obj --save out.png
uv run blenderproc debug main.py ...   # Opens Blender GUI for inspection
uv run blenderproc run main.py -- --help  # CLI help (double-dash passes to Tyro)

# Quality checks (run all after changes)
make check                             # fmt + lint + type + test (recommended)
uv run ruff format .                   # Format only
uv run ruff check .                    # Lint only
uv run pyright                         # Type check only
uv run pytest                          # Unit tests with coverage

# Integration tests (opt-in, requires blenderproc CLI)
BPROC_INTEGRATION=1 uv run pytest tests/test_integration.py

# Single test
uv run pytest tests/test_main.py::test_run_basic -v

# Rebuild example gallery (GPU-friendly)
BPROC_INTEGRATION=1 BPROC_EXAMPLES=1 uv run pytest tests/test_integration.py -k readme_gallery

# Build and release
make build                             # Build sdist + wheel
make release                           # Full release workflow (see docs/agent/releases.md)
```

## Architecture

### Entry Points
- `main.py` (root): Wrapper ensuring package importability; re-exports from `bproc_pubvis.main`
- `bproc_pubvis/main.py`: Core entry with Tyro `Config` dataclass → `run(cfg)` → `main()`

### Core Modules
- `bproc_pubvis/constants.py`: All enums (`Color`, `Shading`, `Shape`, `Look`, `Shadow`, `Engine`, `Primitive`, `Animation`, `Light`, `Strength`)
- `bproc_pubvis/utils.py`: BlenderProc/bpy integration (~1400 lines)
  - Renderer/camera: `init_renderer`, `make_camera`, `render_color`, `render_depth`, `make_animation`
  - Scene setup: `load_data`, `make_obj`, `setup_obj`, `setup_backdrop`, `make_lights`
  - Point cloud: `init_pointcloud`, `_target_count`, `_subsample_indices`
  - Color/material: `get_color`, `set_color`, `set_background_color`, `set_look`

### Pipeline Flow (`run()` in main.py)
1. Seed RNGs, configure logging
2. `init_renderer()` → resolution, transparency, engine, denoising
3. `setup_obj()` → load mesh/pcd, apply materials/colors
4. `make_camera()` → position camera
5. `set_look()` → contrast preset based on data type
6. [Optional] `render_depth()` → depth image or depth→pcd conversion
7. [Conditional] `setup_backdrop()` → gravity physics, HDRI, backdrop plane
8. `add_ambient_occlusion()`, `make_lights()`
9. `make_animation()` or `render_color()`
10. [Optional] `export_obj()`

## Key Constraints

- **BlenderProc first import**: Keep `import blenderproc as bproc` as the first import in `main.py`; ruff silenced with `# noqa: I001,F401`
- **Temp directory**: Set `BPROC_TEMP_DIR` env var if `/dev/shm` is not writable
- **Python version**: 3.11+ required (constrained to <3.12)
- **BlenderProc APIs**: Expects 2.8.0+ (`bproc.renderer.*`, `bproc.camera.*`, `bproc.object.*`, `bproc.world.*`)

## Testing Strategy

| Test File | Purpose |
|-----------|---------|
| `test_main.py` | Unit tests for `run()` pipeline branching (mocks helpers) |
| `test_utils_helpers.py` | Helper functions: `_target_count`, `_subsample_indices`, colors |
| `test_smoke.py` | Minimal smoke test (enum exists) |
| `test_integration.py` | End-to-end BlenderProc CLI (opt-in: `BPROC_INTEGRATION=1`) |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `BPROC_INTEGRATION` | Enable integration tests |
| `BPROC_EXAMPLES` | Rebuild gallery assets |
| `BPROC_TEMP_DIR` | Override BlenderProc temp directory |
| `BPROC_HAVEN_DIR` | Path to HAVEN dataset for HDRI tests |
| `BPROC_EXAMPLES_OUT` | Output directory for gallery |
| `BPROC_README_RES` | Render resolution (default 512) |
| `BPROC_README_FRAMES` | Animation length (default 72) |

## Code Organization Guidelines

- Keep Blender-specific logic in `bproc_pubvis/utils.py`, not in `main.py`
- New presets/enums go in `bproc_pubvis/constants.py`
- `Config` fields map 1:1 to CLI flags
- Functions in `utils.py` should be small and composable
