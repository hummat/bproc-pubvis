# Repository Guidelines

This document describes how to work with `bproc-pubvis` as a contributor.

## Project Structure & Module Organization
- Root: `main.py` is the entry script used with `blenderproc run main.py ...`.
- Library code: `bproc_pubvis/constants.py` and `bproc_pubvis/utils.py` hold reusable logic; treat them as the Python package core.
- Examples: `examples/` contains rendered example outputs; use this for visual regression checks, not as source.
- Packaging: `pyproject.toml` defines build metadata; avoid editing it unless changing dependencies or releases.

## Build, Test, and Development Commands
- Create/use project-local venv with `uv` (recommended): `uv sync --group dev` (creates `.venv` and installs app + dev deps).
- Alternatively, plain venv: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`.
- Run via BlenderProc: `blenderproc run main.py path/to/obj --save out.png` is the primary execution path.
- Debug scenes: `blenderproc debug main.py ...` opens Blender for interactive inspection.
- Get CLI help from Tyro via pass-through: `blenderproc run main.py -- --help` (double dash forwards args past BlenderProc).
- Always run the full tooling after making changes:
  - Format: `uv run ruff format .`
  - Lint: `uv run ruff check .`
  - Type check: `uv run pyright`
  - Tests: `uv run pytest` (uses `--cov` by default via `pyproject.toml`; set `BPROC_INTEGRATION=1` to include integration tests)
- Keep `import blenderproc as bproc` as the first import in `main.py` (required by BlenderProc CLI); ruff is silenced with `# noqa: I001,F401`.

## Coding Style & Naming Conventions
- Use Python 3.11+, 4‑space indentation, and type hints where they add clarity.
- Prefer descriptive, lowercase_with_underscores for variables/functions and UPPER_SNAKE_CASE for constants (see `bproc_pubvis/constants.py`).
- Keep functions in `bproc_pubvis/utils.py` small and composable; avoid Blender-specific logic in `main.py` when it can live in `bproc_pubvis/`.

## Testing Guidelines
- Tests live in `tests/`, named `test_*.py`. Unit tests exercise `main.py` and enums; integration tests live in `tests/test_integration.py`.
- Run unit tests + coverage: `pytest` (integration tests are skipped by default).
- Run BlenderProc integration tests (requires the `blenderproc` CLI on `PATH`): \
  `BPROC_INTEGRATION=1 pytest tests/test_integration.py` \
  (tests call `blenderproc run main.py --temp-dir <tmp> --data suzanne ...` under the hood).
- Rebuild README/example renders (heavy, GPU-friendly): \
  `BPROC_INTEGRATION=1 BPROC_EXAMPLES=1 pytest tests/test_integration.py -k readme_gallery` \
  Optional knobs: `BPROC_HAVEN_DIR` for HAVEN dataset HDRIs, `BPROC_EXAMPLES_OUT` for output dir, `BPROC_README_RES` / `BPROC_README_FRAMES` for size/length.
- For visual behavior beyond integration tests, prefer example CLI snippets and screenshots in the README over pixel-perfect assertions.

## Tooling & Environment Notes
- Pyright is configured for `typeCheckingMode = "basic"` and checks both `main.py` and `bproc_pubvis/utils.py`. Blender/BlenderProc/bpy-heavy code may still emit `reportMissingImports` warnings if those packages are not installed in the active venv.
- To improve editor support and reduce `bpy`-related noise, install `fake-bpy-module-*` into the same Python environment used for this project.
- BlenderProc temporary directories can be redirected by setting `BPROC_TEMP_DIR=/some/writable/dir` (integration tests set this automatically to their temp directory).
- If `/dev/shm` is not writable (common on some hosts), run BlenderProc with an explicit temp dir: `blenderproc run --temp-dir /tmp/bproc_temp main.py ...`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., `Add wireframe color option`, `Fix depth rendering docs`).
- Reference related GitHub issues in commits or PR descriptions using `Fixes #123` when applicable.
- PRs should: describe the change and motivation, mention any new CLI options, include usage examples, and attach example renders when visual output changes.

## AI Agent Quick Map (bproc-pubvis)

Use this section as the first stop when reasoning about the repo; it’s designed so future agents don’t have to re‑scan everything.

- **Entry script & CLI**
  - `main.py`: single Tyro dataclass `Config` → `run(cfg)` → `main()`.
  - Typical invocation: `blenderproc run main.py --data <mesh_or_primitive> [options...]`.
  - `Config` fields map 1:1 to CLI flags; most behavior ultimately calls helpers in `bproc_pubvis/utils.py` and enums in `bproc_pubvis/constants.py`.

- **Core library**
  - `bproc_pubvis/constants.py`: all enums and presets:
    - Visual settings: `Color`, `Shading`, `Shape`, `Look`, `Shadow`, `Engine`, `Light`.
    - Scene/behavior: `Primitive`, `Animation`, `Strength`.
  - `bproc_pubvis/utils.py`: BlenderProc/bpy integration layer:
    - Renderer/camera: `init_renderer`, `make_camera`, `render_color`, `render_depth`, `make_animation`.
    - Scene setup: `load_data`, `make_obj`, `setup_obj`, `setup_backdrop`, `make_lights`, `add_ambient_occlusion`.
    - Geometry/pcd helpers: `_target_count`, `_subsample_indices`, `init_mesh`, `init_pointcloud`, `depth_to_image`.
    - Color/background: `get_color`, `set_color`, `set_background_color`, `set_look`.
    - Export: `export_obj`, `create_mp4_with_ffmpeg`.

- **Run-time pipeline (very condensed)**
  - `run(cfg)` in `main.py`:
    1. Seed RNGs, configure `loguru` logging.
    2. Call `init_renderer(...)` with resolution, transparency, engine, noise/samples, and initial `look`/`exposure`.
    3. Call `setup_obj(...)` → creates mesh/pcd objects from `cfg.data` and options (`pcd`, `wireframe`, `shade`, `color`, `subsample*`).
    4. Call `make_camera(...)` → single camera pose; frame range forced to `[0, 1]` to avoid double renders.
    5. Call `set_look(...)` → choose contrast preset based on `look`, `pcd`, `depth`, `color`.
    6. Optional depth stage via `render_depth(...)` when `cfg.depth` is set (supports depth→pcd conversion and `keep_mesh`).
    7. Determine `gravity` (auto‑disabled for pure point clouds or tumble animations) and compute backdrop offset.
    8. Resolve light intensity from `cfg.light` (float, string, or `Light` enum).
    9. Call `setup_backdrop(...)` when `gravity` or `backdrop` is enabled (supports HDRI via `--backdrop` and `BPROC_TEMP_DIR`).
    10. Configure AO via `add_ambient_occlusion(...)` based on `cfg.ao` and whether the object is a mesh/depth.
    11. Configure scene lighting via `make_lights(...)` with `Shadow` preset and light intensity.
    12. Depending on `cfg.animate`:
        - If animation: `make_animation(...)` → GIF/MP4; no `render_color`.
        - Else: `render_color(...)` once unless `cfg.debug` is set.
    13. Optional mesh export via `export_obj(...)` when `cfg.export` is provided.

- **BlenderProc integration assumptions**
  - Invocation is through `blenderproc run`, so `bproc.init(...)` is available and called in `init_renderer`.
  - Expects BlenderProc 2.8.0 APIs:
    - `bproc.renderer.*`, `bproc.camera.*`, `bproc.object.*`, `bproc.world.*`, `bproc.types.*`, `bproc.utility.*`.
  - Temp directory is controlled via `BPROC_TEMP_DIR` and passed into `bproc.init(temp_dir=...)` when supported.

- **Tests and what they guarantee**
  - `tests/test_main.py`: stubs all high‑level helpers and asserts wiring/branching in `run(cfg)` and `main()`.
  - `tests/test_utils_helpers.py`: validates `_target_count`, `_subsample_indices`, `normalize`, `depth_to_image`, `get_color`, `set_background_color`.
  - `tests/test_smoke.py`: smoke‑tests `bproc_pubvis.constants.Color`.
  - `tests/test_integration.py` (opt‑in via `BPROC_INTEGRATION=1`): sanity‑checks a real BlenderProc CLI render and a depth render.

- **When you need deeper details**
  - For BlenderProc behavior: follow imports from `bproc_pubvis/utils.py` into the local `../BlenderProc` clone (e.g. `blenderproc/api/*`, `blenderproc/python/*`).
  - For user‑facing behavior/CLI examples: see `README.md` plus image outputs under `examples/`.
