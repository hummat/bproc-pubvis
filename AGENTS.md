# Repository Guidelines

This document describes how to work with `bproc-pubvis` as a contributor.

## Project Structure & Module Organization
- Root: `main.py` is the entry script used with `blenderproc run main.py ...`.
- Library code: `src/constants.py` and `src/utils.py` hold reusable logic; treat them as the Python package core.
- Examples: `examples/` contains rendered example outputs; use this for visual regression checks, not as source.
- Packaging: `pyproject.toml` defines build metadata; avoid editing it unless changing dependencies or releases.

## Build, Test, and Development Commands
- Create/use project-local venv with `uv` (recommended): `uv sync --group dev` (creates `.venv` and installs app + dev deps).
- Alternatively, plain venv: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`.
- Run via BlenderProc: `blenderproc run main.py path/to/obj --save out.png` is the primary execution path.
- Debug scenes: `blenderproc debug main.py ...` opens Blender for interactive inspection.
- Lint: `uv run --no-sync ruff check .` (or `ruff check .`)
- Format: `uv run --no-sync ruff format .`
- Type check: `uv run --no-sync pyright` (or `pyright`)
- Tests: `uv run --no-sync pytest` (uses `--cov` by default via `pyproject.toml`)
- When using external BPY (`USE_EXTERNAL_BPY_MODULE=1`), run `python main.py ...` directly; integration tests do this automatically.
- Note: the Linux `bpy==4.2.0` wheel is tagged `cp39`, so `uv run` with syncing will uninstall/reinstall it each time. Use `--no-sync` for dev commands unless you intentionally refresh dependencies.

## Coding Style & Naming Conventions
- Use Python 3.11+, 4‑space indentation, and type hints where they add clarity.
- Prefer descriptive, lowercase_with_underscores for variables/functions and UPPER_SNAKE_CASE for constants (see `src/constants.py`).
- Keep functions in `src/utils.py` small and composable; avoid Blender-specific logic in `main.py` when it can live in `src/`.

## Testing Guidelines
- Tests live in `tests/`, named `test_*.py`. Unit tests exercise `main.py` and enums; integration tests live in `tests/test_integration.py`.
- Run unit tests + coverage: `pytest` (integration tests are skipped by default).
- Run BlenderProc integration tests (requires BlenderProc + Blender with external BPY): \
  `BPROC_INTEGRATION=1 pytest tests/test_integration.py` \
  (tests call `USE_EXTERNAL_BPY_MODULE=1 python main.py --data suzanne ...` under the hood).
- For visual behavior beyond integration tests, prefer example CLI snippets and screenshots in the README over pixel-perfect assertions.

## Tooling & Environment Notes
- Pyright is configured for `typeCheckingMode = "basic"` and checks both `main.py` and `src/utils.py`. Blender/BlenderProc/bpy-heavy code may still emit `reportMissingImports` warnings if those packages are not installed in the active venv.
- To improve editor support and reduce `bpy`-related noise, install `fake-bpy-module-*` into the same Python environment used for this project.
- BlenderProc temporary directories can be redirected by setting `BPROC_TEMP_DIR=/some/writable/dir` (integration tests set this automatically to their temp directory).
- If `/dev/shm` is not writable (common on some hosts), run BlenderProc with an explicit temp dir: `blenderproc run --temp-dir /tmp/bproc_temp main.py ...`.
- External `bpy` / BlenderProc notes (important for integration tests and headless runs):
  - When `USE_EXTERNAL_BPY_MODULE=1`, prefer a sibling `../BlenderProc` source checkout; `src/utils.py` prepends it to `sys.path` so we avoid the PyPI wheel’s “blenderproc run only” guard.
  - Always set `BPROC_TEMP_DIR` to a writable path (e.g., `/tmp/bproc_temp`) to avoid `/dev/shm` permission errors.
  - Camera setup now enforces a single pose (`frame_start=0`, `frame_end=1`) to prevent double renders.
  - `--show` is auto-disabled when `USE_EXTERNAL_BPY_MODULE=1` unless explicitly passed; avoids GUI popups/segfaults headless.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., `Add wireframe color option`, `Fix depth rendering docs`).
- Reference related GitHub issues in commits or PR descriptions using `Fixes #123` when applicable.
- PRs should: describe the change and motivation, mention any new CLI options, include usage examples, and attach example renders when visual output changes.
