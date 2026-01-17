# Contributing to bproc-pubvis

Thanks for your interest in contributing! This document covers development setup and guidelines.

## Development Setup

### Prerequisites

- Python 3.11 (required, <3.12)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- BlenderProc 2.8.0+

### Quick Start

```bash
# Clone the repository
git clone https://github.com/hummat/bproc-pubvis.git
cd bproc-pubvis

# Install development dependencies
uv sync --group dev

# Run all checks
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```

### Running Tests

```bash
# Run unit tests
uv run pytest

# Run specific test
uv run pytest tests/ -k test_name -v

# Run integration tests (requires blenderproc CLI)
BPROC_INTEGRATION=1 uv run pytest tests/test_integration.py
```

## Code Style

### Python

- Python 3.11 required
- 120-character line limit (see `[tool.ruff]` in pyproject.toml)
- Type hints for function signatures
- Run `ruff format .` and `ruff check .` before committing
- Run `pyright` for type checking

### Workflow

1. Read files before editing — understand existing code
2. After changes: `uv run ruff format .` → `uv run ruff check .` → `uv run pyright` → `uv run pytest`
3. Check if docs need updating

## Architecture

- `main.py` (root): Wrapper ensuring package importability
- `bproc_pubvis/main.py`: Core entry with Tyro `Config` dataclass
- `bproc_pubvis/constants.py`: All enums (Color, Shading, Shape, etc.)
- `bproc_pubvis/utils.py`: BlenderProc/bpy integration

### Key Constraints

- **BlenderProc first import**: Keep `import blenderproc as bproc` as first import in `main.py`
- **Python version**: 3.11+ required (constrained to <3.12)
- **Temp directory**: Set `BPROC_TEMP_DIR` if `/dev/shm` is not writable

## Pull Request Process

1. **Create an issue first** for non-trivial changes
2. **Fork and branch** from `main`
3. **Make your changes** following the style guide
4. **Run all checks** — format, lint, typecheck, test
5. **Update documentation** if needed
6. **Submit PR** using the template

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Keep the first line under 72 characters
- Reference issues: "Fix rendering crash (#42)"
- Optionally prefix with type: `feat:`, `fix:`, `docs:`, `refactor:`

## What to Contribute

### Good First Issues

- Documentation improvements
- Adding test coverage
- Bug fixes with clear reproduction steps

### Feature Ideas

- New shading/lighting presets
- Additional export formats
- Animation improvements

### Before Starting Large Features

Please open an issue first to discuss the approach. This helps avoid duplicate work and ensures the feature aligns with project goals.

## Questions?

- Open a [Discussion](https://github.com/hummat/bproc-pubvis/discussions) for questions
- Check existing [Issues](https://github.com/hummat/bproc-pubvis/issues) for known problems
