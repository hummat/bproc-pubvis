#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv build --sdist --wheel --out-dir dist --clear
  exit 0
fi

python -m build --sdist --wheel --outdir dist
