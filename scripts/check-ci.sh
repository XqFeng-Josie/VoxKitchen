#!/usr/bin/env bash
# Run the same fast checks that gate normal GitHub CI.

set -euo pipefail

log() { printf "\n\033[1;34m==>\033[0m %s\n" "$*"; }

log "Ruff lint"
ruff check voxkitchen tests

log "Ruff format check"
ruff format --check voxkitchen tests

log "Mypy"
mypy voxkitchen

log "Pytest"
pytest -q -m "not slow and not gpu" \
    --cov=voxkitchen --cov-report=term-missing \
    --deselect tests/unit/operators/pack/test_pack_huggingface.py \
    --deselect tests/unit/operators/pack/test_pack_parquet.py \
    --deselect tests/unit/viz/test_report.py

log "Local CI checks passed."
