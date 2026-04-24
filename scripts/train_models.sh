#!/usr/bin/env bash
# Run the full walk-forward training pipeline.
# Outputs trained models and the master ledger into ./artifacts/.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m hkjc_engine.models.walk_forward
