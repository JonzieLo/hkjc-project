#!/usr/bin/env bash
# Run the ensemble backtester standalone (requires trained artifacts).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m hkjc_engine.models.backtester
