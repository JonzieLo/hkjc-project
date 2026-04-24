#!/usr/bin/env bash
# Launch all three live services for a race day, each in its own named tmux window.
# Requires: tmux, .env with HKJC_LIVE_VENUE set.
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="hkjc_live"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n scraper "python -m hkjc_engine.live.orchestrator"
tmux new-window -t "$SESSION:1" -n archiver "python -m hkjc_engine.live.odds_archiver \$HKJC_LIVE_VENUE"
tmux new-window -t "$SESSION:2" -n bot "python -m hkjc_engine.live.run_bot"

echo "Live session started. Attach with: tmux attach -t $SESSION"
