# Project Roadmap & Contributing

This repo is primarily a personal research project and portfolio piece. While I am not actively looking to build a massive open-source community around this, I am completely open to discussions, forks, and PRs—especially if you are working on parimutuel pricing yourself!

If you do want to poke around or suggest changes, here is my philosophy for this codebase.

## Development setup

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/hkjc-benter-engine.git
cd hkjc-benter-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env                # edit with real values
docker compose up -d
psql "$HKJC_DB_URL" -f sql/migrations/001_phase_columns.sql
```

## What I'd welcome PRs for

- **Drift-aware retraining**: a version of `models/trainer_residual.py` that uses STOP_SELL-time odds (from `live_odds_history`) as the `base_margin` anchor instead of final odds. This is the single most impactful correctness fix and it's a meaningful amount of work.
- **Unit tests.** There are currently none. The highest-value tests are for `betting_policy.py` (Kelly + caps math), `ensemble.BenterLogLinearStacker`, and the Harville order-probability functions in `live/run_bot.py`.
- **Cleaner backtest reporting.** The current `walk_forward_master_ledger.csv` has everything needed for a tear sheet but no automated rendering beyond the React dashboard.
- **CI pipeline.** GitHub Actions running linting + the test suite on every PR.

## What I will probably decline

- **Strategy changes without backtests.** "I think you should do X differently" is a conversation, not a PR. Show me walk-forward OOF log-loss before and after.
- **Bullish claims.** The backtest is overly optimistic due to late-money drift (see [Known Caveats](README.md#known-caveats)). I will not merge changes that further inflate the headline numbers without addressing the training-execution timing mismatch.

## Code style

- Follow existing patterns. The data/models/live split is intentional.
- Use fully-qualified imports (`from hkjc_engine.models.ensemble import ...`), never relative imports across package subdivisions.
- Secrets come from `hkjc_engine.config`. If you need a new secret, add it there with a sensible default, document it in `.env.example`, and reference it in `SECURITY.md`.
