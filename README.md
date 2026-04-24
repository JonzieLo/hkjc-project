# HKJC Benter Engine

A parimutuel pricing engine for the Hong Kong Jockey Club (HKJC). Combines a dual XGBoost ensemble, Beta calibration, and a Benter log-linear stacker to generate win and exotic-pool probabilities, then sizes bets via fractional Kelly with Smoczyński-Tomkins simultaneous-stake caps.

> **Disclaimer.** This is research code shared for educational and archival purposes. Backtested returns are out-of-sample but walk-forward backtests still systematically overstate live performance — see the [Known Caveats](#known-caveats) section before drawing any conclusions. Gambling carries real financial risk, and parimutuel markets with 17.5% track takeout are particularly unforgiving. Do not deploy this code with money you cannot afford to lose.

---

## Walk-Forward Backtest Results

| Metric            | Value               |
|-------------------|---------------------|
| Period            | 2024-01 → 2026-02 (25 monthly windows) |
| Bets placed       | 80                  |
| Total staked      | $70,658             |
| Net profit        | +$30,212            |
| Win rate          | 17.5%               |
| ROI               | +42.76%             |
| Starting bankroll | $100,000            |
| Ending bankroll   | $130,212.50         |
| OOF log-loss      | 0.23800 (vs 0.23902 public consensus) |
| θ₂, θ₃ (Henery)   | 0.8824, 0.7760 (joint MLE) |

These numbers reflect idealized execution against final settled dividends. See [Known Caveats](#known-caveats) for how much of this is inflated by late-money drift.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │  HKJC Racing Data (Postgres) │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │    HKJCFeatureFactory     │
                     │  (pace EMAs, TrueSkill,   │
                     │   class / track / draw)   │
                     └───┬──────────────────┬────┘
                         │                  │
             ┌───────────▼────────┐  ┌──────▼──────────────┐
             │ Model A: Residual  │  │ Model B: Physics    │
             │  rank:pairwise     │  │  grouped softmax    │
             │  (anchored to mkt) │  │  (conditional logit)│
             └───────────┬────────┘  └──────┬──────────────┘
                         │                  │
             ┌───────────▼────────┐  ┌──────▼──────────────┐
             │ Beta Calibrator A  │  │ Beta Calibrator B   │
             └───────────┬────────┘  └──────┬──────────────┘
                         │                  │
                         └──┬───────────────┘
                            │            ┌──────────────────┐
                            │      ┌────▶│ Public market P  │
                            ▼      │     │  (1/d normalized)│
               ┌────────────────────────┐└────────────┬─────┘
               │  Benter Log-Linear     │  (L-BFGS-B weights
               │  Stacker               │   on OOF log-loss)
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Henery discounted      │  θ₂=0.8824, θ₃=0.7760
               │ Harville for exotics   │  (joint MLE)
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Betting policy:        │
               │  • Baker-McHale hurdle │
               │  • Shrinkage (c ≈ 0.75)│
               │  • α=0.25 Kelly        │
               │  • Tiered ST caps      │
               └────────────┬───────────┘
                            │
                            ▼
                        Live bets
```

---

## Repository Layout

```
hkjc-benter-engine/
├── src/hkjc_engine/
│   ├── config.py                    # Env-var configuration (secrets)
│   ├── data/                        # Ingestion + feature engineering
│   │   ├── scraper_dividends.py
│   │   ├── scraper_horse_numbers.py
│   │   ├── patch_{finish_position,race_class,race_win_odds}.py
│   │   ├── compute_{pace,trueskill,class_and_track}.py
│   │   ├── add_features.py
│   │   ├── update_run_styles.py
│   │   └── daily_updater.py         # Pipeline orchestrator
│   ├── models/                      # Training + backtesting
│   │   ├── feature_factory.py
│   │   ├── ensemble.py              # BetaCalibrator, Benter stacker
│   │   ├── betting_policy.py        # Kelly + hurdle + caps
│   │   ├── theta_optimizer.py       # Henery θ MLE
│   │   ├── trainer_residual.py      # Model A
│   │   ├── trainer_independent.py   # Model B
│   │   ├── backtester.py
│   │   └── walk_forward.py
│   ├── live/                        # Race-day production
│   │   ├── scraper.py               # Playwright HKJC GraphQL interceptor
│   │   ├── orchestrator.py          # Multi-race subprocess launcher
│   │   ├── odds_archiver.py         # Redis → Postgres persistence
│   │   ├── predictor.py             # LiveRacePredictor (5 artifacts)
│   │   └── run_bot.py               # Main live bot (Discord webhook)
│   └── diagnostics/
│       └── drift_diagnostic.py      # Late-money drift analysis
├── sql/migrations/
│   └── 001_phase_columns.sql
├── docker-compose.yml               # Postgres + Redis
├── requirements.txt
├── pyproject.toml
├── .env.example
└── LICENSE
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/hkjc-benter-engine.git
cd hkjc-benter-engine

python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -e .                   # or: pip install -r requirements.txt
playwright install chromium        # one-time, for the live scraper
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and set HKJC_DB_PASSWORD and (optionally) HKJC_DISCORD_WEBHOOK
```

### 3. Start infrastructure

```bash
docker compose up -d              # Postgres + Redis
psql "$HKJC_DB_URL" -f sql/migrations/001_phase_columns.sql
```

### 4. Ingest a race day

```bash
python -m hkjc_engine.data.daily_updater 2026-04-22
```

### 5. Train models (walk-forward)

```bash
python -m hkjc_engine.models.walk_forward
```

This trains Model A, Model B, fits the Benter stacker for each monthly window, computes dynamic shrinkage, and writes `walk_forward_master_ledger.csv` plus `live_config.json` into `./artifacts/`.

### 6. Run the live bot on race day

```bash
# Terminal 1 — scrape the card (sequentially per race)
python -m hkjc_engine.live.orchestrator

# Terminal 2 — archive odds snapshots to Postgres
python -m hkjc_engine.live.odds_archiver HV

# Terminal 3 — pricing + betting signals
python -m hkjc_engine.live.run_bot
```

The bot reads `live_config.json` for shrinkage and fires Discord alerts (or stdout if `HKJC_DISCORD_WEBHOOK` is unset) as each race enters `STOP_SELL`.

---

## How the Pipeline Works

### Feature Factory (`models/feature_factory.py`)

Converts raw race entries into model-ready features: distance-bucketed pace EMAs, TrueSkill μ/σ updated after every race, draw × early-pace interactions, class-drop flags, track geometry (rail × straight length × width), days since last race, jockey α from a prior MCMC fit, and residual-target encoding for Model A.

### Model A — Market-Residual Learner (`models/trainer_residual.py`)

XGBoost with `objective=rank:pairwise` and `eval_metric=ndcg`. The public win odds are injected as `base_margin`, so the model is learning deviations from the market rather than win probability from scratch. This is the core architectural trick behind Benter's original syndicate: the public consensus already encodes most of the truth, so the model's job is to find the 5-10% of mispricing at the margins. Out-of-fold log-loss beats the public baseline by ~0.001, which compounds into real edge through the stacker.

### Model B — Physics Learner (`models/trainer_independent.py`)

XGBoost with a custom grouped-softmax objective (`ensemble.GroupedSoftmaxObjective`) implementing race-level conditional logit. Each race is a group; the gradient enforces Σ P = 1 per race natively. Features focus on physics (pace, class interactions, draw × geometry) rather than odds-anchored signals. Independent error structure from Model A is what allows the log-linear pool to extract real lift.

### Beta Calibration (`models/ensemble.py` → `BetaCalibrator`)

Implements Kull, Silva Filho & Flach (2017). Replaces isotonic regression. A three-parameter sigmoid `σ(a·log(s) − b·log(1−s) + c)` fit via logistic regression. Enforces monotonicity (falls back to `a_only` or `b_only` modes when the full fit is non-monotone). Handles both raw model outputs with correct tail behavior and extrapolation beyond the training distribution.

### Benter Log-Linear Stacker (`models/ensemble.py` → `BenterLogLinearStacker`)

Fits weights `(w_A, w_B, w_mkt)` on out-of-fold predictions by minimizing log-loss of `softmax(Σ wᵢ · log Pᵢ)` per race. L-BFGS-B with bounds `[0, 3]`. Multiplicative fusion in log-space (Genest & Zidek 1986) extracts conditionally-independent information from all three sources. Unlike convex blending, a well-specified log-linear pool can beat the best individual component — which it does here, by about 0.001 log-loss units.

### Henery Discounted Harville (`live/run_bot.py`)

For exotic pools (QIN/QPL/TRI), converts win probabilities into order probabilities via:

```
P(i 1st, j 2nd) = p_i · (p_j^θ₂ / Σ_{k≠i} p_k^θ₂)
P(i, j, k chain) = above × (p_k^θ₃ / Σ_{m≠i,j} p_m^θ₃)
```

`θ₂ = 0.8824` and `θ₃ = 0.7760` are calibrated by joint MLE on 50k+ historical finishing orders (see `models/theta_optimizer.py`). Values near 1.0 would reduce to standard Harville, which systematically overprices longshot exotics because it assumes a horse's 2nd-place probability is just its win probability conditional on not winning — empirically the tail is steeper, which is what θ < 1 captures.

### Betting Policy (`models/betting_policy.py`)

- **Shrinkage**: multiply `P_model` by `c = Σy / Σŷ` from the most recent walk-forward window (typically ~0.75)
- **Hurdle**: require `EV ≥ base + longshot_buffer · I[odds ≥ 15]` (Baker-McHale 2013)
- **Kelly**: `f* = (p_adj · d − 1) / (d − 1)`, multiplied by `α = 0.25` (MacLean-Ziemba)
- **Caps**: per-bet 2%, per-pool 3%, WIN race 4%, master cross-pool 6%

The tiered caps are a practical approximation of the Smoczyński-Tomkins (2010) exact simultaneous-Kelly solution. The approximation always under-stakes vs. exact ST, preserving relative conviction ordering without numerical fragility.

### Live Scraper State Machine (`live/scraper.py`)

Playwright-based GraphQL interceptor. Passive — doesn't click anything except the trifecta "All" button to expand the matrix. State machine:

1. **Pre-STOP_SELL**: poll pool URLs, propagate API status to Redis, write odds snapshots every 5-60s depending on time to jump.
2. **STOP_SELL detected**: immediately write `STOP_SELL` to Redis so `run_bot.py` fires, persist `stop_sell_time` wall-clock, enter 180s late-money capture window with 5s poll cadence.
3. **Late-money window complete**: write `CLOSED` to Redis and exit. The archiver captures its final canonical snapshot at this moment.

This sequence was designed to solve the late-money problem: HKJC's tote continues updating dividends for 1-3 minutes after sell stops. Archiving only pre-jump odds corrupts training data for future walk-forward windows.

---

## Known Caveats

This is the honest version. Read it before getting excited about the numbers.

### 1. Training-execution timing mismatch

Model A is trained on final settled odds (from `race_entries.win_odds`), but the live bot fires at `STOP_SELL` time. If late syndicate money consistently drifts odds after `STOP_SELL`, the model has effectively been trained with information it cannot access live. The size of this gap is measured by `diagnostics/drift_diagnostic.py` — run it against your own data before trusting the backtest ROI. On a 59-race sample I found ~1% aggregate drift on WIN but meaningful tail risk in the TRI pool, where longshot combos can collapse 60-70% after sell stops.

**Mitigation path**: collect 3-6 months of paired STOP_SELL / CLOSED odds via the Plan C phase-tagged archiver, then retrain Model A with STOP_SELL-time odds as the base-margin anchor. Expect the backtest ROI to drop substantially when this is done — the gap between the inflated and honest number is the size of your look-ahead bias.

### 2. Small sample size for exotic validation

Eighty bets across two years is thin for claiming statistical significance, especially split across WIN/QIN/QPL/TRI pools. The 42.76% ROI has wide confidence intervals. A single losing month at the top of the ledger could have wiped out a quarter of the cumulative profit.

### 3. The drift diagnostic is approximate

`diagnostics/drift_diagnostic.py` uses a 90-second-before-final-snapshot heuristic as its T0 anchor (Option B from the methodology). Once you have data tagged with explicit `stop_sell_time` markers (Plan C, which the current scraper writes), upgrade the diagnostic to filter on `phase='PRE_STOP_SELL' AND seconds_vs_stop_sell BETWEEN -30 AND -5` for T0 vs `phase='FINAL'` for settled odds.

### 4. No live data collection paper trail

The repository does not include the raw scraped dataset, just the scraping code. You must run `daily_updater.py` over race days to populate your local Postgres before anything can train or backtest. Building a 5-year historical database from scratch takes ~3-4 days of continuous scraping with HKJC's rate limits.

### 5. Track takeout is merciless

HKJC extracts 17.5% from every parimutuel pool. A "fair" model with zero edge loses 17.5¢ on every dollar bet. Anything that looks like ROI is the model's edge minus 17.5%. The reason this is interesting is that HKJC has deep enough pools and enough late syndicate money that real edge exists for a sufficiently-calibrated model — but that edge is small (single-digit percent on gross turnover), not the 42% you see in backtests.

---

## Why I Built This

Quantitative horse-racing models are a rich case study in the gap between paper edge and live edge. HKJC is one of the most technically interesting markets in the world: $100M+ daily turnover in Hong Kong alone, 40 years of sophisticated syndicates, 17.5% takeout, dense late-money structure, and publicly available historical data going back two decades. William Benter's 1994 paper ([*Computer Based Horse Race Handicapping and Wagering Systems: A Report*](https://www.gwern.net/docs/statistics/decision/1994-benter.pdf)) is still the gold-standard reference, and most of the techniques in this repo trace back to it directly.

The repo evolved through five generations — conditional logistic regression → Bayesian MCMC → naive XGBoost → dual-ensemble → the current Benter stacker. Each stage failed for a specific reason that taught me something about the market. The `ModelEvolution` narrative in my portfolio dashboard documents those failures; see `src/hkjc_engine/models/` for the architectural lineage.

---

## References

- Benter, W. (1994). *Computer Based Horse Race Handicapping and Wagering Systems: A Report.* In *Efficiency of Racetrack Betting Markets* (Hausch, Lo & Ziemba, eds.)
- Bolton, R.N. & Chapman, R.G. (1986). Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races. *Management Science* 32(8).
- Henery, R.J. (1981). Permutation Probabilities as Models for Horse Races. *Journal of the Royal Statistical Society* B 43(1).
- Kull, M., Silva Filho, T. & Flach, P. (2017). Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. *AISTATS*.
- Lo, V.S.Y. & Bacon-Shone, J. (2008). Probability and Statistical Models for Racing. In *Handbook of Sports and Lottery Markets* (Hausch & Ziemba, eds.)
- MacLean, L.C., Ziemba, W.T. & Blazenko, G. (1992). Growth versus security in dynamic investment analysis. *Management Science* 38(11).
- Baker, R.D. & McHale, I.G. (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. *Decision Analysis* 10(3).
- Smoczyński, P. & Tomkins, D. (2010). An explicit solution to the problem of optimizing the allocations of a bettor's wealth when wagering on horse races. *Mathematical Scientist* 35(1).
- Genest, C. & Zidek, J.V. (1986). Combining Probability Distributions: A Critique and an Annotated Bibliography. *Statistical Science* 1(1).

---

## License

MIT. See [LICENSE](LICENSE).
