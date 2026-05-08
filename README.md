# HKJC Benter Engine

A parimutuel pricing engine for the Hong Kong Jockey Club (HKJC). Combines an XGBoost model and Cox proportional hazards model, synthesized via PyMC Bayesian Stacking. Exotic pool (QIN/QPL/TRI) pricing is resolved using N-dimentional Student-t Copulas to model heavy-tailed join probabilities, and bets are sized via a fractional Kelly with Smocynski-Tomkins simultaneous-stake caps.

> **Disclaimer.** This is research code shared for educational and archival purposes. Backtested returns are out-of-sample but walk-forward backtests still systematically overstate live performance — see the [Known Caveats](#known-caveats) section before drawing any conclusions. Gambling carries real financial risk, and parimutuel markets with 17.5% track takeout are particularly unforgiving. Do not deploy this code with money you cannot afford to lose.

---

## Walk-Forward Backtest Results

| Metric            | Value               |
|-------------------|---------------------|
| Period            | 2024-01 → 2026-05 (25 monthly windows) |
| Bets placed       | 799                 |
| Total staked      | $349,835            |
| Net profit        | +$23,181            |
| Win rate          | 6.38%               |
| ROI               | +6.63%              |
| Starting bankroll | $100,000            |
| Ending bankroll   | $123,180.88         |
| OOF log-loss      | 0.23840  (vs 0.23936 public consensus) |

These numbers reflect idealized execution against final settled dividends. See [Known Caveats](#known-caveats) for how much of this is inflated by late-money drift.


--- 
## Per-Pool Breakdown

The multi-agent stacker reveals heavy variance across different parimutuel pools. Fundamental physical edge successfully translates to profit in the Win and heavy-tailed Trifecta pools, while the model struggles to beat the takeout in Quinella and Place markets.
| Pool  | Bets | Win Rate   | Staked    | Profit    | ROI       | 
|-------|------|------------|-----------|-----------|-----------|
| WIN   | 124  | 8.06%      | $53,981   | +$7,473   | +13.84%   |
| PLA   | 38   | 0.00%      | $928      | -$928     | -100%     |
| QIN   | 91   | 3.30%      | $39,286   | -$8,193   | -20.85%   |
| QPL   | 170  | 16.47%     | $98,920   | -$4,618   | -4,67%    |
| TRI   | 376  | 2.66%      | $146,593  | +$52,763  | +35.99%   |

---

## Architecturez

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
             │ XGBoost Model A    │  │ CoxPH Model B       │
             │ (Fundamental /     │  │ (Survival Dynamics &│
             │  Market Edge)      │  │  Pace Collapse)     │
             └───────────┬────────┘  └──────┬──────────────┘
                         │                  │
                         └──┬───────────────┘
                            │            ┌──────────────────┐
                            │      ┌────▶│ Public market P  │
                            ▼      │     │  (1/d normalized)│
               ┌────────────────────────┐└────────────┬─────┘
               │  PyMC Bayesian Stacker │             │
               │  (ADVI Synthesis)      │◀────────────┘
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Student-t Copula       │
               │ (N-Dimensional Joint   │
               │  Exotic Probabilities) │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Betting policy:        │
               │  • Baker-McHale hurdle │
               │  • Exact Kelly via MCMC│
               │  • α=0.25 Fractional   │
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
├── .env.example
├── .gitattributes
├── .gitignore
├── CONTRIBUTING.md
├── docker-compose.yml               # Postgres + Redis local infrastructure
├── pyproject.toml
├── README.md
├── requirements.txt
├── SECURITY.md
├── scripts/                         # Convenience bash wrappers
│   ├── backtest.sh
│   ├── run_live_day.sh
│   └── train_models.sh
├── sql/migrations/                  # DB schema updates
│   ├── 001_phase_columns.sql
│   └── 002_stop_sell_anchor.sql
└── src/hkjc_engine/
    ├── config.py                    # Env-var configuration (secrets & paths)
    ├── data/                        # Ingestion, schema, and feature engineering
    │   ├── add_features.py
    │   ├── audit_missing_dates.py
    │   ├── backfill_dates.py
    │   ├── compute_class_and_track.py
    │   ├── compute_pace.py
    │   ├── compute_trueskill.py
    │   ├── daily_updater.py         # Pipeline orchestrator
    │   ├── patch_finish_position.py
    │   ├── patch_race_class.py
    │   ├── patch_race_win_odds.py
    │   ├── schema.py                # SQLAlchemy declarative base
    │   ├── scraper.py               # Main historical scraper
    │   ├── scraper_dividends.py
    │   ├── scraper_horse_numbers.py
    │   ├── stop_sell_loader.py
    │   └── update_run_styles.py
    ├── diagnostics/                 # Offline analysis and system checks
    │   ├── diagnose_dividend_mapping.py
    │   ├── drift_diagnostic.py      # Late-money drift analysis
    │   ├── exotics_backtest.py
    │   ├── place_projection_diagnostic.py
    │   ├── prepare_backtest_inputs.py
    │   ├── rank_calibration_check.py
    │   ├── stern_exotics_backtest.py
    │   └── theta_place_diagnostic.py
    ├── live/                        # Race-day execution
    │   ├── account_statement_ingester.py
    │   ├── odds_archiver.py         # Redis → Postgres persistence
    │   ├── orchestrator.py          # Multi-race subprocess launcher
    │   ├── predictor.py             # Inference wrappers
    │   ├── run_bot.py               # Main live bot (Discord webhook)
    │   ├── scraper.py               # Playwright HKJC GraphQL interceptor
    │   ├── slippage_attribution.py
    │   └── snapshot_logger.py
    └── models/                      # Training, ensembles, and simulation
        ├── backtester.py
        ├── betting_policy.py        # Kelly + hurdle + caps
        ├── build_pace_copula.py
        ├── drift_forecaster.py
        ├── ensemble.py              # Bayesian Stacking & Calibration
        ├── feature_factory.py
        ├── multi_pool_backtester.py
        ├── stern_simulator.py       # Copula Gamma Simulator
        ├── theta_optimizer.py
        ├── trainer_independent.py   # CoxPH Survival Agent
        ├── trainer_pla_residual.py  # XGBoost PLA Agent
        ├── trainer_residual.py      # XGBoost WIN Agent
        └── walk_forward.py          # Master training loop
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

This trains the XGBoost agents (Residual & Physics), fits the PyMC Bayesian stacker via ADVI for each monthly window, computes dynamic drift-aware shrinkage, and writes `walk_forward_master_ledger.csv` plus `live_config.json` into `./artifacts/`.

### 6. Run the live bot on race day

```bash
# Terminal 1 — scrape the card (sequentially per race)
python -m hkjc_engine.live.orchestrator

# Terminal 2 — archive odds snapshots to Postgres
python -m hkjc_engine.live.odds_archiver HV

# Terminal 3 — pricing + betting signals
python -m hkjc_engine.live.run_bot
```

The bot reads the latest PyMC posterior traces and fires Discord alerts as each race enters `STOP_SELL`.

---

## How the Pipeline Works

### Feature Factory (`models/feature_factory.py`)

Converts raw race entries into model-ready features. The pipeline computes distance-bucketed pace EMAs, TrueSkill $\mu/\sigma$ updated chronologically, draw $\times$ early-pace interactions, class-drop flags, and track geometry (rail $\times$ straight length $\times$ width).Crucially, it prepares the residual target for Model A by converting point-in-time STOP_SELL decimal odds ($d_i$) into a true probability distribution ($p_{mkt,i}$), discounting the track takeout using a power-exponent $z = \frac{\ln(\sum d_i^{-1})}{\ln N} + 1.0$, followed by a Hayek-Hong-Stutzer longshot bias correction.

### Model A — Market-Residual Learner (`models/trainer_residual.py`)

An XGBoost agent trained with `objective=rank:pairwise` to optimize normalized discounted cumulative gain (NDCG). Instead of learning win probabilities from scratch, it learns the residuals (deviations) from the public consensus.The public log-odds are injected directly into the trees as the base_margin:$$b_i = \ln\left(\frac{p_{mkt, i}}{1 - p_{mkt, i}}\right)$$ The model learns a function $f(X_i)$ such that the final predicted probability incorporates both the market prior and the physical edge:$$P_{A, i} = \frac{\exp(b_i + f(X_i))}{\sum_j \exp(b_j + f(X_j))}$$Note: A parallel architecture (trainer_pla_residual.py) executes this exact mechanism natively for the Place (PLA) pool, abandoning Henery place-projection approximations.

### Model B — Physics Learner (`models/trainer_independent.py`)

Replaces the standard classification trees to natively model survival dynamics and pace collapse using a Cox Proportional Hazards model.Let $T_i$ represent the "time" (finish position) of a horse. The hazard function $h_i(t)$ evaluates the risk of finishing at position $t$:$$h_i(t) = h_0(t) \exp(\beta^T X_i)$$By stratifying on race_id and clustering on a pace_archetype Gamma frailty, the model evaluates exact Plackett-Luce likelihoods. The partial hazard $\exp(\beta^T X_i)$ is extracted, where a higher hazard translates to a higher win probability, independent of public odds.

### Calibration & PyMC Bayesian Stacker (`models/ensemble.py`)

Model predictions are first calibrated via a `StratifiedSmoothedIsotonicCalibrator` conditioned on early-pace archetypes.The ensemble relies on a Bayesian Hierarchical Stacker fitted via ADVI in PyMC. Multiplicative fusion in log-space correctly synthesizes conditionally-independent agents (Genest & Zidek, 1986). The ensemble computes posterior distributions for the weights $w$:$$P_{ens, i} = \frac{\exp(w_A \log P_{A,i} + w_B \log P_{B,i} + w_{mkt} \log P_{mkt,i})}{\sum_j \exp(w_A \log P_{A,j} + w_B \log P_{B,j} + w_{mkt} \log P_{mkt,j})}$$By evaluating $w_{mkt}$ dynamically based on `I_valid` (presence of an uncorrupted `STOP_SELL` anchor), the stacker guarantees mathematical debiasing against late-money timing mismatches.

### N-Dimensional Student-t Copula for Exotics (`models/stern_simulator.py`)

Instead of extrapolating prices through the Henery-discounted Harville method, we deploy a Quasi-Monte Carlo **Copula Gamma (Stern) Simulator**.

Each horse's race time $T_i$ is modeled as a Gamma distribution $T_i \sim \Gamma(r_i, \lambda_i)$. The shape parameter $r_i$ is heteroskedastic, derived dynamically from pace $z$-scores to capture "traffic frailty" (closers have higher variance right-tails than front-runners). 

Joint probabilities are modeled by coupling the marginals via a Student-t Copula:
1. Generate correlated normals: $Z \sim t_\nu(0, \Sigma_{pace})$
2. Transform to uniforms via t-CDF: $U_i = \Phi_{t,\nu}(Z_i)$
3. Map to empirical race times: $T_i = F_{\Gamma}^{-1}(U_i; r_i, \lambda_i)$

By counting combinatorial rank configurations across 16,384 paths, we generate natively correlated $N$-dimensional probabilities for all parimutuel pools simultaneously.

### Betting Policy (`models/betting_policy.py`)

Standard fractional Kelly assumes static odds. In parimutuel systems, odds drift after execution. We expand the MacLean-Ziemba (1992) fractional Kelly to account for stochastic payoffs using a second-order Taylor expansion.

* **Drift-Aware EV:** Calculates expected value using the Jensen-correct conditional median drift.
  $$EV_{eff} = (P_{shrunk} \cdot d_{stop\_sell} \cdot \tilde{R}_{median}) - 1$$
* **Stochastic Kelly Penalty:** Scales down conviction dynamically based on the ratio of pool drift variance ($\sigma_R$) to the expected edge.
  $$\alpha_{pool} = \frac{\alpha_{base}}{1 + \kappa \left(\frac{\sigma_R}{EV_{eff}}\right)^2}$$
* **Simultaneous Stakes Caps:** Approximates Smoczyński-Tomkins (2010). Caps are mathematically tightened in high-variance exotic pools: $C_{pool} = C_{base} \sqrt{\frac{\sigma_{R, win}}{\sigma_{R, pool}}}$.

### Live Scraper State Machine (`live/scraper.py` & `live/odds_archiver.py`)

Playwright-based GraphQL interceptor. Passive — doesn't click anything except the trifecta "All" button to expand the matrix. State machine:

1. **PRE_STOP_SELL**: poll pool URLs, propagate API status to Redis, write odds snapshots every 5-60s depending on time to jump.
2. **STOP_SELL detected**: immediately write `STOP_SELL` to Redis so `run_bot.py` fires, persist `stop_sell_time` wall-clock, enter 600s late-money capture window with 5s poll cadence.
3. **CLOSED**: write `CLOSED` to Redis and exit. The archiver captures the true settled odds as ground truth for future walk-forward iterations.

This sequence was designed to solve the late-money problem: HKJC's tote continues updating dividends for 3-5 minutes after sell stops. Archiving only pre-jump odds corrupts training data for future walk-forward windows.

---

## Known Caveats

### 1. Training-execution timing mismatch

Model A is trained on final settled odds (from `race_entries.win_odds`), but the live bot fires at `STOP_SELL` time. If late syndicate money consistently drifts odds after `STOP_SELL` (~3-5min HKJC aggregation lag), the model has effectively been trained with information it cannot access live. The size of this gap is measured by `diagnostics/drift_diagnostic.py` — run it against your own data before trusting the backtest ROI. On a 59-race sample I found ~1% aggregate drift on WIN but meaningful tail risk in the TRI pool, where longshot combos can collapse 60-70% after sell stops. 

**Mitigation path**: collect 3-6 months of paired STOP_SELL / CLOSED odds via phase-tagged archiver, then retrain Model A with STOP_SELL-time odds as the base-margin anchor. Expect the backtest ROI to drop substantially when this is done — the gap between the inflated and honest number is the size of your look-ahead bias.

### 2. Small sample size for exotic validation

Bets over two months is thin for claiming statistical significance, especially split across WIN/QIN/QPL/TRI pools. The ROI has wide confidence intervals. A single losing month at the top of the ledger could have wiped out a quarter of the cumulative profit.

### 3. The drift diagnostic is approximate

`diagnostics/drift_diagnostic.py` uses a 90-second-before-final-snapshot heuristic as its T0 anchor (Option B from the methodology). Once you have data tagged with explicit `stop_sell_time` markers (which the current scraper writes), upgrade the diagnostic to filter on `phase='PRE_STOP_SELL' AND seconds_vs_stop_sell BETWEEN -30 AND -5` for T0 vs `phase='FINAL'` for settled odds.

### 4. No live data collection paper trail

The repository does not include the raw scraped dataset, just the scraping code. You must run `daily_updater.py` over race days to populate your local Postgres before anything can train or backtest. Building a 5-year historical database from scratch takes ~3-4 days of continuous scraping with HKJC's rate limits.

### 5. Track takeout is merciless

HKJC extracts at least 17.5% from every parimutuel pool. A "fair" model with zero edge loses 17.5¢ on every dollar bet. Anything that looks like ROI is the model's edge minus 17.5%. The reason this is interesting is that HKJC has deep enough pools and enough late syndicate money that real edge exists for a sufficiently-calibrated model — but that edge is small (single-digit percent on gross turnover).

---

## Why I Built This

Quantitative horse-racing models are a rich case study in the gap between paper edge and live edge. HKJC is one of the most technically interesting markets in the world: $100M+ daily turnover in Hong Kong alone, 40 years of sophisticated syndicates, 17.5% takeout, dense late-money structure, and publicly available historical data going back two decades. William Benter's 1994 paper ([*Computer Based Horse Race Handicapping and Wagering Systems: A Report*](https://www.gwern.net/docs/statistics/decision/1994-benter.pdf)) is still the gold-standard reference, and most of the techniques in this repo trace back to it directly.

The repo evolved through six generations — conditional logistic regression → Bayesian MCMC → naive XGBoost → dual-ensemble → Benter Stacker → the current Multi-Agent Copula architecture. Each stage failed for a specific reason that taught me something about market microstructure and statistics.

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
