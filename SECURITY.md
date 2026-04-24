# Security Policy

## Reporting a Vulnerability

This is a personal research project without a security team. If you find a vulnerability — particularly a secret leaked in commit history, a credential in the codebase, or a way the live bot could be induced to place unintended bets — please open a private GitHub Security Advisory rather than a public issue.

## Known Operational Risks
While financial execution is air-gapped, the data pipeline has known points of operational risk:

**Signal Tampering:** the bot reads live_config.json for dynamic shrinkage values. If filesystem permissions are poorly configured, a bad actor could alter the shrinkage, causing the dashboard to recommend artificially inflated stake sizes.

**State Corruption:** The bot trusts Redis for race_status. A corrupted or manipulated Redis cache could cause the bot to evaluate stale odds, leading the human operator to place a bet based on a ghost state.

**Webhook Hijacking:** The bot does not currently verify Discord webhook response signatures. A leaked webhook means unauthorized payloads could be posted to the private signal channel.