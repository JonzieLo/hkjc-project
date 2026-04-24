"""
Central configuration loaded from environment variables.

All modules should import `DB_URL`, `WEBHOOK_URL`, `REDIS_HOST`, etc. from here
rather than hardcoding values. Real values live in `.env` (gitignored); the
template lives in `.env.example`.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    _repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Copy .env.example → .env and fill it in, or export {name} in your shell."
        )
    return val


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, default)

DB_URL: str = _optional(
    "HKJC_DB_URL",
    "postgresql://hkjc:CHANGE_ME@localhost:5432/hkjc_racing",
)
REDIS_HOST: str = _optional("HKJC_REDIS_HOST", "localhost")
REDIS_PORT: int = int(_optional("HKJC_REDIS_PORT", "6379"))
REDIS_DB: int   = int(_optional("HKJC_REDIS_DB", "0"))
WEBHOOK_URL: str | None = os.environ.get("HKJC_DISCORD_WEBHOOK")
LIVE_VENUE: str         = _optional("HKJC_LIVE_VENUE", "HV")
LIVE_BANKROLL: float    = float(_optional("HKJC_LIVE_BANKROLL", "100000"))

# Artifacts directory (trained models, OOF CSVs, stacker, live_config.json). Override for deployments that need a persistent absolute path.
from pathlib import Path as _Path
ARTIFACTS_DIR: _Path = _Path(_optional("HKJC_ARTIFACTS_DIR", "./artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def artifact(name: str) -> str:
    return str(ARTIFACTS_DIR / name)


def redis_client():
    import redis
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
    )
