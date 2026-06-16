from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _normalize_openai_base_url(raw: str) -> str:
    """Strip whitespace; fix common OpenRouter mistake (models page vs API)."""
    u = (raw or "").strip().rstrip("/")
    if not u:
        return "https://api.openai.com/v1"
    low = u.lower()
    if "openrouter.ai" in low and "/api/v1" not in low:
        return "https://openrouter.ai/api/v1"
    return u


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url: str = _normalize_openai_base_url(
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
    database_url: str = os.getenv(
        "DATABASE_URL", "sqlite:///data/app.db"
    ).strip()
    jwt_secret: str = os.getenv(
        "JWT_SECRET", "dev-simulated-system-change-in-production"
    ).strip()
    jwt_expire_hours: int = int(os.getenv("JWT_EXPIRE_HOURS", "168"))
    max_chat_history_turns: int = int(os.getenv("MAX_CHAT_HISTORY_TURNS", "10"))


settings = Settings()
