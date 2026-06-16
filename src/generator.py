from __future__ import annotations

import os
from functools import lru_cache

from openai import DEFAULT_MAX_RETRIES, DefaultHttpxClient, OpenAI

from src.config import settings


SYSTEM_PROMPT_WITH_RAG = """You are a rare-disease medical support assistant (not a diagnosis system).
Rules:
1) Use only provided context for disease knowledge claims.
2) Never give a definitive diagnosis; provide differential directions only.
3) Do not provide exact drug dosing instructions or prescription-level treatment plans.
4) If evidence is weak or missing, clearly say uncertainty and suggest what data to collect next.
5) Keep answer concise and structured in Chinese with these sections:
   - 可能方向（非确诊）
   - 依据（对应症状/检查）
   - 建议补充检查
   - 可讨论的缓解思路（非处方级）
   - 何时应尽快线下就医
6) End with: “以上仅为医学信息辅助，不替代医生面诊与诊断。”"""

SYSTEM_PROMPT_NO_RAG = """You are a medical information assistant for rare diseases.
Rules:
1) Answer using general knowledge and clearly state uncertainty.
2) Never give a definitive diagnosis; provide only possible directions.
3) Do not provide exact drug dosing instructions.
4) Keep answer concise and structured in Chinese with these sections:
   - 可能方向（非确诊）
   - 依据（对应症状/检查）
   - 建议补充检查
   - 可讨论的缓解思路（非处方级）
   - 何时应尽快线下就医
5) End with: “以上仅为医学信息辅助，不替代医生面诊与诊断。”"""


def _effective_http_trust_env(base_url: str) -> bool:
    """Whether httpx should honor HTTP(S)_PROXY from the environment.

    OpenRouter often returns opaque 500s when a system proxy intercepts the request.
    Default: trust env for normal OpenAI; skip env proxies for openrouter.ai unless overridden.
    """
    raw = os.getenv("OPENAI_HTTP_TRUST_ENV", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return "openrouter.ai" not in base_url.lower()


def _effective_max_retries(base_url: str) -> int:
    if "openrouter.ai" not in base_url.lower():
        return DEFAULT_MAX_RETRIES
    try:
        return max(0, int(os.getenv("OPENAI_MAX_RETRIES", "10")))
    except ValueError:
        return 10


@lru_cache(maxsize=4)
def _openai_client(
    api_key: str,
    base_url: str,
    trust_env: bool,
    max_retries: int,
    referer: str,
    title: str,
) -> OpenAI:
    headers: dict[str, str] | None = None
    if "openrouter.ai" in base_url.lower():
        h: dict[str, str] = {}
        if referer:
            h["HTTP-Referer"] = referer
        if title:
            h["X-Title"] = title
        headers = h or None
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        default_headers=headers,
        http_client=DefaultHttpxClient(trust_env=trust_env),
    )


def _build_user_content(
    question: str,
    context: str,
    *,
    use_rag: bool,
    has_patient_profile: bool,
) -> str:
    if use_rag:
        return (
            f"Task: Provide medical support advice (non-diagnostic). "
            f"Patient profile provided: {'yes' if has_patient_profile else 'no'}.\n\n"
            f"Question:\n{question}\n\nContext:\n{context}"
        )
    return (
        f"Task: Provide medical support advice (non-diagnostic). "
        f"Patient profile provided: {'yes' if has_patient_profile else 'no'}.\n\n"
        f"Question:\n{question}"
    )


def generate_answer(
    question: str,
    context: str,
    use_rag: bool = True,
    has_patient_profile: bool = False,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    if not settings.openai_api_key:
        if not use_rag:
            return (
                "I cannot call an LLM because OPENAI_API_KEY is empty.\n"
                "Current mode: no RAG retrieval.\n\n"
                "Safety note: This is informational only and not medical advice."
            )
        return (
            "I cannot call an LLM because OPENAI_API_KEY is empty.\n"
            "Retrieved context summary:\n"
            + context[:1200]
            + "\n\nSafety note: This is informational only and not medical advice."
        )

    base = settings.openai_base_url
    trust = _effective_http_trust_env(base)
    retries = _effective_max_retries(base)
    ref, ttl = "", ""
    if "openrouter.ai" in base.lower():
        ref = os.getenv("OPENROUTER_HTTP_REFERER", "http://127.0.0.1:8000").strip()
        ttl = os.getenv("OPENROUTER_APP_TITLE", "Rare Disease RAG").strip()

    client = _openai_client(
        settings.openai_api_key,
        base,
        trust,
        retries,
        ref,
        ttl,
    )
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_WITH_RAG if use_rag else SYSTEM_PROMPT_NO_RAG,
        },
    ]
    if chat_history:
        for turn in chat_history:
            role = turn.get("role", "")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": _build_user_content(
                question,
                context,
                use_rag=use_rag,
                has_patient_profile=has_patient_profile,
            ),
        }
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.2,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()
