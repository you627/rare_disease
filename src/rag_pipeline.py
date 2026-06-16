from __future__ import annotations

import json
from typing import Any

from src.config import settings
from src.generator import generate_answer
from src.retriever import HybridRetriever
from src.safety import SafetyResult, check_question_safety


def patient_profile_from_dict(data: dict[str, Any]) -> str:
    field_map = [
        ("age", "Age"),
        ("sex", "Sex"),
        ("chief_complaint", "Chief complaint"),
        ("key_symptoms", "Key symptoms"),
        ("duration", "Duration/course"),
        ("history", "History"),
        ("labs_or_imaging", "Labs/imaging"),
    ]
    parts: list[str] = []
    for key, label in field_map:
        val = str(data.get(key, "") or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def merge_question_with_profile(question: str, patient_profile: str) -> str:
    if not patient_profile:
        return question
    return f"{question}\n\nPatient profile:\n{patient_profile}"


def build_retrieval_query(question: str, chat_history: list[dict[str, str]]) -> str:
    """Combine current question with recent turns for better follow-up retrieval."""
    if not chat_history:
        return question
    recent = chat_history[-4:]
    snippets: list[str] = []
    for msg in recent:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if content:
            snippets.append(f"{role}: {content[:400]}")
    if not snippets:
        return question
    return question + "\n\nRecent dialogue:\n" + "\n".join(snippets)


def trim_chat_history(
    history: list[dict[str, str]], max_turns: int | None = None
) -> list[dict[str, str]]:
    limit = max_turns if max_turns is not None else settings.max_chat_history_turns
    if limit <= 0:
        return []
    # Each turn = user + assistant => 2 messages per turn
    max_messages = limit * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def run_rag_qa(
    *,
    question: str,
    retriever: HybridRetriever | None,
    use_rag: bool,
    top_k: int,
    patient_profile: str = "",
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict], SafetyResult, str]:
    history = trim_chat_history(chat_history or [])
    merged_question = merge_question_with_profile(question, patient_profile)
    safety = check_question_safety(merged_question)

    search_query = build_retrieval_query(merged_question, history)
    hits = retriever.search(search_query, top_k=top_k) if use_rag and retriever else []
    context = "\n\n".join(
        [
            f"[{i+1}] {h['title']} | {h['field']} | score={h['score']:.3f}\n{h['text']}"
            for i, h in enumerate(hits)
        ]
    )
    context = context[: settings.max_context_chars]

    answer = generate_answer(
        merged_question,
        context,
        use_rag=use_rag,
        has_patient_profile=bool(patient_profile),
        chat_history=history,
    )
    return answer, hits, safety, context


def hits_to_citations(hits: list[dict]) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "title": h["title"],
            "field": h["field"],
            "score": h["score"],
            "source_url": h["source_url"],
            "chunk_id": h["chunk_id"],
        }
        for i, h in enumerate(hits)
    ]


def loads_patient_profile_json(raw: str) -> dict[str, str]:
    try:
        data = json.loads(raw or "{}")
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        pass
    return {}
