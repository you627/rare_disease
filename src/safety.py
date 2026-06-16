from __future__ import annotations

from dataclasses import dataclass


HIGH_RISK_KEYWORDS = [
    "dose",
    "stop drug",
    "switch drug",
    "pregnant",
    "child",
    "emergency",
    "suicide",
    "chest pain",
    "shortness of breath",
    "seizure",
    "剂量",
    "停药",
    "换药",
    "孕妇",
    "儿童",
    "急救",
    "自杀",
    "胸痛",
    "呼吸困难",
    "抽搐",
]


@dataclass
class SafetyResult:
    level: str
    flagged_terms: list[str]
    disclaimer: str


def check_question_safety(question: str) -> SafetyResult:
    lower_q = question.lower()
    matched = [term for term in HIGH_RISK_KEYWORDS if term in lower_q or term in question]
    level = "high" if matched else "normal"
    disclaimer = (
        "Medical information only. This system cannot replace a licensed clinician. "
        "For urgent symptoms, seek in-person care immediately."
    )
    return SafetyResult(level=level, flagged_terms=matched, disclaimer=disclaimer)
