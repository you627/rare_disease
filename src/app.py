from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.auth import (
    create_access_token,
    get_current_user,
    get_db,
    hash_password,
    verify_password,
)
from src.config import settings
from src.db import (
    Conversation,
    Message,
    User,
    dumps_citations,
    init_db,
    loads_citations,
)
from src.rag_pipeline import (
    hits_to_citations,
    loads_patient_profile_json,
    patient_profile_from_dict,
    run_rag_qa,
)
from src.retriever import HybridRetriever
from src.safety import SafetyResult

app = FastAPI(title="Rare Disease RAG API", version="0.2.0")
WEB_DIR = Path("web")
RETRIEVER_PATH = Path("data/retriever.pkl")

retriever: HybridRetriever | None
if not RETRIEVER_PATH.exists():
    retriever = None
else:
    retriever = HybridRetriever.load(RETRIEVER_PATH)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


class PatientProfileFields(BaseModel):
    age: str = Field(default="", max_length=80)
    sex: str = Field(default="", max_length=40)
    chief_complaint: str = Field(default="", max_length=800)
    key_symptoms: str = Field(default="", max_length=2000)
    duration: str = Field(default="", max_length=400)
    history: str = Field(default="", max_length=1200)
    labs_or_imaging: str = Field(default="", max_length=2000)

    def as_dict(self) -> dict[str, str]:
        return self.model_dump()


class AskRequest(PatientProfileFields):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=settings.top_k_default, ge=1, le=20)
    use_rag: bool = Field(default=True, description="Whether to use retriever context")


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(default="")


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(default="")


class ConversationCreateRequest(PatientProfileFields):
    title: str = Field(default="新对话", max_length=200)


class ConversationMessageRequest(PatientProfileFields):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.top_k_default, ge=1, le=20)
    use_rag: bool = True


def _safety_payload(safety: SafetyResult) -> dict:
    return {
        "level": safety.level,
        "flagged_terms": safety.flagged_terms,
        "disclaimer": safety.disclaimer,
    }


def _llm_error_detail(exc: Exception) -> str:
    detail = f"LLM request failed: {exc}"
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError) and exc.response is not None:
            snippet = (exc.response.text or "")[:800].strip()
            if snippet:
                detail = f"LLM HTTP {exc.status_code}: {snippet}"
    except ImportError:
        pass
    return detail


def _conversation_or_404(
    db: Session, user: User, conversation_id: int
) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


def _history_from_conversation(conv: Conversation) -> list[dict[str, str]]:
    return [
        {"role": m.role, "content": m.content}
        for m in conv.messages
        if m.role in ("user", "assistant")
    ]


def _conversation_summary(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "patient_profile": loads_patient_profile_json(conv.patient_profile_json),
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "message_count": len(conv.messages),
    }


def _message_summary(msg: Message) -> dict:
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "citations": loads_citations(msg.citations_json),
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "retriever_loaded": retriever is not None,
        "openai_enabled": bool(settings.openai_api_key),
    }


@app.get("/")
def home() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=302)


@app.get("/login")
def login_page() -> FileResponse:
    path = WEB_DIR / "login.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="web/login.html not found")
    return FileResponse(path)


@app.get("/chat")
def chat_page() -> FileResponse:
    path = WEB_DIR / "chat.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="web/chat.html not found")
    return FileResponse(path)


@app.get("/legacy")
def legacy_page() -> FileResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(path)


@app.post("/auth/register")
def register(req: AuthRegisterRequest, db: Session = Depends(get_db)) -> dict:
    username = req.username.strip()
    existing = db.scalar(select(User).where(User.username == username))
    if existing:
        raise HTTPException(status_code=400, detail="该用户名已注册，请直接登录")
    user = User(username=username, password_hash=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username},
    }


@app.post("/auth/login")
def login(req: AuthLoginRequest, db: Session = Depends(get_db)) -> dict:
    username = req.username.strip()
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user.id, user.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username},
    }


@app.get("/auth/me")
def me(user: User = Depends(get_current_user)) -> dict:
    return {"id": user.id, "username": user.username}


@app.get("/conversations")
def list_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    convs = db.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()
    return {"conversations": [_conversation_summary(c) for c in convs]}


@app.post("/conversations")
def create_conversation(
    req: ConversationCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conv = Conversation(
        user_id=user.id,
        title=req.title.strip() or "新对话",
        patient_profile_json=json.dumps(req.as_dict(), ensure_ascii=False),
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"conversation": _conversation_summary(conv)}


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conv = _conversation_or_404(db, user, conversation_id)
    return {
        "conversation": _conversation_summary(conv),
        "messages": [_message_summary(m) for m in conv.messages],
    }


@app.post("/conversations/{conversation_id}/messages")
def post_conversation_message(
    conversation_id: int,
    req: ConversationMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if req.use_rag and retriever is None:
        raise HTTPException(
            status_code=500,
            detail="Retriever not found. Run ingest first: python src/ingest.py --csv <path>",
        )

    conv = _conversation_or_404(db, user, conversation_id)
    profile_data = loads_patient_profile_json(conv.patient_profile_json)
    for key, val in req.as_dict().items():
        if str(val).strip():
            profile_data[key] = str(val).strip()
    if any(profile_data.values()):
        conv.patient_profile_json = json.dumps(profile_data, ensure_ascii=False)

    patient_profile = patient_profile_from_dict(profile_data)
    history = _history_from_conversation(conv)

    try:
        answer, hits, safety, _context = run_rag_qa(
            question=req.question.strip(),
            retriever=retriever,
            use_rag=req.use_rag,
            top_k=req.top_k,
            patient_profile=patient_profile,
            chat_history=history,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_llm_error_detail(exc)) from exc

    citations = hits_to_citations(hits)
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=req.question.strip(),
        citations_json="[]",
    )
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        citations_json=dumps_citations(citations),
    )
    db.add(user_msg)
    db.add(assistant_msg)
    if conv.title == "新对话" and req.question.strip():
        conv.title = req.question.strip()[:40]
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return {
        "conversation_id": conv.id,
        "question": req.question,
        "patient_profile_used": bool(patient_profile),
        "mode": "rag" if req.use_rag else "no_rag",
        "answer": answer,
        "safety": _safety_payload(safety),
        "citations": citations,
        "messages": [
            _message_summary(user_msg),
            _message_summary(assistant_msg),
        ],
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conv = _conversation_or_404(db, user, conversation_id)
    db.delete(conv)
    db.commit()
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Stateless single-turn endpoint (backward compatible)."""
    if req.use_rag and retriever is None:
        raise HTTPException(
            status_code=500,
            detail="Retriever not found. Run ingest first: python src/ingest.py --csv <path>",
        )

    patient_profile = patient_profile_from_dict(req.as_dict())
    try:
        answer, hits, safety, _context = run_rag_qa(
            question=req.question,
            retriever=retriever,
            use_rag=req.use_rag,
            top_k=req.top_k,
            patient_profile=patient_profile,
            chat_history=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_llm_error_detail(exc)) from exc

    return {
        "question": req.question,
        "patient_profile_used": bool(patient_profile),
        "mode": "rag" if req.use_rag else "no_rag",
        "answer": answer,
        "safety": _safety_payload(safety),
        "citations": hits_to_citations(hits),
    }
