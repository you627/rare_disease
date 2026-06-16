# Rare Disease RAG Starter

Minimal runnable RAG project for rare disease QA from CSV.

## Features

- Build knowledge index from CSV
- Hybrid retrieval (BM25 + TF-IDF)
- FastAPI QA endpoint
- Basic medical safety checks
- Optional OpenAI answer generation
- Optional patient profile input for non-diagnostic medical support answers
- User login (simulated, no password rules) with JWT
- Multi-turn conversations persisted in SQLite
- Basic crawler-noise cleanup during ingest (footer/contact/record-number removal)

## Structure

```text
rag_rare_disease/
  |- data/
  |- src/
  |- .env.example
  |- requirements.txt
  `- README.md
```

## 1) Install

```bash
cd /d D:\rag_rare_disease
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configure

```bash
copy .env.example .env
```

Edit `.env` and set your [智谱 API Key](https://bigmodel.cn/usercenter/proj-mgmt/apikeys). Default LLM:

- `OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4`
- `OPENAI_MODEL=glm-4-flash-250414` ([GLM-4-Flash 免费版](https://docs.bigmodel.cn/cn/guide/models/free/glm-4-flash-250414))

If `OPENAI_API_KEY` is empty, retrieval-only mode is used (no generated answer).

## 3) Build index

```bash
python src\ingest.py --csv "C:\Users\Administrator\homework\raredisease_encyclopedia\diseases_data.csv"
```

## 4) Start API

```bash
uvicorn src.app:app --reload --port 8000
```

Open http://127.0.0.1:8000/ (redirects to login). **New users: register first**, then you are auto-logged in and sent to `/chat`.
The legacy single-turn page remains at `/legacy`.

## Docker

Build index first (if `data/retriever.pkl` is missing):

```bash
python src\ingest.py --csv "path\to\diseases_data.csv"
```

Copy env and set your API key:

```bash
copy .env.example .env
```

Build and run:

```bash
docker compose up --build -d
```

Open http://127.0.0.1:8000/ . User data (`data/app.db`) is persisted via volume mount.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Stop:

```bash
docker compose down
```

## 5) Example request

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"基于这些信息有哪些可能方向？\",\"top_k\":5,\"use_rag\":true,\"age\":\"8岁\",\"sex\":\"女\",\"chief_complaint\":\"反复咳嗽和活动后气促\",\"key_symptoms\":\"夜间加重，偶发胸闷\",\"duration\":\"6个月\",\"history\":\"母亲有哮喘史\",\"labs_or_imaging\":\"肺功能轻度异常\"}"
```

## Presentation materials

Chinese PPT outline, speaker notes, and experiment templates:

- `docs/PPT汇报材料.md` — slide content + 演讲者备注（18 页）
- `docs/对比实验记录表.md` — RAG 对比 / 多轮 / 安全实验填写表

## Notes

- This is an MVP and not a clinical diagnosis system.
- Output is medical decision support only (possible directions, not definitive diagnosis).
- Keep clinician review for high-risk cases.
