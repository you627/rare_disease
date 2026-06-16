from pathlib import Path
from src.retriever import HybridRetriever
from src.generator import generate_answer
from src.config import settings

q = '什么是视网膜母细胞瘤？'
r = HybridRetriever.load(Path('data/retriever.pkl'))
hits = r.search(q, top_k=3)
ctx_parts = []
for i, h in enumerate(hits):
    ctx_parts.append(f"[{i+1}] {h['title']} | {h['field']}\n{h['text'][:300]}")
ctx = "\n\n".join(ctx_parts)
ans = generate_answer(q, ctx)
print('hits=', len(hits))
print('top1=', hits[0]['title'], hits[0]['field'], round(hits[0]['score'], 3))
print('openai_enabled=', bool(settings.openai_api_key))
print('answer_preview=', ans[:400])
