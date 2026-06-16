from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
print('health=', client.get('/health').status_code, client.get('/health').json())
resp = client.post('/ask', json={'question': '什么是视网膜母细胞瘤？', 'top_k': 3})
print('ask_status=', resp.status_code)
body = resp.json()
print('answer_len=', len(body.get('answer', '')))
print('citations=', len(body.get('citations', [])))
print('safety=', body.get('safety', {}).get('level'))
