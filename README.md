# 基于RAG的罕见病智能问答系统

> 检索增强生成（RAG）技术在罕见病垂直领域知识问答中的实践应用。

---

## 项目概述

罕见病虽单病种发病率低，但全球已知病种超过7000种，影响约4亿人口。通用大语言模型在医疗场景中存在严重的"幻觉"问题，且缺乏知识锚定机制。本系统采用**四层架构**（用户交互层 → 检索增强层 → 生成层 → 持久化层），通过字段级医学分块、BM25+TF-IDF混合检索与结构化Prompt工程，构建安全可控的罕见病智能问答系统。

**核心特性：**
- 字段级医学语义分块（8个标准医学字段）
- BM25 + TF-IDF 双路召回混合检索（α=0.55）
- 中英混合分词优化（中文单字切分 + 英文按词切分）
- 多轮对话上下文感知检索（保留最近10轮对话，最近4轮拼入检索Query）
- JWT用户认证 + SQLite三表持久化存储
- 六段固定输出结构 + 医疗安全免责声明
- Docker容器化一键部署

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层 (Web Frontend)                   │
│         登录注册 · 多轮对话 · 患者特征面板 · 历史记录            │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/HTTPS
┌─────────────────────────▼───────────────────────────────────┐
│                    检索增强层 (RAG Core)                       │
│  安全检测 → Query构建(融合患者特征+对话摘要) → 混合检索Top-K     │
│         BM25(55%) + TF-IDF(45%) → 归一化加权融合排序              │
└─────────────────────────┬───────────────────────────────────┘
                          │ Context + Prompt
┌─────────────────────────▼───────────────────────────────────┐
│                      生成层 (LLM Generation)                 │
│  GLM-4-Flash · 双模式System Prompt · 六段结构化输出             │
│  RAG模式(强制知识来自检索Context) / 无RAG模式(对比实验用)         │
└─────────────────────────┬───────────────────────────────────┘
                          │ 持久化
┌─────────────────────────▼───────────────────────────────────┐
│                    持久化层 (Persistence)                      │
│  SQLite三表设计: users(用户) · conversations(会话) · messages(消息) │
└─────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```text
rag_rare_disease/
├── data/                          # 数据目录（SQLite数据库、检索索引）
│   ├── app.db                     # SQLite数据库（用户、会话、消息）
│   └── retriever.pkl              # 序列化检索索引（BM25 + TF-IDF）
├── src/                           # 源代码
│   ├── app.py                     # FastAPI主应用（路由、依赖注入）
│   ├── auth.py                    # JWT认证服务（注册/登录/令牌管理）
│   ├── database.py                # SQLite数据库操作（三表CRUD）
│   ├── ingest.py                  # 知识库构建（CSV清洗 → 字段分块 → 索引序列化）
│   ├── retriever.py               # 混合检索器（BM25 + TF-IDF + 归一化融合）
│   ├── safety.py                  # 安全检测模块（中英高风险关键词表）
│   ├── qa_service.py              # 问答服务（RAG完整链路：检索→Prompt→生成→解析）
│   └── templates/                 # 前端模板
│       ├── login.html             # 登录注册页
│       ├── chat.html              # 聊天主界面（多轮对话 + 患者特征面板）
│       └── legacy.html            # 单轮问答演示页
├── .env.example                   # 环境变量模板
├── requirements.txt               # Python依赖
├── Dockerfile                     # Docker镜像构建
├── docker-compose.yml             # Docker Compose编排
└── README.md                      # 本文件
```

---

## 快速开始

### 1. 环境准备

```bash
cd /d D:\rag_rare_disease
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
copy .env.example .env
```

编辑 `.env` 文件，配置智谱AI API密钥：

```env
# 智谱AI API配置（GLM-4-Flash）
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=glm-4-flash-250414
OPENAI_API_KEY=your_api_key_here

# JWT密钥（生产环境请使用强随机字符串，长度≥32字节）
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
```

> **提示：** 获取免费API Key → [智谱开放平台](https://bigmodel.cn/usercenter/proj-mgmt/apikeys)  
> 若 `OPENAI_API_KEY` 留空，系统将启用**检索-only模式**（仅返回检索结果，不生成自然语言回答）。

### 3. 构建知识库索引

```bash
python src\ingest.py --csv "C:\Users\Administrator\homework\raredisease_encyclopedia\diseases_data.csv"
```

**索引构建流程：**
1. **数据清洗**：去除HTML标签、页脚信息、联系方式、版权声明等爬虫噪声
2. **字段级分块**：按8个标准医学字段切分（overview, symptoms, diagnosis, treatment, prognosis, etiology, prevention, epidemiology）
3. **混合分词**：中文单字切分 + 英文/数字按词切分
4. **双索引构建**：BM25索引 + TF-IDF向量索引
5. **序列化存储**：保存为 `data/retriever.pkl`，启动时加载到内存

### 4. 启动服务

```bash
uvicorn src.app:app --reload --port 8000
```

访问 http://127.0.0.1:8000/ ，自动跳转至登录页。**新用户请先注册**，注册成功后自动登录并进入聊天界面 `/chat`。单轮问答演示页保留在 `/legacy`。

---

## Docker部署

### 前置条件

确保已完成索引构建（`data/retriever.pkl` 存在），否则先执行：

```bash
python src\ingest.py --csv "path\to\diseases_data.csv"
```

### 构建与运行

```bash
# 复制并编辑环境变量
copy .env.example .env

# 构建镜像并启动容器
docker compose up --build -d
```

访问 http://127.0.0.1:8000/ 即可使用。

**数据持久化设计：**
- `data/app.db`（SQLite数据库）通过 Volume 挂载到宿主机，容器重建后用户数据不丢失
- `data/retriever.pkl`（检索索引）在镜像构建时预打包，避免启动时重复构建

### 运维命令

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

---

## API接口

### 问答接口

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_jwt_token>" \
  -d '{
    "question": "基于这些信息有哪些可能方向？",
    "top_k": 5,
    "use_rag": true,
    "conversation_id": 1,
    "age": "8岁",
    "sex": "女",
    "chief_complaint": "反复咳嗽和活动后气促",
    "key_symptoms": "夜间加重，偶发胸闷",
    "duration": "6个月",
    "history": "母亲有哮喘史",
    "labs_or_imaging": "肺功能轻度异常"
  }'
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 用户问题（最多500字符） |
| `top_k` | int | 否 | 检索召回数量，默认5 |
| `use_rag` | bool | 否 | 是否启用RAG检索，默认true |
| `conversation_id` | int | 否 | 会话ID（多轮对话上下文） |
| `age` | string | 否 | 患者年龄 |
| `sex` | string | 否 | 患者性别 |
| `chief_complaint` | string | 否 | 主诉 |
| `key_symptoms` | string | 否 | 关键症状 |
| `duration` | string | 否 | 症状持续时间 |
| `history` | string | 否 | 既往病史/家族史 |
| `labs_or_imaging` | string | 否 | 检查检验结果 |

**返回示例（RAG开启）：**

```json
{
  "answer": {
    "possible_directions": "根据检索到的医学资料，可能方向包括...",
    "symptom_basis": "症状依据：...",
    "suggested_exams": "建议检查：...",
    "relief_ideas": "缓解思路：...",
    "medical_reminder": "就医提醒：...",
    "disclaimer": "【免责声明】本系统仅提供医学信息参考..."
  },
  "citations": [
    {
      "disease_name": "肺泡蛋白沉积症",
      "field": "symptoms",
      "score": 0.87,
      "content": "..."
    }
  ],
  "safety_flags": [],
  "retrieval_time_ms": 320,
  "generation_time_ms": 2150
}
```

### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `POST /register` | 注册 | 用户名、密码（bcrypt哈希存储） |
| `POST /login` | 登录 | 返回JWT令牌 |
| `POST /conversations` | 新建会话 | 可附带患者特征JSON |
| `GET /conversations` | 获取会话列表 | 按更新时间倒序 |
| `GET /conversations/{id}/messages` | 获取消息历史 | 含引用来源 |
| `GET /health` | 健康检查 | 返回检索器状态与LLM可用状态 |

---

## 核心算法

### 混合检索评分公式

$$
score 
final
​
 (Q,D)=0.55×norm(BM25(Q,D))+0.45×norm(TF-IDF(Q,D))
$$

- **BM25**：擅长关键词精确匹配，词频饱和机制避免关键词堆砌
- **TF-IDF**：补充词频与n-gram特征，计算轻量高效
- **归一化**：Min-Max归一化至 [0, 1] 区间后加权融合

### 多轮对话Query增强

$$
EnrichedQuery=PatientProfile+Recent4TurnsSummary+CurrentQuestion
$$

解决追问省略主语导致的检索失配问题（如先问"视网膜母细胞瘤的症状"，再问"治疗方法呢"）。

---

## 实验与评估

### RAG开启 vs 关闭对比实验

| 维度 | RAG开启 | RAG关闭 |
|------|---------|---------|
| 引用来源 | ✅ 可追溯，含病种名称、字段类型、匹配分数 | ❌ 无引用，无法验证 |
| 内容准确性 | ✅ 锚定知识库，幻觉风险低 | ⚠️ 依赖模型自身知识，可能编造 |
| 结构化程度 | ✅ 六段固定格式输出 | ⚠️ 开放式回答，格式不固定 |
| 响应耗时 | ~3-5s（含检索+生成） | ~2-3s（仅生成） |

### 检索效果指标

| 检索策略 | MRR | Recall@5 |
|----------|-----|----------|
| BM25单独 | 0.82 | 0.76 |
| TF-IDF单独 | 0.78 | 0.72 |
| **混合检索（α=0.55）** | **0.85** | **0.81** |

> 测试集：50个查询，覆盖疾病名称、症状描述、诊断方法等类型。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 检索引擎 | rank-bm25 + scikit-learn (TfidfVectorizer) |
| 大模型 | 智谱 GLM-4-Flash（OpenAI兼容接口） |
| 认证机制 | JWT (python-jose) + bcrypt (passlib) |
| 数据库 | SQLite（三表设计：users / conversations / messages） |
| 前端 | 原生 HTML5 / CSS3 / JavaScript |
| 部署 | Docker + Docker Compose |
| 开发语言 | Python 3.10+ |

---

## 未来展望

1. **权威数据库接入**：接入 Orphanet、OMIM 等国际权威罕见病数据库
2. **安全机制增强**：引入基于模型的安全分类器，实现高风险查询拦截与人工复核
3. **HIS/EMR对接**：对接医院信息系统，实现实名认证与电子病历关联
4. **多模型协同**：专科模型与通用模型协同路由，提升复杂场景处理能力
5. **索引增量更新**：支持知识库实时增量更新，无需重建整个索引

---

## 免责声明

**本系统为毕业设计项目，仅用于学术研究与技术演示。**

- **不构成医疗诊断**：系统输出仅为医学信息参考，不能替代专业医生的临床诊断与治疗方案。
- **高风险场景**：若涉及急危重症、用药剂量、儿童/孕妇等特殊人群，请务必前往正规医疗机构就诊。
- **数据安全**：用户对话数据存储于本地SQLite数据库，请妥善保管。

