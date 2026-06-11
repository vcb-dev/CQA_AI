# CQA CRM - AI Service

## 🚀 Công nghệ sử dụng

- **FastAPI** - Web framework
- **LangChain** - LLM orchestration
- **DeepSeek / OpenAI** - LLM provider
- **FAISS** - Vector store cho RAG
- **PyMuPDF** - Đọc PDF

---

## 📁 Cấu trúc thư mục

```
CQA_AI/
├── main.py                          # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── .env                             # Cấu hình môi trường
├── src/
│   ├── api/                         # (Tương lai: router phân tách)
│   ├── core/
│   │   └── ingestion.py             # Trích xuất nội dung PDF
│   ├── models/
│   │   └── assistant_models.py      # Pydantic request/response models
│   ├── prompts/
│   │   └── system_prompt.py         # System prompt templates
│   ├── services/
│   │   ├── assistant_service.py     # Logic chat LLM chính
│   │   └── rag_store.py             # FAISS vector store
│   └── utils/
│       └── user_facing_error.py     # Chuyển lỗi kỹ thuật → thân thiện
├── data/
│   ├── docs/                        # PDF và tài liệu nội bộ
│   ├── knowledge/                   # File TXT/MD bổ sung
│   └── faiss_index/                 # (Auto-generated) FAISS index
├── scripts/
│   └── index_knowledge.py           # Build FAISS index từ tài liệu
└── docs/
    └── cqa-assistant.md             # Tài liệu API
```

---

## ⚙️ Cài đặt

### 1. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình `.env`

```env
PORT=8000
HOST=0.0.0.0

DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
ASSISTANT_LLM_MODEL=deepseek-chat

ASSISTANT_INTERNAL_SECRET=your-internal-secret
```

### 4. Chạy server

```bash
python main.py
# hoặc
uvicorn main:app --reload --port 8000
```

---

## 📋 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| GET | `/` | Thông tin service |
| GET | `/health` | Health check |
| GET | `/assistant/health` | Assistant health |
| POST | `/assistant/chat` | Chat với AI |

### POST `/assistant/chat`

**Header**: `X-Assistant-Secret: <secret>`

```json
{
  "message": "Xin chào, tôi cần hỗ trợ",
  "history": [],
  "user_context": {
    "user_id": "123",
    "email": "user@cqa.vn",
    "full_name": "Nguyễn Văn A",
    "app_role": "USER",
    "department": "Kinh doanh"
  },
  "session_facts": {}
}
```

---

## 🗂️ RAG - Knowledge Base

Để thêm tài liệu vào knowledge base:

1. Đặt file PDF/TXT vào `data/docs/` hoặc `data/knowledge/`
2. Chạy script build index:

```bash
python scripts/index_knowledge.py
```

3. Restart server

---

## 🐳 Docker

```bash
docker build -t cqa-ai .
docker run -p 8000:8080 --env-file .env cqa-ai
```
