# UniAssist — University AI Assistant 🎓

An intelligent, modular **Retrieval-Augmented Generation (RAG)** platform for university students and staff to query official policies, placement guidelines, dress codes, academic benefit plans, library rules, and campus maps.

---

## 🌟 Highlights

- **Fast & Accurate Retrieval**: ChromaDB persistent vector database powered by HuggingFace `all-MiniLM-L6-v2` embeddings.
- **Reranking Intelligence**: Cohere cross-encoder reranker for high-precision semantic selection.
- **Multi-Provider LLM Integration**: OpenRouter, xAI Grok, or any OpenAI-compatible model with structured citations.
- **Multi-Format Ingestion**: PDF (`pypdf`/`pdfplumber`), DOCX (`python-docx`), and UTF-8 `.txt`/`.md` loaders with zero heavy native dependencies.
- **ChatGPT-Style UI**: Next.js 16 App Router chat interface with markdown tables, multi-turn history, source citations, and per-chat persistence.

---

## 🏗️ Project Architecture

```
LPUASSIST/
├── backend/
│   ├── app/
│   │   ├── config.py             # Pydantic environment configuration
│   │   ├── main.py               # FastAPI application & endpoints
│   │   ├── ingestion/            # Partition, chunk, summarize & embed
│   │   └── query/                # Retrieve, rerank & generate
│   ├── requirements.txt          # Python dependencies
│   └── .env.example              # Environment variables template
├── frontend/
│   ├── app/                      # Next.js 15 App Router (ChatGPT UI)
│   ├── public/                   # Static assets & icons
│   └── package.json              # Frontend dependencies
├── Data/                         # University documents (PDF, DOCX, TXT, JPG)
└── README.md                     # Project documentation
```

---

## 🚀 Quickstart

### 1. Backend Setup
```bash
cd backend
python -m venv venv

# Activate the virtualenv
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\Activate.ps1   # Windows (PowerShell)

pip install -r requirements.txt

# Configure environment variables, then edit .env
cp .env.example .env
# Required for answers/summaries: LLM_API_KEY (legacy GROK_API_KEY still works)
# Optional: COHERE_API_KEY (reranking), CHROMA_HOST (shared server),
#           INGEST_TOKEN (enables POST /api/ingest), ALLOWED_ORIGINS (CORS)

# Run offline ingestion (recommended way to (re)build the index)
python -m app.ingestion.pipeline

# Start API server
uvicorn app.main:app --reload --port 8000
```

> Ingestion can also be triggered over HTTP with `POST /api/ingest`, but only
> when `INGEST_TOKEN` is set — send it as `Authorization: Bearer <token>`.

### 2. Frontend Setup
```bash
cd frontend
npm install

# Optional: point the UI at a non-default backend
cp .env.example .env.local      # then set NEXT_PUBLIC_API_BASE

npm run dev
```

Visit **http://localhost:3000** in your browser.

> Note: image files (e.g. the campus map `.jpg`) are not ingested; provide a
> `.txt`/`.md` companion for any content that must be searchable.
