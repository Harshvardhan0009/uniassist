# UniAssist — University AI Assistant 🎓

An intelligent, modular **Retrieval-Augmented Generation (RAG)** platform for university students and staff to query official policies, placement guidelines, dress codes, academic benefit plans, library rules, and campus maps.

---

## 🌟 Highlights

- **Fast & Accurate Retrieval**: ChromaDB persistent vector database powered by HuggingFace `all-MiniLM-L6-v2` embeddings.
- **Reranking Intelligence**: Cohere cross-encoder reranker for high-precision semantic selection.
- **Multi-Provider LLM Integration**: OpenRouter, xAI Grok, or OpenAI-compatible models with structured citations.
- **Multi-Format Ingestion**: PyPDF, python-docx, and UTF-8 text/markdown document loaders with zero heavy native dependencies.
- **ChatGPT-Style UI**: Next.js 15 App Router chat interface with real-time feedback, typing indicators, suggestion chips, and source citations.

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
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Run offline ingestion
python3 -m app.ingestion.pipeline

# Start API server
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:3000** in your browser.
