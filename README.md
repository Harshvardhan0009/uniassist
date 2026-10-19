# UniAssist — University AI Assistant 🎓

An intelligent, modular **Retrieval-Augmented Generation (RAG)** platform designed for university students and staff to query official policies, placement guidelines, dress codes, academic benefit plans, library rules, and campus maps with exact page-level citations.

> **Evaluation Program Status: 100% Complete (Phases 0–25).**  
> Complete decision records, ablation studies, and benchmark results are documented in **[`evaluation/README.md`](evaluation/README.md)** and summarized in **[`docs/PROJECT_HANDOFF.md`](docs/PROJECT_HANDOFF.md)** and **[`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md)**.

---

## 🌟 Production System Highlights

* **High-Precision Embeddings**: Powered by HuggingFace `intfloat/e5-base-v2` (768-dim) in cosine space, outperforming standard baseline models with prefix-guided queries (`query: `) and passages (`passage: `).
* **Reranking Intelligence**: Cohere `rerank-v3.5` cross-encoder reranker narrows Top-20 retrieved candidates down to Top-5 high-relevance chunks.
* **Calibrated Generation & LLM Failover**: Primary generation powered by Google `gemini-3.6-flash` with zero phantom citations and automatic failover to Groq (`openai/gpt-oss-120b`).
* **Conversational Query Rewriter**: Multi-turn history resolution (+50% conversational recall) resolving ambiguous follow-ups into standalone search queries.
* **Adaptive Context Compression**: Trims redundant distractor chunks (-15.8% prompt tokens) while retaining 100% source answerability.
* **Real-Time Streaming UX**: Next.js 16 App Router interface featuring Server-Sent Events (SSE) token streaming, inline citation badges (`📄 clean_name`), follow-up context pills, and per-chat history.
* **Live Observability**: Runtime telemetry window tracking TTFT, retrieval, rerank, and generation percentiles via `GET /api/metrics`.

---

## 🏗️ Project Architecture

```
UNIASSIST/
├── backend/
│   ├── app/
│   │   ├── config.py             # Pydantic v2 environment settings
│   │   ├── main.py               # FastAPI application, streaming & health routes
│   │   ├── telemetry.py          # Rolling latency and cost observability
│   │   ├── ingestion/
│   │   │   ├── partition.py      # Table-aware PDF/DOCX/TXT extraction (pdfplumber)
│   │   │   ├── chunker.py        # 1500 chars / 200 overlap text splitting
│   │   │   ├── embedder.py       # E5-base-v2 prefixed vector embeddings
│   │   │   └── pipeline.py       # Offline and CLI ingestion orchestrator
│   │   └── query/
│   │       ├── rewriter.py       # Turn 2+ conversational query resolution
│   │       ├── retriever.py      # Prefix-aware ChromaDB similarity retrieval
│   │       ├── reranker.py       # Cohere rerank-v3.5 cross-encoder
│   │       ├── compressor.py     # Adaptive context tail pruning
│   │       ├── generator.py      # Calibrated LLM generation & SSE streaming
│   │       └── chain.py          # Unified RAG query pipeline orchestrator
│   ├── requirements.txt          # Python dependencies
│   └── .env.example              # Environment variables template
├── frontend/
│   ├── app/                      # Next.js 16 App Router (Turbopack, SSE streaming)
│   │   ├── page.js               # Chat UI, streaming reader & citation renderer
│   │   ├── page.module.css       # Responsive styling, inline badges & cursors
│   │   └── globals.css           # Design tokens, color scheme & typography
│   ├── public/                   # Static assets & logos
│   └── package.json              # Frontend dependencies
├── evaluation/                   # 25-Phase evaluation framework & benchmarks
│   ├── benchmarks/               # Frozen 63-question evaluation dataset
│   ├── configs/                  # Reproducible experiment configurations
│   ├── metrics/                  # Retrieval, generation & faithfulness evaluators
│   └── reports/                  # Formal decision records & ablation reports
├── docs/                         # System architecture, handoff & state docs
├── Data/                         # University documents (14 PDFs, 144 pages)
└── docker-compose.yml            # Local containerized ChromaDB setup
```

---

## 🚀 Quickstart

### 1. Backend Setup

```bash
cd backend
python -m venv venv

# Activate the virtual environment
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\Activate.ps1   # Windows (PowerShell)

pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env to add your LLM_API_KEY and COHERE_API_KEY.

# Ingest university documents into ChromaDB
python -m app.ingestion.pipeline

# Start the API server
uvicorn app.main:app --reload --port 8000
```

> **API Endpoints:**
> - `POST /api/query/stream` — Real-time SSE token stream with citation metadata
> - `POST /api/query` — Standard synchronous JSON query endpoint
> - `GET /api/health` — System status, ChromaDB connection & indexed document count
> - `GET /api/metrics` — Runtime latency percentiles (p50, p95) and token metrics

### 2. Frontend Setup

```bash
cd frontend
npm install

# Point the UI to the backend (defaults to http://localhost:8000)
cp .env.example .env.local

# Run development server
npm run dev
```

Visit **http://localhost:3000** in your browser.

---

## 📊 Evaluation & Benchmarking Results

| Component | Winning Configuration | Metric / Improvement |
| :--- | :--- | :--- |
| **Embedding** | `intfloat/e5-base-v2` (768-dim) | Hit@5: **88.9%**, MRR: **0.782** |
| **Reranker** | Cohere `rerank-v3.5` | Top-5 Precision: **+22.4%** |
| **Chunking** | 1500 chars / 200 overlap | Hit@5: **88.9%**, Zero table truncation |
| **Parser** | `pdfplumber` (table-preserving) | Numerical extraction accuracy: **100%** |
| **Primary LLM** | `gemini-3.6-flash` | Human eval: **4.68 / 5.00**, 0 hallucinated citations |
| **Fallback LLM** | Groq `openai/gpt-oss-120b` | Zero-downtime failover |
| **Rewriter** | History-conditioned LLM | Follow-up conversational recall: **+50.0%** |
| **Compression**| Score-aware tail pruning | Prompt token reduction: **-15.8%** |
