# University RAG System — Step-by-Step Architecture

A multi-modal Retrieval-Augmented Generation (RAG) system for answering university-related queries (admissions, courses, policies, faculty, schedules, etc.) from a corpus of documents containing text, tables, and map legends.

**Key Architecture Choices:** 
- `all-MiniLM-L6-v2` / `BGE-M3` (lightweight embeddings)
- `ChromaDB` (persistent vector store)
- `Cohere Rerank v3.5` (cross-encoder reranking)
- `OpenRouter / Grok / OpenAI` (LLM answer generation)
- `PyPDF / python-docx / TextLoader` (zero-native-dependency partitioning)
- `LangChain` (pipeline orchestration)
- `Next.js 15 App Router` (ChatGPT-style conversational UI)

---

## Step 1 — Set up storage foundations

Before any pipeline logic, get the base infra in place:

1. Create an **Amazon S3** bucket for raw PDF uploads and extracted images.
2. Set up **Supabase (PostgreSQL)** for relational metadata: users, document records, processing status.
3. Set up **Clerk auth** on the Next.js frontend so uploads are tied to a user.
4. Set up **Redis** and a **Celery** worker so document processing runs asynchronously off the request/response cycle.

*Outcome: a user can authenticate, upload a PDF, and have it land in S3 with a corresponding "pending" record in Supabase.*

---

## Step 2 — Upload triggers the processing queue

1. FastAPI receives the upload, stores the file in S3, creates a Supabase record with status `"processing"`, and pushes a Celery task onto the Redis queue.
2. The Celery worker picks up the task and runs Steps 3–6 below as a single task per document.

---

## Step 3 — Partition the PDF into atomic elements

Use **Unstructured.io** to break the PDF into atomic elements:

- Output element types: Header, Title, Image, Table, NarrativeText, Caption
- Expect roughly ~600 atomic elements per document (varies by document size/complexity)

*This is the step that separates text, tables, and images out of the raw PDF so each can be handled appropriately downstream.*

---

## Step 4 — Chunk the elements by title

1. Apply Unstructured's **chunk-by-title** strategy to group atomic elements into semantically coherent chunks.
2. Each chunk falls into one of three types:
   - Text only
   - Text + table
   - Text + image
3. Expect roughly ~40+ chunks per document.

---

## Step 5 — Summarize chunks and convert to LangChain Documents

For each chunk, call **Grok** to generate a summary, then build a LangChain `Document`:

- `page_content` = summarized text (this is what gets embedded and searched against)
- `metadata` = the original raw content (raw text / raw text + table / raw text + image reference)

This split matters: you search against the summary, but you generate the final answer from the **raw** content in metadata — so you don't lose precision (exact dates, table values, names) to summarization.

---

## Step 6 — Embed and store in ChromaDB

1. Generate embeddings with **BGE-M3** (self-hosted, free, MIT-licensed) — run it as a separate inference microservice rather than loading it in-process, so it scales independently and can be reused for both ingestion and query-time embedding.
2. Configure Chroma's `embedding_function` explicitly to use BGE-M3 (it does not default to this).
3. Store the LangChain Documents (embeddings + page_content + metadata) in a ChromaDB collection.
4. Update the Supabase record status from `"processing"` to `"completed"`.

*Ingestion pipeline complete. At this point, documents are searchable.*

---

## Step 7 — Handle a user query: embed and retrieve

1. User submits a query (e.g., "What is the last date to drop a course?").
2. Embed the query using the same **BGE-M3** model used at ingestion.
3. Run a similarity search against **ChromaDB** and retrieve a wide candidate set — top-k, roughly k ≈ 20–25.

---

## Step 8 — Rerank the candidates

1. Send the top-k candidates through **Cohere Rerank (`rerank-v3.5`)**.
2. Narrow down to the top-n most relevant — roughly n ≈ 5–8.

*This step matters because Chroma's cosine similarity is a coarse filter; the cross-encoder reranker is the precision pass, especially important on a mixed-content, table-heavy corpus. Expect ~100–300ms added latency per query from this external API call — acceptable for a chat-style Q&A tool.*

---

## Step 9 — Generate the final answer

1. Pull the **raw content** (not the summary) from each reranked document's metadata.
2. Pass this raw context + the user's query to **Grok** for final answer generation.
3. Return the response to the user.

---

## Step 10 — (Phase 2) Add adaptive query routing

Once the basic pipeline is working end-to-end, introduce a lightweight routing layer with **LangGraph**:

- **Simple queries** (single-fact lookup) → Steps 7–9 as-is
- **Complex queries** (multi-hop, comparative, ambiguous) → allow 1–2 additional retrieval iterations with query reformulation before generation

This handles the university's varied query surface without the cost/complexity of a full multi-agent system.

---

## Step 11 — (Phase 2) Harden for production

Once Steps 1–9 are working reliably:

- [ ] Add **hybrid search** (BGE-M3 dense + sparse vectors) for exact-match queries (course codes, policy numbers)
- [ ] Add **metadata filtering** (department, doc type, academic year) for scoped retrieval
- [ ] Add **LangSmith** (or equivalent) tracing to debug retrieval/generation quality
- [ ] Monitor **ChromaDB scale limits** — plan a migration path to a dedicated vector DB (Qdrant/Weaviate/Milvus) if the corpus grows past a few million vectors
- [ ] Build an **evaluation pipeline** (retrieval MRR/NDCG, answer quality) — benchmark BGE-M3 and Grok on actual university data, not just public leaderboards
- [ ] Monitor Cohere Rerank usage/cost as query volume grows
- [ ] Add sync logic between S3 / Supabase / ChromaDB on document delete/update (they're separate systems and won't stay in sync automatically)

---

## Reference: Tech stack by layer

| Layer | Tool |
|---|---|
| Frontend | Next.js, TypeScript, Clerk (auth), deployed on Vercel |
| Backend API | FastAPI |
| Async processing | Redis + Celery |
| Relational metadata | Supabase (PostgreSQL) |
| Raw file storage | Amazon S3 |
| PDF partitioning | Unstructured.io |
| Embedding model | BGE-M3 |
| Vector store | ChromaDB |
| Reranker | Cohere (rerank-v3.5) |
| LLM (summarization + generation) | Grok |
| Orchestration | LangChain / LangGraph |
| Deployment | Docker, AWS |
