"""Experiment orchestration — glue between snapshot, index, harness and runners.

Provides:
* ``load_questions`` — read the benchmark dataset.
* ``prepare_store`` — ensure the frozen snapshot + isolated local index exist for
  a config, returning the ready-to-query store plus provenance.
* ``run_retrieval`` — one question -> raw ranked retrieval + rerank record.
* ``run_full_rag`` — one question -> end-to-end record incl. generated answer.

No production state is touched: retrieval uses the caller's local store; only
generation reuses the production LLM client (``app.query.generator``), which
does not touch the vector store.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from . import paths
from . import snapshot as snap
from .harness import (
    ExperimentConfig,
    doc_ref,
    rerank_candidates,
    retrieve_candidates,
)
from .indexer import build_index
from .rewriter import rewrite_query

paths.ensure_backend_on_path()

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from app.config import settings  # noqa: E402
from app.query.generator import (  # noqa: E402
    ANSWER_PROMPT,
    _build_clean_answer,
    _clean_answer_text,
    _format_context,
    _to_history_messages,
    generate_answer,
)

logger = logging.getLogger(__name__)


def load_questions(path: Path | None = None) -> dict:
    """Load the benchmark dataset JSON (full document, incl. metadata + questions)."""
    path = path or paths.QUESTIONS_PATH
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_store(
    config: ExperimentConfig, rebuild_snapshot: bool = False, rebuild_index: bool = False
) -> tuple[Chroma, dict, dict, int]:
    """Ensure snapshot + isolated index exist; return (store, manifest, index_info, n_docs)."""
    if rebuild_snapshot or not snap.snapshot_exists(config.snapshot):
        logger.info(
            "Snapshot '%s' missing or rebuild requested — building it (chunk %d/%d).",
            config.snapshot, config.chunk_size, config.chunk_overlap,
        )
        snap.build_snapshot(
            name=config.snapshot,
            use_llm=(config.content_indexed == "llm_summary"),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            parser=config.parser,
        )

    docs = snap.load_snapshot(config.snapshot)
    manifest = snap.load_manifest(config.snapshot)
    store, index_info = build_index(
        docs, collection=config.collection, embedding_model=config.embedding_model,
        force=rebuild_index, query_prefix=config.query_prefix, passage_prefix=config.passage_prefix,
    )
    return store, manifest, index_info, len(docs)


def _retrieve_and_rerank(store: Chroma, config: ExperimentConfig, item: dict) -> tuple[dict, list]:
    """Retrieve (+rerank) once for one question.

    Returns ``(record, reranked_docs)`` where ``record`` is the raw ranked
    record and ``reranked_docs`` are the actual reranked Documents (so callers
    that also generate an answer reuse the *same* reranked context instead of
    issuing a second, redundant rerank call — important on a rate-limited key).

    If config.query_rewriting is True and the question has conversational history,
    the question is reformulated into a standalone query before retrieval/reranking.
    """
    question = item["question"]
    is_conversational = bool(item.get("history"))

    effective_query = question
    rewrite_ms = 0.0
    if config.query_rewriting and is_conversational:
        effective_query, rewrite_ms = rewrite_query(question, item.get("history"))

    lat_bd = None
    if config.hybrid:
        from .hybrid import retrieve_hybrid_candidates
        candidates, ret_ms, lat_bd = retrieve_hybrid_candidates(
            store=store,
            snapshot_name=config.snapshot,
            query=effective_query,
            top_k=config.top_k,
            rrf_k=config.hybrid_rrf_k,
            dense_weight=config.hybrid_dense_weight,
            sparse_weight=config.hybrid_bm25_weight,
            min_score=config.min_relevance_score,
        )
    else:
        candidates, ret_ms = retrieve_candidates(store, effective_query, config.top_k, config.min_relevance_score)
    reranked, rr_ms, reranker_used = rerank_candidates(
        effective_query, candidates, config.rerank_top_n, config.rerank_model, config.rerank_enabled
    )

    latencies = {"rewriting": rewrite_ms, "retrieval": ret_ms, "reranking": rr_ms}
    if lat_bd:
        latencies["hybrid_breakdown"] = lat_bd

    record = {
        "id": item["id"],
        "category": item.get("category"),
        "topic": item.get("topic"),
        "difficulty": item.get("difficulty"),
        "answerable": item.get("answerable", True),
        "question": question,
        "query_used": effective_query,
        "rewritten_query": effective_query if (config.query_rewriting and is_conversational) else None,
        "conversational": is_conversational,
        "history_used_in_retrieval": bool(config.query_rewriting and is_conversational),
        "expected_sources": item.get("expected_sources", []),
        "expected_pages": item.get("expected_pages", []),
        "retrieved": [doc_ref(d, i + 1) for i, d in enumerate(candidates)],
        "reranked": [doc_ref(d, i + 1) for i, d in enumerate(reranked)],
        "reranker_used": reranker_used,
        "n_retrieved": len(candidates),
        "n_reranked": len(reranked),
        "latency_ms": latencies,
    }
    return record, reranked


def run_retrieval(store: Chroma, config: ExperimentConfig, item: dict) -> dict:
    """Run retrieve (+rerank) for one question and return a raw ranked record."""
    record, _ = _retrieve_and_rerank(store, config, item)
    return record


def load_snapshot_chunks(snapshot_name: str) -> dict[str, Document]:
    """Load all Document objects from snapshots/<snapshot_name>.jsonl keyed by chunk_id."""
    path = paths.SNAPSHOT_DIR / f"{snapshot_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    chunks: dict[str, Document] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            meta = data.get("metadata", {})
            raw = meta.get("raw_content", data.get("page_content", ""))
            cid = f"{meta.get('source_file')}__chunk_{meta.get('chunk_index')}"
            chunks[cid] = Document(page_content=raw, metadata=meta)
    return chunks


def _resolve_eval_llm(config: ExperimentConfig) -> tuple[str, str, str]:
    model = config.llm_model
    base_url = config.llm_base_url
    provider = (config.llm_provider or "").lower()

    if "groq" in provider or "groq" in base_url or "gpt-oss" in model or "qwen" in model:
        api_key = settings.FALLBACK_LLM_API_KEY
        if not base_url:
            base_url = settings.FALLBACK_LLM_BASE_URL
    else:
        api_key = settings.LLM_API_KEY
        if not base_url:
            base_url = settings.LLM_BASE_URL
    return api_key, base_url, model


def generate_answer_for_eval(
    query: str,
    documents: list[Document],
    config: ExperimentConfig,
    history: list[dict] | None = None,
    *,
    no_fallback: bool = False,
) -> dict:
    """Generate an answer using the exact LLM specified in config (faithful to production prompt).

    When *no_fallback* is ``True`` the function will **not** silently switch to
    the production generator when the target LLM errors out (e.g. rate-limited).
    Instead it returns an error record so the caller can decide to retry later.
    This prevents contaminating A/B LLM experiments with answers from a
    different model.
    """
    sources = list(dict.fromkeys(doc.metadata.get("source_file", "Unknown") for doc in documents))
    if not documents:
        return {
            "answer": "I couldn't find relevant information about that in the university documents.",
            "sources": [],
            "has_llm": False,
            "model_used": None,
            "llm_role": None,
        }

    api_key, base_url, model = _resolve_eval_llm(config)
    if not api_key:
        logger.warning("No API key available for %s — using extractive fallback.", model)
        return {
            "answer": _build_clean_answer(query, documents),
            "sources": sources,
            "has_llm": False,
            "model_used": None,
            "llm_role": None,
        }

    try:
        context = _format_context(documents)
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke(
            {"context": context, "query": query, "history": _to_history_messages(history)}
        )
        cleaned = _clean_answer_text(answer)
        if not cleaned:
            raise ValueError(f"empty answer returned from {model}")
        return {
            "answer": cleaned,
            "sources": sources,
            "has_llm": True,
            "model_used": model,
            "llm_role": config.llm_provider or "eval_target",
        }
    except Exception as e:
        if no_fallback:
            logger.error("Eval LLM (%s) failed: %s. no_fallback=True — recording error.", model, e)
            return {
                "answer": f"[ERROR] {model} failed: {e}",
                "sources": sources,
                "has_llm": False,
                "model_used": model,
                "llm_role": "error",
                "error": str(e),
            }
        logger.error("Eval LLM (%s) failed: %s. Falling back to default generator failover.", model, e)
        return generate_answer(query, documents, history=history)


def run_full_rag(
    store: Chroma | None,
    config: ExperimentConfig,
    item: dict,
    *,
    reranked_docs: list[Document] | None = None,
    prior_retrieval_record: dict | None = None,
    no_fallback: bool = False,
) -> dict:
    """Run the full pipeline for one question (retrieve -> rerank -> generate).

    If prior retrieval results and snapshot chunks are provided, reuses the
    exact frozen candidates and latencies from the prior retrieval run.
    """
    if reranked_docs is not None and prior_retrieval_record is not None:
        record = json.loads(json.dumps(prior_retrieval_record))
        reranked = reranked_docs
    else:
        assert store is not None, "Store required when not using prior retrieval"
        record, reranked = _retrieve_and_rerank(store, config, item)

    question = item["question"]

    t0 = time.perf_counter()
    result = generate_answer_for_eval(
        question, reranked, config, history=item.get("history"), no_fallback=no_fallback,
    )
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    record["answer"] = result.get("answer")
    record["answer_sources"] = result.get("sources", [])
    record["has_llm"] = result.get("has_llm", False)
    record["model_used"] = result.get("model_used")
    record["llm_role"] = result.get("llm_role")
    if result.get("error"):
        record["error"] = result["error"]
    if "latency_ms" not in record:
        record["latency_ms"] = {}
    record["latency_ms"]["generation"] = gen_ms
    record["latency_ms"]["total"] = round(
        record["latency_ms"].get("retrieval", 0.0)
        + record["latency_ms"].get("reranking", 0.0)
        + gen_ms,
        2,
    )
    return record
