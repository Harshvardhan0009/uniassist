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

paths.ensure_backend_on_path()

from langchain_chroma import Chroma  # noqa: E402

from app.query.generator import generate_answer  # noqa: E402

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
        logger.info("Snapshot '%s' missing or rebuild requested — building it.", config.snapshot)
        snap.build_snapshot(name=config.snapshot, use_llm=(config.content_indexed == "llm_summary"))

    docs = snap.load_snapshot(config.snapshot)
    manifest = snap.load_manifest(config.snapshot)
    store, index_info = build_index(
        docs, collection=config.collection, embedding_model=config.embedding_model, force=rebuild_index
    )
    return store, manifest, index_info, len(docs)


def run_retrieval(store: Chroma, config: ExperimentConfig, item: dict) -> dict:
    """Run retrieve (+rerank) for one question and return a raw ranked record.

    Faithful to production: conversational follow-ups are retrieved on the bare
    question (history is NOT used for retrieval today — see chain.py). This is
    recorded so Phase 15 (query rewriting) can be measured against it.
    """
    question = item["question"]
    is_conversational = bool(item.get("history"))

    candidates, ret_ms = retrieve_candidates(store, question, config.top_k, config.min_relevance_score)
    reranked, rr_ms, reranker_used = rerank_candidates(
        question, candidates, config.rerank_top_n, config.rerank_model, config.rerank_enabled
    )

    return {
        "id": item["id"],
        "category": item.get("category"),
        "topic": item.get("topic"),
        "difficulty": item.get("difficulty"),
        "answerable": item.get("answerable", True),
        "question": question,
        "conversational": is_conversational,
        "history_used_in_retrieval": False,
        "expected_sources": item.get("expected_sources", []),
        "expected_pages": item.get("expected_pages", []),
        "retrieved": [doc_ref(d, i + 1) for i, d in enumerate(candidates)],
        "reranked": [doc_ref(d, i + 1) for i, d in enumerate(reranked)],
        "reranker_used": reranker_used,
        "n_retrieved": len(candidates),
        "n_reranked": len(reranked),
        "latency_ms": {"retrieval": ret_ms, "reranking": rr_ms},
    }


def run_full_rag(store: Chroma, config: ExperimentConfig, item: dict) -> dict:
    """Run the full pipeline for one question (retrieve -> rerank -> generate).

    Captures the generated answer + sources for later generation/human eval
    (Phases 13-14). Reuses the production generator so answers match live behaviour.
    """
    record = run_retrieval(store, config, item)

    # Rebuild the reranked Documents for generation (records only hold refs).
    question = item["question"]
    candidates, _ = retrieve_candidates(store, question, config.top_k, config.min_relevance_score)
    reranked, _, _ = rerank_candidates(
        question, candidates, config.rerank_top_n, config.rerank_model, config.rerank_enabled
    )

    t0 = time.perf_counter()
    result = generate_answer(question, reranked, history=item.get("history"))
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    record["answer"] = result.get("answer")
    record["answer_sources"] = result.get("sources", [])
    record["has_llm"] = result.get("has_llm", False)
    record["latency_ms"]["generation"] = gen_ms
    record["latency_ms"]["total"] = round(
        record["latency_ms"]["retrieval"] + record["latency_ms"]["reranking"] + gen_ms, 2
    )
    return record
