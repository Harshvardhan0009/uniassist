"""Evaluation harness — configuration + faithful retrieve/rerank against an
isolated store.

These helpers mirror the production ``app.query.retriever`` and
``app.query.reranker`` behaviour (same Chroma call, same Cohere reranker) but
operate on a **caller-supplied local store** instead of the production
singleton, and they capture per-step latency and both cosine + rerank scores so
the runners can emit rich raw records.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

paths.ensure_backend_on_path()

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app.config import settings  # noqa: E402

logger = logging.getLogger(__name__)


# ── Experiment configuration ─────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """A resolved experiment configuration.

    Loaded from an ``evaluation/configs/<id>.json`` file (the Baseline V1 schema).
    ``snapshot`` and ``collection`` are evaluation-only fields: they identify the
    frozen corpus snapshot to embed and the isolated local Chroma collection to
    build. They are derived with sensible defaults if the config file omits them
    (so the frozen ``baseline_v1.json`` never needs editing).
    """

    id: str
    name: str
    embedding_model: str
    content_indexed: str  # "llm_summary" | "raw"
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance_score: float
    rerank_enabled: bool
    rerank_model: str
    rerank_top_n: int
    llm_model: str
    snapshot: str
    collection: str
    source_path: str
    raw: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        """Compact, machine-readable description for result artifacts."""
        return {
            "id": self.id,
            "embedding": {"name": self.embedding_model, "content_indexed": self.content_indexed},
            "chunking": {"size": self.chunk_size, "overlap": self.chunk_overlap},
            "retrieval": {"top_k": self.top_k, "min_relevance_score": self.min_relevance_score},
            "reranker": {"name": self.rerank_model, "top_n": self.rerank_top_n, "enabled": self.rerank_enabled},
            "llm": {"name": self.llm_model},
            "snapshot": self.snapshot,
            "collection": self.collection,
        }


def load_config(config_id_or_path: str, snapshot: str | None = None, collection: str | None = None) -> ExperimentConfig:
    """Load and resolve an experiment configuration.

    Args:
        config_id_or_path: A config id (looked up in ``evaluation/configs/``) or a path.
        snapshot: Override the frozen snapshot name (default: config's or ``baseline_v1``).
        collection: Override the local Chroma collection (default: ``eval_<id>``).
    """
    p = Path(config_id_or_path)
    if not p.exists():
        p = paths.CONFIGS_DIR / f"{config_id_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_id_or_path} (looked in {paths.CONFIGS_DIR})")

    cfg = json.loads(p.read_text(encoding="utf-8"))
    eval_block = cfg.get("evaluation", {})

    cfg_id = cfg.get("id", p.stem)
    resolved_snapshot = snapshot or eval_block.get("snapshot") or "baseline_v1"
    resolved_collection = collection or eval_block.get("collection") or f"eval_{cfg_id}"

    return ExperimentConfig(
        id=cfg_id,
        name=cfg.get("name", cfg_id),
        embedding_model=cfg["embedding"]["name"],
        content_indexed=cfg.get("indexing", {}).get("content_indexed", "llm_summary"),
        chunk_size=cfg["chunking"]["size"],
        chunk_overlap=cfg["chunking"]["overlap"],
        top_k=cfg["retrieval"]["top_k"],
        min_relevance_score=cfg["retrieval"].get("min_relevance_score", 0.0),
        rerank_enabled=cfg["reranker"].get("enabled", True),
        rerank_model=cfg["reranker"]["name"],
        rerank_top_n=cfg["reranker"]["top_n"],
        llm_model=cfg["generation"]["llm"],
        snapshot=resolved_snapshot,
        collection=resolved_collection,
        source_path=str(p.relative_to(paths.PROJECT_ROOT)) if p.is_relative_to(paths.PROJECT_ROOT) else str(p),
        raw=cfg,
    )


# ── Document reference helper ────────────────────────────────────────
def chunk_id(doc: Document) -> str:
    return f"{doc.metadata.get('source_file', 'unknown')}__chunk_{doc.metadata.get('chunk_index')}"


def doc_ref(doc: Document, rank: int) -> dict:
    """Compact ranked reference used in raw result records (no raw text dumped)."""
    return {
        "rank": rank,
        "chunk_id": chunk_id(doc),
        "source_file": doc.metadata.get("source_file"),
        "page_number": doc.metadata.get("page_number"),
        "chunk_type": doc.metadata.get("chunk_type"),
        "cosine_score": doc.metadata.get("retrieval_cosine_score"),
        "rerank_score": doc.metadata.get("rerank_score"),
    }


# ── Retrieval (mirrors app.query.retriever) ──────────────────────────
def retrieve_candidates(
    store: Chroma, query: str, top_k: int, min_score: float = 0.0
) -> tuple[list[Document], float]:
    """Dense similarity search against the given local store. Returns (docs, latency_ms)."""
    t0 = time.perf_counter()
    results = store.similarity_search_with_relevance_scores(query=query, k=top_k)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    docs: list[Document] = []
    for doc, score in results:
        if score < min_score:
            continue
        # Distinct key so Cohere's later `relevance_score` doesn't clobber the cosine score.
        doc.metadata["retrieval_cosine_score"] = round(float(score), 4)
        docs.append(doc)
    return docs, latency_ms


# ── Reranking (mirrors app.query.reranker) ───────────────────────────
def rerank_candidates(
    query: str, documents: list[Document], top_n: int, model: str, enabled: bool
) -> tuple[list[Document], float, bool]:
    """Cohere cross-encoder rerank. Returns (docs, latency_ms, reranker_used).

    Falls back to retrieval order (truncated) when disabled, no key, or on error —
    exactly like production.
    """
    if not documents:
        return [], 0.0, False

    if not enabled or not settings.has_cohere:
        return documents[:top_n], 0.0, False

    t0 = time.perf_counter()
    try:
        from langchain_cohere import CohereRerank

        reranker = CohereRerank(cohere_api_key=settings.COHERE_API_KEY, model=model, top_n=top_n)
        reranked = list(reranker.compress_documents(documents=documents, query=query))
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        # CohereRerank writes its score into metadata['relevance_score']; normalise the key.
        for d in reranked:
            if "relevance_score" in d.metadata and "rerank_score" not in d.metadata:
                d.metadata["rerank_score"] = round(float(d.metadata["relevance_score"]), 4)
        return reranked, latency_ms, True
    except Exception as e:  # pragma: no cover
        logger.error("Reranking failed: %s. Falling back to retrieval order.", e)
        return documents[:top_n], round((time.perf_counter() - t0) * 1000, 2), False
