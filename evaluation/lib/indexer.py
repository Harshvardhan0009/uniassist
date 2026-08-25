"""Isolated index builder — embed a frozen snapshot into a *local* Chroma collection.

Reproducibility & isolation (freeze rule #6 — never touch production):

* Production ``app.ingestion.embedder.get_vector_store()`` is a module-global
  singleton pointed at the **remote** Chroma. We never call it here.
* Instead each experiment gets its own **local, persistent** Chroma collection
  under ``evaluation/.chroma/<collection>`` (gitignored, regenerated from the
  committed snapshot).
* Only the embedding model varies between embedding experiments; the frozen
  snapshot (chunks + summaries) stays identical.

We reuse production's exact metadata sanitisation and device selection so the
stored vectors match what the live system would produce for the same model.

CLI::

    python -m evaluation.lib.indexer --snapshot baseline_v1 \
        --collection eval_minilm_baseline --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from . import paths

paths.ensure_backend_on_path()

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

# Reuse production helpers so experiment vectors are byte-for-byte faithful.
from app.ingestion.embedder import _sanitize_metadata, _select_device  # noqa: E402

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50


def _persist_dir_for(collection: str) -> Path:
    return paths.CHROMA_DIR / collection


class _PrefixedEmbeddings(Embeddings):
    """Wrap an Embeddings backend to prepend instruction prefixes.

    Several retrieval models expect **asymmetric** query/passage prefixes and
    perform poorly without them:

    * **E5 family** (``intfloat/e5-*``): queries must start with ``"query: "`` and
      passages with ``"passage: "`` (**required** — omitting them badly degrades E5).
    * **BGE v1.5** (``BAAI/bge-*-en-v1.5``): queries use the instruction
      ``"Represent this sentence for searching relevant passages: "``; passages
      need no prefix.

    Prefixes are applied consistently at index time (``embed_documents`` →
    passages) and query time (``embed_query`` → queries), so each model is used as
    its authors intended and the embedding comparison stays fair. MiniLM uses no
    prefixes and is passed through unchanged.
    """

    def __init__(self, base: Embeddings, query_prefix: str = "", passage_prefix: str = ""):
        self._base = base
        self._query_prefix = query_prefix or ""
        self._passage_prefix = passage_prefix or ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._passage_prefix:
            texts = [self._passage_prefix + t for t in texts]
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._query_prefix + text)


def make_embedding_function(
    model_name: str, query_prefix: str = "", passage_prefix: str = ""
) -> Embeddings:
    """Create a fresh (non-singleton) embedding function for a given model.

    Mirrors ``app.ingestion.embedder.get_embedding_function`` (normalised,
    cosine-ready) but is *not* cached globally, so multiple models can coexist
    within one experiment run. When ``query_prefix``/``passage_prefix`` are given,
    the function is wrapped so those instruction prefixes are applied at both
    index and query time (needed for E5/BGE — see ``_PrefixedEmbeddings``).
    """
    device = _select_device()
    logger.info(
        "Loading embedding model '%s' (device=%s%s)",
        model_name, device,
        f", query_prefix={query_prefix!r}, passage_prefix={passage_prefix!r}"
        if (query_prefix or passage_prefix) else "",
    )
    base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    if query_prefix or passage_prefix:
        return _PrefixedEmbeddings(base, query_prefix=query_prefix, passage_prefix=passage_prefix)
    return base


def _deterministic_ids(docs: list[Document]) -> list[str]:
    """Same ID scheme as production embedder: ``{source_file}__chunk_{chunk_index}``."""
    return [
        f"{d.metadata.get('source_file', 'unknown')}__chunk_{d.metadata.get('chunk_index', i)}"
        for i, d in enumerate(docs)
    ]


def build_index(
    snapshot_docs: list[Document],
    collection: str,
    embedding_model: str,
    persist_dir: Path | None = None,
    force: bool = False,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> tuple[Chroma, dict]:
    """Embed ``snapshot_docs`` into a local, isolated Chroma collection.

    Returns ``(store, info)`` where ``info`` records reuse/build stats, timing,
    dimensions, the embedding model, and any query/passage instruction prefixes.
    """
    persist_dir = persist_dir or _persist_dir_for(collection)
    embedding_fn = make_embedding_function(embedding_model, query_prefix, passage_prefix)
    dimensions = len(embedding_fn.embed_query("dimension probe"))

    # Force rebuild: drop the directory *before* opening a client handle.
    if force and persist_dir.exists():
        logger.info("--force: removing existing store at %s", persist_dir)
        shutil.rmtree(persist_dir, ignore_errors=True)

    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=collection,
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    existing = 0
    try:
        existing = store._collection.count()
    except Exception:  # pragma: no cover - fresh collection
        existing = 0

    if existing == len(snapshot_docs) and not force:
        logger.info(
            "Reusing index '%s' (%d vectors already present at %s)", collection, existing, persist_dir
        )
        return store, {
            "reused": True,
            "collection": collection,
            "embedding_model": embedding_model,
            "dimensions": dimensions,
            "vectors": existing,
            "index_seconds": 0.0,
            "persist_dir": str(persist_dir),
            "query_prefix": query_prefix,
            "passage_prefix": passage_prefix,
        }

    if existing and not force:
        logger.warning(
            "Index '%s' has %d vectors but snapshot has %d; rebuilding from scratch.",
            collection, existing, len(snapshot_docs),
        )
        store.reset_collection()

    sanitized = _sanitize_metadata(snapshot_docs)
    ids = _deterministic_ids(sanitized)

    t0 = time.perf_counter()
    stored = 0
    for i in range(0, len(sanitized), _BATCH_SIZE):
        batch = sanitized[i : i + _BATCH_SIZE]
        store.add_documents(batch, ids=ids[i : i + _BATCH_SIZE])
        stored += len(batch)
        logger.info("  embedded %d/%d", stored, len(sanitized))
    index_seconds = round(time.perf_counter() - t0, 2)

    logger.info(
        "Index '%s' built: %d vectors (%d-dim, %s) in %.2fs",
        collection, stored, dimensions, embedding_model, index_seconds,
    )
    return store, {
        "reused": False,
        "collection": collection,
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "vectors": stored,
        "index_seconds": index_seconds,
        "persist_dir": str(persist_dir),
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
    }


def open_index(
    collection: str, embedding_model: str, persist_dir: Path | None = None,
    query_prefix: str = "", passage_prefix: str = "",
) -> Chroma:
    """Open an already-built local Chroma collection (no writes)."""
    persist_dir = persist_dir or _persist_dir_for(collection)
    if not persist_dir.exists():
        raise FileNotFoundError(f"No index at {persist_dir}. Build it with evaluation.lib.indexer first.")
    return Chroma(
        collection_name=collection,
        embedding_function=make_embedding_function(embedding_model, query_prefix, passage_prefix),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    from .snapshot import load_snapshot

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Embed a frozen snapshot into a local Chroma collection.")
    parser.add_argument("--snapshot", default="baseline_v1", help="Snapshot name to embed")
    parser.add_argument("--collection", required=True, help="Target local Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--query-prefix", default="", help="Instruction prefix for queries (E5/BGE)")
    parser.add_argument("--passage-prefix", default="", help="Instruction prefix for passages (E5)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if vectors already exist")
    args = parser.parse_args()

    docs = load_snapshot(args.snapshot)
    _, info = build_index(
        docs, collection=args.collection, embedding_model=args.model, force=args.force,
        query_prefix=args.query_prefix, passage_prefix=args.passage_prefix,
    )
    print(json.dumps(info, indent=2))
