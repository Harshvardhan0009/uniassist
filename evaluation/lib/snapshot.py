"""Corpus snapshot — freeze the ingestion front-half so experiments are reproducible.

The production pipeline is::

    partition_directory -> chunk_documents -> summarize_chunks -> embed_and_store

The searchable ``page_content`` produced by ``summarize_chunks`` is an **LLM
summary**, which is non-deterministic (see docs/CURRENT_STATE.md §11.4). If every
embedding experiment re-ran summarisation it would change the indexed text and
violate "one variable at a time". So we run partition -> chunk -> summarize
**once**, write the result to a versioned JSONL snapshot, and every experiment
re-embeds *that exact frozen set*.

The snapshot is production-faithful: it reuses the real ``app.ingestion`` modules
(same parser, same 2500/300 chunking, same summariser model/params). Only the
embedding step is deferred to :mod:`evaluation.lib.indexer`.

CLI::

    python -m evaluation.lib.snapshot --name baseline_v1
    python -m evaluation.lib.snapshot --name baseline_v1_raw --no-llm   # raw text, deterministic
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from . import paths

paths.force_utf8_stdio()
paths.ensure_backend_on_path()

from langchain_core.documents import Document  # noqa: E402

from app.config import settings  # noqa: E402
from app.ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents  # noqa: E402
from app.ingestion.partition import partition_directory  # noqa: E402
from app.ingestion.summarizer import _passthrough, summarize_chunks  # noqa: E402

logger = logging.getLogger(__name__)

# Metadata keys carried from the pipeline into the frozen snapshot.
_META_KEYS = (
    "source_file",
    "page_number",
    "chunk_index",
    "chunk_type",
    "category",
    "title",
    "raw_content",
)


@dataclass
class SnapshotPaths:
    """Resolved on-disk locations for a named snapshot."""

    name: str
    jsonl: Path
    manifest: Path

    @classmethod
    def for_name(cls, name: str) -> "SnapshotPaths":
        base = paths.SNAPSHOT_DIR / name
        return cls(name=name, jsonl=base.with_suffix(".jsonl"), manifest=Path(f"{base}.manifest.json"))


def _sha256_of(strings) -> str:
    h = hashlib.sha256()
    for s in strings:
        h.update(s.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def build_snapshot(name: str = "baseline_v1", data_dir: Path | None = None, use_llm: bool = True) -> dict:
    """Partition -> chunk -> (summarize) the corpus and freeze it to JSONL.

    Args:
        name: Snapshot name (file stem under ``evaluation/snapshots/``).
        data_dir: Corpus root (defaults to ``Data/``).
        use_llm: When True (default) and an LLM key is configured, freeze the
            production LLM summaries. When False, freeze raw chunk text
            (deterministic; useful as a control).

    Returns:
        The manifest dict describing the frozen snapshot.
    """
    data_dir = data_dir or paths.DATA_DIR
    paths.ensure_dirs()
    sp = SnapshotPaths.for_name(name)

    logger.info("Building snapshot '%s' from %s", name, data_dir)

    # ── Step 3: Partition ────────────────────────────────────────────
    t0 = time.perf_counter()
    all_elements = partition_directory(data_dir)
    if not all_elements:
        raise RuntimeError(f"No ingestible documents found under {data_dir}")
    n_pages = sum(len(v) for v in all_elements.values())

    # ── Step 4: Chunk (per file, exactly like pipeline.run_pipeline) ──
    all_chunks: list[Document] = []
    per_file_chunks: dict[str, int] = {}
    for filename, elements in all_elements.items():
        chunks = chunk_documents(elements, source_file=filename)
        per_file_chunks[filename] = len(chunks)
        all_chunks.extend(chunks)
    partition_chunk_seconds = round(time.perf_counter() - t0, 2)

    # ── Step 5: Summarize (or passthrough) ───────────────────────────
    summarize_used_llm = bool(use_llm and settings.has_llm)
    t1 = time.perf_counter()
    if summarize_used_llm:
        logger.info("Summarizing %d chunks with %s ...", len(all_chunks), settings.LLM_MODEL)
        frozen = summarize_chunks(all_chunks)
    else:
        logger.info("Freezing raw chunk text (no LLM summarisation).")
        frozen = _passthrough(all_chunks)
    summarize_seconds = round(time.perf_counter() - t1, 2)

    # Proxy for summaries that silently fell back to raw text.
    summary_equals_raw = sum(
        1 for d in frozen if d.page_content.strip() == str(d.metadata.get("raw_content", "")).strip()
    )
    # If the LLM was requested but every chunk fell back, summarisation did not
    # actually happen (e.g. the provider returned 402/limit errors) — record this
    # honestly so the snapshot's indexed content is not mislabelled.
    summarization_effective = summarize_used_llm and summary_equals_raw < len(frozen)
    effective_content = "llm_summary" if summarization_effective else "raw_text"
    if summarize_used_llm and not summarization_effective:
        logger.warning(
            "LLM summarisation was requested but ALL %d chunks fell back to raw text "
            "(provider error such as 402/quota?). Snapshot content is RAW TEXT, not summaries.",
            len(frozen),
        )

    # ── Write JSONL ──────────────────────────────────────────────────
    with sp.jsonl.open("w", encoding="utf-8") as f:
        for d in frozen:
            meta = {k: d.metadata.get(k) for k in _META_KEYS if k in d.metadata}
            f.write(json.dumps({"page_content": d.page_content, "metadata": meta}, ensure_ascii=False) + "\n")

    # Corpus fingerprint (independent of summaries) + full snapshot fingerprint.
    raw_fingerprint = _sha256_of(
        f"{d.metadata.get('source_file')}|{d.metadata.get('chunk_index')}|{d.metadata.get('raw_content', '')}"
        for d in frozen
    )
    snapshot_fingerprint = _sha256_of(
        f"{d.metadata.get('source_file')}|{d.metadata.get('chunk_index')}|{d.page_content}" for d in frozen
    )

    manifest = {
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": str(data_dir.relative_to(paths.PROJECT_ROOT)) if data_dir.is_relative_to(paths.PROJECT_ROOT) else str(data_dir),
        "jsonl": sp.jsonl.name,
        "chunking": {"splitter": "RecursiveCharacterTextSplitter", "size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "summarizer": {
            "requested_llm": summarize_used_llm,
            "effective": summarization_effective,
            "effective_content": effective_content,
            "model": settings.LLM_MODEL if summarize_used_llm else None,
            "base_url": settings.LLM_BASE_URL if summarize_used_llm else None,
            "temperature": 0.0,
            "max_tokens": 256,
            "concurrency": settings.SUMMARY_CONCURRENCY if summarize_used_llm else None,
        },
        "counts": {
            "files": len(all_elements),
            "pages": n_pages,
            "chunks": len(all_chunks),
            "frozen": len(frozen),
            "summary_equals_raw": summary_equals_raw,
        },
        "per_file_chunks": per_file_chunks,
        "timing_seconds": {"partition_chunk": partition_chunk_seconds, "summarize": summarize_seconds},
        "raw_content_sha256": raw_fingerprint,
        "snapshot_sha256": snapshot_fingerprint,
        "note": (
            "Frozen ingestion front-half (partition->chunk->summarize). Re-embed this exact set for "
            "every experiment. raw_content_sha256 fingerprints the corpus (detects Data/ drift); "
            "snapshot_sha256 fingerprints the exact indexed text."
        ),
    }
    sp.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Snapshot '%s' written: %d chunks from %d files (%d pages) -> %s",
        name, len(frozen), len(all_elements), n_pages, sp.jsonl.name,
    )
    return manifest


def load_snapshot(name: str = "baseline_v1") -> list[Document]:
    """Load a frozen snapshot back into LangChain Documents."""
    sp = SnapshotPaths.for_name(name)
    if not sp.jsonl.exists():
        raise FileNotFoundError(
            f"Snapshot '{name}' not found at {sp.jsonl}. Build it with: "
            f"python -m evaluation.lib.snapshot --name {name}"
        )
    docs: list[Document] = []
    with sp.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec.get("metadata", {})))
    return docs


def load_manifest(name: str = "baseline_v1") -> dict:
    """Load a snapshot manifest (or {} if absent)."""
    sp = SnapshotPaths.for_name(name)
    if not sp.manifest.exists():
        return {}
    return json.loads(sp.manifest.read_text(encoding="utf-8"))


def snapshot_exists(name: str = "baseline_v1") -> bool:
    return SnapshotPaths.for_name(name).jsonl.exists()


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Freeze a reproducible corpus snapshot for evaluation.")
    parser.add_argument("--name", default="baseline_v1", help="Snapshot name (default: baseline_v1)")
    parser.add_argument("--no-llm", action="store_true", help="Freeze raw chunk text instead of LLM summaries")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the snapshot already exists")
    args = parser.parse_args()

    if snapshot_exists(args.name) and not args.force:
        print(f"Snapshot '{args.name}' already exists. Use --force to rebuild.")
        m = load_manifest(args.name)
        print(json.dumps(m.get("counts", {}), indent=2))
    else:
        manifest = build_snapshot(name=args.name, use_llm=not args.no_llm)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
