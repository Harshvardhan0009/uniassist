"""Retrieval evaluation runner (Phase 4).

Executes the benchmark against a configuration and writes a **raw** result
artifact: for every question, the ranked dense-retrieval candidates (top-k) and
the reranked set (top-n), with cosine + rerank scores, page numbers and per-step
latency.

This runner intentionally does **no metric scoring** — that is Phase 5
(`evaluation/metrics/retrieval_metrics.py`), which will consume this artifact and
compute Recall@1/5/10/20, MRR, Precision@K and Hit Rate@K. A tiny
``sanity_preview`` (does an expected source appear at all?) is included only as a
smoke-test signal that the pipeline retrieves sensibly.

Usage::

    python -m evaluation.runners.retrieval_eval --config baseline_v1
    python -m evaluation.runners.retrieval_eval --config baseline_v1 --limit 5
    python -m evaluation.runners.retrieval_eval --config baseline_v1 --rebuild-index
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..lib import paths
from ..lib.experiment import load_questions, prepare_store, run_retrieval
from ..lib.harness import load_config
from . import _common

logger = logging.getLogger("eval.retrieval")


def _source_hit(record: dict, ranked_key: str) -> bool | None:
    """Sanity preview only: does any expected source appear in the ranked list?"""
    expected = set(record.get("expected_sources") or [])
    if not expected:
        return None  # unanswerable / no ground-truth sources
    got = {r.get("source_file") for r in record.get(ranked_key, [])}
    return len(expected & got) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw retrieval evaluation runner (Phase 4).")
    parser.add_argument("--config", default="baseline_v1", help="Config id or path (default: baseline_v1)")
    parser.add_argument("--snapshot", default=None, help="Override frozen snapshot name")
    parser.add_argument("--collection", default=None, help="Override local Chroma collection name")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--no-rerank", action="store_true", help="Disable Cohere reranking for this run")
    parser.add_argument("--rebuild-snapshot", action="store_true", help="Rebuild the corpus snapshot first")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the local index first")
    parser.add_argument("--out", default=None, help="Output path (default: experiments/results/<id>_retrieval_raw.json)")
    parser.add_argument("--verbose", action="store_true", help="Show underlying app/library logs")
    args = parser.parse_args()

    _common.configure_logging(args.verbose)
    paths.ensure_dirs()

    config = load_config(args.config, snapshot=args.snapshot, collection=args.collection)
    if args.no_rerank:
        config.rerank_enabled = False

    logger.info("Config '%s' | embedding=%s | snapshot=%s | collection=%s",
                config.id, config.embedding_model, config.snapshot, config.collection)

    store, manifest, index_info, n_docs = prepare_store(
        config, rebuild_snapshot=args.rebuild_snapshot, rebuild_index=args.rebuild_index
    )
    logger.info("Index ready: %d vectors (%s, %d-dim)%s",
                index_info.get("vectors"), index_info.get("embedding_model"),
                index_info.get("dimensions"), " [reused]" if index_info.get("reused") else "")

    data = load_questions()
    items = data.get("questions", [])
    if args.limit:
        items = items[: args.limit]

    logger.info("Running retrieval on %d question(s)...", len(items))
    records: list[dict] = []
    started = time.time()
    for i, item in enumerate(items, 1):
        records.append(run_retrieval(store, config, item))
        if i % 10 == 0 or i == len(items):
            logger.info("  %d/%d", i, len(items))

    # ── Aggregate latency + sanity preview (NOT the Phase 5 metric) ──
    ret_lat = _common.latency_stats([r["latency_ms"].get("retrieval") for r in records])
    rr_lat = _common.latency_stats([r["latency_ms"].get("reranking") for r in records])

    answerable = [r for r in records if r.get("answerable")]
    hits_topn = [_source_hit(r, "reranked") for r in answerable]
    hits_topk = [_source_hit(r, "retrieved") for r in answerable]
    hits_topn = [h for h in hits_topn if h is not None]
    hits_topk = [h for h in hits_topk if h is not None]

    envelope = {
        "experiment": f"{config.id}_retrieval",
        "kind": "retrieval",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 1),
        "dataset": _common.dataset_meta(data, evaluated=len(records)),
        "config": config.summary(),
        "config_source": config.source_path,
        "snapshot": {
            "name": manifest.get("name"),
            "snapshot_sha256": manifest.get("snapshot_sha256"),
            "raw_content_sha256": manifest.get("raw_content_sha256"),
            "counts": manifest.get("counts"),
            "summarizer": manifest.get("summarizer"),
        },
        "index": index_info,
        "environment": {**_common.environment_meta(), "reranker_available": store is not None and _reranker_available()},
        "latency_ms": {"retrieval": ret_lat, "reranking": rr_lat},
        "sanity_preview": {
            "note": "NOT the Phase 5 metric. Boolean 'expected source appears at all' over answerable questions.",
            "answerable_evaluated": len(answerable),
            "expected_source_in_retrieved_top_k": _fraction(hits_topk),
            "expected_source_in_reranked_top_n": _fraction(hits_topn),
        },
        "results": records,
    }

    out_path = Path(args.out) if args.out else paths.RESULTS_DIR / f"{config.id}_retrieval_raw.json"
    _common.write_result(out_path, envelope)

    _print_summary(envelope, out_path)
    return 0


def _reranker_available() -> bool:
    from app.config import settings

    return settings.has_cohere


def _fraction(bools: list[bool]) -> dict:
    n = len(bools)
    hit = sum(1 for b in bools if b)
    return {"hit": hit, "total": n, "fraction": round(hit / n, 4) if n else None}


def _print_summary(envelope: dict, out_path: Path) -> None:
    d = envelope["dataset"]
    ret = envelope["latency_ms"]["retrieval"]
    rr = envelope["latency_ms"]["reranking"]
    sp = envelope["sanity_preview"]
    print("\n" + "=" * 68)
    print(f"RETRIEVAL EVAL — {envelope['experiment']}")
    print("=" * 68)
    print(f"dataset          : {d['name']} v{d['version']} | evaluated {d['evaluated']}/{d['questions']}")
    print(f"embedding        : {envelope['config']['embedding']['name']} ({envelope['index'].get('dimensions')}-dim)")
    print(f"index            : {envelope['index'].get('vectors')} vectors | collection {envelope['config']['collection']}")
    print(f"retrieval latency: avg {ret.get('avg')}ms  p95 {ret.get('p95')}ms  max {ret.get('max')}ms")
    print(f"rerank latency   : avg {rr.get('avg')}ms  p95 {rr.get('p95')}ms  max {rr.get('max')}ms")
    print("sanity (not a metric — Phase 5 computes Recall/MRR):")
    print(f"  expected source in retrieved top-k : {sp['expected_source_in_retrieved_top_k']['hit']}/{sp['expected_source_in_retrieved_top_k']['total']}")
    print(f"  expected source in reranked top-n  : {sp['expected_source_in_reranked_top_n']['hit']}/{sp['expected_source_in_reranked_top_n']['total']}")
    print(f"written          : {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    raise SystemExit(main())
