"""Full-RAG evaluation runner (Phase 4).

Executes the complete pipeline (retrieve -> rerank -> generate) for each
question and captures the generated answer, cited sources and per-step latency.
The output feeds Phase 6 (baseline answer set) and Phases 13-14
(generation/human evaluation). Answer *scoring* is not done here.

This runner makes live LLM calls, so during Phase 4 it is typically run with a
small ``--limit`` as a smoke test; the full baseline answer set is produced in
Phase 6.

Usage::

    python -m evaluation.runners.full_rag_eval --config baseline_v1 --limit 2
    python -m evaluation.runners.full_rag_eval --config baseline_v1            # all questions
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..lib import paths
from ..lib.experiment import load_questions, prepare_store, run_full_rag
from ..lib.harness import load_config
from . import _common

logger = logging.getLogger("eval.full_rag")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-RAG evaluation runner (Phase 4).")
    parser.add_argument("--config", default="baseline_v1", help="Config id or path (default: baseline_v1)")
    parser.add_argument("--snapshot", default=None, help="Override frozen snapshot name")
    parser.add_argument("--collection", default=None, help="Override local Chroma collection name")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--no-rerank", action="store_true", help="Disable Cohere reranking for this run")
    parser.add_argument("--rebuild-snapshot", action="store_true", help="Rebuild the corpus snapshot first")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the local index first")
    parser.add_argument("--out", default=None, help="Output path (default: experiments/results/<id>_full_rag_raw.json)")
    parser.add_argument("--verbose", action="store_true", help="Show underlying app/library logs")
    args = parser.parse_args()

    _common.configure_logging(args.verbose)
    paths.ensure_dirs()

    config = load_config(args.config, snapshot=args.snapshot, collection=args.collection)
    if args.no_rerank:
        config.rerank_enabled = False

    logger.info("Config '%s' | embedding=%s | llm=%s | collection=%s",
                config.id, config.embedding_model, config.llm_model, config.collection)

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

    logger.info("Running full RAG on %d question(s) (live LLM calls)...", len(items))
    records: list[dict] = []
    started = time.time()
    for i, item in enumerate(items, 1):
        records.append(run_full_rag(store, config, item))
        logger.info("  %d/%d  [%s] %s", i, len(items), item.get("id"), item.get("question", "")[:60])

    gen_lat = _common.latency_stats([r["latency_ms"].get("generation") for r in records])
    total_lat = _common.latency_stats([r["latency_ms"].get("total") for r in records])
    ret_lat = _common.latency_stats([r["latency_ms"].get("retrieval") for r in records])
    rr_lat = _common.latency_stats([r["latency_ms"].get("reranking") for r in records])

    envelope = {
        "experiment": f"{config.id}_full_rag",
        "kind": "full_rag",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 1),
        "dataset": _common.dataset_meta(data, evaluated=len(records)),
        "config": config.summary(),
        "config_source": config.source_path,
        "snapshot": {
            "name": manifest.get("name"),
            "snapshot_sha256": manifest.get("snapshot_sha256"),
            "counts": manifest.get("counts"),
        },
        "index": index_info,
        "environment": _common.environment_meta(),
        "latency_ms": {
            "retrieval": ret_lat,
            "reranking": rr_lat,
            "generation": gen_lat,
            "total": total_lat,
        },
        "results": records,
    }

    out_path = Path(args.out) if args.out else paths.RESULTS_DIR / f"{config.id}_full_rag_raw.json"
    _common.write_result(out_path, envelope)

    _print_summary(envelope, out_path)
    return 0


def _print_summary(envelope: dict, out_path: Path) -> None:
    d = envelope["dataset"]
    lat = envelope["latency_ms"]
    print("\n" + "=" * 68)
    print(f"FULL-RAG EVAL — {envelope['experiment']}")
    print("=" * 68)
    print(f"dataset          : {d['name']} v{d['version']} | evaluated {d['evaluated']}/{d['questions']}")
    print(f"embedding / llm  : {envelope['config']['embedding']['name']} / {envelope['config']['llm']['name']}")
    print(f"latency (avg ms) : retrieval {lat['retrieval'].get('avg')} | rerank {lat['reranking'].get('avg')} | "
          f"generation {lat['generation'].get('avg')} | total {lat['total'].get('avg')}")
    print(f"answers captured : {len(envelope['results'])}  (answer scoring is Phase 13)")
    print(f"written          : {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    raise SystemExit(main())
