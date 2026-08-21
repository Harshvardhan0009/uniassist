"""Retrieval metrics (Phase 5).

Consumes a raw retrieval artifact produced by
``evaluation.runners.retrieval_eval`` and computes standard retrieval-quality
metrics. Scoring is intentionally separate from execution (Phase 4) so the same
metrics apply unchanged to every future experiment.

--------------------------------------------------------------------------------
DEFINITIONS (documented precisely, per the plan)
--------------------------------------------------------------------------------
For one answerable question we have a rank-ordered list of retrieved items
r1, r2, …, rn (rank 1 = best). Ground truth is the set of expected source files
(and, for the stricter *page* level, expected page numbers).

Relevance of item r_i:
  • source level : r_i.source_file ∈ expected_sources
  • page level   : r_i.source_file ∈ expected_sources AND
                   r_i.page_number ∈ expected_pages   (stricter)

Let rel(i) ∈ {0,1} be that indicator. Then, for a cutoff K:

  Recall@K  = 1 if Σ_{i≤K} rel(i) ≥ 1 else 0
              → "was a correct source found within the top-K?" (binary).
  Hit Rate@K = mean of Recall@K over all scored questions.
              → identical to mean Recall@K under binary relevance; reported once.
  Precision@K = (Σ_{i≤K} rel(i)) / min(K, n)
              → fraction of the top-K that are relevant (n = list length).
  Reciprocal Rank = 1 / (rank of the first relevant item), or 0 if none.
  MRR       = mean Reciprocal Rank over all scored questions.
  Set-Recall@K (source level) = |distinct expected sources present in top-K|
                                 / |expected sources|
              → for multi-source questions, the fraction of required sources found.

Aggregates are the arithmetic mean over the N scored questions. Unanswerable
questions (no expected sources) are **excluded** from retrieval metrics (they are
evaluated for abstention in Phase 18).

MRR uses the first relevant rank anywhere in the provided list (capped by the
list length: 20 for dense retrieval, 5 for the reranked list).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean
from typing import Callable

from ..lib import paths

logger = logging.getLogger("eval.metrics.retrieval")

RETRIEVED_KS = (1, 5, 10, 20)
RERANKED_KS = (1, 5)


# ── Relevance predicates ─────────────────────────────────────────────
def _make_relevance(expected_sources, expected_pages, level: str) -> Callable[[dict], bool]:
    exp_sources = set(expected_sources or [])
    exp_pages = set()
    for p in expected_pages or []:
        try:
            exp_pages.add(int(p))
        except (TypeError, ValueError):
            pass

    def is_relevant(ref: dict) -> bool:
        if ref.get("source_file") not in exp_sources:
            return False
        if level == "page":
            pn = ref.get("page_number")
            try:
                return int(pn) in exp_pages
            except (TypeError, ValueError):
                return False
        return True

    return is_relevant


# ── Per-question evaluation of one ranked list ───────────────────────
def evaluate_ranked_list(
    ranked: list[dict], expected_sources, expected_pages, level: str, ks
) -> dict:
    """Compute per-question metrics for one ranked list at one relevance level."""
    is_relevant = _make_relevance(expected_sources, expected_pages, level)
    rels = [1 if is_relevant(r) else 0 for r in ranked]
    n = len(ranked)

    # Reciprocal rank (first relevant anywhere in the list).
    rr = 0.0
    for i, x in enumerate(rels, start=1):
        if x:
            rr = 1.0 / i
            break

    out: dict[str, float] = {"reciprocal_rank": rr}
    for k in ks:
        kk = min(k, n) if n else 0
        topk = rels[:k]
        out[f"recall@{k}"] = 1.0 if any(topk) else 0.0
        out[f"precision@{k}"] = (sum(topk) / kk) if kk else 0.0

    # Set-recall (source level only; page-level pairing is ambiguous).
    if level == "source":
        exp_sources = set(expected_sources or [])
        if exp_sources:
            for k in ks:
                found = {r.get("source_file") for r in ranked[:k]} & exp_sources
                out[f"set_recall@{k}"] = len(found) / len(exp_sources)
    return out


# ── Aggregation ──────────────────────────────────────────────────────
def _aggregate(per_question: list[dict]) -> dict:
    """Mean each metric across questions; MRR = mean reciprocal_rank."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    agg: dict[str, float] = {}
    for key in keys:
        vals = [pq[key] for pq in per_question if key in pq]
        if not vals:
            continue
        agg["mrr" if key == "reciprocal_rank" else key] = round(mean(vals), 4)
    return agg


def _score_list_over_questions(records: list[dict], ranked_key: str, level: str, ks) -> dict:
    per_q = [
        evaluate_ranked_list(r.get(ranked_key, []), r.get("expected_sources"), r.get("expected_pages"), level, ks)
        for r in records
    ]
    return _aggregate(per_q)


def _scored_records(records: list[dict]) -> list[dict]:
    """Answerable questions that have ground-truth sources (retrieval is measurable)."""
    return [r for r in records if r.get("answerable") and (r.get("expected_sources"))]


# ── Public scorer ────────────────────────────────────────────────────
def score_artifact(artifact: dict) -> dict:
    """Score a raw retrieval artifact -> nested metrics (overall + breakdowns)."""
    all_records = artifact.get("results", [])
    records = _scored_records(all_records)
    excluded = len(all_records) - len(records)

    def block(recs: list[dict]) -> dict:
        return {
            "retrieved": {
                "source": _score_list_over_questions(recs, "retrieved", "source", RETRIEVED_KS),
                "page": _score_list_over_questions(recs, "retrieved", "page", RETRIEVED_KS),
            },
            "reranked": {
                "source": _score_list_over_questions(recs, "reranked", "source", RERANKED_KS),
                "page": _score_list_over_questions(recs, "reranked", "page", RERANKED_KS),
            },
        }

    # Breakdowns (retrieved / source only, to stay readable).
    def breakdown(field: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(r.get(field)), []).append(r)
        return {
            g: {
                "n": len(recs),
                "retrieved_source": _score_list_over_questions(recs, "retrieved", "source", RETRIEVED_KS),
            }
            for g, recs in sorted(groups.items())
        }

    reranker_used = sum(1 for r in all_records if r.get("reranker_used"))
    rerank_enabled = bool(((artifact.get("config") or {}).get("reranker") or {}).get("enabled", True))

    return {
        "scored_from": artifact.get("experiment"),
        "kind": "retrieval_metrics",
        "dataset": artifact.get("dataset"),
        "config": artifact.get("config"),
        "snapshot": {
            "name": (artifact.get("snapshot") or {}).get("name"),
            "summarizer_effective": ((artifact.get("snapshot") or {}).get("summarizer") or {}).get("effective"),
        },
        "index": artifact.get("index"),
        "scored": {"answerable_scored": len(records), "excluded_unanswerable_or_no_gt": excluded},
        "rerank_status": {
            "rerank_enabled": rerank_enabled,
            "reranker_used_questions": reranker_used,
            "total_questions": len(all_records),
            "note": (
                "rerank disabled for this run (dense-only)" if not rerank_enabled
                else "if reranker_used < total, some/all reranking fell back to retrieval order (e.g. Cohere rate-limit)"
            ),
        },
        "metrics": block(records),
        "by_category": breakdown("category"),
        "by_difficulty": breakdown("difficulty"),
    }


# ── Markdown report ──────────────────────────────────────────────────
def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def render_markdown(scored: dict) -> str:
    m = scored["metrics"]
    ds = scored.get("dataset") or {}
    idx = scored.get("index") or {}
    cfg = scored.get("config") or {}
    emb = (cfg.get("embedding") or {}).get("name", "?")

    rs = m["retrieved"]["source"]
    rp = m["retrieved"]["page"]
    kk = RETRIEVED_KS

    lines = []
    lines.append(f"# Retrieval metrics — {scored.get('scored_from')}")
    lines.append("")
    lines.append(
        f"- **Embedding:** {emb} ({idx.get('dimensions')}-dim) · **index:** {idx.get('vectors')} vectors "
        f"· **dataset:** {ds.get('name')} v{ds.get('version')}"
    )
    lines.append(
        f"- **Scored:** {scored['scored']['answerable_scored']} answerable questions "
        f"(excluded {scored['scored']['excluded_unanswerable_or_no_gt']} unanswerable/no-GT)"
    )
    rr = scored["rerank_status"]
    if not rr.get("rerank_enabled", True):
        lines.append("- **Reranking:** disabled for this run (dense-only baseline)")
    else:
        lines.append(
            f"- **Reranking:** applied on {rr['reranker_used_questions']}/{rr['total_questions']} questions"
            + ("" if rr["reranker_used_questions"] == rr["total_questions"] else " — **degraded** (see note)")
        )
    snap = scored.get("snapshot") or {}
    if snap.get("summarizer_effective") is False:
        lines.append("- **Index content:** RAW chunk text (LLM summaries unavailable — 402).")
    lines.append("")

    # Main table: dense retrieval, source vs page level.
    lines.append("## Dense retrieval (top-20)")
    lines.append("")
    header = "| Level | " + " | ".join(f"Recall@{k}" for k in kk) + " | MRR | " + " | ".join(f"P@{k}" for k in kk) + " |"
    sep = "|" + "---|" * (1 + len(kk) + 1 + len(kk))
    lines.append(header)
    lines.append(sep)
    for level_name, blk in (("source", rs), ("page", rp)):
        row = [level_name]
        row += [_fmt(blk.get(f"recall@{k}")) for k in kk]
        row.append(_fmt(blk.get("mrr")))
        row += [_fmt(blk.get(f"precision@{k}")) for k in kk]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # Set-recall (source)
    lines.append(
        "> Source set-recall (distinct expected sources found): "
        + ", ".join(f"@{k} {_fmt(rs.get(f'set_recall@{k}'))}" for k in kk)
    )
    lines.append("")

    # Reranked block
    rrs = m["reranked"]["source"]
    lines.append("## Reranked (top-5)")
    lines.append("")
    lines.append("| Level | Recall@1 | Recall@5 | MRR | P@5 |")
    lines.append("|---|---|---|---|---|")
    for level_name, blk in (("source", m["reranked"]["source"]), ("page", m["reranked"]["page"])):
        lines.append(
            f"| {level_name} | {_fmt(blk.get('recall@1'))} | {_fmt(blk.get('recall@5'))} "
            f"| {_fmt(blk.get('mrr'))} | {_fmt(blk.get('precision@5'))} |"
        )
    if not scored["rerank_status"].get("rerank_enabled", True):
        lines.append("")
        lines.append(
            "> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows "
            "equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9."
        )
    elif scored["rerank_status"]["reranker_used_questions"] < scored["rerank_status"]["total_questions"]:
        lines.append("")
        lines.append(
            "> ⚠ Reranking was rate-limited (Cohere Trial 429); reranked rows above largely reflect "
            "dense top-5, not a true rerank. A faithful reranker comparison is Phase 9."
        )
    lines.append("")

    # By category
    lines.append("## By category (dense retrieval, source level)")
    lines.append("")
    lines.append("| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|")
    for cat, blk in scored["by_category"].items():
        s = blk["retrieved_source"]
        lines.append(
            f"| {cat} | {blk['n']} | {_fmt(s.get('recall@1'))} | {_fmt(s.get('recall@5'))} "
            f"| {_fmt(s.get('recall@10'))} | {_fmt(s.get('mrr'))} |"
        )
    lines.append("")

    # By difficulty
    lines.append("## By difficulty (dense retrieval, source level)")
    lines.append("")
    lines.append("| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|")
    for diff, blk in scored["by_difficulty"].items():
        s = blk["retrieved_source"]
        lines.append(
            f"| {diff} | {blk['n']} | {_fmt(s.get('recall@1'))} | {_fmt(s.get('recall@5'))} "
            f"| {_fmt(s.get('recall@10'))} | {_fmt(s.get('mrr'))} |"
        )
    lines.append("")
    lines.append("> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.")
    lines.append("")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────
def _print_console(scored: dict) -> None:
    rs = scored["metrics"]["retrieved"]["source"]
    print("\n" + "=" * 68)
    print(f"RETRIEVAL METRICS — {scored.get('scored_from')}")
    print("=" * 68)
    print(f"scored answerable : {scored['scored']['answerable_scored']}")
    print("dense retrieval (source level):")
    print(f"  Recall@1={_fmt(rs.get('recall@1'))}  Recall@5={_fmt(rs.get('recall@5'))}  "
          f"Recall@10={_fmt(rs.get('recall@10'))}  Recall@20={_fmt(rs.get('recall@20'))}  MRR={_fmt(rs.get('mrr'))}")
    rp = scored["metrics"]["retrieved"]["page"]
    print("dense retrieval (page level):")
    print(f"  Recall@1={_fmt(rp.get('recall@1'))}  Recall@5={_fmt(rp.get('recall@5'))}  "
          f"Recall@10={_fmt(rp.get('recall@10'))}  Recall@20={_fmt(rp.get('recall@20'))}  MRR={_fmt(rp.get('mrr'))}")
    print("=" * 68)


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths.force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Score a raw retrieval artifact (Phase 5).")
    parser.add_argument("--result", default=None, help="Path to *_retrieval_raw.json (default: baseline_v1)")
    parser.add_argument("--out", default=None, help="Scored JSON output path")
    parser.add_argument("--report", default=None, help="Markdown report output path")
    args = parser.parse_args()

    result_path = Path(args.result) if args.result else paths.RESULTS_DIR / "baseline_v1_retrieval_raw.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Retrieval artifact not found: {result_path}. Run evaluation.runners.retrieval_eval first."
        )

    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    scored = score_artifact(artifact)

    stem = (scored.get("scored_from") or result_path.stem).replace("_retrieval", "")
    out_path = Path(args.out) if args.out else paths.RESULTS_DIR / f"{stem}_retrieval_scored.json"
    report_path = Path(args.report) if args.report else paths.REPORTS_DIR / f"retrieval_metrics_{stem}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_markdown(scored), encoding="utf-8")

    _print_console(scored)
    print(f"scored json      : {out_path}")
    print(f"markdown report  : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
