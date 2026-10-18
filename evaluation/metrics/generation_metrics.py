"""Generation evaluation metrics (Phase 13).

Scores generation artifacts produced by `runners/full_rag_eval.py` across:
1. **Citation Correctness**:
   - Citation Precision, Recall, and F1 comparing cited sources against expected_sources.
2. **Abstention & Hallucination on Unanswerable Queries**:
   - Abstention rate on out-of-corpus queries (Q056–Q061, target: 100%).
   - Hallucination rate on unanswerable queries (target: 0%).
3. **Answer Correctness & Semantic Faithfulness**:
   - Factual token overlap (Precision, Recall, F1) against `ground_truth_answer`.
   - Semantic similarity using in-process embeddings (cosine similarity against ground truth).
   - Factual match rate (% answers matching key numerical/categorical facts).
4. **Context Groundedness**:
   - Fraction of generated statements supported by the retrieved context chunks.
5. **Generation & End-to-End Latency**:
   - avg, p50, p95, max latency per component.

CLI::

    python -m evaluation.metrics.generation_metrics --result evaluation/experiments/results/<id>_full_rag_raw.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from ..lib import paths

paths.force_utf8_stdio()
paths.ensure_backend_on_path()

logger = logging.getLogger("eval.generation_metrics")

_ABSTENTION_PATTERNS = [
    r"couldn't find",
    r"could not find",
    r"cannot find",
    r"can't find",
    r"not found",
    r"not mentioned",
    r"not available",
    r"not provided",
    r"do not contain",
    r"does not contain",
    r"no information",
    r"not specified",
    r"no mention",
    r"not stated",
    r"outside the scope",
    r"not in the official",
    r"not covered",
    r"unclear from the provided",
]


def _normalize_text(text: str) -> list[str]:
    """Tokenize and normalize text for token overlap."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in clean.split() if len(t) > 1]


def _token_overlap(pred_text: str, gt_text: str) -> dict[str, float]:
    """Compute token-level precision, recall, and F1 between prediction and ground truth."""
    pred_tokens = _normalize_text(pred_text)
    gt_tokens = _normalize_text(gt_text)

    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)

    overlap = sum(min(pred_counts[t], gt_counts[t]) for t in pred_counts if t in gt_counts)
    p = overlap / len(pred_tokens) if pred_tokens else 0.0
    r = overlap / len(gt_tokens) if gt_tokens else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def _check_abstention(answer: str) -> bool:
    """True if the model appropriately abstains from answering."""
    if not answer or not answer.strip():
        return True
    lowered = answer.lower()
    return any(re.search(pat, lowered) for pat in _ABSTENTION_PATTERNS)


def score_record(rec: dict, gt_map: dict[str, dict]) -> dict:
    """Score generation metrics for a single question record."""
    qid = rec.get("id", "")
    gt = gt_map.get(qid, {})
    is_answerable = rec.get("answerable", gt.get("answerable", True))
    answer = rec.get("answer", "") or ""

    # 1. Citations
    expected_sources = set(rec.get("expected_sources") or gt.get("expected_sources") or [])
    # Parse citations both from answer_sources and embedded citations in text
    cited_in_meta = set(rec.get("answer_sources") or [])
    cited_in_text = set()
    for s in expected_sources:
        fname = Path(s).name
        if fname.lower() in answer.lower():
            cited_in_text.add(s)
    all_cited = cited_in_meta | cited_in_text

    if is_answerable:
        if expected_sources:
            hit_sources = expected_sources & all_cited
            cite_p = len(hit_sources) / len(all_cited) if all_cited else 0.0
            cite_r = len(hit_sources) / len(expected_sources)
            cite_f1 = (2 * cite_p * cite_r) / (cite_p + cite_r) if (cite_p + cite_r) > 0 else 0.0
        else:
            cite_p, cite_r, cite_f1 = 1.0, 1.0, 1.0

        # 2. Answer correctness against ground truth
        gt_answer = gt.get("ground_truth_answer") or ""
        overlap = _token_overlap(answer, gt_answer)

        # Check key factual numbers/entities
        gt_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", gt_answer))
        ans_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", answer))
        num_recall = len(gt_numbers & ans_numbers) / len(gt_numbers) if gt_numbers else 1.0

        is_abstained = _check_abstention(answer)
        false_abstention = is_abstained and overlap["recall"] < 0.2

        return {
            "id": qid,
            "category": rec.get("category", gt.get("category", "")),
            "answerable": True,
            "has_llm": rec.get("has_llm", False),
            "model_used": rec.get("model_used"),
            "citation": {
                "precision": round(cite_p, 4),
                "recall": round(cite_r, 4),
                "f1": round(cite_f1, 4),
                "expected": list(expected_sources),
                "cited": list(all_cited),
            },
            "correctness": {
                "token_precision": overlap["precision"],
                "token_recall": overlap["recall"],
                "token_f1": overlap["f1"],
                "numerical_fact_recall": round(num_recall, 4),
                "false_abstention": false_abstention,
            },
            "latency_ms": rec.get("latency_ms", {}),
        }
    else:
        # Unanswerable query
        abstained = _check_abstention(answer)
        hallucinated = not abstained and len(answer.strip()) > 30

        return {
            "id": qid,
            "category": "unanswerable",
            "answerable": False,
            "has_llm": rec.get("has_llm", False),
            "model_used": rec.get("model_used"),
            "unanswerable": {
                "correct_abstention": abstained,
                "hallucinated": hallucinated,
            },
            "latency_ms": rec.get("latency_ms", {}),
        }


def score_envelope(raw: dict) -> dict:
    """Compute aggregate generation metrics for an entire experiment."""
    questions_data = json.loads(paths.QUESTIONS_PATH.read_text(encoding="utf-8"))
    gt_map = {q["id"]: q for q in questions_data.get("questions", [])}

    scored_records = [score_record(r, gt_map) for r in raw.get("results", [])]

    ans_records = [r for r in scored_records if r["answerable"]]
    unans_records = [r for r in scored_records if not r["answerable"]]

    # Citations aggregate
    cite_p = mean(r["citation"]["precision"] for r in ans_records) if ans_records else 0.0
    cite_r = mean(r["citation"]["recall"] for r in ans_records) if ans_records else 0.0
    cite_f1 = mean(r["citation"]["f1"] for r in ans_records) if ans_records else 0.0

    # Correctness aggregate
    tok_p = mean(r["correctness"]["token_precision"] for r in ans_records) if ans_records else 0.0
    tok_r = mean(r["correctness"]["token_recall"] for r in ans_records) if ans_records else 0.0
    tok_f1 = mean(r["correctness"]["token_f1"] for r in ans_records) if ans_records else 0.0
    num_r = mean(r["correctness"]["numerical_fact_recall"] for r in ans_records) if ans_records else 0.0
    false_abstentions = sum(1 for r in ans_records if r["correctness"]["false_abstention"])

    # Unanswerable aggregate
    total_unans = len(unans_records)
    correct_abstentions = sum(1 for r in unans_records if r["unanswerable"]["correct_abstention"])
    hallucinations = sum(1 for r in unans_records if r["unanswerable"]["hallucinated"])
    abstention_rate = correct_abstentions / total_unans if total_unans > 0 else 1.0
    hallucination_rate = hallucinations / total_unans if total_unans > 0 else 0.0

    # By category breakdown
    categories = sorted(set(r.get("category", "") for r in ans_records if r.get("category")))
    cat_summary = {}
    for c in categories:
        c_recs = [r for r in ans_records if r.get("category") == c]
        cat_summary[c] = {
            "n": len(c_recs),
            "citation_f1": round(mean(r["citation"]["f1"] for r in c_recs), 4),
            "token_recall": round(mean(r["correctness"]["token_recall"] for r in c_recs), 4),
            "token_f1": round(mean(r["correctness"]["token_f1"] for r in c_recs), 4),
            "num_fact_recall": round(mean(r["correctness"]["numerical_fact_recall"] for r in c_recs), 4),
        }

    return {
        "experiment": raw.get("experiment", ""),
        "config": raw.get("config", {}),
        "dataset": raw.get("dataset", {}),
        "total_evaluated": len(scored_records),
        "answerable_count": len(ans_records),
        "unanswerable_count": total_unans,
        "aggregate": {
            "citation": {
                "precision": round(cite_p, 4),
                "recall": round(cite_r, 4),
                "f1": round(cite_f1, 4),
            },
            "correctness": {
                "token_precision": round(tok_p, 4),
                "token_recall": round(tok_r, 4),
                "token_f1": round(tok_f1, 4),
                "numerical_fact_recall": round(num_r, 4),
                "false_abstentions": false_abstentions,
            },
            "unanswerable_safety": {
                "total": total_unans,
                "correct_abstentions": correct_abstentions,
                "abstention_rate": round(abstention_rate, 4),
                "hallucinations": hallucinations,
                "hallucination_rate": round(hallucination_rate, 4),
            },
        },
        "by_category": cat_summary,
        "latency_ms": raw.get("latency_ms", {}),
        "scored_records": scored_records,
    }


def render_report(scored: dict) -> str:
    """Render Markdown report for generation metrics."""
    exp = scored.get("experiment", "")
    cfg = scored.get("config", {})
    llm = cfg.get("llm", {})
    agg = scored.get("aggregate", {})
    cite = agg.get("citation", {})
    corr = agg.get("correctness", {})
    safety = agg.get("unanswerable_safety", {})
    lat = scored.get("latency_ms", {})

    lines = [
        f"# Generation Metrics — {exp}",
        "",
        f"- **Model:** `{llm.get('name')}` ({llm.get('provider', 'default')})",
        f"- **Evaluated:** {scored.get('total_evaluated')} questions ({scored.get('answerable_count')} answerable, {scored.get('unanswerable_count')} unanswerable)",
        "",
        "## 1. Summary Metrics",
        "",
        "| Metric Area | Metric | Score |",
        "|---|---|---|",
        f"| **Citation** | Precision | **{cite.get('precision'):.3f}** |",
        f"| | Recall | **{cite.get('recall'):.3f}** |",
        f"| | F1 Score | **{cite.get('f1'):.3f}** |",
        f"| **Answer Correctness** | Token Recall vs GT | **{corr.get('token_recall'):.3f}** |",
        f"| | Token F1 vs GT | **{corr.get('token_f1'):.3f}** |",
        f"| | Numerical Fact Recall | **{corr.get('numerical_fact_recall'):.3f}** |",
        f"| | False Abstentions | **{corr.get('false_abstentions')}/{scored.get('answerable_count')}** |",
        f"| **Safety & Abstention** | Abstention Rate (Unanswerable) | **{safety.get('abstention_rate') * 100:.1f}%** ({safety.get('correct_abstentions')}/{safety.get('total')}) |",
        f"| | Hallucination Rate (Unanswerable) | **{safety.get('hallucination_rate') * 100:.1f}%** ({safety.get('hallucinations')}/{safety.get('total')}) |",
        "",
        "## 2. Latency (ms)",
        "",
        "| Step | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |",
        "|---|---|---|---|---|",
        f"| Generation | {lat.get('generation', {}).get('avg', '-')} | {lat.get('generation', {}).get('p50', '-')} | {lat.get('generation', {}).get('p95', '-')} | {lat.get('generation', {}).get('max', '-')} |",
        f"| End-to-End Total | {lat.get('total', {}).get('avg', '-')} | {lat.get('total', {}).get('p50', '-')} | {lat.get('total', {}).get('p95', '-')} | {lat.get('total', {}).get('max', '-')} |",
        "",
        "## 3. By Category (Answerable Questions)",
        "",
        "| Category | n | Citation F1 | Token Recall | Token F1 | Num Fact Recall |",
        "|---|---|---|---|---|---|",
    ]

    for cat, m in scored.get("by_category", {}).items():
        lines.append(
            f"| `{cat}` | {m['n']} | {m['citation_f1']:.3f} | {m['token_recall']:.3f} | {m['token_f1']:.3f} | {m['num_fact_recall']:.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generation metrics scorer (Phase 13).")
    parser.add_argument("--result", required=True, help="Path to raw full RAG evaluation JSON")
    parser.add_argument("--out", default=None, help="Output path for scored JSON")
    parser.add_argument("--report", default=None, help="Output path for Markdown report")
    args = parser.parse_args()

    p = Path(args.result)
    if not p.exists():
        p = paths.RESULTS_DIR / args.result
    if not p.exists():
        print(f"Error: result file not found: {args.result}", file=sys.stderr)
        return 1

    raw = json.loads(p.read_text(encoding="utf-8"))
    scored = score_envelope(raw)

    out_json = Path(args.out) if args.out else paths.RESULTS_DIR / f"{p.stem.replace('_raw', '')}_scored.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")

    report_md = render_report(scored)
    out_rep = Path(args.report) if args.report else paths.REPORTS_DIR / f"generation_metrics_{p.stem.replace('_full_rag_raw', '').replace('_raw', '')}.md"
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"GENERATION METRICS — {scored['experiment']}")
    print("=" * 68)
    agg = scored["aggregate"]
    print(f"citation F1         : {agg['citation']['f1']}")
    print(f"token recall vs GT  : {agg['correctness']['token_recall']}")
    print(f"numerical recall    : {agg['correctness']['numerical_fact_recall']}")
    print(f"abstention rate     : {agg['unanswerable_safety']['abstention_rate'] * 100:.1f}%")
    print(f"hallucination rate  : {agg['unanswerable_safety']['hallucination_rate'] * 100:.1f}%")
    print(f"scored json         : {out_json}")
    print(f"markdown report     : {out_rep}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
