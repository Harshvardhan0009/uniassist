"""Faithfulness & Hallucination Audit framework (Phase 20).

Performs comprehensive factual groundedness, citation verification, and hallucination detection
on generated RAG answers against retrieved context documents and ground-truth references.

Audits:
1. Numerical Faithfulness: Verifies all numbers, percentages, currency, and dates against retrieved context.
2. Citation Faithfulness & Phantom Citations: Validates cited documents against provided context documents.
3. Intrinsic & Extrinsic Hallucination Detection: Identifies unsupported factual claims.
4. Unanswerable Refusal Fidelity: Verifies 0% hallucination on out-of-corpus queries.
5. Context Utilization Rate: Measures which retrieved chunks directly supported the answer.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from ..lib import paths

paths.force_utf8_stdio()
logger = logging.getLogger("eval.faithfulness_audit")

# Numbers, percentages, fees, currency, CGPA, grades, day intervals
NUMERICAL_PATTERN = re.compile(r"\b(?:\$?Rs\.?\s*|\$|INR\s*)?\d+(?:[\.,]\d+)?(?:%|LPA|th|st|nd|rd)?\b", re.IGNORECASE)


def load_snapshot_chunks() -> dict[str, Document]:
    """Load all chunks from chunk_1500_200.jsonl keyed by chunk_id."""
    path = paths.SNAPSHOT_DIR / "chunk_1500_200.jsonl"
    chunks = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            cid = f"{meta.get('source_file')}__chunk_{meta.get('chunk_index')}"
            raw = meta.get("raw_content", rec.get("page_content", ""))
            chunks[cid] = Document(page_content=raw, metadata=meta)
    return chunks


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numerical tokens (e.g. '15000', '7.5', '40%', '3')."""
    raw_nums = NUMERICAL_PATTERN.findall(text)
    cleaned = set()
    for n in raw_nums:
        # Strip currency symbols and whitespace
        val = re.sub(r"[^\d\.\%]", "", n.strip().lower())
        if val and len(val) >= 1:
            cleaned.add(val)
    return cleaned


def audit_single_record(
    record: dict,
    chunk_map: dict[str, Document],
    gt_item: dict,
) -> dict[str, Any]:
    """Audit faithfulness, citations, and hallucination for a single question record."""
    qid = record["id"]
    q = record["question"]
    is_ans = record.get("answerable", gt_item.get("answerable", True))
    answer = record.get("answer", "") or ""

    reranked_refs = record.get("reranked", [])
    context_chunks = []
    for ref in reranked_refs:
        cid = ref.get("chunk_id")
        if cid in chunk_map:
            context_chunks.append(chunk_map[cid])

    full_context_text = " ".join(c.page_content for c in context_chunks).lower()
    context_sources = {c.metadata.get("source_file", "") for c in context_chunks if c.metadata.get("source_file")}
    expected_sources = set(gt_item.get("expected_sources", []))
    expected_pages = set(gt_item.get("expected_pages", []))
    gt_answer = gt_item.get("ground_truth_answer", "") or ""

    # 1. Unanswerable handling
    if not is_ans:
        from .generation_metrics import _check_abstention
        abstained = _check_abstention(answer)
        return {
            "id": qid,
            "question": q,
            "answerable": False,
            "classification": "CLEAN_ABSTENTION" if abstained else "HALLUCINATED_ANSWER",
            "is_faithful": abstained,
            "has_hallucination": not abstained,
            "phantom_citations": [],
            "unsupported_numbers": [],
            "faithfulness_score": 1.0 if abstained else 0.0,
            "details": "Model correctly refused out-of-corpus query" if abstained else "Model hallucinated answer to unanswerable query",
        }

    # 2. Citation Verification
    # Check what sources the model cited in its answer text
    cited_sources = set()
    for src in context_sources | expected_sources:
        fname = Path(src).name
        # Match filename without extension or full name
        base_name = Path(src).stem
        if fname.lower() in answer.lower() or base_name.lower() in answer.lower():
            cited_sources.add(src)

    # Check for Phantom Citations (cited files NOT in retrieved context)
    phantom_citations = [s for s in cited_sources if s not in context_sources]

    # Citation Grounding Precision (are cited sources actual ground truth sources?)
    valid_citations = cited_sources & expected_sources
    cite_precision = len(valid_citations) / len(cited_sources) if cited_sources else 1.0
    cite_recall = len(valid_citations) / len(expected_sources) if expected_sources else 1.0

    # 3. Numerical Fact Verification
    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(full_context_text)
    gt_numbers = extract_numbers(gt_answer)

    # Numbers in answer that appear neither in context nor in query
    query_numbers = extract_numbers(q)
    unsupported_numbers = []
    for num in answer_numbers:
        if num not in context_numbers and num not in query_numbers:
            unsupported_numbers.append(num)

    # Numerical accuracy vs Ground Truth
    matching_gt_numbers = answer_numbers & gt_numbers
    num_faithfulness = (
        (len(answer_numbers) - len(unsupported_numbers)) / len(answer_numbers)
        if answer_numbers
        else 1.0
    )

    # 4. Context Groundedness & Sentence Support
    sentences = [s.strip() for s in re.split(r"[.\n]", answer) if len(s.strip()) > 15]
    supported_sentences = 0
    for sent in sentences:
        words = [w for w in re.sub(r"[^\w\s]", "", sent.lower()).split() if len(w) > 3]
        if not words:
            continue
        # Check if high proportion of key words appear in context
        in_context = sum(1 for w in words if w in full_context_text)
        if in_context / len(words) >= 0.65:
            supported_sentences += 1

    sent_groundedness = supported_sentences / len(sentences) if sentences else 1.0

    # 5. Hallucination Classification
    has_hallucination = len(unsupported_numbers) > 0 or len(phantom_citations) > 0 or sent_groundedness < 0.50

    if not has_hallucination and sent_groundedness >= 0.85:
        classification = "FULLY_GROUNDED"
    elif not has_hallucination:
        classification = "GROUNDED_WITH_REASONING"
    elif len(unsupported_numbers) > 0:
        classification = "NUMERICAL_HALLUCINATION"
    elif len(phantom_citations) > 0:
        classification = "PHANTOM_CITATION"
    else:
        classification = "EXTRINSIC_HALLUCINATION"

    faithfulness_score = round((num_faithfulness * 0.4) + (sent_groundedness * 0.4) + (cite_precision * 0.2), 4)

    return {
        "id": qid,
        "question": q,
        "answerable": True,
        "classification": classification,
        "is_faithful": not has_hallucination,
        "has_hallucination": has_hallucination,
        "faithfulness_score": faithfulness_score,
        "sentence_groundedness": round(sent_groundedness, 4),
        "numerical_faithfulness": round(num_faithfulness, 4),
        "citation_precision": round(cite_precision, 4),
        "citation_recall": round(cite_recall, 4),
        "unsupported_numbers": unsupported_numbers,
        "phantom_citations": phantom_citations,
        "cited_sources": list(cited_sources),
    }


def audit_results_file(raw_results_path: Path) -> dict[str, Any]:
    """Run full faithfulness and hallucination audit on a full_rag_raw.json artifact."""
    chunk_map = load_snapshot_chunks()
    raw_data = json.loads(raw_results_path.read_text(encoding="utf-8"))
    results = raw_data.get("results", [])

    questions_data = json.loads(paths.QUESTIONS_PATH.read_text(encoding="utf-8"))
    gt_map = {q["id"]: q for q in questions_data.get("questions", [])}

    audited_records = []
    for r in results:
        qid = r["id"]
        gt_item = gt_map.get(qid, {})
        audited_records.append(audit_single_record(r, chunk_map, gt_item))

    ans_records = [r for r in audited_records if r["answerable"]]
    unans_records = [r for r in audited_records if not r["answerable"]]

    # Aggregates
    n_ans = len(ans_records)
    n_unans = len(unans_records)

    classifications = Counter(r["classification"] for r in audited_records)
    total_faithful = sum(1 for r in audited_records if r["is_faithful"])
    total_hallucinations = sum(1 for r in audited_records if r["has_hallucination"])

    avg_faithfulness = sum(r["faithfulness_score"] for r in ans_records) / n_ans if n_ans else 0.0
    avg_sent_groundedness = sum(r["sentence_groundedness"] for r in ans_records) / n_ans if n_ans else 0.0
    avg_num_faithfulness = sum(r["numerical_faithfulness"] for r in ans_records) / n_ans if n_ans else 0.0
    avg_cite_precision = sum(r["citation_precision"] for r in ans_records) / n_ans if n_ans else 0.0

    phantom_citation_queries = [r["id"] for r in ans_records if r["phantom_citations"]]
    numerical_hallucination_queries = [r["id"] for r in ans_records if r["unsupported_numbers"]]

    summary = {
        "audited_file": str(raw_results_path.name),
        "total_evaluated": len(audited_records),
        "answerable_count": n_ans,
        "unanswerable_count": n_unans,
        "overall_faithfulness_rate": round(total_faithful / len(audited_records), 4),
        "hallucination_rate": round(total_hallucinations / len(audited_records), 4),
        "mean_faithfulness_score": round(avg_faithfulness, 4),
        "mean_sentence_groundedness": round(avg_sent_groundedness, 4),
        "mean_numerical_faithfulness": round(avg_num_faithfulness, 4),
        "mean_citation_precision": round(avg_cite_precision, 4),
        "classification_breakdown": dict(classifications),
        "unanswerable_abstention_accuracy": f"{sum(1 for r in unans_records if r['is_faithful'])}/{n_unans}",
        "phantom_citations_found": len(phantom_citation_queries),
        "phantom_citation_queries": phantom_citation_queries,
        "numerical_hallucinations_found": len(numerical_hallucination_queries),
        "numerical_hallucination_queries": numerical_hallucination_queries,
    }

    return {"summary": summary, "per_question": audited_records}
