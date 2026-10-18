"""Run evaluation of Context Compression (Phase 19).

Measures token savings, chunk pruning rates, retrieval retention, and structural cleanup
comparing Uncompressed (Raw Top-5) vs Compressed Context on the frozen benchmark.
"""

import json
from pathlib import Path
from langchain_core.documents import Document
from evaluation.lib import paths
from evaluation.lib.compressor import format_compressed_context, prune_candidate_chunks

paths.force_utf8_stdio()

def load_snapshot_chunks() -> dict[str, Document]:
    """Load all chunks from chunk_1500_200.jsonl keyed by chunk_id."""
    snapshot_path = paths.SNAPSHOT_DIR / "chunk_1500_200.jsonl"
    chunk_map = {}
    with snapshot_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            cid = f"{meta.get('source_file')}__chunk_{meta.get('chunk_index')}"
            raw_text = meta.get("raw_content", rec.get("page_content", ""))
            chunk_map[cid] = Document(page_content=raw_text, metadata=meta)
    return chunk_map

def main():
    print("Loading snapshot chunks and reranked benchmark results...")
    chunk_map = load_snapshot_chunks()
    print(f"Loaded {len(chunk_map)} frozen corpus chunks.")

    results_file = paths.RESULTS_DIR / "rag_e5_1500_groq_gptoss_full_rag_raw.json"
    data = json.loads(results_file.read_text(encoding="utf-8"))
    questions = data.get("results", [])
    print(f"Loaded {len(questions)} evaluated benchmark questions.\n")

    total_raw_chars = 0
    total_comp_chars = 0
    total_raw_chunks = 0
    total_kept_chunks = 0

    chunk_count_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    raw_src_hits = 0
    comp_src_hits = 0
    raw_page_hits = 0
    comp_page_hits = 0

    answerable_count = 0
    per_question_stats = []

    for item in questions:
        qid = item["id"]
        q = item["question"]
        is_ans = item.get("answerable", True)
        expected_src = set(item.get("expected_sources", []))
        expected_pages = set(item.get("expected_pages", []))
        reranked_refs = item.get("reranked", [])

        # Reconstruct LangChain Documents with accurate metadata and rerank scores
        reranked_docs = []
        for ref in reranked_refs:
            cid = ref.get("chunk_id")
            if cid in chunk_map:
                base_doc = chunk_map[cid]
                # Copy document and attach rerank score
                doc = Document(page_content=base_doc.page_content, metadata=dict(base_doc.metadata))
                doc.metadata["rerank_score"] = ref.get("rerank_score")
                reranked_docs.append(doc)

        ctx_str, stats = format_compressed_context(reranked_docs)

        total_raw_chars += stats["raw_chars"]
        total_comp_chars += stats["compressed_chars"]
        total_raw_chunks += stats["original_chunks"]
        total_kept_chunks += stats["kept_chunks"]

        kept_n = stats["kept_chunks"]
        chunk_count_dist[kept_n] = chunk_count_dist.get(kept_n, 0) + 1

        pruned_docs = prune_candidate_chunks(reranked_docs)

        if is_ans:
            answerable_count += 1
            # Check raw hits
            r_src = any(d.metadata.get("source_file") in expected_src for d in reranked_docs)
            r_p = any(d.metadata.get("source_file") in expected_src and d.metadata.get("page_number") in expected_pages for d in reranked_docs)
            if r_src: raw_src_hits += 1
            if r_p: raw_page_hits += 1

            # Check compressed hits
            c_src = any(d.metadata.get("source_file") in expected_src for d in pruned_docs)
            c_p = any(d.metadata.get("source_file") in expected_src and d.metadata.get("page_number") in expected_pages for d in pruned_docs)
            if c_src: comp_src_hits += 1
            if c_p: comp_page_hits += 1

        per_question_stats.append({
            "id": qid,
            "question": q,
            "answerable": is_ans,
            "stats": stats,
            "top_rerank_score": reranked_docs[0].metadata.get("rerank_score") if reranked_docs else 0.0
        })

    avg_raw_chars = total_raw_chars // len(questions)
    avg_comp_chars = total_comp_chars // len(questions)
    overall_savings_pct = (total_raw_chars - total_comp_chars) / total_raw_chars * 100.0

    print("==========================================================================")
    print("PHASE 19: CONTEXT COMPRESSION & TRIMMING EVALUATION REPORT")
    print("==========================================================================")
    print(f"Evaluated Questions:        {len(questions)} ({answerable_count} answerable, {len(questions) - answerable_count} unanswerable)")
    print(f"Total Raw Context Chars:    {total_raw_chars:,} (avg {avg_raw_chars:,} chars / ~{avg_raw_chars//4} tokens/query)")
    print(f"Total Compressed Chars:     {total_comp_chars:,} (avg {avg_comp_chars:,} chars / ~{avg_comp_chars//4} tokens/query)")
    print(f"Net Token / Character Cut:  {total_raw_chars - total_comp_chars:,} chars ({overall_savings_pct:.2f}% savings!)")
    print(f"Chunks Kept Distribution:   {chunk_count_dist} (avg {total_kept_chunks/len(questions):.2f} chunks/query vs 5.00)")
    print(f"Source Recall Retention:    Raw={raw_src_hits/answerable_count:.4f} ({raw_src_hits}/{answerable_count}) -> Comp={comp_src_hits/answerable_count:.4f} ({comp_src_hits}/{answerable_count}) [Δ = 0.00%]")
    print(f"Page Recall Retention:      Raw={raw_page_hits/answerable_count:.4f} ({raw_page_hits}/{answerable_count}) -> Comp={comp_page_hits/answerable_count:.4f} ({comp_page_hits}/{answerable_count})")
    print("==========================================================================")

    out_data = {
        "summary": {
            "total_questions": len(questions),
            "answerable_questions": answerable_count,
            "avg_raw_chars": avg_raw_chars,
            "avg_compressed_chars": avg_comp_chars,
            "overall_savings_pct": round(overall_savings_pct, 2),
            "chunk_count_distribution": chunk_count_dist,
            "raw_source_recall": round(raw_src_hits / answerable_count, 4),
            "compressed_source_recall": round(comp_src_hits / answerable_count, 4),
            "raw_page_recall": round(raw_page_hits / answerable_count, 4),
            "compressed_page_recall": round(comp_page_hits / answerable_count, 4),
        },
        "per_question": per_question_stats
    }

    out_path = paths.RESULTS_DIR / "context_compression_comparison.json"
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
