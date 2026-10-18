"""Run Phase 20 Faithfulness & Hallucination Audit across model runs."""

from __future__ import annotations

import json
from pathlib import Path
from evaluation.lib import paths
from evaluation.metrics.faithfulness_audit import audit_results_file

paths.force_utf8_stdio()

def main():
    results_dir = paths.RESULTS_DIR
    
    files_to_audit = [
        ("groq_e5_1500", results_dir / "rag_e5_1500_groq_gptoss_full_rag_raw.json"),
        ("gemini_e5_1500_sample", results_dir / "rag_e5_1500_gemini_full_rag_raw.json"),
        ("gemini_baseline_v1", results_dir / "baseline_v1_gemini36_full_rag_raw.json"),
    ]
    
    audit_outputs = {}
    for label, path in files_to_audit:
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        print(f"Auditing {label} ({path.name})...")
        res = audit_results_file(path)
        audit_outputs[label] = res
        summary = res["summary"]
        print(f"  Total: {summary['total_evaluated']}")
        print(f"  Faithfulness Rate: {summary['overall_faithfulness_rate']*100:.1f}%")
        print(f"  Hallucination Rate: {summary['hallucination_rate']*100:.1f}%")
        print(f"  Mean Faithfulness Score: {summary['mean_faithfulness_score']:.4f}")
        print(f"  Numerical Faithfulness: {summary['mean_numerical_faithfulness']*100:.1f}%")
        print(f"  Citation Precision: {summary['mean_citation_precision']*100:.1f}%")
        print(f"  Unanswerable Abstention: {summary['unanswerable_abstention_accuracy']}")
        print(f"  Phantom Citations: {summary['phantom_citations_found']}")
        print(f"  Numerical Hallucinations: {summary['numerical_hallucinations_found']}")
        print()

    out_file = results_dir / "faithfulness_audit_scored.json"
    out_file.write_text(json.dumps(audit_outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit results written to {out_file}")

if __name__ == "__main__":
    main()
