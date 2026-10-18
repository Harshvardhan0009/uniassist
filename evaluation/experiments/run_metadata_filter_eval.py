import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.lib.harness import load_config, retrieve_candidates, doc_ref
from evaluation.lib.experiment import prepare_store
from evaluation.lib.metadata_filter import classify_domain, matches_domain, FILE_TO_DOMAINS
from evaluation.metrics.retrieval_metrics import _make_relevance

# Load config and isolated store
cfg = load_config('chunk_1500_200_e5')
store, manifest, idx_info, n_docs = prepare_store(cfg)

# Load benchmark questions
questions_data = json.load(open('evaluation/dataset/questions.json', encoding='utf-8'))['questions']

# Pre-rewritten conversational queries from Phase 15 so retrieval is best-in-class
rewritten_queries = {
    'Q044': 'LPU block libraries operating hours',
    'Q045': 'Consequences of CGPA falling below the minimum required after registering for Career Services.',
    'Q046': 'minimum duration of a two-semester OJT internship',
    'Q047': 'Maximum duration of semester abroad for a 3-year degree program',
    'Q048': 'What grade is assigned for a published patent under grade upgradation?',
    'Q049': 'Which block is the School of Law located in?'
}

results = {
    'unfiltered': {'src_r1': [], 'src_r5': [], 'pg_r1': [], 'pg_r5': []},
    'oracle_filtered': {'src_r1': [], 'src_r5': [], 'pg_r1': [], 'pg_r5': []},
    'automated_filtered': {'src_r1': [], 'src_r5': [], 'pg_r1': [], 'pg_r5': [], 'router_acc': []}
}

misclassified_cases = []

answerable_count = 0
for q in questions_data:
    if not q.get('answerable', True):
        continue
    answerable_count += 1
    qid = q['id']
    query = rewritten_queries.get(qid, q['question'])
    exp_sources = q.get('expected_sources', [])
    exp_pages = q.get('expected_pages', [])
    gt_topic = q.get('topic')

    src_rel_fn = _make_relevance(exp_sources, exp_pages, 'source')
    pg_rel_fn = _make_relevance(exp_sources, exp_pages, 'page')

    # 1. Unfiltered dense retrieval (top 20)
    cands_unfiltered, _ = retrieve_candidates(store, query, top_k=20)
    u_src_r1 = 1 if cands_unfiltered and src_rel_fn(doc_ref(cands_unfiltered[0], 1)) else 0
    u_src_r5 = 1 if any(src_rel_fn(doc_ref(d, i+1)) for i, d in enumerate(cands_unfiltered[:5])) else 0
    u_pg_r1 = 1 if cands_unfiltered and pg_rel_fn(doc_ref(cands_unfiltered[0], 1)) else 0
    u_pg_r5 = 1 if any(pg_rel_fn(doc_ref(d, i+1)) for i, d in enumerate(cands_unfiltered[:5])) else 0

    results['unfiltered']['src_r1'].append(u_src_r1)
    results['unfiltered']['src_r5'].append(u_src_r5)
    results['unfiltered']['pg_r1'].append(u_pg_r1)
    results['unfiltered']['pg_r5'].append(u_pg_r5)

    # 2. Oracle Filtered (filter candidates by gt_topic)
    cands_oracle = [d for d in cands_unfiltered if matches_domain(d.metadata.get('source_file', ''), gt_topic)]
    o_src_r1 = 1 if cands_oracle and src_rel_fn(doc_ref(cands_oracle[0], 1)) else 0
    o_src_r5 = 1 if any(src_rel_fn(doc_ref(d, i+1)) for i, d in enumerate(cands_oracle[:5])) else 0
    o_pg_r1 = 1 if cands_oracle and pg_rel_fn(doc_ref(cands_oracle[0], 1)) else 0
    o_pg_r5 = 1 if any(pg_rel_fn(doc_ref(d, i+1)) for i, d in enumerate(cands_oracle[:5])) else 0

    results['oracle_filtered']['src_r1'].append(o_src_r1)
    results['oracle_filtered']['src_r5'].append(o_src_r5)
    results['oracle_filtered']['pg_r1'].append(o_pg_r1)
    results['oracle_filtered']['pg_r5'].append(o_pg_r5)

    # 3. Automated Filtered (Intent Router)
    pred_domain, router_ms = classify_domain(query)
    router_correct = (pred_domain == gt_topic) or any(pred_domain in FILE_TO_DOMAINS.get(s, []) for s in exp_sources)
    results['automated_filtered']['router_acc'].append(1 if router_correct else 0)

    cands_auto = [d for d in cands_unfiltered if matches_domain(d.metadata.get('source_file', ''), pred_domain)]
    a_src_r1 = 1 if cands_auto and src_rel_fn(doc_ref(cands_auto[0], 1)) else 0
    a_src_r5 = 1 if any(src_rel_fn(doc_ref(d, i+1)) for i, d in enumerate(cands_auto[:5])) else 0
    a_pg_r1 = 1 if cands_auto and pg_rel_fn(doc_ref(cands_auto[0], 1)) else 0
    a_pg_r5 = 1 if any(pg_rel_fn(doc_ref(d, i+1)) for i, d in enumerate(cands_auto[:5])) else 0

    results['automated_filtered']['src_r1'].append(a_src_r1)
    results['automated_filtered']['src_r5'].append(a_src_r5)
    results['automated_filtered']['pg_r1'].append(a_pg_r1)
    results['automated_filtered']['pg_r5'].append(a_pg_r5)

    if not router_correct or a_src_r1 < u_src_r1:
        misclassified_cases.append({
            'id': qid,
            'query': query,
            'gt_topic': gt_topic,
            'pred_domain': pred_domain,
            'expected_sources': exp_sources,
            'unfiltered_hit': u_src_r1,
            'filtered_hit': a_src_r1
        })

acc = mean(results['automated_filtered']['router_acc'])
print(f'=== PHASE 16 METADATA FILTERING RESULTS (N={answerable_count}) ===')
print(f'Router Accuracy: {acc:.3f}')
print('Unfiltered:        Src@1={:.3f}, Src@5={:.3f}, Pg@1={:.3f}, Pg@5={:.3f}'.format(
    mean(results['unfiltered']['src_r1']), mean(results['unfiltered']['src_r5']),
    mean(results['unfiltered']['pg_r1']), mean(results['unfiltered']['pg_r5'])
))
print('Oracle Filtered:   Src@1={:.3f}, Src@5={:.3f}, Pg@1={:.3f}, Pg@5={:.3f}'.format(
    mean(results['oracle_filtered']['src_r1']), mean(results['oracle_filtered']['src_r5']),
    mean(results['oracle_filtered']['pg_r1']), mean(results['oracle_filtered']['pg_r5'])
))
print('Automated Filtered: Src@1={:.3f}, Src@5={:.3f}, Pg@1={:.3f}, Pg@5={:.3f}'.format(
    mean(results['automated_filtered']['src_r1']), mean(results['automated_filtered']['src_r5']),
    mean(results['automated_filtered']['pg_r1']), mean(results['automated_filtered']['pg_r5'])
))
print(f'Misclassified/Degraded Cases count: {len(misclassified_cases)}')
for c in misclassified_cases[:5]:
    qid_c = c['id']
    gt_c = c['gt_topic']
    pred_c = c['pred_domain']
    uhit = c['unfiltered_hit']
    fhit = c['filtered_hit']
    q_str = c['query'][:50]
    print(f'  {qid_c}: GT={gt_c} Pred={pred_c} | UnfiltHit={uhit} FiltHit={fhit} | Q: {q_str}...')

out_path = Path('evaluation/experiments/results/metadata_filtering_comparison.json')
out_path.write_text(json.dumps({
    'summary': {
        'unfiltered': {k: mean(v) for k, v in results['unfiltered'].items()},
        'oracle_filtered': {k: mean(v) for k, v in results['oracle_filtered'].items()},
        'automated_filtered': {k: mean(v) for k, v in results['automated_filtered'].items()},
    },
    'misclassified_cases': misclassified_cases
}, indent=2), encoding='utf-8')
print('Saved to', out_path)
