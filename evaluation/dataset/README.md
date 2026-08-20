# UniAssist Evaluation Dataset (`uniassist_eval_v1`)

Permanent benchmark of real student queries, grounded in the LPU document corpus under `Data/`.
This is the fixed question set every experiment runs against (freeze rule #2).

- **File:** [`questions.json`](./questions.json)
- **Current size:** 63 questions (target 100+; expandable without renumbering existing IDs)
- **Built:** Phase 2 (2026-08-20). Every question was written after reading the actual source
  document it targets, so answerable questions are genuinely answerable from the corpus.

> **Do not fabricate ground truth.** Answers/sources/pages are attached in **Phase 3** strictly
> from the real documents.

---

## Schema

Each entry in `questions[]` has:

| Field | Phase | Type | Meaning |
|---|---|---|---|
| `id` | 2 | string | Stable unique ID (`Q001`…). Never renumber; append new IDs only. |
| `category` | 2 | enum | One of the 9 categories below. |
| `topic` | 2 | enum | Coarse document area (balance tracking), **not** the precise source. |
| `difficulty` | 2 | enum | `easy` / `medium` / `hard`. |
| `answerable` | 2 | bool | `false` = intentionally out-of-corpus (abstention test). |
| `question` | 2 | string | The user query (for conversational items, the follow-up turn). |
| `history` | 2 | array | Conversational items only: prior `{role, content}` turns. |
| `expected_sources` | **3** | string[] | Corpus file(s) (relative to `Data/`) that contain the answer. |
| `expected_pages` | **3** | int[] | Page number(s) supporting the answer. |
| `ground_truth_answer` | **3** | string\|null | Reference answer (null for unanswerable). |
| `expected_behavior` | **3** | string | For unanswerable: the required abstention response. |

Top-level keys: `dataset`, `version`, `created`, `phase`, `description`, `categories`, `topics`,
`notes`, `questions`.

---

## Categories (the 9)

1. **direct_factual** — single-fact lookups (e.g. *"What are the library timings?"*).
2. **policy** — rule/procedure questions (e.g. *"What are the hostel refund rules?"*).
3. **academic** — academic-benefit / CGPA / grade questions (EDU REV, Academic Benefits).
4. **placement** — career-services / OJT / placement-eligibility questions.
5. **table_based** — answers that live inside a table (charges, salary/stipend slabs, score→grade).
6. **multi_condition** — require combining several facts / conditional reasoning.
7. **conversational** — multi-turn; the follow-up is ambiguous without `history` (tests follow-up
   understanding and, later, query rewriting in Phase 15).
8. **exact_terminology** — abbreviations, codes, exact program/division names (OJT, PPO, PEP, LPA,
   course codes such as ENG/PEL/PEV).
9. **unanswerable** — the answer is **not** in the corpus; the system must abstain rather than invent
   an answer. Expected response pattern: *"I couldn't find this information in the official
   university documents available to me."*

---

## Current distribution (63 questions)

**By category:** policy 9 · direct_factual 8 · academic 7 · placement 7 · table_based 7 ·
exact_terminology 7 · multi_condition 6 · conversational 6 · unanswerable 6.

**By topic:** placement_career 19 · academic_benefits 10 · library 7 · residential 7 ·
semester_abroad 6 · ojt 5 · nss 4 · campus_map 3 · dress_code 2.

**By answerability:** 57 answerable · 6 unanswerable.
**By difficulty:** easy 18 · medium 40 · hard 5.

> `placement_career` is intentionally the largest topic: it spans the Career Services policy, the OJT
> policy, and the three placement/competitive-exam academic-benefit documents — the biggest slice of
> the corpus. `dress_code` is smallest, proportionate to its 2-page source.

---

## Corpus coverage

Questions are grounded across all 8 document areas: DressCode, EDU REV (Academic Benefits),
LibraryPolicy, NSSPolicy, PlacementPloicy (Career Services + OJT + 3 Academic Benefit Plans),
ResidentialFacilities, SemesterExchange, and UNImap (campus guide).

## Regenerating / extending

- Add new questions by appending new `QNNN` IDs (do not reuse or renumber).
- Keep categories balanced; prefer adding grounded, corpus-verifiable questions.
- Validate with the Phase 4 dataset checker (unique IDs, valid enums, conversational `history`).
