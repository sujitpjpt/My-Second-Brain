# eval/ — Phase 2: retrieval evaluation

Build, freeze and score the retrieval eval set. Files live in `Journal-Entires/eval/` (gitignored).

| File | Purpose |
|---|---|
| `eval_sampler.py` | Interactive helper for writing the eval set (`retrieval_v1.jsonl`). Modes: `--draft N` (local LLM drafts, you keep/edit/drop), `--manual`, `--add` (multi-entry/date questions), `--grep`, `--show ID`, `--stats`. **Prints journal text — run in your own terminal only.** |
| `freeze_eval.py` | Validates the eval set, writes a manifest (hash + link to the corpus), makes the file read-only. Do this before measuring a baseline. |
| `eval_retrieval.py` | Scores a Chroma collection against the frozen set: recall@k, hit@k, MRR. Retrieval only (no generator/LLM judge). Search is exact by default (brute-force cosine over all chunk vectors in numpy, ties broken by chunk id), so repeated runs give identical rankings. `--ann` switches to Chroma's approximate HNSW search, which can flip a few questions between separate runs; use it only to measure how far HNSW drifts from exact. Terminal shows ids/ranks/numbers only; per-question results go to `Journal-Entires/eval/results/`. `--collection` compares variants; The date filter is on by default and restricts candidates to a date named in the question; `--no-date-filter` turns it off and adds a `__nodatefilter` suffix to the result file (see `../retrieve/`) and prints which questions it fired on. `--rerank [--rerank-depth 40]` reorders each question's top chunks with the local cross-encoder in `../retrieve/rerank.py` (after the date filter, before chunks collapse to entries), adds `top1_rerank_score` / `n_truncated` per question, and adds a `__rerank40` suffix; needs the reranker downloaded once and pinned in `PINNED_REVISIONS`. |

Order: `eval_sampler.py` → `freeze_eval.py` → `eval_retrieval.py` (after `../index/build_index.py`).
