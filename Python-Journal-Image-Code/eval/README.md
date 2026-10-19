# eval/ — Phase 2: retrieval evaluation

Build, freeze and score the retrieval eval set. Files live in `Journal-Entires/eval/` (gitignored).

| File | Purpose |
|---|---|
| `eval_sampler.py` | Interactive helper for writing the eval set (`retrieval_v1.jsonl`). Modes: `--draft N` (local LLM drafts, you keep/edit/drop), `--manual`, `--add` (multi-entry/date questions), `--grep`, `--show ID`, `--stats`. **Prints journal text — run in your own terminal only.** |
| `freeze_eval.py` | Validates the eval set, writes a manifest (hash + link to the corpus), makes the file read-only. Do this before measuring a baseline. |
| `eval_retrieval.py` | Scores a Chroma collection against the frozen set: recall@k, hit@k, MRR. Retrieval only (no generator/LLM judge), so deterministic. Terminal shows ids/ranks/numbers only; per-question results go to `Journal-Entires/eval/results/`. `--collection` compares variants. |

Order: `eval_sampler.py` → `freeze_eval.py` → `eval_retrieval.py` (after `../index/build_index.py`).
