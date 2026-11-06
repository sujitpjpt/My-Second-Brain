# retrieve/ — Phase 2: query-time logic shared by the eval and the RAG app

Pure-Python helpers that sit between a question and the vector index. No journal text is read or written here.

| File | Purpose |
|---|---|
| `date_filter.py` | Finds an explicit date in a question (`3 November 2024`, `November 2024`, `2024-11-03`, `in 2023`) and returns the concrete indexed dates inside that window. Callers match them with Chroma `where={"date": {"$in": [...]}}` (Chroma's `$gte`/`$lte` only work on numbers; `date` is an ISO string). Relative or fuzzy phrases ("last spring", "early 2022 to early 2024") deliberately do not fire: a wrong filter silently drops the right entry, a missed one falls back to normal ranking. |
| `date_filter_test.py` | Synthetic-string checks for the parser (`python retrieve/date_filter_test.py`). No Ollama, no Chroma, no journal data. |

Used by `../eval/eval_retrieval.py`, on by default (`--no-date-filter` turns it off). No index rebuild needed: the `date` metadata is already stored per chunk.
