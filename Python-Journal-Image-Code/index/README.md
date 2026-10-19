# index/ — Phase 2: chunks → vector index

Local embeddings (Ollama) into ChromaDB at `chroma_db/` (gitignored). The vectors are journal content — treat them like the corpus.

| File | Purpose |
|---|---|
| `build_index.py` | Verifies the chunks hash against the manifest, embeds in batches, upserts into a Chroma collection (default `nomic-embed-text_chunks_v1`), asserts count == chunk count, writes a read-only index manifest. `--dry-run` embeds nothing; `--rebuild` drops and recreates; `--model` / `--doc-prefix` / `--query-prefix` build a variant to compare. |
| `build_index_test.py` | Debug aid. Dumps exactly what each batch (and each string sent to the embedder) contains to `Journal-Entires/corpus/batch_debug_v1.json`. Embeds and indexes nothing. Output holds journal text. |

Order: `../corpus/` → `build_index.py` → `../eval/eval_retrieval.py`.
