# corpus/ — Phase 1, step 3: OCR output → frozen corpus

Turns per-page OCR records into versioned, read-only corpus files in `Journal-Entires/corpus/`.

| File | Purpose |
|---|---|
| `freeze_corpus.py` | Freezes `ocr_v1.jsonl`: writes a manifest (content hash + OCR model digest) and makes the file read-only. This is what pins the corpus for reproducibility. |
| `build_entries.py` | Groups per-page records by entry ID, orders pages, joins text → `entries_v1.jsonl` (one record per entry). Flags entries needing a manual check. |
| `build_chunks.py` | Splits entries into retrieval chunks → `chunks_<version>.jsonl` + manifest. One chunk per entry unless longer than `--split-above`; long ones are cut with overlap. `--dry-run` prints stats only. `--version` / `--chunk-chars` make variants to compare via the eval set. |

Order: `../ocr/` → `freeze_corpus.py` → `build_entries.py` → `build_chunks.py` → `../index/`.
Output files contain journal text; they are gitignored (`*.jsonl`, `Journal-Entires/`).
