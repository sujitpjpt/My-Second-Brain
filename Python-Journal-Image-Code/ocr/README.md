# ocr/ — Phase 1, step 2: images → text

Local vision-model OCR via Ollama. Nothing leaves the machine.

| File | Purpose |
|---|---|
| `ocr_images.py` | The real OCR run. Reads every journal image, appends one record per page to `Journal-Entires/corpus/ocr_v1.jsonl`. Append-only; Ctrl-C and re-run resumes. `--limit N` for a smoke test. |
| `ocr_compare_models.py` | One-off model bake-off. Runs a small hand-picked set (`test_journal_images/`) through the candidate models, one folder of `.txt` per model in `ocr_test_output/`, for comparing by eye. Does not produce the corpus. |

Order: `../extract/` → `ocr_images.py` → `../corpus/`.
