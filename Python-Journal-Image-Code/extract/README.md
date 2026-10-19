# extract/ — Phase 1, step 1: get the images

Pulls journal photos out of Notion and normalises them. Run from the repo root; data lands in `Journal-Entires/` (gitignored).

| File | Purpose |
|---|---|
| `download_journal_images.py` | Downloads images attached to Notion journal pages → `journal_images/<date>_<title>_<n>.<ext>`. Needs `NOTION_API_TOKEN` and `NOTION_DATABASE_ID` env vars. `--start` required, `--end` optional. Re-runnable: skips images already on disk or hand-moved to `rejected_journal_images/`. |
| `convert_to_jpg.py` | Converts non-JPG images (png, heic, …) to JPG at quality 95 with macOS `sips`. Deletes each original only after the JPG is verified. `--dry-run` previews. |

Order: download → convert → `../ocr/`.
