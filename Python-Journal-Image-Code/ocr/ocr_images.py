"""OCR all journal images via local Ollama -> append-only JSONL corpus.

    python ocr/ocr_images.py --limit 5   # smoke test
    python ocr/ocr_images.py             # full run; Ctrl-C and re-run resumes
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import ollama

# Paths are relative to this file, so the script works from any working directory
ROOT = Path(__file__).parent.parent.parent / "Journal-Entires"
IMAGES_DIR = ROOT / "journal_images"
OUTPUT_PATH = ROOT / "corpus" / "ocr_v1.jsonl"  # bump the version if model/prompt/options change; *.jsonl is gitignored

# Pinned settings (dev-log Step 8). Also saved in every record so each line says how it was made.
MODEL = "qwen2.5vl:7b"  # digest 5ced39df...611cc
PROMPT = (
    "Transcribe all handwritten text in this image exactly as written, "
    "preserving line breaks. Output only the transcription, no commentary."
)
OPTIONS = {
    "num_ctx": 8192,        # total workspace: image (~4k tokens) + prompt + response
    "num_predict": 2048,    # cap on response length only; safety net against runaway output
    "repeat_penalty": 1.15,  # small penalty to stop repetition loops (>1 discourages repeats)
    "repeat_last_n": 256,   # how many recent tokens the penalty looks back over
    "temperature": 0,       # deterministic; set explicitly instead of the model default (0.0001)
    "seed": 42,             # fixed seed, so a re-run gives the same text
}

# Filename format from the Notion download: YYYY-MM-DD_Entry-Title_N.jpg
FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)_(\d+)$")


def check_output(response, text):
    """Flags for silent OCR failures; empty list means clean."""
    flags = []
    if not text:
        flags.append("EMPTY")  # model returned nothing
    if response.get("done_reason") == "length":
        flags.append("TRUNCATED")  # hit a token cap before finishing; "stop" means it finished itself
    # Looping model: many lines but few distinct ones
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:
        flags.append("REPETITIVE")
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="process at most N images")
    args = parser.parse_args()

    images = sorted(IMAGES_DIR.glob("*.jpg"))  # sorted = chronological, since filenames start with the date
    if not images:
        # an empty glob would otherwise fail silently (this bit us in Step 4)
        raise FileNotFoundError(f"No .jpg files in {IMAGES_DIR.resolve()}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Resume: collect filenames already in the corpus so a re-run skips them
    done = set()
    if OUTPUT_PATH.exists():
        done = {json.loads(line)["filename"] for line in OUTPUT_PATH.read_text().splitlines() if line}
    todo = [p for p in images if p.name not in done][: args.limit]  # [:None] = no limit
    print(f"{len(images)} images, {len(done)} done, {len(todo)} to process")

    for i, path in enumerate(todo, 1):
        # Pull date/title/page out of the filename; None if the name doesn't match the pattern
        m = FILENAME_RE.match(path.stem)
        date, title, page = m.groups() if m else (None, None, None)

        # The actual OCR call; runs locally through Ollama, nothing leaves the machine
        response = ollama.generate(model=MODEL, prompt=PROMPT, images=[str(path)], options=OPTIONS)
        text = response["response"].strip()
        flags = check_output(response, text)

        record = {
            "filename": path.name,  # also the resume key
            "entry_id": f"{date}_{title}" if m else None,  # pages sharing an entry_id belong to one entry
            "date": date,
            "title": title,
            "page": int(page) if page else None,  # page order within the entry
            "text": text,
            "flags": flags,  # review any record with non-empty flags before trusting it
            "model": MODEL,
            "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:12],  # detects a changed prompt
            "options": OPTIONS,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Append one line per image and close the file each time, so a crash loses at most the current image
        with OUTPUT_PATH.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")  # ensure_ascii=False keeps non-ASCII characters readable

        print(f"[{i}/{len(todo)}] {path.name} chars={len(text)} {flags or 'OK'}")


if __name__ == "__main__":
    main()
