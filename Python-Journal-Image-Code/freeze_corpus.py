"""Freeze the OCR corpus: write a manifest (hash + model digest) and make the file read-only.

    python freeze_corpus.py
"""
import hashlib
import json
import os
import stat
from pathlib import Path

import ollama

# Paths are relative to this file, so the script works from any working directory
ROOT = Path(__file__).parent.parent / "Journal-Entires"
CORPUS = ROOT / "corpus" / "ocr_v1.jsonl"
MANIFEST = ROOT / "corpus" / "manifest_v1.json"  # one manifest per corpus version
IMAGES_DIR = ROOT / "journal_images"


def main():
    # Load every OCR record (one JSON object per line; skip blank lines)
    records = [json.loads(ln) for ln in CORPUS.read_text().splitlines() if ln]
    n_images = len(list(IMAGES_DIR.glob("*.jpg")))  # how many photos should have been OCR'd
    done = {r["filename"] for r in records}  # set removes duplicates, so sizes can be compared

    # Refuse to freeze a partial or dirty corpus: once frozen, mistakes are baked in
    if len(records) != len(done):
        raise SystemExit(f"duplicate filenames: {len(records)} records, {len(done)} unique")
    if len(done) != n_images:
        raise SystemExit(f"OCR incomplete: {len(done)}/{n_images} images")
    flagged = [r["filename"] for r in records if r["flags"]]  # EMPTY / TRUNCATED / REPETITIVE
    if flagged:
        raise SystemExit(f"{len(flagged)} flagged records, review first: {flagged[:5]}")

    # Exactly one model/prompt/options combo allowed; mixed settings means a mixed corpus
    # (options are dumped with sorted keys so dict ordering can't create a false difference)
    combos = {(r["model"], r["prompt_hash"], json.dumps(r["options"], sort_keys=True)) for r in records}
    if len(combos) != 1:
        raise SystemExit(f"{len(combos)} different model/prompt/option combos in corpus")
    model, prompt_hash, _ = next(iter(combos))

    # Exact weights behind the model tag, from the local Ollama install. A tag like
    # "qwen2.5vl:7b" can point to different weights over time; the digest can't
    digest = next(m.digest for m in ollama.list().models if m.model == model)

    manifest = {
        "file": CORPUS.name,
        "sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),  # fingerprint of the exact bytes
        "records": len(records),
        "model": model,
        "model_digest": digest,
        "prompt_hash": prompt_hash,
        "options": records[0]["options"],  # identical across records (checked above)
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    os.chmod(CORPUS, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only: nobody can edit/append by accident
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
