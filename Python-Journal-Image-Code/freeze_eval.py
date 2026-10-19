"""Freeze the retrieval eval set: validate, write a manifest (hash + corpus link), make the file read-only.

    python freeze_eval.py
"""
import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent / "Journal-Entires"
EVAL = ROOT / "eval" / "retrieval_v1.jsonl"
MANIFEST = ROOT / "eval" / "retrieval_v1_manifest.json"
ENTRIES = ROOT / "corpus" / "entries_v1.jsonl"

# Allowed number of relevant ids per question type
RULES = {
    "single": lambda n: n == 1,
    "multi": lambda n: n >= 2,
    "date": lambda n: n >= 1,
    "unanswerable": lambda n: n == 0,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} exists: eval set already frozen. Make retrieval_v2 instead of editing v1")

    recs = [json.loads(ln) for ln in EVAL.read_text().splitlines() if ln.strip()]
    entry_ids = {json.loads(ln)["entry_id"] for ln in ENTRIES.read_text().splitlines() if ln.strip()}

    # Refuse to freeze a dirty set: once frozen, label mistakes are baked into every baseline
    errors = []
    for k in ("qid", "question"):
        dups = [v for v, c in Counter(r[k] for r in recs).items() if c > 1]
        if dups:
            errors.append(f"duplicate {k}: {dups}")
    for r in recs:
        q, rel = r["qid"], r["relevant"]
        if r["type"] not in RULES:
            errors.append(f"{q}: unknown type {r['type']!r}")
        elif not RULES[r["type"]](len(rel)):
            errors.append(f"{q}: type {r['type']} with {len(rel)} relevant ids")
        if len(set(rel)) != len(rel):
            errors.append(f"{q}: repeated ids in relevant")
        bad = [i for i in rel if i not in entry_ids]
        if bad:
            errors.append(f"{q}: ids not in corpus {bad}")
    if errors:
        raise SystemExit("not frozen:\n  " + "\n  ".join(errors))

    manifest = {
        "file": EVAL.name,
        "sha256": sha256(EVAL),
        "questions": len(recs),
        "by_type": dict(Counter(r["type"] for r in recs)),
        "by_source": dict(Counter(r.get("source", "hand") for r in recs)),
        # ties the labels to the exact corpus they were written against (ids are checked against it above)
        "entries_file": ENTRIES.name,
        "entries_sha256": sha256(ENTRIES),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    os.chmod(EVAL, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only: no accidental edits/appends
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
