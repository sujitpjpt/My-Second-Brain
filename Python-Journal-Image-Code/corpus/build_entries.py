"""Group per-photo OCR records into one record per journal entry -> entries_v1.jsonl.

    python corpus/build_entries.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

# Paths are relative to this file, so the script works from any working directory
ROOT = Path(__file__).parent.parent.parent / "Journal-Entires"
CORPUS = ROOT / "corpus" / "ocr_v1.jsonl"
MANIFEST = ROOT / "corpus" / "manifest_v1.json"
OUTPUT = ROOT / "corpus" / "entries_v1.jsonl"  # gitignored via *.jsonl

# Rough date-like strings ("12/03", "3-4-24", "March 4"); a hint for the multi-day check, not a parser
DATE_RE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b"  # numeric: 12/03, 3-4-24
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}\b",  # month name + day
    re.I,
)


def main():
    manifest = json.loads(MANIFEST.read_text())  # fails if the corpus was never frozen: entries are built only from a frozen corpus
    records = [json.loads(ln) for ln in CORPUS.read_text().splitlines() if ln]

    # Bucket the per-photo records by entry
    groups = defaultdict(list)
    for r in records:
        # Filenames that didn't match the pattern have no entry_id: treat each as its own entry
        groups[r["entry_id"] or r["filename"]].append(r)

    entries, review = [], []
    for entry_id, pages in sorted(groups.items()):  # sorted = chronological, since ids start with the date
        pages.sort(key=lambda r: r["page"] or 0)  # page order within the entry
        nums = [r["page"] for r in pages]
        notes = []  # review flags for this entry

        # Pages should run 1..N with no holes
        if None not in nums and nums != list(range(1, len(nums) + 1)):
            notes.append("PAGE_GAP")  # e.g. pages 1,3: a photo is missing or was rejected

        # Two or more distinct date-like strings in one entry may mean it holds two days.
        # findall returns a str or a tuple depending on the regex group; keep only plain strings
        dates = {d.lower() for p in pages for d in DATE_RE.findall(p["text"]) if isinstance(d, str)}
        if len(dates) >= 2:
            notes.append("POSSIBLE_MULTI_DAY")  # a hint only; review by eye

        entry = {
            "entry_id": entry_id,
            "date": pages[0]["date"],  # from the filename, i.e. the Notion page date
            "title": pages[0]["title"],
            "text": "\n\n".join(p["text"] for p in pages),  # blank line between pages
            "sources": [p["filename"] for p in pages],  # trace any answer back to its photos
            "n_pages": len(pages),
            "review": notes,  # empty list = nothing to check
            "source_corpus_sha256": manifest["sha256"],  # which frozen corpus this came from
        }
        entries.append(entry)
        if notes:
            review.append((entry_id, notes))

    # One JSON object per line; ensure_ascii=False keeps non-ASCII characters readable
    OUTPUT.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    print(f"{len(records)} pages -> {len(entries)} entries; {len(review)} need review")
    for entry_id, notes in review:
        print(f"  {entry_id}: {notes}")


if __name__ == "__main__":
    main()
