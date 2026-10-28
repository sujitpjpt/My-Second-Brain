"""Chunk entries_v1.jsonl -> chunks_<version>.jsonl (+ manifest). One chunk per entry unless the entry is long.

    python corpus/build_chunks.py --dry-run                 # print stats only, write nothing
    python corpus/build_chunks.py                           # writes chunks_v1.jsonl, read-only
    python corpus/build_chunks.py --version v2 --chunk-chars 600   # a variant to compare against v1 with the eval set

Entries longer than --split-above are cut at paragraph, then sentence, then word boundaries into
roughly equal pieces, each (except the first) starting with the tail of the previous piece (overlap).
Prints counts and sizes only, never journal text.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

# Paths are relative to this file, so the script works from any working directory
ROOT = Path(__file__).parent.parent.parent / "Journal-Entires" / "corpus"
ENTRIES = ROOT / "entries_v1.jsonl"  # gitignored via *.jsonl

# Sentence boundary: whitespace after . ! or ? (lookbehind keeps the punctuation with its sentence).
# Naive on purpose: "Dr. Smith" splits wrongly, which only moves a cut point slightly
SENT_RE = re.compile(r"(?<=[.!?])\s+")


def units(text, limit):
    """Split text into (piece, separator-before) units no longer than limit: paragraphs, else sentences, else words."""
    out = []
    for para in re.split(r"\n\s*\n", text):  # blank line = paragraph break
        para = para.strip()
        if not para:
            continue
        # Keep a paragraph whole if it fits; otherwise fall back to its sentences
        parts = [para] if len(para) <= limit else SENT_RE.split(para)
        for i, s in enumerate(parts):
            sep = "\n\n" if i == 0 else " "  # first unit of a paragraph gets a blank-line gap; later sentences join with a space
            if len(s) <= limit:
                out.append((s, sep))
                continue
            # Single sentence longer than the limit (e.g. OCR with no punctuation): wrap on word boundaries
            line = ""
            for w in s.split():
                if line and len(line) + 1 + len(w) > limit:
                    out.append((line, sep))
                    line, sep = w, " "
                else:
                    line = f"{line} {w}".strip()
            if line:
                out.append((line, sep))
    return out


def split_text(text, chunk_chars, min_chunk):
    """Greedy-pack units into near-equal pieces (n = ceil(len / chunk_chars))."""
    # Equal-sized pieces beat "1200 + 1200 + 80": a 2500-char entry becomes 3 x ~833, not 2 x 1200 + a stub
    n = max(1, math.ceil(len(text) / chunk_chars))
    target = math.ceil(len(text) / n)
    pieces = []
    for unit, sep in units(text, target):
        # Append to the current piece while it stays near the target; 10% slack avoids a tiny last piece
        if pieces and len(pieces[-1]) + len(sep) + len(unit) <= target * 1.1:
            pieces[-1] += sep + unit
        else:
            pieces.append(unit)
    # A stub chunk has too little context to embed well: fold it into the previous piece
    if len(pieces) > 1 and len(pieces[-1]) < min_chunk:
        last = pieces.pop()
        pieces[-1] += "\n\n" + last
    return pieces


def overlap_tail(prev, overlap):
    """Whole trailing sentences of prev totalling <= overlap chars; else the last words that fit."""
    acc = ""
    for s in reversed(SENT_RE.split(prev)):  # walk backwards from the end of the previous piece
        cand = f"{s} {acc}".strip()
        if len(cand) > overlap:
            break
        acc = cand
    # No whole sentence fits (one long sentence): take the last `overlap` chars, dropping the cut-off first word
    return acc or prev[-overlap:].split(" ", 1)[-1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    # Version label in the output filename (chunks_v1.jsonl). Change it for each variant so v1 stays untouched for comparison
    ap.add_argument("--version", default="v1")
    # Length cutoff. 2000 sits above ~94% of entries (p90 is ~1750), so only the long outliers get split
    ap.add_argument("--split-above", type=int, default=2000, help="entries up to this many chars stay whole")
    # Size of each piece when an entry is split. 1200 is near the typical entry length (median ~1000),
    # so split pieces look like normal entries; it sets the piece count, pieces end up near-equal
    ap.add_argument("--chunk-chars", type=int, default=1200, help="target piece size when splitting")
    # Context carried across a cut. ~10-15% of chunk size is the usual rule of thumb; 150 chars is about two sentences
    ap.add_argument("--overlap", type=int, default=150, help="max chars repeated from the previous piece")
    # Floor for the last piece: anything shorter has too little context to embed well, so it is merged back
    ap.add_argument("--min-chunk", type=int, default=300, help="a final piece shorter than this is merged back")
    # Print the stats and exit without writing files; use this to try settings before freezing a version
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / f"chunks_{args.version}.jsonl"
    manifest_path = ROOT / f"chunks_manifest_{args.version}.json"
    # A frozen version is never overwritten: downstream indexes record its hash
    if not args.dry_run and out_path.exists():
        raise SystemExit(f"{out_path.name} already exists (frozen). Use --version for a new variant.")

    entries = [json.loads(ln) for ln in ENTRIES.read_text().splitlines() if ln]
    chunks = []
    for e in entries:
        text = e["text"].strip()
        # Most entries (~94%) are short enough to stay whole: one entry = one chunk = one idea
        pieces = [text] if len(text) <= args.split_above else split_text(text, args.chunk_chars, args.min_chunk)
        # Date + title go into the embedded text so date-flavoured queries ("March 2022") can match.
        # They come from Notion/the filename, not the handwriting, so embedding the body alone would leave them out of the vector.
        # Kept out of `text` so displayed/generated context isn't cluttered with a repeated header.
        # Embeddings match digits only loosely: if date queries stay weak in the eval, filter on the `date` metadata instead
        header = f"{e['date']} · {e['title'].replace('-', ' ')}\n"
        for i, piece in enumerate(pieces):
            # Every piece after the first starts with the end of the previous one, so a thought cut at the boundary survives
            body = piece if i == 0 else f"{overlap_tail(pieces[i - 1], args.overlap)} {piece}"
            chunks.append({
                "chunk_id": f"{e['entry_id']}#{i}",  # stable id: eval and citations refer to this
                "entry_id": e["entry_id"],  # links back to the entry, which the eval set labels
                "date": e["date"],
                "title": e["title"],
                "sources": e["sources"],  # trace any chunk back to its photos
                "chunk_index": i,
                "n_chunks": len(pieces),
                "text": body,  # what the reader sees and the generator receives
                "embed_text": header + body,  # what gets embedded
            })

    # Size summary only; no text is printed
    sizes = sorted(len(c["text"]) for c in chunks)
    n_split = sum(1 for c in chunks if c["n_chunks"] > 1 and c["chunk_index"] == 0)
    stats = {
        "n_entries": len(entries), "n_chunks": len(chunks), "n_entries_split": n_split,
        "chars_min_med_max": [sizes[0], sizes[len(sizes) // 2], sizes[-1]],
    }
    print(stats)
    if args.dry_run:
        return

    out_path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks))
    out_path.chmod(0o444)  # frozen: a new variant gets a new version, never an overwrite
    # The manifest ties this file to its exact inputs and settings, so any index built on it is traceable
    manifest = {
        "file": out_path.name, "sha256": sha256(out_path),
        "entries_file": ENTRIES.name, "entries_sha256": sha256(ENTRIES),
        "source_corpus_sha256": entries[0]["source_corpus_sha256"],  # the frozen OCR corpus upstream
        "params": {k: getattr(args, k.replace("-", "_")) for k in ("split-above", "chunk-chars", "overlap", "min-chunk")},
        **stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o444)
    print(f"wrote {out_path.name} and {manifest_path.name}")


if __name__ == "__main__":
    main()
