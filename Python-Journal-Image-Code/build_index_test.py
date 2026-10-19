"""Dump what every `batch` in build_index.py contains, to one file. No Ollama, no ChromaDB, nothing is embedded or indexed.

    python build_index_test.py                    # all batches -> Journal-Entires/corpus/batch_debug_v1.json

The output has journal text in it, so it lives in the gitignored corpus folder. Treat it like the corpus.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
CORPUS = REPO / "Journal-Entires" / "corpus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--doc-prefix", default="search_document: ")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    chunks = [json.loads(ln) for ln in (CORPUS / f"chunks_{args.version}.jsonl").read_text().splitlines() if ln]

    out = []
    # Same loop as build_index.py: i = 0, B, 2B, ... ; slice i:i+B takes up to B chunks (last one may be shorter)
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        out.append({
            "slice": f"chunks[{i}:{i + args.batch_size}]",
            "n_in_batch": len(batch),
            # Exactly the strings build_index.py would send to ollama.embed (doc_prefix + embed_text)
            "embed_inputs": [args.doc_prefix + c["embed_text"] for c in batch],
            # The full chunk dicts, as loaded from the jsonl
            "batch": batch,
        })
        print(f"batched {min(i + args.batch_size, len(chunks))}/{len(chunks)}")

    out_path = CORPUS / f"batch_debug_{args.version}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} batches to {out_path.name}")


if __name__ == "__main__":
    main()
