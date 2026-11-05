"""Seed a new eval version from an older one, dropping the qids you name.

    python eval/carry_over_eval.py --drop q007 q041 q042          # v1 -> v2, minus those qids
    python eval/carry_over_eval.py --from v1 --to v2 --drop ...   # explicit versions

Local file copy only; prints ids and counts, never journal text. Kept qids stay unchanged so v1 and v2
results can be compared question by question; new questions added later get max+1 (see eval_sampler.save).
Run before eval_sampler.py --version v2, then freeze_eval.py --version v2.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent.parent / "Journal-Entires" / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="v1")
    ap.add_argument("--to", dest="dst", default="v2")
    ap.add_argument("--drop", nargs="*", default=[], help="qids to leave out (e.g. vague questions)")
    args = ap.parse_args()

    src = EVAL_DIR / f"retrieval_{args.src}.jsonl"
    dst = EVAL_DIR / f"retrieval_{args.dst}.jsonl"
    # Never overwrite or extend a target: it may already hold hand-written questions or be frozen
    if dst.exists():
        raise SystemExit(f"{dst.name} already exists; refusing to overwrite")

    recs = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]
    unknown = set(args.drop) - {r["qid"] for r in recs}
    if unknown:
        raise SystemExit(f"--drop ids not in {src.name}: {sorted(unknown)}")

    kept = []
    for r in recs:
        if r["qid"] in args.drop:
            continue
        # Provenance: lets later analysis separate carried questions from ones written fresh for v2
        kept.append({**r, "carried_from": args.src})
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))

    print(f"{src.name}: {len(recs)} -> {dst.name}: {len(kept)} (dropped {len(recs) - len(kept)}: {sorted(args.drop)})")
    print("by type:", dict(Counter(r["type"] for r in kept)))


if __name__ == "__main__":
    main()
