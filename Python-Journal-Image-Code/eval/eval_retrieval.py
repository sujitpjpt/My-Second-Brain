"""Baseline retrieval eval: recall@k, hit@k and MRR for a Chroma collection against the frozen eval set.

    python eval/eval_retrieval.py                                      # default collection nomic-embed-text_chunks_v1
    python eval/eval_retrieval.py --collection bge-m3_chunks_v1        # compare a variant (prefixes come from its manifest)

Retrieval only: no generator and no LLM judge, so the run is deterministic and needs no model but the embedder.
Everything is local (Ollama on localhost + chroma_db/). Terminal output has ids, ranks and numbers, never journal text.
Per-question results go to Journal-Entires/eval/results/ (gitignored via Journal-Entires/).
"""
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import ollama

# This file lives at <repo>/Python-Journal-Image-Code/eval/, so the repo root is three levels up
REPO = Path(__file__).parent.parent.parent
CORPUS = REPO / "Journal-Entires" / "corpus"
EVAL_DIR = REPO / "Journal-Entires" / "eval"
CHROMA_DIR = REPO / "chroma_db"

KS = (1, 3, 5, 10)  # cutoffs reported; the largest is also how many entries we keep per query
N_CHUNKS = 50       # chunks fetched per query; enough to yield >= max(KS) distinct entries after collapsing


def sha256(path):
    # Whole-file hash; fine at this corpus size (a few MB)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    # One JSON object per line; skip blank lines
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def verify_inputs(index_manifest):
    """Refuse to score anything whose inputs drifted from their frozen manifests (otherwise numbers aren't comparable)."""
    # Fixed v1 filenames: a new eval/corpus version means a new frozen file here, never an in-place edit
    eval_path = EVAL_DIR / "retrieval_v1.jsonl"
    eval_man = json.loads((EVAL_DIR / "retrieval_v1_manifest.json").read_text())
    chunks_man = json.loads((CORPUS / "chunks_manifest_v1.json").read_text())
    # Each check is True when the artifact still matches the hash recorded when it was frozen
    checks = {
        "eval file vs its manifest": sha256(eval_path) == eval_man["sha256"],
        "entries file vs the corpus the labels were written against": sha256(CORPUS / "entries_v1.jsonl") == eval_man["entries_sha256"],
        "index was built from the frozen chunks": index_manifest["chunks_sha256"] == chunks_man["sha256"],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"hash mismatch, refusing to score: {bad}")
    return eval_path, eval_man


def retrieve(col, questions, model, query_prefix):
    """Embed all questions in one local call, then return, per question, a ranked list of (entry_id, distance)."""
    # Same prefix the index manifest recorded: a mismatched prefix silently degrades nomic-embed-text
    # truncate=False makes Ollama error on over-long input instead of silently cutting the query
    emb = ollama.embed(model=model, input=[query_prefix + q for q in questions], truncate=False).embeddings
    # Pass vectors directly so Chroma does not re-embed with its own default model.
    # Result lists are parallel: res["metadatas"][i] / res["distances"][i] belong to question i.
    res = col.query(query_embeddings=emb, n_results=N_CHUNKS, include=["metadatas", "distances"])
    ranked = []
    for metas, dists in zip(res["metadatas"], res["distances"]):
        # Labels are entry ids but the index holds chunks: collapse to entries, keeping each entry's best chunk.
        # Chroma returns hits best-first, so the first time we see an entry is its best rank.
        seen = {}  # dict keeps insertion order, so the result is already best-first by entry
        for m, d in zip(metas, dists):
            seen.setdefault(m["entry_id"], d)
        ranked.append(list(seen.items()))
    return ranked


def score(relevant, ranked_ids):
    """Metrics for one answerable question. recall@k = fraction of relevant entries found in the top k entries."""
    rel = set(relevant)  # gold entry ids from the eval set; a question can have several
    ranks = [i + 1 for i, e in enumerate(ranked_ids) if e in rel]  # 1-based ranks of relevant hits
    # MRR = 1 / rank of the first relevant entry (0 if none retrieved); first_rank is kept for miss reporting
    out = {"mrr": 1 / ranks[0] if ranks else 0.0, "first_rank": ranks[0] if ranks else None}
    for k in KS:
        found = len(rel & set(ranked_ids[:k]))
        out[f"recall@{k}"] = found / len(rel)
        out[f"hit@{k}"] = float(found > 0)  # at least one relevant entry in top k
    return out


def mean(rows, key):
    # Average one metric across per-question rows
    return statistics.fmean(r[key] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="nomic-embed-text_chunks_v1")
    args = ap.parse_args()

    # The manifest written at index-build time is the source of truth for model, prefix and chunk hash
    index_manifest = json.loads((CORPUS / f"index_manifest_{args.collection}.json").read_text())
    eval_path, eval_man = verify_inputs(index_manifest)
    recs = load_jsonl(eval_path)

    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(args.collection)
    assert col.count() == index_manifest["n_vectors"], "collection changed since its manifest was written"

    # ranked[i] aligns with recs[i]: [(entry_id, best_chunk_distance), ...], best first
    ranked = retrieve(col, [r["question"] for r in recs], index_manifest["model"], index_manifest["query_prefix"])

    # per_q holds one row per question; the two top-1 lists feed the abstention analysis below
    per_q, answerable_top1, unanswerable_top1 = [], [], []
    for r, hits in zip(recs, ranked):
        if r["type"] == "unanswerable":
            # Nothing to find: the signal is how confidently the retriever returns *something* (top-1 distance)
            unanswerable_top1.append(hits[0][1])
            per_q.append({"qid": r["qid"], "type": r["type"], "top1_distance": hits[0][1]})
        else:
            s = score(r["relevant"], [e for e, _ in hits])
            answerable_top1.append(hits[0][1])
            per_q.append({"qid": r["qid"], "type": r["type"], "n_relevant": len(r["relevant"]), "top1_distance": hits[0][1], **s})

    # Group answerable rows by question type (e.g. single vs multi-entry) for the per-type table
    answerable = [p for p in per_q if p["type"] != "unanswerable"]
    by_type = defaultdict(list)
    for p in answerable:
        by_type[p["type"]].append(p)

    # Report per type, plus a macro average (types weighted equally so 22 singles don't drown 8 multis)
    # recall@k for every k, plus hit@5/hit@10 (hit@1 == recall@1 for single-entry questions, so it is omitted)
    cols = [f"recall@{k}" for k in KS] + ["hit@5", "hit@10", "mrr"]
    print(f"\ncollection={args.collection}  questions={len(recs)}  chunks={col.count()}")
    print(f"{'type':<14}{'n':>3}  " + "  ".join(f"{c:>9}" for c in cols))
    for t, rows in by_type.items():
        print(f"{t:<14}{len(rows):>3}  " + "  ".join(f"{mean(rows, c):>9.3f}" for c in cols))
    # Macro = mean of the per-type means, not of all questions
    macro = {c: statistics.fmean(mean(rows, c) for rows in by_type.values()) for c in cols}
    print(f"{'MACRO':<14}{len(answerable):>3}  " + "  ".join(f"{macro[c]:>9.3f}" for c in cols))

    # Unanswerable: can a distance threshold separate "no answer" from "has an answer"?
    # If unanswerable top-1 distances sit in the same range as answerable ones, the retriever can't abstain on its own.
    print("\nunanswerable vs answerable top-1 cosine distance (lower = more confident)")
    print(f"  unanswerable  min={min(unanswerable_top1):.3f}  median={statistics.median(unanswerable_top1):.3f}  n={len(unanswerable_top1)}")
    print(f"  answerable    median={statistics.median(answerable_top1):.3f}  p90={sorted(answerable_top1)[int(.9 * len(answerable_top1))]:.3f}")

    # Misses worth a look: ids and ranks only (look up text yourself with eval_sampler.py --show)
    misses = [p for p in answerable if p["first_rank"] is None or p["first_rank"] > max(KS)]
    print(f"\nnot in top {max(KS)} entries ({len(misses)}): " + ", ".join(f"{p['qid']}({p['type']})" for p in misses))

    # Save everything needed to compare runs later: inputs' hashes, embedder digest, per-question rows
    out_dir = EVAL_DIR / "results"  # under Journal-Entires/, so gitignored
    out_dir.mkdir(exist_ok=True)
    # UTC timestamp in the filename: every run is kept, nothing is overwritten
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{args.collection}__{stamp}.json"
    # Hashes + embedder digest pin exactly what was scored, so two result files are only comparable if these match
    out.write_text(json.dumps({
        "run_at": stamp, "collection": args.collection,
        "embed_model": index_manifest["model"], "embed_model_digest": index_manifest["model_digest"],
        "query_prefix": index_manifest["query_prefix"], "eval_sha256": eval_man["sha256"],
        "chunks_sha256": index_manifest["chunks_sha256"], "ks": KS,
        "macro": macro, "by_type": {t: {c: mean(rows, c) for c in cols} for t, rows in by_type.items()},
        "per_question": per_q,
    }, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
