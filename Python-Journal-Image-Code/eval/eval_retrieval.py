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
    """Embed all questions in one local call, then return, per question, a ranked list of (entry_id, distance).

    Shape (outer list parallel to `questions`; distances ascend, lower = closer):
        [
            [("entry_042", 0.18), ("entry_007", 0.31), ...],  # questions[0]
            [("entry_090", 0.22), ("entry_042", 0.29), ...],  # questions[1]
        ]
    """
    # Same prefix the index manifest recorded: a mismatched prefix silently degrades nomic-embed-text
    # truncate=False makes Ollama error on over-long input instead of silently cutting the query
    emb = ollama.embed(model=model, input=[query_prefix + q for q in questions], truncate=False).embeddings
    # Pass vectors directly so Chroma does not re-embed with its own default model.
    # Result lists are parallel: res["metadatas"][i] / res["distances"][i] belong to question i.
    res = col.query(query_embeddings=emb, n_results=N_CHUNKS, include=["metadatas", "distances"])
    ranked = []
    for metas, dists in zip(res["metadatas"], res["distances"]):
        # Labels are entry ids but the index holds chunks, and one entry can own several chunks.
        # Collapse to entries by max-pooling: an entry scores as its closest chunk (lower distance = better).
        # Chroma returns hits best-first, so the first chunk seen for an entry is its closest one; later ones are
        # dropped, e.g. E1(.2) E2(.3) E1(.4) E3(.5) -> [(E1,.2), (E2,.3), (E3,.5)].
        # Caveat: this yields <= N_CHUNKS entries, so N_CHUNKS must comfortably exceed max(KS) or
        # duplicate chunks crowd out entries and understate recall@k / MRR.
        seen = {}  # dict keeps insertion order, so the result is already best-first by entry
        for m, d in zip(metas, dists):
            seen.setdefault(m["entry_id"], d)
        ranked.append(list(seen.items()))
    return ranked


def score(relevant, ranked_ids):
    """Metrics for one answerable question. recall@k = fraction of relevant entries found in the top k entries.

    Worked example (used in the comments below):
        relevant   = ["entry_007", "entry_090"]                            # gold entries
        ranked_ids = ["entry_042", "entry_007", "entry_113", "entry_090"]  # retrieved, best first
        k=1: found 0 -> recall 0.0, hit 0.0 | k=3: found 1 -> recall 0.5, hit 1.0 | k=5: found 2 -> recall 1.0, hit 1.0
        result: {"mrr": 0.5, "first_rank": 2, "recall@1": 0.0, "hit@1": 0.0, "recall@3": 0.5, "hit@3": 1.0, ...}
    Entries the retriever never returned are absent from ranked_ids, so they count as misses.
    """
    rel = set(relevant)  # gold entry ids from the eval set; a question can have several
    # {"entry_007", "entry_090"}; a set makes the membership test and the intersection below simple
    ranks = [i + 1 for i, e in enumerate(ranked_ids) if e in rel]  # 1-based ranks of relevant hits
    # [2, 4]: enumerate is 0-based, hence + 1; ascending, so ranks[0] is the earliest relevant hit
    # MRR = 1 / rank of the first relevant entry (0 if none retrieved); first_rank is kept for miss reporting
    # Only the first hit counts for MRR (entry_090 at rank 4 is ignored): 1/2 = 0.5. Miss -> mrr 0.0, first_rank None
    out = {"mrr": 1 / ranks[0] if ranks else 0.0, "first_rank": ranks[0] if ranks else None}
    for k in KS:
        # ranked_ids[:k] = top-k retrieved entries; & keeps only the ones that are gold
        # k=1: {042} & {007,090} = {}         -> found 0
        # k=3: {042,007,113} & {007,090} = {007} -> found 1
        # k=5: slice past the end is safe (returns all 4) -> {007,090} -> found 2
        found = len(rel & set(ranked_ids[:k]))
        # recall@k = found / total gold: "how much of what I needed did I get?"
        # k=1: 0/2 = 0.0 | k=3: 1/2 = 0.5 | k=5: 2/2 = 1.0
        # len(rel) > 0 since unanswerable questions never reach here
        out[f"recall@{k}"] = found / len(rel)
        # hit@k = 1.0 if found at least one gold entry: "did I get anything useful?"
        # k=1: 0.0 | k=3: 1.0 (007 is enough) | k=5: 1.0
        out[f"hit@{k}"] = float(found > 0)
        # Single gold entry: found is 0 or 1 and len(rel) == 1, so recall@k == hit@k.
        # They differ only for multi-entry questions, e.g. k=3 above: recall 0.5 vs hit 1.0
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

    # per_q holds one row per question; The other two hold only top-1 distances, split by question type.
    per_q, answerable_top1, unanswerable_top1 = [], [], []
    for r, hits in zip(recs, ranked):
        # r    = one eval record, e.g. {"qid": "q03", "type": "single", "question": "...", "relevant": ["entry_007"]}
        #        (unanswerable records have no "relevant" list)
        # hits = that question's ranked entries, e.g. [("entry_042", 0.18), ("entry_007", 0.31), ("entry_113", 0.35)]
        #        so hits[0] is the best entry and hits[0][1] its distance (top-1 distance = 0.18)
        if r["type"] == "unanswerable":
            # Nothing to find: the signal is how confidently the retriever returns *something* (top-1 distance)
            # Row example: {"qid": "q07", "type": "unanswerable", "top1_distance": 0.41}
            unanswerable_top1.append(hits[0][1])
            per_q.append({"qid": r["qid"], "type": r["type"], "top1_distance": hits[0][1]})
        else:
            # score() needs only the ranked entry ids, so drop the distances (`_`):
            #   [e for e, _ in hits] -> ["entry_042", "entry_007", "entry_113"]
            # and compares them to the gold ids in r["relevant"] (["entry_007"], found at rank 2):
            #   s -> {"mrr": 0.5, "first_rank": 2, "recall@1": 0.0, "recall@3": 1.0, "hit@3": 1.0, ...}
            s = score(r["relevant"], [e for e, _ in hits])
            answerable_top1.append(hits[0][1])
            # Row example: {"qid": "q03", "type": "single", "n_relevant": 1, "top1_distance": 0.18, "mrr": 0.5, ...}
            # (**s merges the metric dict into the row)
            per_q.append({"qid": r["qid"], "type": r["type"], "n_relevant": len(r["relevant"]), "top1_distance": hits[0][1], **s})

    # Group answerable rows by question type (e.g. single vs multi-entry) for the per-type table
    # Unanswerable questions have no gold entry, so recall/MRR are undefined for them; they're only used for distances below
    answerable = [p for p in per_q if p["type"] != "unanswerable"]
    # by_type example: {"single": [row_q01, row_q02, ...], "multi": [row_q23, ...]}
    by_type = defaultdict(list)
    for p in answerable:
        by_type[p["type"]].append(p)

    # Report per type, plus a macro average (types weighted equally so 22 singles don't drown 8 multis)
    # recall@k for every k, plus hit@5/hit@10 (hit@1 == recall@1 for single-entry questions, so it is omitted)
    # e.g. with KS = [1, 3, 5, 10]: ["recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "hit@10", "mrr"]
    cols = [f"recall@{k}" for k in KS] + ["hit@5", "hit@10", "mrr"]
    print(f"\ncollection={args.collection}  questions={len(recs)}  chunks={col.count()}")
    # Illustrative output (numbers made up):
    #   type            n   recall@1   recall@3  ...      mrr
    #   single         22      0.636      0.864  ...    0.745
    #   multi           8      0.250      0.500  ...    0.410
    #   MACRO          30      0.443      0.682  ...    0.578
    print(f"{'type':<14}{'n':>3}  " + "  ".join(f"{c:>9}" for c in cols))
    for t, rows in by_type.items():
        print(f"{t:<14}{len(rows):>3}  " + "  ".join(f"{mean(rows, c):>9.3f}" for c in cols))
    # Macro = mean of the per-type means, not of all questions
    # e.g. recall@3: single 0.864, multi 0.500 -> macro 0.682 (a pooled mean over 30 questions would give 0.773, hiding the multi weakness)
    macro = {c: statistics.fmean(mean(rows, c) for rows in by_type.values()) for c in cols}
    print(f"{'MACRO':<14}{len(answerable):>3}  " + "  ".join(f"{macro[c]:>9.3f}" for c in cols))

    # Unanswerable: can a distance threshold separate "no answer" from "has an answer"?
    # If unanswerable top-1 distances sit in the same range as answerable ones, the retriever can't abstain on its own.
    # Example: unanswerable min=0.41 vs answerable p90=0.38 -> a cutoff near 0.40 would separate them (good).
    #          unanswerable min=0.22 vs answerable p90=0.38 -> any cutoff that keeps real answers also lets junk through (bad).
    # p90 = value at index int(0.9 * n) of the sorted list, i.e. 90% of answerable top-1 distances are at or below it
    print("\nunanswerable vs answerable top-1 cosine distance (lower = more confident)")
    print(f"  unanswerable  min={min(unanswerable_top1):.3f}  median={statistics.median(unanswerable_top1):.3f}  n={len(unanswerable_top1)}")
    print(f"  answerable    median={statistics.median(answerable_top1):.3f}  p90={sorted(answerable_top1)[int(.9 * len(answerable_top1))]:.3f}")

    # Misses worth a look: ids and ranks only (look up text yourself with eval_sampler.py --show)
    # first_rank is the 1-based rank of the first gold entry: None = never retrieved, 14 = found but past the largest k (10)
    # Example printed line: "not in top 10 entries (2): q07(single), q26(multi)"
    misses = [p for p in answerable if p["first_rank"] is None or p["first_rank"] > max(KS)]
    print(f"\nnot in top {max(KS)} entries ({len(misses)}): " + ", ".join(f"{p['qid']}({p['type']})" for p in misses))

    # Save everything needed to compare runs later: inputs' hashes, embedder digest, per-question rows
    out_dir = EVAL_DIR / "results"  # under Journal-Entires/, so gitignored
    out_dir.mkdir(exist_ok=True)
    # UTC timestamp in the filename: every run is kept, nothing is overwritten
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{args.collection}__{stamp}.json"
    # Hashes + embedder digest pin exactly what was scored, so two result files are only comparable if these match
    # Example file: results/nomic-embed-text_chunks_v1__20261028T143005Z.json
    #   {"run_at": "20261028T143005Z", "collection": "nomic-embed-text_chunks_v1", "embed_model": "nomic-embed-text",
    #    "embed_model_digest": "0a109f...", "eval_sha256": "9c1e...", "chunks_sha256": "b47d...", "ks": [1, 3, 5, 10],
    #    "macro": {"recall@1": 0.443, ...}, "by_type": {"single": {...}, "multi": {...}}, "per_question": [{...}, ...]}
    # To compare two runs: check eval_sha256 / chunks_sha256 / embed_model_digest match, then diff "macro" and per_question rows
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
