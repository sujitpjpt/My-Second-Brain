"""Baseline retrieval eval: recall@k, hit@k and MRR for a Chroma collection against the frozen eval set.

    python eval/eval_retrieval.py                                      # default collection nomic-embed-text_chunks_v1
    python eval/eval_retrieval.py --collection bge-m3_chunks_v1        # compare a variant (prefixes come from its manifest)
    python eval/eval_retrieval.py --eval-version v2 --date-filter      # restrict to the date named in the question, if any
    python eval/eval_retrieval.py --ann                                # Chroma's approximate HNSW search instead of exact

Retrieval only: no generator and no LLM judge, and it needs no model but the embedder. Search is EXACT by default
(brute-force cosine over every chunk vector in numpy), so the same inputs give the same ranking in every process.
Chroma's HNSW search is approximate and was seen to return different rankings in separate processes on the same index
(q015, q033, q051, q069, 2026-11-05); it stays available behind --ann to measure how far it drifts from exact.
Everything is local (Ollama on localhost + chroma_db/). Terminal output has ids, ranks and numbers, never journal text.
Per-question results go to Journal-Entires/eval/results/ (gitignored via Journal-Entires/).
"""
import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
import ollama

sys.path.insert(0, str(Path(__file__).parent.parent))  # so `retrieve` (a sibling of eval/) imports from any cwd
from retrieve.date_filter import date_window  # noqa: E402

# This file lives at <repo>/Python-Journal-Image-Code/eval/, so the repo root is three levels up
REPO = Path(__file__).parent.parent.parent
CORPUS = REPO / "Journal-Entires" / "corpus"
EVAL_DIR = REPO / "Journal-Entires" / "eval"
CHROMA_DIR = REPO / "chroma_db"

KS = (1, 3, 5, 10, 20, 30)  # cutoffs reported; the largest is also how many entries we keep per query
N_CHUNKS = 60       # chunks fetched per query; enough to yield >= max(KS) distinct entries after collapsing
MISS_K = 10         # "miss" list cutoff, fixed so it stays comparable when KS grows
# Multi-chunk entries collapse, so demand 2x headroom or hit@30 silently reads low
assert N_CHUNKS >= 2 * max(KS), "N_CHUNKS too small for max(KS)"


def sha256(path):
    # Whole-file hash; fine at this corpus size (a few MB)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    # One JSON object per line; skip blank lines
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def verify_inputs(index_manifest, version):
    """Refuse to score anything whose inputs drifted from their frozen manifests (otherwise numbers aren't comparable)."""
    # A new eval version means a new frozen file, never an in-place edit
    eval_path = EVAL_DIR / f"retrieval_{version}.jsonl"
    eval_man = json.loads((EVAL_DIR / f"retrieval_{version}_manifest.json").read_text())
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


def load_exact_index(col):
    """Pull every chunk vector out of Chroma once, in a fixed order, as a unit-length float64 matrix.

    Returns (V, metas): V is (n_chunks, dim), metas[i] is the metadata dict of row i (entry_id, date, ...).
    Rows are sorted by chunk id, so row order never depends on Chroma's internal storage order. That fixed order is
    what makes the tie-break in exact_rank() deterministic.
    """
    got = col.get(include=["embeddings", "metadatas"])
    order = np.argsort(got["ids"], kind="stable")  # chunk ids are strings; sorted order is the canonical row order
    V = np.asarray(got["embeddings"], dtype=np.float64)[order]  # float64: removes float32 accumulation noise
    metas = [got["metadatas"][i] for i in order]
    return unit(V), metas


def unit(x):
    """L2-normalise rows. For unit vectors, dot product == cosine similarity, so cosine distance = 1 - dot."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def exact_rank(V, metas, q_vec, n_results, dates=None):
    """Brute-force nearest chunks for one query: score ALL chunks, sort, keep the top n_results.

    Returns (metas_list, dists_list) best-first, the same shape col.query() gives per question, so the code below
    (entry collapsing, scoring) is unchanged whichever search produced it.

    dates: optional list of ISO dates. Chunks on other dates are excluded, which is what Chroma's
    where={"date": {"$in": dates}} did, but exactly.

    Worked example (4 chunks, rows already sorted by chunk id), distances to the query:
        row0 0.200000 | row1 0.150000 | row2 0.150000 | row3 0.400000
        round to 6 decimals, stable sort -> rows 1, 2, 0, 3. Rows 1 and 2 tie, and the lower row (= lower chunk id)
        wins, every run. Without the rounding, 0.1500000001 vs 0.1499999999 could swap them between runs.
    """
    q = unit(np.asarray(q_vec, dtype=np.float64))
    dist = np.round(1.0 - V @ q, 6)  # cosine distance to every chunk: 0 = identical direction, 1 = unrelated
    if dates is not None:
        allowed = np.array([m["date"] in set(dates) for m in metas])
        dist = np.where(allowed, dist, np.inf)  # excluded chunks sort last...
        n_results = min(n_results, int(allowed.sum()))  # ...and are cut off by the count, never returned
    # kind="stable" keeps equal-distance rows in their original (chunk-id) order, which is the tie-break.
    idx = np.argsort(dist, kind="stable")[:n_results]
    return [metas[i] for i in idx], [float(dist[i]) for i in idx]


def retrieve(col, questions, model, query_prefix, date_filter=False, ann=False):
    """Embed all questions in one local call, then return (ranked, windows).

    Search is exact brute force by default (see exact_rank); ann=True uses Chroma's approximate HNSW index instead.

    ranked: per question, a ranked list of (entry_id, distance).
    windows: parallel list; None = no filter applied, else (kind, n_dates_in_window) for questions that named a date.

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
    # n_results means how many nearest chunks to return per query, which is your top-k. The recall@k and MRR numbers depend on it. Recall@k can't be computed for any k larger than N_CHUNKS, so we keep it at least as large as the biggest k we evaluate.
    # Result lists are parallel: res["metadatas"][i] / res["distances"][i] belong to question i.
    if ann:
        res = col.query(query_embeddings=emb, n_results=N_CHUNKS, include=["metadatas", "distances"])
        metas_all, dists_all = res["metadatas"], res["distances"]
    else:
        # Exact path: load the 248 vectors once, then score every chunk for every question. No graph, no shortcut.
        V, all_metas = load_exact_index(col)
        pairs = [exact_rank(V, all_metas, e, N_CHUNKS) for e in emb]
        metas_all, dists_all = [p[0] for p in pairs], [p[1] for p in pairs]
    windows = [None] * len(questions)
    if date_filter:
        # Chroma's $gte/$lte only handle numbers and `date` is an ISO string, so expand each window into the concrete
        # dates the index holds and match with $in. Questions that name no explicit date keep the unfiltered ranking above.
        n_chunks_on = {}  # date -> number of chunks, so n_results never exceeds what the filter can return
        for m in col.get(include=["metadatas"])["metadatas"]:
            n_chunks_on[m["date"]] = n_chunks_on.get(m["date"], 0) + 1
        for i, q in enumerate(questions):
            hit = date_window(q, n_chunks_on)
            if hit is None:
                continue
            kind, dates = hit
            windows[i] = (kind, len(dates))
            if not dates:
                continue  # named a date nothing is indexed on: keep the unfiltered ranking rather than return nothing
            if ann:
                r = col.query(query_embeddings=[emb[i]], n_results=min(N_CHUNKS, sum(n_chunks_on[d] for d in dates)),
                              where={"date": {"$in": dates}}, include=["metadatas", "distances"])
                metas_all[i], dists_all[i] = r["metadatas"][0], r["distances"][0]
            else:
                metas_all[i], dists_all[i] = exact_rank(V, all_metas, emb[i], N_CHUNKS, dates)
    ranked = []
    for metas, dists in zip(metas_all, dists_all):
        # Labels are entry ids but the index holds chunks, and one entry can own several chunks.
        # Collapse to entries by max-pooling: an entry scores as its closest chunk (lower distance = better).
        # Chroma returns hits best-first, so the first chunk seen for an entry is its closest one; later ones are dropped, e.g. E1(.2) E2(.3) E1(.4) E3(.5) -> [(E1,.2), (E2,.3), (E3,.5)].
        # Caveat: this yields <= N_CHUNKS entries, so N_CHUNKS must comfortably exceed max(KS) or
        # duplicate chunks crowd out entries and understate recall@k / MRR.
        seen = {}  # dict keeps insertion order, so the result is already best-first by entry
        for m, d in zip(metas, dists):
            seen.setdefault(m["entry_id"], d)
        ranked.append(list(seen.items()))
    return ranked, windows


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
    # Final out for the worked example (KS = 1, 3, 5, 10; k=10 behaves like k=5 since only 4 were retrieved):
    # inputs: relevant   = ["entry_007", "entry_090"]
    #         ranked_ids = ["entry_042", "entry_007", "entry_113", "entry_090"]
    # output: {"mrr": 0.5, "first_rank": 2,
    #         "recall@1": 0.0, "hit@1": 0.0,
    #         "recall@3": 0.5, "hit@3": 1.0,
    #         "recall@5": 1.0, "hit@5": 1.0,
    #         "recall@10": 1.0, "hit@10": 1.0}
    # Total miss (no gold retrieved): mrr 0.0, first_rank None, every recall@k / hit@k 0.0
    return out


def mean(rows, key):
    # Average one metric across per-question rows
    return statistics.fmean(r[key] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="nomic-embed-text_chunks_v1")
    ap.add_argument("--eval-version", default="v1", help="frozen eval set to score against (retrieval_<version>.jsonl)")
    ap.add_argument("--ann", action="store_true", help="use Chroma's approximate HNSW search instead of exact brute force (not reproducible across processes)")
    ap.add_argument("--date-filter", action="store_true", help="restrict candidates to the date/month/year named in the question")
    args = ap.parse_args()

    # The manifest written at index-build time is the source of truth for model, prefix and chunk hash
    index_manifest = json.loads((CORPUS / f"index_manifest_{args.collection}.json").read_text())
    eval_path, eval_man = verify_inputs(index_manifest, args.eval_version)
    recs = load_jsonl(eval_path)

    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(args.collection)
    assert col.count() == index_manifest["n_vectors"], "collection changed since its manifest was written"

    # ranked[i] aligns with recs[i]: [(entry_id, best_chunk_distance), ...], best first
    ranked, windows = retrieve(col, [r["question"] for r in recs], index_manifest["model"], index_manifest["query_prefix"], args.date_filter, args.ann)

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
            # Record the best entry's distance (hits[0] = closest entry, [1] = its distance, e.g. 0.18) regardless of
            # whether it's actually a gold entry. Compared against unanswerable_top1 below to see if a distance
            # threshold could separate "has an answer" from "no answer"
            answerable_top1.append(hits[0][1])
            # Row example: {"qid": "q03", "type": "single", "n_relevant": 1, "top1_distance": 0.18, "mrr": 0.5, ...}
            # (**s merges the metric dict into the row)
            per_q.append({"qid": r["qid"], "type": r["type"], "n_relevant": len(r["relevant"]), "top1_distance": hits[0][1], **s})

    # Group answerable rows by question type (e.g. single vs multi-entry) for the per-type table
    # Unanswerable questions have no gold entry, so recall/MRR are undefined for them; they're only used for distances below
    answerable = [p for p in per_q if p["type"] != "unanswerable"]
    # defaultdict(list): the first by_type["single"] auto-creates an empty list, so no "key exists?" check is needed
    by_type = defaultdict(list)
    for p in answerable:
        # p = one per_q row; append it to the list for its type
        by_type[p["type"]].append(p)
    # Actual value after the loop (rows are the full per_q dicts; metrics abbreviated, k=10 keys omitted):
    # {
    #   "single": [
    #     {"qid": "q01", "type": "single", "n_relevant": 1, "top1_distance": 0.18, "mrr": 1.0, "first_rank": 1,
    #      "recall@1": 1.0, "hit@1": 1.0, "recall@3": 1.0, "hit@3": 1.0, "recall@5": 1.0, "hit@5": 1.0, ...},
    #     {"qid": "q03", "type": "single", "n_relevant": 1, "top1_distance": 0.18, "mrr": 0.5, "first_rank": 2,
    #      "recall@1": 0.0, "hit@1": 0.0, "recall@3": 1.0, "hit@3": 1.0, "recall@5": 1.0, "hit@5": 1.0, ...},
    #     ...
    #   ],
    #   "multi": [
    #     {"qid": "q23", "type": "multi", "n_relevant": 2, "top1_distance": 0.22, "mrr": 0.5, "first_rank": 2,
    #      "recall@1": 0.0, "hit@1": 0.0, "recall@3": 0.5, "hit@3": 1.0, "recall@5": 1.0, "hit@5": 1.0, ...},
    #     ...
    #   ],
    # }   # no "unanswerable" key: those rows were filtered out above
    # mean(rows, "recall@3") later averages that column over one list, e.g. over by_type["single"]

    # Report per type, plus a macro average (types weighted equally so 22 singles don't drown 8 multis)
    # recall@k for every k, plus hit@k for k >= 5 (hit@1 == recall@1 for single-entry questions, so it is omitted)
    # e.g. with KS = [1, 3, 5, 10]: ["recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "hit@10", "hit@20", "hit@30", "mrr"]
    cols = [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS if k >= 5] + ["mrr"]
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
    # p90 = the value 90% of the way along the sorted answerable top-1 distances (index int(0.9 * n)).
    #   e.g. sorted: 0.10 0.12 0.15 0.18 0.20 0.22 0.25 0.28 0.31 0.40 -> n=10, index 9 -> p90 = 0.40
    # Why it matters: a "no answer" cutoff must sit at about p90 to keep 90% of real questions.
    #   unanswerable distances all above p90 -> the cutoff separates them (good)
    #   some unanswerable distances below p90 -> that cutoff would let them through (bad)
    # Rough with ~30 questions: the top 10% is only ~3 questions, so one odd question can move p90 a lot.
    print("\nunanswerable vs answerable top-1 cosine distance (lower = more confident)")
    print(f"  unanswerable  min={min(unanswerable_top1):.3f}  median={statistics.median(unanswerable_top1):.3f}  n={len(unanswerable_top1)}")
    print(f"  answerable    median={statistics.median(answerable_top1):.3f}  p90={sorted(answerable_top1)[int(.9 * len(answerable_top1))]:.3f}")

    # Date-filter audit (ids and counts only). Date gold is "every entry on that day/month", so a correct parse makes date
    # questions near-trivial: this measures parser coverage, not retrieval. The check that matters is no false fires elsewhere.
    fired = {r["qid"]: w for r, w in zip(recs, windows) if w}
    if args.date_filter:
        fired_by_type = defaultdict(int)
        type_of = {r["qid"]: r["type"] for r in recs}
        for qid in fired:
            fired_by_type[type_of[qid]] += 1
        print("\ndate filter fired on:", dict(fired_by_type) or "nothing")
        unparsed = [r["qid"] for r in recs if r["type"] == "date" and r["qid"] not in fired]
        false_fires = [q for q in fired if type_of[q] != "date"]
        print(f"  date questions NOT parsed ({len(unparsed)}): {', '.join(unparsed) or '-'}")
        print(f"  fired on non-date questions ({len(false_fires)}): {', '.join(false_fires) or '-'}")
        empty = [q for q, (_, n) in fired.items() if n == 0]
        print(f"  parsed but no indexed date in window ({len(empty)}): {', '.join(empty) or '-'}")

    # Misses worth a look: ids and ranks only (look up text yourself with eval_sampler.py --show)
    # first_rank is the 1-based rank of the first gold entry: None = never retrieved, 14 = found but past the largest k (10)
    # Example printed line: "not in top 10 entries (2): q07(single), q26(multi)"
    misses = [p for p in answerable if p["first_rank"] is None or p["first_rank"] > MISS_K]
    print(f"\nnot in top {MISS_K} entries ({len(misses)}): " + ", ".join(f"{p['qid']}({p['type']})" for p in misses))

    # Save everything needed to compare runs later: inputs' hashes, embedder digest, per-question rows
    out_dir = EVAL_DIR / "results"  # under Journal-Entires/, so gitignored
    out_dir.mkdir(exist_ok=True)
    # UTC timestamp in the filename: every run is kept, nothing is overwritten
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = ("__ann" if args.ann else "") + ("__datefilter" if args.date_filter else "")  # filtered and baseline runs must not be mistaken for each other
    out = out_dir / f"{args.collection}__eval-{args.eval_version}{suffix}__{stamp}.json"
    # Hashes + embedder digest pin exactly what was scored, so two result files are only comparable if these match
    # Example file: results/nomic-embed-text_chunks_v1__20261028T143005Z.json
    #   {"run_at": "20261028T143005Z", "collection": "nomic-embed-text_chunks_v1", "embed_model": "nomic-embed-text",
    #    "embed_model_digest": "0a109f...", "eval_sha256": "9c1e...", "chunks_sha256": "b47d...", "ks": [1, 3, 5, 10],
    #    "macro": {"recall@1": 0.443, ...}, "by_type": {"single": {...}, "multi": {...}}, "per_question": [{...}, ...]}
    # To compare two runs: check eval_sha256 / chunks_sha256 / embed_model_digest match, then diff "macro" and per_question rows
    out.write_text(json.dumps({
        "run_at": stamp, "collection": args.collection, "eval_version": args.eval_version, "date_filter": args.date_filter,
        "search": "ann_hnsw" if args.ann else "exact_numpy",
        "embed_model": index_manifest["model"], "embed_model_digest": index_manifest["model_digest"],
        "query_prefix": index_manifest["query_prefix"], "eval_sha256": eval_man["sha256"],
        "chunks_sha256": index_manifest["chunks_sha256"], "ks": KS,
        "macro": macro, "by_type": {t: {c: mean(rows, c) for c in cols} for t, rows in by_type.items()},
        "per_question": per_q, "date_filter_fired": {q: {"kind": k, "n_dates": n} for q, (k, n) in fired.items()},
    }, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
