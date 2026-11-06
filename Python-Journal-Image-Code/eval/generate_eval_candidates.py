"""Generate candidate retrieval-eval questions (single / date / multi / unanswerable) from all entries.

Two steps, both local (Ollama on localhost only; nothing leaves the machine):

  python eval/generate_eval_candidates.py generate --version v2 --model llama3.1:8b
      Reads entries_v1.jsonl, writes eval/candidates_v2.jsonl. Prints counts and ids only (no journal text).

  python eval/generate_eval_candidates.py review --version v2      # RUN IN YOUR OWN TERMINAL: prints journal text
      You accept / edit / drop each candidate. Accepted ones are appended to retrieval_v2.jsonl
      (same schema as eval_sampler.py, source = llm-candidate[-edited]). Nothing is auto-accepted.

How each type is made:
  single       LLM writes 1 question from 1 sampled entry (spread across years); rejected if it copies a 4-word phrase.
  date         NO LLM: templated from entry dates ("on 3 November 2024" / "in November 2024"); gold = every entry in that day/month.
  multi        LLM tags every entry (cached); tags shared by 3-6 entries spread over >=60 days form a group; LLM writes
               one question needing the group. The review screen lists OTHER entries that also mention the tag
               (possible missing gold labels).
  unanswerable LLM writes a near-miss question about an entry's topic whose answer is NOT in it; rejected if any of
               its distinctive keywords appears anywhere in the corpus. You still confirm absence in review.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import eval_sampler as es  # reuse entries path, shares_phrase(), show(), save()

OLLAMA = "http://localhost:11434"  # local only; never change to a remote host
EVAL_DIR = es.ROOT / "Journal-Entires" / "eval"  # gitignored via Journal-Entires/
MIN_SPAN_DAYS = 60  # multi groups must span this long, or they are one episode, not a pattern


def _post(path, body, timeout=300):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_digest(model):
    """Digest of the pulled model, recorded so the candidate set is reproducible (tags can move)."""
    try:
        tags = json.loads(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10).read())["models"]
    except OSError as err:
        sys.exit(f"Ollama not reachable at {OLLAMA}: {err}")
    for m in tags:
        if m["name"] == model or m["name"] == f"{model}:latest":
            return m["digest"]
    sys.exit(f"model {model!r} not pulled; run: ollama pull {model}")


def chat(model, prompt, schema, temperature, seed):
    """One structured-output call. Returns the parsed dict, or None if the reply is unusable."""
    out = _post("/api/chat", {
        "model": model, "stream": False, "format": schema,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192},
    })["message"]["content"]
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def progress(label, kept, target, tried):
    """Overwrite one status line (counts only, no journal text) so a slow LLM call doesn't look like a hang."""
    print(f"  {label}: {kept}/{target} kept, {tried} LLM calls made", end="\r", flush=True)


def str_list_schema(key):
    return {"type": "object", "required": [key], "properties": {key: {"type": "array", "items": {"type": "string"}}}}


def str_schema(key):
    return {"type": "object", "required": [key], "properties": {key: {"type": "string"}}}


# ---------- single ----------
SINGLE_PROMPT = """You build a retrieval test for a personal journal search tool.
Write ONE question the author might later ask that this entry answers.
Rules: first person; paraphrase, never reuse 4+ word phrases from the entry; no dates or names unless
essential; must make sense without seeing the entry. Reply as JSON: {"question": "..."}.

ENTRY:
"""


def gen_single(entries, n, model, rng):
    by_year = defaultdict(list)
    for e in entries.values():
        by_year[e["date"][:4]].append(e["entry_id"])
    for ids in by_year.values():
        rng.shuffle(ids)
    order, years = [], sorted(by_year)
    while any(by_year.values()):  # round-robin over years so one year doesn't dominate
        for y in years:
            if by_year[y]:
                order.append(by_year[y].pop())
    out = []
    for tried, eid in enumerate(order, 1):
        if len(out) >= n:
            break
        progress("single", len(out), n, tried)
        r = chat(model, SINGLE_PROMPT + entries[eid]["text"], str_schema("question"), 0.7, rng.randint(0, 10**6))
        q = (r or {}).get("question", "").strip()
        if q and not es.shares_phrase(q, entries[eid]["text"]):
            out.append({"question": q, "relevant": [eid], "type": "single", "note": ""})
    return out


# ---------- date (no LLM) ----------
def gen_date(entries, n, rng):
    by_day = {e["date"]: [e["entry_id"]] for e in entries.values()}
    by_month = defaultdict(list)
    for e in entries.values():
        by_month[e["date"][:7]].append(e["entry_id"])
    days = [d for d, ids in by_day.items() if len(ids) == 1]
    months = [m for m, ids in by_month.items() if 2 <= len(ids) <= 15]  # 1 entry = a day question; >15 is too broad to review
    rng.shuffle(days), rng.shuffle(months)
    out = []
    for m in months[: n // 2 + n % 2]:
        label = datetime.strptime(m, "%Y-%m").strftime("%B %Y")
        out.append({"question": f"What was going on with me in {label}?", "relevant": sorted(by_month[m]), "type": "date", "note": ""})
    for d in days[: n // 2]:
        label = datetime.strptime(d, "%Y-%m-%d").strftime("%-d %B %Y")
        out.append({"question": f"What did I write about on {label}?", "relevant": by_day[d], "type": "date", "note": ""})
    return out


# ---------- multi ----------
TAG_PROMPT = """List 2 to 4 short topic tags (1-2 lowercase words each) naming the main specific subjects of this
journal entry: people, places, activities, recurring concerns. Reply as JSON: {"tags": ["..."]}.

ENTRY:
"""
MULTI_PROMPT = """Below are several dated journal entries by one person. Write ONE first-person question that
needs information from at least 3 of them to answer fully (e.g. how something recurred or changed over time).
Paraphrase; no dates; must make sense without seeing the entries. Reply as JSON: {"question": "..."}.

"""


def tag_entries(entries, model, rng, cache_path):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for i, (eid, e) in enumerate(entries.items(), 1):
        if eid not in cache:
            r = chat(model, TAG_PROMPT + e["text"], str_list_schema("tags"), 0.0, 42)
            cache[eid] = sorted({t.strip().lower() for t in (r or {}).get("tags", []) if isinstance(t, str) and t.strip()})
            cache_path.write_text(json.dumps(cache, indent=0))
            print(f"  tagged {i}/{len(entries)}", end="\r", flush=True)
    print()
    return cache


def gen_multi(entries, n, model, rng, cache_path):
    tags = tag_entries(entries, model, rng, cache_path)
    groups = defaultdict(list)
    for eid, ts in tags.items():
        for t in ts:
            groups[t].append(eid)
    ok = []
    for t, ids in groups.items():
        days = sorted(datetime.strptime(entries[i]["date"], "%Y-%m-%d") for i in ids)
        if 3 <= len(ids) <= 6 and (days[-1] - days[0]).days >= MIN_SPAN_DAYS:
            ok.append((t, sorted(ids)))
    rng.shuffle(ok)
    out, used, tried = [], set(), 0
    for t, ids in ok:
        if len(out) >= n:
            break
        if used & set(ids):  # keep groups disjoint so the questions cover different entries
            continue
        tried += 1
        progress("multi", len(out), n, tried)
        ctx = "\n\n".join(f"[{entries[i]['date']}]\n{entries[i]['text'][:1500]}" for i in ids)
        r = chat(model, MULTI_PROMPT + ctx, str_schema("question"), 0.7, rng.randint(0, 10**6))
        q = (r or {}).get("question", "").strip()
        if not q:
            continue
        used |= set(ids)
        also = sorted(e["entry_id"] for e in entries.values() if t in e["text"].lower() and e["entry_id"] not in ids)
        out.append({"question": q, "relevant": ids, "type": "multi", "note": f"tag={t!r}; also mention it (check gold): {also}"})
    return out


# ---------- unanswerable ----------
UNANS_PROMPT = """Read this journal entry. Write ONE first-person question about a closely related topic whose
answer is NOT stated in the entry (a plausible near-miss, not an absurd one). Also give 2-3 distinctive
single-word keywords the answer would have to contain. Reply as JSON: {"question": "...", "keywords": ["..."]}.

ENTRY:
"""


def gen_unanswerable(entries, n, model, rng):
    corpus = " ".join(e["text"].lower() for e in entries.values())
    ids = list(entries)
    rng.shuffle(ids)
    schema = {"type": "object", "required": ["question", "keywords"],
              "properties": {"question": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}}}
    out = []
    for tried, eid in enumerate(ids, 1):
        if len(out) >= n:
            break
        progress("unanswerable", len(out), n, tried)
        r = chat(model, UNANS_PROMPT + entries[eid]["text"], schema, 0.7, rng.randint(0, 10**6)) or {}
        q, kws = r.get("question", "").strip(), [k.lower() for k in r.get("keywords", []) if isinstance(k, str) and k.strip()]
        if q and kws and not any(re.search(rf"\b{re.escape(k)}\b", corpus) for k in kws):  # any keyword in corpus -> maybe answerable
            out.append({"question": q, "relevant": [], "type": "unanswerable", "note": f"near-miss to {eid}; keywords {kws}"})
    return out


# ---------- commands ----------
def cmd_generate(a):
    path = EVAL_DIR / f"candidates_{a.version}.jsonl"
    if path.exists() and not a.force:
        sys.exit(f"{path.name} exists; pass --force to regenerate (this discards review state)")
    entries = es.load_entries()
    digest = model_digest(a.model)
    rng = random.Random(a.seed)
    safe = re.sub(r"\W+", "_", a.model)
    cands = []
    for label, fn in [
        ("single", lambda: gen_single(entries, a.n_single, a.model, rng)),
        ("date", lambda: gen_date(entries, a.n_date, rng)),
        ("multi", lambda: gen_multi(entries, a.n_multi, a.model, rng, EVAL_DIR / f"tags_cache_{safe}.json")),
        ("unanswerable", lambda: gen_unanswerable(entries, a.n_unans, a.model, rng)),
    ]:
        print(f"[{label}] starting...")
        got = fn()
        print(f"{label}: {len(got)} candidates" + " " * 30)  # trailing spaces clear the \r progress line
        cands += got
    entries_sha = hashlib.sha256(es.ENTRIES.read_bytes()).hexdigest()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, c in enumerate(cands, 1):
            c.update(cid=f"c{i:03d}", status="pending", draft_model=a.model, model_digest=digest, seed=a.seed, entries_sha256=entries_sha)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(cands)} candidates -> {path}")


def cmd_review(a):
    path = EVAL_DIR / f"candidates_{a.version}.jsonl"
    es.EVAL = EVAL_DIR / f"retrieval_{a.version}.jsonl"  # es.save() writes here
    if es.EVAL.exists() and not os.access(es.EVAL, os.W_OK):
        sys.exit(f"{es.EVAL.name} is frozen (read-only); pass a new --version")
    entries = es.load_entries()
    cands = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for c in cands:
        if c["status"] != "pending":
            continue
        print(f"\n{'#' * 70}\n{c['cid']}  [{c['type']}]  {c['note']}")
        if c["type"] in ("single", "multi"):
            for i in c["relevant"]:
                es.show(entries[i])
        else:
            print("relevant:", [f"{i}" for i in c["relevant"]] or "NONE (must be absent from the whole journal)")
        print(f"\nQUESTION: {c['question']}")
        while True:
            act = input("Enter=accept, e=edit question, i=edit entry ids, x=drop, q=quit: ").strip().lower()
            if act == "e":
                c["question"], c["edited"] = input("new question: ").strip() or c["question"], True
            elif act == "i":
                ids = [x.strip() for x in input("ids (comma-sep): ").split(",") if x.strip()]
                bad = [x for x in ids if x not in entries]
                if bad:
                    print("unknown:", bad)
                else:
                    c["relevant"], c["edited"] = ids, True
            elif act in ("", "x", "q"):
                break
        if act == "q":
            break
        if act == "x":
            c["status"] = "dropped"
        else:
            es.save(c["question"], c["relevant"], c["type"], "llm-candidate-edited" if c.get("edited") else "llm-candidate", c["draft_model"])
            c["status"] = "accepted"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cands))  # persist after every decision


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", default="qwen2.5:14b-instruct", help="local Ollama text model, ideally a different family from your answerer/embedder")
    g.add_argument("--n-single", type=int, default=25)
    g.add_argument("--n-date", type=int, default=10)
    g.add_argument("--n-multi", type=int, default=10)
    g.add_argument("--n-unans", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--force", action="store_true")
    r = sub.add_parser("review")
    for p in (g, r):
        p.add_argument("--version", default="v2")
    a = ap.parse_args()
    {"generate": cmd_generate, "review": cmd_review}[a.cmd](a)


if __name__ == "__main__":
    main()
