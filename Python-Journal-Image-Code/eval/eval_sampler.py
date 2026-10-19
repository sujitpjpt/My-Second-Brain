"""Interactive helper for writing the retrieval eval set by hand.

Run in YOUR terminal only: it prints journal text. Output must not go through any
tool/agent/API. Eval file lives under Journal-Entires/ (gitignored).

  python eval/eval_sampler.py                # = --draft 40 (local LLM drafts; you keep/edit/drop)
  python eval/eval_sampler.py --manual       # random entries, you write every question
  python eval/eval_sampler.py --grep term    # list entries containing a term (for multi-entry questions)
  python eval/eval_sampler.py --show ID ...  # print specific entries
  python eval/eval_sampler.py --stats        # counts per question type
  python eval/eval_sampler.py --add          # write a question + list its entry_ids yourself (multi/date); no random seed entry
  python eval/eval_sampler.py --draft 40     # local LLM drafts questions; you keep/edit/drop

--draft talks only to the local Ollama server (localhost). Nothing leaves the machine.
"""
import argparse
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENTRIES = ROOT / "Journal-Entires" / "corpus" / "entries_v1.jsonl"  # input: one record per entry (build_entries.py)
EVAL = ROOT / "Journal-Entires" / "eval" / "retrieval_v1.jsonl"  # output: append-only, one question per line

GREP_OUT = ROOT / "Journal-Entires" / "eval" / "grep_last.txt"  # scratch: overwritten on every grep (gitignored via Journal-Entires/)

OLLAMA_URL = "http://localhost:11434/api/chat"  # local only; never change to a remote host
DRAFT_MODEL = "qwen2.5vl:7b"  # already pulled for OCR, so no extra download; any local chat model works via --model
MAX_CHARS = 12000  # keep entry inside num_ctx
# The "paraphrase, never reuse 4+ word phrases" rule matches shares_phrase() below, which flags violations after the fact
DRAFT_PROMPT = """You help build a retrieval test for a personal journal search tool.
Read the journal entry and write 2 questions the author might later ask that THIS entry answers.
Rules: first-person ("when did I...", "what was I worried about..."); paraphrase, never reuse
phrases of 4+ words from the entry; no dates or names unless essential; each question must make
sense without seeing the entry. Reply as JSON: {"questions": ["...", "..."]}.

ENTRY:
"""

TIPS = """\
Writing good questions:
  - Paraphrase. Don't copy phrases from the entry, or you test keyword match, not retrieval.
  - Ask what you'd really ask later ("when did I last feel burnt out?", "what was I worried about in June 2021?").
  - Mix types: single (one entry), date (lookup by day/period), multi (answer spans 3+ entries).
  - A question can have several relevant entries; list every one that genuinely answers it.
"""


def load_entries():
    """All entries keyed by entry_id."""
    with ENTRIES.open() as f:
        return {e["entry_id"]: e for e in map(json.loads, f)}


def load_eval():
    """Questions saved so far; empty list if the eval file doesn't exist yet."""
    if not EVAL.exists():
        return []
    with EVAL.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def grep_to_file(entries, term):
    """Write the full text of every entry containing `term` to GREP_OUT, replacing the previous result."""
    t = term.lower()
    hits = [e for e in sorted(entries.values(), key=lambda e: e["date"]) if t in e["text"].lower()]
    bar = "=" * 70
    body = "\n\n".join(f"{bar}\n{e['entry_id']}  ({e['date']})\n{'-' * 70}\n{e['text']}" for e in hits)
    GREP_OUT.parent.mkdir(parents=True, exist_ok=True)
    GREP_OUT.write_text(f"grep: {term}  ({len(hits)} entries)\n\n{body}\n")
    print(f"  {len(hits)} entries -> {GREP_OUT}")


def show(e):
    """Print one entry's full text, framed. This is journal content: terminal only."""
    print(f"\n{'=' * 70}\n{e['entry_id']}  ({e['date']})  pages={e['n_pages']}\n{'-' * 70}\n{e['text']}\n{'=' * 70}")


def append(rec):
    """Add one JSON line to the eval file, creating the folder on first use."""
    EVAL.parent.mkdir(parents=True, exist_ok=True)
    with EVAL.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save(question, ids, qtype, source, model=None):
    """Save one eval record. `relevant` lists every entry_id that answers the question (recall is scored against it)."""
    n = max((int(r["qid"][1:]) for r in load_eval()), default=0) + 1  # max+1, not len+1: dropped questions leave gaps, so len would reuse an id
    append({
        "qid": f"q{n:03d}", "question": question, "relevant": ids, "type": qtype,
        # source records provenance: hand / llm-draft / llm-draft-edited, so LLM-written questions can be audited later
        "source": source, "draft_model": model,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"  saved q{n:03d} ({qtype}, {source})")


def draft_questions(text, model):
    """Ask the local model for candidate questions about one entry. Returns [] if the reply isn't usable."""
    body = {
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": DRAFT_PROMPT + text[:MAX_CHARS]}],
        # Structured output: Ollama constrains the reply to this JSON schema, so parsing rarely fails
        "format": {"type": "object", "required": ["questions"],
                   "properties": {"questions": {"type": "array", "items": {"type": "string"}}}},
        # temperature 0.7 for varied questions (unlike OCR, which wants determinism); fixed seed keeps a re-run reproducible
        "options": {"temperature": 0.7, "seed": 42, "num_ctx": 8192},
    }
    req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())["message"]["content"]
    try:
        qs = json.loads(out)["questions"]
    except (json.JSONDecodeError, KeyError):
        return []
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()]


def shares_phrase(question, text, n=4):
    """True if question copies an n-word run from the entry (tests keywords, not retrieval)."""
    words = lambda s: re.findall(r"\w+", s.lower())  # lowercase, punctuation stripped, so only the words are compared
    qw, tw = words(question), words(text)
    # All n-word windows of the entry as a set, then check each n-word window of the question against it
    grams = {tuple(tw[i:i + n]) for i in range(len(tw) - n + 1)}
    return any(tuple(qw[i:i + n]) in grams for i in range(len(qw) - n + 1))


def draft_mode(entries, limit, model, seed):
    """Default flow: show an entry, show LLM-drafted questions, and let me keep, rewrite, drop or replace them."""
    # Entries already cited by a saved question are skipped, so repeat runs cover new ground
    used = {i for r in load_eval() for i in r["relevant"]}
    pool = [i for i in entries if i not in used]
    random.Random(seed + len(used)).shuffle(pool)  # same seed + same saved set = same order; saving a question reshuffles
    for eid in pool[:limit]:
        e = entries[eid]
        print(f"\ndrafting for {eid} …")
        try:
            drafts = draft_questions(e["text"], model)
        except OSError as err:
            sys.exit(f"Ollama not reachable at {OLLAMA_URL}: {err}")
        show(e)
        # Flag drafts that copy wording from the entry: they'd test keyword match rather than retrieval
        for i, q in enumerate(drafts, 1):
            flag = "  (!) copies a phrase from the entry" if shares_phrase(q, e["text"]) else ""
            print(f"  [{i}] {q}{flag}")
        choice = input("keep which (e.g. 1 2), w=write own, blank=skip, q=quit: ").strip().lower()
        if choice == "q":
            break
        for tok in choice.split():  # input like "1 2 w": keep drafts 1 and 2, then write one of my own
            if tok == "w":
                q = input("your question: ").strip()
                if q:
                    save(q, [eid], "single", "hand")
            elif tok.isdigit() and 1 <= int(tok) <= len(drafts):
                orig = drafts[int(tok) - 1]
                print(f"  [{tok}] {orig}")
                final = input("  Enter=keep as is, or type a rewrite (x=drop): ").strip()
                if final.lower() == "x":
                    continue
                final = final or orig  # blank input keeps the draft as is
                save(final, [eid], "single",
                     "llm-draft" if final == orig else "llm-draft-edited", model)


def ask_question(entries, seed_id=None):
    """Prompt for one eval record. Returns False if user quits."""
    q = input("Question (blank=skip, q=quit): ").strip()
    if q == "q":
        return False
    if not q:
        return True
    extra = input(f"Relevant entry_ids{f' besides {seed_id}' if seed_id else ''} (comma-sep, blank=none): ")
    ids = ([seed_id] if seed_id else []) + [i.strip() for i in extra.split(",") if i.strip()]
    bad = [i for i in ids if i not in entries]  # a typo'd id would silently make the question unanswerable
    if bad:
        print(f"  unknown entry_id(s) {bad}; not saved")
        return True
    qtype = input("Type [s]ingle / [d]ate / [m]ulti (default by count): ").strip().lower()
    # Unrecognised input falls back to the count of relevant entries: one id = single, several = multi
    qtype = {"s": "single", "d": "date", "m": "multi"}.get(qtype, "single" if len(ids) == 1 else "multi")
    save(q, ids, qtype, "hand")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep")
    ap.add_argument("--show", nargs="+")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--draft", type=int, metavar="N", help="entries to draft for (default 40)")
    ap.add_argument("--add", action="store_true", help="add questions with any number of entry_ids (use for multi/date)")
    ap.add_argument("--manual", action="store_true", help="no LLM drafts; write every question yourself")
    ap.add_argument("--model", default=DRAFT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    entries = load_entries()

    # Modes are checked in order: --stats, then draft (the default), then the --manual helpers (--grep, --show, free sampling)
    if args.stats:
        recs = load_eval()
        print(len(recs), "questions", dict(Counter(r["type"] for r in recs)),
              dict(Counter(r.get("source", "hand") for r in recs)))
        return
    if args.add:
        # Loop: type "grep <term>" to find candidate entries, or "show <id>" to read one, or write a question
        print(TIPS)
        while True:
            cmd = input("\nQuestion, or 'grep <term>' / 'show <id>' (q=quit): ").strip()
            if cmd.lower() == "q":
                return
            if cmd.startswith("grep "):
                grep_to_file(entries, cmd[5:])
            elif cmd.startswith("show "):
                i = cmd[5:].strip()
                show(entries[i]) if i in entries else print(f"unknown {i}")
            elif cmd:
                extra = input("Relevant entry_ids (comma-sep): ")
                ids = [i.strip() for i in extra.split(",") if i.strip()]
                bad = [i for i in ids if i not in entries]
                if bad or not ids:
                    print(f"  invalid/empty ids {bad}; not saved")
                    continue
                save(cmd, ids, "single" if len(ids) == 1 else "multi", "hand")
    if not args.manual:  # drafting is the default; --manual gives the old write-everything mode
        print(TIPS)
        draft_mode(entries, args.draft or 40, args.model, args.seed)
        return
    if args.grep:
        grep_to_file(entries, args.grep)
        return
    if args.show:
        for i in args.show:
            show(entries[i]) if i in entries else print(f"unknown {i}")
        return

    print(TIPS)
    # skip entries already used as a seed, so repeat runs don't re-show them
    used = {i for r in load_eval() for i in r["relevant"]}
    pool = [i for i in entries if i not in used]
    random.Random(args.seed + len(used)).shuffle(pool)
    for eid in pool:
        show(entries[eid])
        if not ask_question(entries, seed_id=eid):
            break


if __name__ == "__main__":
    sys.exit(main())
