"""Isolated checks for date_filter.py. Synthetic strings only, no journal data, no Ollama/Chroma.

    python retrieve/date_filter_test.py
"""
from date_filter import date_window, parse_date_range

# (question, expected) -- expected is (kind, start, end) or None
CASES = [
    # the two shapes generate_eval_candidates.py templates
    ("What did I write about on 3 November 2024?", ("day", "2024-11-03", "2024-11-03")),
    ("What was going on with me in November 2024?", ("month", "2024-11-01", "2024-11-30")),
    # other explicit spellings
    ("What happened on 2024-11-03?", ("day", "2024-11-03", "2024-11-03")),
    ("What happened on November 3, 2024?", ("day", "2024-11-03", "2024-11-03")),
    ("What happened on the 3rd of November 2024?", ("day", "2024-11-03", "2024-11-03")),
    ("What happened on 21 Sept 2023?", ("day", "2023-09-21", "2023-09-21")),
    ("Anything in Feb 2022?", ("month", "2022-02-01", "2022-02-28")),
    ("How was I feeling in 2023?", ("year", "2023-01-01", "2023-12-31")),
    # a day wins over the month it sits in
    ("What did I write on 3 March 2024?", ("day", "2024-03-03", "2024-03-03")),
    # must NOT fire: no explicit date, fuzzy ranges, bare month/year without a trigger, impossible dates
    ("How did my worry about work show up across entries?", None),
    ("How did this evolve from early 2022 to early 2024?", None),
    ("What was I doing last March?", None),
    ("I may have been tired.", None),
    ("What happened on 31 February 2024?", None),
    ("Why did I stop at 2000 steps a day?", None),
]

for q, want in CASES:
    got = parse_date_range(q)
    assert got == want, f"{q!r}: got {got}, want {want}"

# window expansion: only dates the index actually holds
AVAILABLE = ["2024-10-30", "2024-11-03", "2024-11-17", "2024-12-01", "2024-11-03"]  # dupes as chunks of one entry
assert date_window("in November 2024", AVAILABLE) == ("month", ["2024-11-03", "2024-11-17"])
assert date_window("on 3 November 2024", AVAILABLE) == ("day", ["2024-11-03"])
assert date_window("on 9 November 2024", AVAILABLE) == ("day", [])  # named a date, nothing indexed there
assert date_window("no date here", AVAILABLE) is None
print(f"ok: {len(CASES)} parse cases + 4 window cases")
