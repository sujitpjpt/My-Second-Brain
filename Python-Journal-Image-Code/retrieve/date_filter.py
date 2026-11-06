"""Pull an explicit calendar date out of a question so retrieval can filter on chunk date metadata.

    from retrieve.date_filter import date_window
    date_window("What did I write about on 3 November 2024?", available_dates)  # -> ("day", ["2024-11-03"])

Dense embeddings don't match dates, so "in November 2024" retrieves on topic words only. When the question names an
explicit date, restrict candidates to that window and let the embedding rank within it.

Pure stdlib, no I/O, never touches journal text: the input is the question and the index's date strings.
Only EXPLICIT dates fire (day, month+year, ISO, "in <year>"). Relative or fuzzy phrases ("last spring", "early 2022 to
early 2024", "lately") are left alone on purpose: a wrong filter silently drops the right entry, a missed one just
falls back to the normal ranking.
"""
import re
from calendar import monthrange
from datetime import date

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december"]
MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, 1)}              # "november" -> 11
MONTHS.update({name[:3]: i for i, name in enumerate(_MONTH_NAMES, 1)})    # "nov" -> 11
MONTHS["sept"] = 9                                                        # "sept" -> 9
# Longest alternatives first so "sept"/"september" win over "sep"
# e.g. matches "November", "Nov", "Nov." (optional trailing dot)
_MON = r"(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?"
_ORD = r"(?:st|nd|rd|th)?"  # optional ordinal suffix: "3", "3rd", "21st"

# Tried in order, first match wins: most specific first, so "3 March 2024" is a day, not the month "March 2024"
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_FIRST = re.compile(rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON},?\s+(\d{{4}})\b", re.I)    # 3 November 2024
_MONTH_FIRST = re.compile(rf"\b{_MON}\s+(\d{{1,2}}){_ORD},?\s+(\d{{4}})\b", re.I)            # November 3, 2024
_MONTH_YEAR = re.compile(rf"\b{_MON},?\s+(\d{{4}})\b", re.I)                                  # November 2024
_YEAR = re.compile(r"\b(?:in|during|throughout)\s+(?:the\s+year\s+)?((?:19|20)\d{2})\b", re.I)  # in 2023


def parse_date_range(question):
    """-> (kind, start_iso, end_iso) with inclusive bounds, or None. kind is 'day' | 'month' | 'year'.

    parse_date_range("entry from 2024-11-03")        -> ("day",   "2024-11-03", "2024-11-03")
    parse_date_range("on 3rd of Nov, 2024")          -> ("day",   "2024-11-03", "2024-11-03")
    parse_date_range("November 3rd, 2024")           -> ("day",   "2024-11-03", "2024-11-03")
    parse_date_range("what happened in Feb 2024?")   -> ("month", "2024-02-01", "2024-02-29")  # leap year handled
    parse_date_range("how was I in 2023?")           -> ("year",  "2023-01-01", "2023-12-31")
    parse_date_range("31 February 2024")             -> None  # invalid date, no filter
    parse_date_range("last spring")                  -> None  # relative/fuzzy, left to normal ranking
    """
    try:
        if m := _ISO.search(question):
            d = date(int(m[1]), int(m[2]), int(m[3])).isoformat()
            return "day", d, d
        if m := _DAY_FIRST.search(question):
            d = date(int(m[3]), MONTHS[m[2].lower()], int(m[1])).isoformat()
            return "day", d, d
        if m := _MONTH_FIRST.search(question):
            d = date(int(m[3]), MONTHS[m[1].lower()], int(m[2])).isoformat()
            return "day", d, d
        if m := _MONTH_YEAR.search(question):
            y, mo = int(m[2]), MONTHS[m[1].lower()]
            return "month", f"{y:04d}-{mo:02d}-01", f"{y:04d}-{mo:02d}-{monthrange(y, mo)[1]:02d}"
        if m := _YEAR.search(question):
            return "year", f"{m[1]}-01-01", f"{m[1]}-12-31"
    except ValueError:  # e.g. 31 February: not a real date, so don't filter
        return None
    return None


def date_window(question, available_dates):
    """Concrete dates the index holds inside the question's window: (kind, sorted_dates), or None if no explicit date.

    Returns an empty list when the question names a date but nothing is indexed there; the caller decides the fallback.
    Chroma's range operators only handle numbers and `date` is a string, so callers match these with `$in`.

    available = ["2024-10-30", "2024-11-03", "2024-11-17", "2025-01-02"]
    date_window("entry on 3 November 2024", available)  -> ("day",   ["2024-11-03"])
    date_window("anything in November 2024?", available) -> ("month", ["2024-11-03", "2024-11-17"])
    date_window("what about 5 March 2020?", available)   -> ("day",   [])   # date named, nothing indexed there
    date_window("what did I think lately?", available)   -> None           # no explicit date, don't filter
    # caller usage: where={"date": {"$in": dates}}
    """
    parsed = parse_date_range(question)
    if parsed is None:
        return None  # no explicit date in the question -> caller skips filtering entirely
    kind, start, end = parsed
    # explicit date found -> (kind, [dates in window]); ISO strings compare correctly as plain strings.
    # set() dedupes (many chunks share one date), sorted() makes the output deterministic.
    # Empty list = date named but nothing indexed there -> caller decides the fallback (e.g. unfiltered search).
    # kind tells the caller how specific the match was: "day" | "month" | "year".
    # Example: question "November 2024" -> kind="month", start="2024-11-01", end="2024-11-30"
    #   available_dates = ["2024-11-17", "2024-10-30", "2024-11-03", "2024-11-03", "2025-01-02"]
    #   set(...)    -> {"2024-11-17", "2024-10-30", "2024-11-03", "2025-01-02"}   (duplicate 11-03 dropped)
    #   filter      -> "2024-11-17", "2024-11-03"      (10-30 < start, 2025-01-02 > end)
    #   sorted(...) -> ["2024-11-03", "2024-11-17"]
    #   returns     -> ("month", ["2024-11-03", "2024-11-17"])
    return kind, sorted(d for d in set(available_dates) if start <= d <= end)
