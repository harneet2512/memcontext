"""Deterministic event-date extraction — zero LLM, regex + calendar arithmetic.

``event_ts`` (schema.py) records WHEN the fact's described thing happened, distinct
from ``created_ts`` (when the turn was ingested). It is the signal the supersession
``_event_blocks`` guard reads to keep two distinct dated occurrences ("ran a 5K on
March 3" vs "ran a 5K on June 12") both active instead of letting one clobber the
other.

Until now nothing populated ``event_ts`` at ingest, so the guard was inert. This
module fills it deterministically: it pulls an explicit calendar date out of a
fact's value/text and converts it to a UTC-midnight nanosecond timestamp. No LLM,
no randomness, no network — pure regex + ``datetime``.

Conservative by design: it fires ONLY on an explicit, unambiguous date (ISO
``YYYY-MM-DD``, ``Month D, YYYY``, ``D Month YYYY``, or ``MM/DD/YYYY``). Vague
references ("yesterday", "last week", "in the spring") return ``None`` — a wrong
event_ts is worse than none, because it would falsely split or falsely merge
claims. Returning ``None`` simply leaves the existing behaviour unchanged.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# ISO 8601: 2024-03-09
_ISO_RE = re.compile(r"\b(?P<y>(?:19|20)\d\d)-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])\b")

# "March 9, 2024" / "Mar 9 2024" (optional ordinal suffix + comma)
_MDY_RE = re.compile(
    rf"\b(?P<mon>{_MONTH_ALT})\.?\s+(?P<d>[0-3]?\d)(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d\d)\b",
    re.IGNORECASE,
)

# "9 March 2024" / "9th of March, 2024"
_DMY_RE = re.compile(
    rf"\b(?P<d>[0-3]?\d)(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_MONTH_ALT})\.?,?\s+(?P<y>(?:19|20)\d\d)\b",
    re.IGNORECASE,
)

# Numeric MM/DD/YYYY (US ordering — the dominant convention in the conversational
# corpora this serves). Requires a 4-digit year to avoid matching fractions/ratios.
_NUM_RE = re.compile(r"\b(?P<m>0?[1-9]|1[0-2])/(?P<d>0?[1-9]|[12]\d|3[01])/(?P<y>(?:19|20)\d\d)\b")

_NS_PER_SEC = 1_000_000_000


def _to_ns(year: int, month: int, day: int) -> int | None:
    try:
        dt = datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None  # impossible calendar date (e.g. Feb 30) — reject
    return int(dt.timestamp()) * _NS_PER_SEC


_NS_PER_DAY = 86_400 * _NS_PER_SEC


def format_elapsed(event_ts_ns: int | None, ref_ts_ns: int | None) -> str | None:
    """Render the time between ``event_ts_ns`` and ``ref_ts_ns`` in multiple human
    units, so an elapsed/interval question ("how many weeks/days/months ago …") can
    be read off DIRECTLY without the reader doing any date arithmetic.

    Pure arithmetic — deterministic, zero LLM, no network. Returns ``None`` when
    either timestamp is ``None`` (graceful: nothing to ground, caller leaves the
    served item unchanged).

    Examples (ref AFTER event -> "ago"):
        28 days  -> "~28 days (~4 weeks, ~1 month) ago"
        6 days   -> "~6 days ago"
        35 days  -> "~35 days (~5 weeks, ~1 month) ago"
    A future event (ref BEFORE event) renders "… from now". A zero delta renders
    "today / same day".
    """
    if event_ts_ns is None or ref_ts_ns is None:
        return None

    delta_ns = ref_ts_ns - event_ts_ns
    future = delta_ns < 0
    abs_ns = -delta_ns if future else delta_ns

    days = abs_ns // _NS_PER_DAY
    if days == 0:
        return "today (same day)"

    weeks = round(days / 7)
    months = round(days / 30)

    def _unit(n: int, singular: str) -> str:
        return f"~{n} {singular}" if n == 1 else f"~{n} {singular}s"

    parts = [_unit(int(days), "day")]
    if weeks >= 1:
        parts.append(_unit(int(weeks), "week"))
    if months >= 1:
        parts.append(_unit(int(months), "month"))

    head = parts[0]
    tail = parts[1:]
    rendered = head if not tail else f"{head} ({', '.join(tail)})"
    suffix = "from now" if future else "ago"
    return f"{rendered} {suffix}"


def extract_event_ts(*texts: str) -> int | None:
    """Return a UTC-midnight nanosecond timestamp for the FIRST explicit calendar
    date found across ``texts`` (checked in order), or ``None``.

    Deterministic and side-effect-free. ``texts`` are searched in the order given,
    so callers should pass the most specific field first (e.g. the claim VALUE
    before the full turn TEXT).
    """
    for text in texts:
        if not text:
            continue
        m = _ISO_RE.search(text)
        if m:
            ts = _to_ns(int(m["y"]), int(m["m"]), int(m["d"]))
            if ts is not None:
                return ts
        m = _MDY_RE.search(text)
        if m:
            ts = _to_ns(int(m["y"]), _MONTHS[m["mon"].lower()], int(m["d"]))
            if ts is not None:
                return ts
        m = _DMY_RE.search(text)
        if m:
            ts = _to_ns(int(m["y"]), _MONTHS[m["mon"].lower()], int(m["d"]))
            if ts is not None:
                return ts
        m = _NUM_RE.search(text)
        if m:
            ts = _to_ns(int(m["y"]), int(m["m"]), int(m["d"]))
            if ts is not None:
                return ts
    return None
