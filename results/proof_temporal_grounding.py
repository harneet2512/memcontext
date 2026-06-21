"""TDD proof: temporal grounding for elapsed/interval questions (zero-LLM).

Real AMB failures: temporal elapsed/interval questions fail because the reader is
served DATED memories but refuses to do the date arithmetic ("I'm not able to
determine how many weeks ago …"). The substrate already populates event_ts at
ingest (temporal.extract_event_ts); the gap was (a) no directly-usable elapsed
rendering and (b) handle_memory_query never exposed event_ts on served items.

This proves both fixes, deterministically:

  PART A (no embedder, pure arithmetic): temporal.format_elapsed renders elapsed
  time in MULTIPLE human units so "how many weeks/days/months ago" is read off
  directly; handles past/future, singular/plural, None.

  PART B (real-embedder ingest -> serve): handle_memory_query now exposes a
  non-null event_ts on the served claim AND the served episode for a dated event,
  and orders dated items chronologically under temporal intent.
"""
import os, sqlite3, sys

os.environ.setdefault("SUBSTRATE_PACKS_DIR", os.path.abspath("predicate_packs"))
os.environ.setdefault("ACTIVE_PACK", "general")
os.environ.setdefault("MEMCONTEXT_EMBED_EPISODES", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.abspath("."))

from memcontext.temporal import format_elapsed, extract_event_ts
from memcontext.schema import open_database, Speaker
from memcontext.on_new_turn import on_new_turn
from memcontext.extractors import PassthroughExtractor
from memcontext.supersession_semantic import SemanticSupersession
from memcontext.retrieval import EmbeddingClient
from memcontext.predicate_packs import active_pack
from memcontext.mcp_tools import handle_memory_query

_NS_PER_DAY = 86_400 * 1_000_000_000


# ============================ PART A — format_elapsed =========================

def case_28_days_weeks():
    ref = 100 * _NS_PER_DAY
    ev = ref - 28 * _NS_PER_DAY
    out = format_elapsed(ev, ref)
    ok = out is not None and "4 week" in out and "ago" in out
    return ("format_elapsed 28d -> '~4 weeks' & 'ago'", ok, f"out={out!r}")


def case_6_days_singular_only():
    ref = 100 * _NS_PER_DAY
    ev = ref - 6 * _NS_PER_DAY
    out = format_elapsed(ev, ref)
    # 6 days: weeks rounds to 1 (round(6/7)=1) -> still must read '6 day' directly.
    ok = out is not None and "6 day" in out and "ago" in out
    return ("format_elapsed 6d -> '6 day'", ok, f"out={out!r}")


def case_35_days_weeks_and_month():
    ref = 100 * _NS_PER_DAY
    ev = ref - 35 * _NS_PER_DAY
    out = format_elapsed(ev, ref)
    ok = out is not None and "5 week" in out and "1 month" in out
    return ("format_elapsed 35d -> '~5 weeks' & '~1 month'", ok, f"out={out!r}")


def case_none_graceful():
    ok = format_elapsed(None, 100) is None and format_elapsed(100, None) is None
    return ("format_elapsed(None,*) / (*,None) -> None", ok, "graceful")


def case_future_from_now():
    ref = 100 * _NS_PER_DAY
    ev = ref + 14 * _NS_PER_DAY  # event is AFTER ref -> future
    out = format_elapsed(ev, ref)
    ok = out is not None and "from now" in out and "2 week" in out
    return ("format_elapsed future -> 'from now'", ok, f"out={out!r}")


def case_same_day():
    ref = 100 * _NS_PER_DAY
    out = format_elapsed(ref, ref)
    ok = out is not None and "today" in out
    return ("format_elapsed 0 delta -> 'today'", ok, f"out={out!r}")


# ============================ PART B — serve exposes event_ts =================

_EMB = None
def emb():
    global _EMB
    if _EMB is None:
        _EMB = EmbeddingClient(); _EMB.embed(["warmup"])
    return _EMB


def ingest(c, sid, text, value, predicate="user_fact"):
    ex = PassthroughExtractor([{"subject": "user", "predicate": predicate, "value": value}])
    on_new_turn(c, session_id=sid, speaker=Speaker.USER, text=text, extractor=ex,
                semantic=SemanticSupersession(emb()), embedder=emb(), namespace="user")


def fresh():
    active_pack.cache_clear()
    return open_database(":memory:")


def case_serve_exposes_event_ts():
    c = fresh()
    # A dated event — the value itself carries the explicit calendar date so
    # extract_event_ts(value, text) fires at ingest.
    ingest(c, "s1",
           "I met my aunt on 2024-03-01 and got a chandelier",
           "met aunt and received a chandelier on 2024-03-01")
    res = handle_memory_query(c, query="when did I meet my aunt and get the chandelier",
                              session_id=None, namespace="user")
    claims = res.get("claims", [])
    epis = res.get("episodes", [])

    expected = extract_event_ts("met aunt and received a chandelier on 2024-03-01")
    claim_ev = next((x.get("event_ts") for x in claims if x.get("event_ts") is not None), None)
    epi_ev = next((e.get("event_ts") for e in epis if e.get("event_ts") is not None), None)

    claim_ok = claim_ev is not None and claim_ev == expected
    epi_ok = epi_ev is not None and epi_ev == expected
    # created_ts must still be present (kept, not stripped) on the served claim.
    created_ok = all("created_ts" in x for x in claims) if claims else False

    ok = claim_ok and epi_ok and created_ok
    return ("serve exposes non-null event_ts on claim+episode (gold=2024-03-01)", ok,
            f"claim event_ts={claim_ev} (expected {expected})",
            f"episode event_ts={epi_ev}",
            f"created_ts present on claims={created_ok}")


def case_timeline_ordering():
    c = fresh()
    # Two dated events ingested out of chronological order; a temporal query must
    # serve them oldest-first so an interval is read off the ends.
    ingest(c, "s1", "I started project Atlas on 2024-04-10",
           "started project Atlas on 2024-04-10")
    ingest(c, "s1", "I finished the kickoff on 2024-04-04",
           "finished kickoff on 2024-04-04")
    res = handle_memory_query(c, query="how many days passed between the kickoff and starting Atlas",
                              session_id=None, namespace="user")
    claims = res.get("claims", [])
    evs = [x.get("event_ts") for x in claims if x.get("event_ts") is not None]
    # temporal intent -> dated claims ascending
    ordered = evs == sorted(evs)
    ok = len(evs) >= 2 and ordered
    # Demonstrate the elapsed rendering the reader would read off the two ends.
    elapsed = format_elapsed(evs[0], evs[-1]) if len(evs) >= 2 else None
    return ("temporal query orders dated claims chronologically; elapsed readable", ok,
            f"event_ts (served order)={evs}", f"format_elapsed(ends)={elapsed!r}")


if __name__ == "__main__":
    part_a = [
        case_28_days_weeks(), case_6_days_singular_only(),
        case_35_days_weeks_and_month(), case_none_graceful(),
        case_future_from_now(), case_same_day(),
    ]
    part_b = [case_serve_exposes_event_ts(), case_timeline_ordering()]
    print("=" * 80)
    allok = True
    print("PART A — format_elapsed (pure arithmetic, no embedder)")
    for name, ok, *detail in part_a:
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for d in detail:
            print("          ", d)
    print("-" * 80)
    print("PART B — handle_memory_query exposes event_ts (real-embedder ingest)")
    for name, ok, *detail in part_b:
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for d in detail:
            print("          ", d)
    print("=" * 80)
    print("RESULT:", "ALL PASS" if allok else "SOME FAIL — see above")
    sys.exit(0 if allok else 1)
