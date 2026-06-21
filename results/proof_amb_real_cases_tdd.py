"""TDD from REAL past-run failures (AMB run 27885224285, LongMemEval-S).

Each case reconstructs a real wrong-answer question as an ingest->serve test against
the CURRENT integrated substrate, using the ACTUAL distractor memories that were in
the served context for that question (so the noise is real, not synthetic). Where the
OLD product dropped the answer-bearing evidence entirely (mortgage: the $400k update
was never served), we RECREATE that turn faithfully -- the question's gold proves it
existed in the haystack -- and prove the substrate now RESOLVES + SERVES it cleanly.

What this proves: the resolution/serve layer (supersession + stale-episode filter +
dedup) turns the real noisy memory set into clean current context. What it does NOT
prove (needs the full run): upstream extraction/retrieval recovering a dropped update.
"""
import os, sqlite3, sys

os.environ.setdefault("SUBSTRATE_PACKS_DIR", os.path.abspath("predicate_packs"))
os.environ.setdefault("ACTIVE_PACK", "general")
os.environ.setdefault("MEMCONTEXT_EMBED_EPISODES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.abspath("."))

from memcontext.schema import open_database, Speaker
from memcontext.on_new_turn import on_new_turn
from memcontext.extractors import PassthroughExtractor
from memcontext.supersession_semantic import SemanticSupersession
from memcontext.retrieval import EmbeddingClient
from memcontext.predicate_packs import active_pack
from memcontext.mcp_tools import handle_memory_query

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

def served(c, query):
    res = handle_memory_query(c, query=query, session_id=None, namespace="user")
    claims = [x["value"] for x in res.get("claims", [])]
    epis = [e["text"] for e in res.get("episodes", [])]
    return res, claims, epis


# === CASE 1: knowledge-update mortgage. REAL distractors from run E's served context
# ($325k closing costs, AI-regulation Q, cable setup, road trip) + the $350k pre-approval
# that WAS served + the $400k update the old product dropped (gold proves it existed). ===
def case_mortgage():
    c = fresh()
    ingest(c, "s1", "provided estimated closing costs for a $325,000 home purchase",
           "estimated closing costs for a $325,000 home purchase", "assistant_action")
    ingest(c, "s1", "I got pre-approved for a $350,000 mortgage loan from Wells Fargo",
           "pre-approved for a $350,000 mortgage loan")
    ingest(c, "s2", "what does this article say about AI regulation",
           "asked about AI regulation article", "user_topic")
    ingest(c, "s3", "I'm planning a road trip and want my car checked",
           "planning a road trip", "user_topic")
    # the UPDATE the haystack contained (gold=$400,000) -- recreated faithfully
    ingest(c, "s4", "Wells Fargo raised my pre-approval to a $400,000 mortgage loan",
           "pre-approved for a $400,000 mortgage loan")
    res, claims, epis = served(c, "What was the amount I was pre-approved for on my mortgage?")
    has400 = any("400,000" in x for x in claims)        # current value served
    no350_claim = not any("350,000" in x for x in claims)  # stale not in claims
    no350_epi = not any("350,000" in e for e in epis)      # stale not leaked via episodes
    no_empty = all(str(x).strip() for x in claims)         # no degenerate empty claim
    # NB: $325,000 (closing costs) is a DISTINCT fact, legitimately served — not stale.
    ok = has400 and no350_claim and no350_epi and no_empty
    return ("knowledge-update mortgage  $350k->$400k vs real noise", ok,
            f"served claims={claims}", f"served episodes={[e[:42] for e in epis]}")


# === CASE 2: multi-session COUNT. REAL clothing evidence from run E (gold=3 items to
# pick up / return): boots exchanged-not-picked-up, sweater lent to sister, jacket. ===
def case_clothing_count():
    c = fresh()
    ingest(c, "s1", "I exchanged boots at Zara for a larger size, haven't picked them up yet",
           "boots exchanged at Zara, not yet picked up", "user_event")
    ingest(c, "s1", "my green sweater is lent to my sister with no return date",
           "green sweater lent to sister, awaiting return", "user_event")
    ingest(c, "s2", "I need to return a jacket that didn't fit",
           "jacket to be returned", "user_event")
    ingest(c, "s2", "recommended grouping clothes by type when organizing",
           "advice: group clothes by type", "assistant_action")  # distractor
    res, claims, epis = served(c, "How many clothing items do I need to pick up or return?")
    en = res.get("enumeration") or {}
    dc = en.get("distinct_count")
    ok = dc == 3
    return ("multi-session count clothing  (gold=3)", ok,
            f"enumeration.distinct_count={dc}", f"reps={[r.get('representative') for r in en.get('representatives',[])]}")


if __name__ == "__main__":
    results = [case_mortgage(), case_clothing_count()]
    print("=" * 80)
    allok = True
    for name, ok, *detail in results:
        allok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for d in detail:
            print("        ", d)
    print("=" * 80)
    print("RESULT:", "ALL PASS" if allok else "SOME FAIL — see above")
