"""Synthesis/preference-intent detection + fact-biased two-tier fusion.

Preference/advice queries ("suggest…", "recommend…", "what do you think") want
the user's distilled preference *facts*, not raw episodes that crowd them out.
``detect_synthesis_intent`` flags such queries (deterministic, zero-LLM) and
``_fuse_memory(..., fact_bias=True)`` strongly down-weights the episode tier.
"""
from __future__ import annotations

from memcontext.claims import now_ns
from memcontext.retrieval import _fuse_memory, detect_synthesis_intent
from memcontext.schema import Claim, ClaimStatus, Speaker, Turn


def test_fires_on_advice_and_preference_queries() -> None:
    for q in [
        "Can you suggest a hotel for my upcoming trip to Miami?",
        "Can you recommend some interesting cultural events this weekend?",
        "I'm planning my meal prep, any suggestions for new recipes?",
        "Should I buy a NAS device now or wait? What do you think?",
        "What kind of activities can I do during my commute?",
        "Help me decide which laptop to get.",
    ]:
        assert detect_synthesis_intent(q), q


def test_silent_on_factual_lookups() -> None:
    for q in [
        "What play did I attend at the local community theater?",
        "How many days did I spend camping this year?",
        "Remind me what color the Plesiosaur's body was.",
        "When did I start my new job?",
        "What is my sister's name?",
    ]:
        assert not detect_synthesis_intent(q), q


def _claim(cid: str) -> Claim:
    return Claim(
        claim_id=cid, session_id="s", subject="user", predicate="user_preference",
        value="prefers oceanfront hotels", value_normalised=None, confidence=0.9,
        source_turn_id="t_fact", status=ClaimStatus.ACTIVE, created_ts=now_ns(),
    )


def _turn(tid: str) -> Turn:
    return Turn(turn_id=tid, session_id="s", speaker=Speaker.USER, text="raw chatter", ts=now_ns())


def test_fact_bias_downweights_episodes() -> None:
    # A top-ranked episode outranks a lower-ranked fact under normal fusion, but
    # fact_bias flips it: the distilled fact must come first for synthesis.
    facts = [(_claim("c_low"), 0.1)]  # single fact, rank 1
    episodes = [(_turn("e_top"), 9.9)]  # single episode, rank 1
    normal = _fuse_memory(facts, episodes, top_k=2)
    biased = _fuse_memory(facts, episodes, top_k=2, fact_bias=True)
    # Under both, the fact (w=1.0) already beats the episode (w<=0.9) at equal rank;
    # the key invariant: fact_bias never lets an episode outrank the fact.
    assert normal[0][0].kind == "fact"
    assert biased[0][0].kind == "fact"
    # And the episode's fused weight is strictly lower under fact_bias.
    ep_normal = next(s for h, s in normal if h.kind == "episode")
    ep_biased = next(s for h, s in biased if h.kind == "episode")
    assert ep_biased < ep_normal
