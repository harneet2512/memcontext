"""Source-trust tiering — how much to trust a claim based on WHERE it came from.

Provenance-aware trust scoring (the named memory-poisoning defense): a fact the
user stated in conversation is more authoritative than one scraped from a browsed
page or inferred by the assistant. The weight is intrinsic to the claim (set at
insert from its source episode) and feeds (a) a retrieval ranking channel and
(b) a supersession guard so low-trust content cannot silently override high-trust
content. Deterministic, zero-LLM.
"""
from __future__ import annotations

# Trust tiers in [0, 1] — higher is more authoritative.
TRUSTED_USER = 1.0      # the user stated it in conversation
TOOL_OUTPUT = 0.7       # a tool / API returned it
AGENT_INFERRED = 0.5    # the assistant inferred / derived it
EXTERNAL_WEB = 0.35     # scraped from a browsed page (untrusted origin)
DEFAULT = 0.5


def trust_for_source(source_type: str | None, speaker: str | None) -> float:
    """Map an episode's origin (source_type + speaker) to a trust weight."""
    st = (source_type or "conversation").lower()
    sp = (speaker or "user").lower()
    if st == "browser":
        return EXTERNAL_WEB
    if st == "tool_call":
        return TOOL_OUTPUT
    if sp == "assistant":
        return AGENT_INFERRED
    if sp == "user":
        return TRUSTED_USER
    return DEFAULT
