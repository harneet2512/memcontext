"""Claim extractors — local-only, no external API calls.

PassthroughExtractor: accepts pre-structured claims (default for MCP clients).
SimpleExtractor: regex/heuristic extraction from raw text (dev/test fallback only).

Neither calls any external LLM API (Claude, OpenAI, Anthropic, LiteLLM, Bedrock).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from memcontext.on_new_turn import ExtractedClaim
from memcontext.schema import Turn


class PassthroughExtractor:
    """Default extractor — accepts pre-structured claims from the MCP client.

    The MCP client (e.g., Claude Code) performs extraction and passes
    structured claims directly. This extractor just converts them.
    """

    def __init__(self, claims: list[dict]) -> None:
        self._claims = claims

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        result = []
        for c in self._claims:
            result.append(ExtractedClaim(
                subject=c.get("subject", "user"),
                predicate=c.get("predicate", "user_fact"),
                value=c["value"],
                confidence=float(c.get("confidence", 0.9)),
                char_start=c.get("char_start"),
                char_end=c.get("char_end"),
            ))
        return result


class SimpleExtractor:
    """Local regex-only claim extractor. Dev/test fallback — not for production use.

    Uses pattern matching to extract basic claims. All claims get low
    confidence (0.5) since extraction is heuristic. Does NOT call any
    external API.
    """

    _PATTERNS: list[tuple[str, str]] = [
        (r"(?:I|i) (?:prefer|like|love|enjoy|favor)\b(.+)", "user_preference"),
        (r"(?:I|i) (?:hate|dislike|avoid|don't like|can't stand)\b(.+)", "user_preference"),
        (r"(?:I|i) (?:am|'m)\b(.+)", "user_fact"),
        (r"(?:I|i) (?:have|'ve)\b(.+)", "user_fact"),
        (r"(?:I|i) (?:work|worked) (?:at|for)\b(.+)", "user_fact"),
        (r"(?:I|i) (?:live|lived) (?:in|at)\b(.+)", "user_fact"),
        (r"(?:I|i) (?:use|used|using)\b(.+)", "user_preference"),
        (r"(?:my|My) (?:name is|name's)\b(.+)", "user_fact"),
    ]

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        from memcontext.predicate_packs import active_pack
        families = active_pack().predicate_families

        claims: list[ExtractedClaim] = []
        text = turn.text.strip()
        if not text:
            return claims

        for pattern, predicate in self._PATTERNS:
            if predicate not in families:
                continue
            m = re.search(pattern, text)
            if m:
                value = m.group(1).strip().rstrip(".,;!?")
                if value:
                    claims.append(ExtractedClaim(
                        subject="user",
                        predicate=predicate,
                        value=value,
                        confidence=0.5,
                    ))

        # Fallback: if no patterns matched, store whole text as user_fact
        if not claims:
            predicate = "user_fact" if "user_fact" in families else next(iter(families))
            claims.append(ExtractedClaim(
                subject="user",
                predicate=predicate,
                value=text,
                confidence=0.5,
            ))

        return claims
