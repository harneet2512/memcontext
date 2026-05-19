"""Claim extractors — three tiers for different contexts.

PassthroughExtractor: accepts pre-structured claims (default for MCP clients).
LLMExtractor: local Ollama-backed extraction (benchmark + production).
SimpleExtractor: regex/heuristic fallback (dev/test smoke only).

No cloud API calls. LLMExtractor uses Ollama (localhost:11434) for local
inference. SimpleExtractor uses regex only. PassthroughExtractor does
no extraction at all.

Architecture ported from RobbyMD src/extraction/claim_extractor/ and
generalized: removed clinical dependencies, replaced OpenAI client with
Ollama HTTP, kept the prompt composition, JSON parsing, predicate
validation, and char span resolution logic.
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from memcontext.on_new_turn import ExtractedClaim
from memcontext.schema import Turn

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt composition (from RobbyMD prompt.py, generalized)
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    """Build the extraction system prompt from the active predicate pack."""
    from memcontext.predicate_packs import active_pack

    pack = active_pack()
    families = tuple(sorted(pack.predicate_families))
    examples_text = _render_examples(pack.few_shot_examples)

    return f"""\
You extract structured claims from one conversation turn. You emit **only**
JSON — a list of claim objects, possibly empty.

## Rules

1. **Predicate families (closed set)**. Every claim's `predicate` MUST be
   one of: {", ".join(families)}.
   Emitting any other predicate is a failure.

2. **One fact per claim**. If a turn contains multiple independent facts,
   emit multiple claims.

3. **Negations / denials**. When the speaker explicitly denies or dislikes
   something, emit a claim with value `dislikes:<thing>` for preferences
   or a clear negation in the value.

4. **Honesty over fluency**. If the turn is ambiguous, emit with low
   confidence (≤ 0.5) or emit an empty list. NEVER fabricate.

5. **No invented predicates**. Use the nearest listed family.

6. **Supersession is downstream**. Emit the new claim; the substrate
   decides what older claim it supersedes.

7. **Confidence scale** [0, 1]. Use 0.9+ only for explicit unambiguous
   statements. Hedge words drop confidence below 0.7.

8. **Exact fact preservation**. Preserve names, locations, money, durations,
   counts, dates, percentages, product names, degree names verbatim.
   Do not round, paraphrase, or replace these values.

## Output schema

```json
[{{"subject": "str", "predicate": "str", "value": "str", "confidence": float}}]
```

Return a JSON array (possibly empty). No commentary. No markdown.

## Few-shot examples

{examples_text}
"""


def _render_examples(examples: tuple) -> str:
    """Render few-shot examples from the active pack."""
    if not examples:
        return "(no examples available)"
    blocks: list[str] = []
    for i, ex in enumerate(examples, 1):
        prior = "\n".join(f"    {spk}: {txt}" for spk, txt in ex.prior_turns)
        cur_spk, cur_txt = ex.current_turn
        blocks.append(
            f"### Example {i}: {ex.name}\n"
            f"Scenario: {ex.scenario}\n\n"
            f"Prior turns:\n{prior}\n"
            f"Current turn:\n    {cur_spk}: {cur_txt}\n"
            f"Active claims: {ex.active_claims_summary}\n\n"
            f"Expected output:\n{ex.expected_output}\n"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# JSON parsing (from RobbyMD extractor.py, kept as-is)
# ---------------------------------------------------------------------------


def _parse_claims(raw: str) -> list[dict[str, Any]]:
    """Parse an LLM response into a list of claim dicts.

    Handles: bare list, wrapper dict with known keys, single-key dict,
    single claim dict, empty dict. Returns [] on malformed input.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("extractor.parse_failed", raw=raw[:200])
        return []

    items: list[Any] | None = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("claims", "extracted_claims", "items", "list", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                break
        if items is None and len(parsed) == 1:
            only_value = next(iter(parsed.values()))
            if isinstance(only_value, list):
                items = only_value
        if items is None and {"subject", "predicate", "value", "confidence"}.issubset(
            parsed.keys()
        ):
            items = [parsed]
        if items is None and len(parsed) == 0:
            return []
        if items is None:
            log.warning("extractor.parse_unexpected_shape", keys=list(parsed.keys())[:10])
            return []
    else:
        return []

    return [item for item in items if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Claim validation (from RobbyMD extractor.py, kept as-is)
# ---------------------------------------------------------------------------


def _to_extracted_claims(
    raw_claims: list[dict[str, Any]],
    turn: Turn,
    allowed_predicates: frozenset[str],
) -> list[ExtractedClaim]:
    """Convert raw claim dicts into validated ExtractedClaim objects.

    Drops claims with: missing fields, invalid predicate, bad confidence.
    Resolves char spans via text.find(value).
    """
    result: list[ExtractedClaim] = []
    drop_count = 0

    for raw in raw_claims:
        subject = raw.get("subject")
        predicate = raw.get("predicate")
        value = raw.get("value")
        confidence_raw = raw.get("confidence")

        if not isinstance(subject, str) or not subject.strip():
            drop_count += 1
            continue
        if not isinstance(predicate, str) or predicate not in allowed_predicates:
            drop_count += 1
            continue
        if not isinstance(value, str) or not value.strip():
            drop_count += 1
            continue
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            drop_count += 1
            continue
        if not (0.0 <= confidence <= 1.0):
            drop_count += 1
            continue

        char_start, char_end = _find_char_span(turn.text, value)
        value_normalised = raw.get("value_normalised")
        if not isinstance(value_normalised, str):
            value_normalised = None

        result.append(
            ExtractedClaim(
                subject=subject.strip(),
                predicate=predicate,
                value=value.strip(),
                confidence=confidence,
                value_normalised=value_normalised,
                char_start=char_start,
                char_end=char_end,
            )
        )

    if drop_count:
        log.info(
            "extractor.dropped", turn_id=turn.turn_id,
            kept=len(result), dropped=drop_count,
        )
    return result


def _find_char_span(text: str, value: str) -> tuple[int | None, int | None]:
    """Best-effort char-span resolution. (None, None) when no match."""
    if not text or not value:
        return None, None
    idx = text.find(value)
    if idx < 0:
        lowered = text.lower().find(value.lower())
        if lowered < 0:
            return None, None
        return lowered, lowered + len(value)
    return idx, idx + len(value)


# ---------------------------------------------------------------------------
# LLMExtractor — Ollama-backed, local-only, no cloud API
# ---------------------------------------------------------------------------


class LLMExtractor:
    """Local LLM claim extractor via Ollama. Production/benchmark grade.

    Calls a local Ollama instance (default: localhost:11434) with the same
    prompt architecture used for the 88.4% LongMemEval baseline. No cloud
    API calls — fully offline once the model is pulled.

    Setup:
        1. Install Ollama: https://ollama.com
        2. Pull a model: ollama pull qwen3:8b
        3. The extractor auto-connects to localhost:11434

    Environment variables:
        MEMCONTEXT_OLLAMA_URL: Ollama base URL (default: http://localhost:11434)
        MEMCONTEXT_OLLAMA_MODEL: Model name (default: qwen3:8b)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        import os
        self._model = model or os.environ.get("MEMCONTEXT_OLLAMA_MODEL", "qwen3:8b")
        self._base_url = (
            base_url or os.environ.get("MEMCONTEXT_OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self._timeout = timeout
        self._system_prompt: str | None = None
        self._allowed_predicates: frozenset[str] | None = None

    def _ensure_prompt(self) -> None:
        if self._system_prompt is None:
            from memcontext.predicate_packs import active_pack
            self._system_prompt = _build_system_prompt()
            self._allowed_predicates = active_pack().predicate_families

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        text = (turn.text or "").strip()
        if not text:
            return []

        self._ensure_prompt()
        assert self._system_prompt is not None
        assert self._allowed_predicates is not None

        speaker_label = (
            turn.speaker.value if hasattr(turn.speaker, "value") else str(turn.speaker)
        )
        user_content = f"Current turn:\n    {speaker_label}: {text}"

        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "{}")
        except Exception as exc:
            log.warning("extractor.ollama_error", turn_id=turn.turn_id, err=str(exc)[:200])
            return []

        raw_claims = _parse_claims(content)
        return _to_extracted_claims(raw_claims, turn, self._allowed_predicates)

    @staticmethod
    def is_available() -> bool:
        """Check if Ollama is running and reachable."""
        import os
        base_url = os.environ.get("MEMCONTEXT_OLLAMA_URL", "http://localhost:11434")
        try:
            import requests
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PassthroughExtractor — MCP interactive path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SimpleExtractor — dev/test smoke only, NOT for benchmarks
# ---------------------------------------------------------------------------


class SimpleExtractor:
    """Local regex-only claim extractor. Dev/test fallback — NOT for production
    or benchmark use. Stores unmatched text as full-turn claims, which produces
    garbage retrieval context. Use LLMExtractor for benchmarks.
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

        if not claims:
            predicate = "user_fact" if "user_fact" in families else next(iter(families))
            claims.append(ExtractedClaim(
                subject="user",
                predicate=predicate,
                value=text,
                confidence=0.5,
            ))

        return claims


# ---------------------------------------------------------------------------
# Factory — auto-select best available extractor
# ---------------------------------------------------------------------------


def auto_extractor() -> PassthroughExtractor | LLMExtractor | SimpleExtractor:
    """Return the best available extractor for the current environment.

    Priority: LLMExtractor (Ollama running) > SimpleExtractor (fallback).
    PassthroughExtractor is not returned here — it requires explicit claims.
    """
    if LLMExtractor.is_available():
        log.info("extractor.auto_selected", extractor="LLMExtractor")
        return LLMExtractor()
    log.warning(
        "extractor.auto_fallback",
        extractor="SimpleExtractor",
        reason="Ollama not available at localhost:11434",
    )
    return SimpleExtractor()
