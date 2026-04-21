"""Admission filter — noise-regex only.

Rejects turns that are obviously non-content:
- fewer than 3 content words
- only filler tokens ("uh", "um", "mhm", "ok", "right", ...)
- silence / system markers (e.g. "[silence]", "[noise]")
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


_FILLER_WORDS: frozenset[str] = frozenset({
    "uh", "um", "er", "hmm", "mhm", "mm", "ah",
    "ok", "okay", "yeah", "yes", "no", "right",
    "well", "so", "like",
})

_SILENCE_MARKER = re.compile(r"^\s*\[(?:silence|noise|pause|inaudible|music)\]\s*$", re.I)
_WORD_RE = re.compile(r"[A-Za-z']+")

MIN_CONTENT_WORDS = 3


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admitted: bool
    reason: str


def admit(text: str) -> AdmissionResult:
    """Decide whether a raw turn is worth sending to extraction."""
    if not text or not text.strip():
        return AdmissionResult(False, "empty")
    if _SILENCE_MARKER.match(text):
        return AdmissionResult(False, "silence_marker")

    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    if not tokens:
        return AdmissionResult(False, "no_word_tokens")
    content = [t for t in tokens if t not in _FILLER_WORDS]
    if len(content) < MIN_CONTENT_WORDS:
        return AdmissionResult(False, f"only_{len(content)}_content_words")
    return AdmissionResult(True, "admitted")
