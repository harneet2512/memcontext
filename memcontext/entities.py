"""Lightweight entity extraction — regex-based, zero dependencies beyond stdlib."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_LOCATION_RE = re.compile(
    r"(?:in|at|to|from|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)
_ORG_SUFFIX_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
    r"(?:Inc|Corp|LLC|Ltd|Co|Foundation|University|Institute|Hospital|Group)\b"
)

_STOP_WORDS = frozenset({
    "I", "The", "This", "That", "These", "Those", "My", "Your",
    "His", "Her", "Its", "Our", "Their", "We", "He", "She", "It",
    "Not", "But", "And", "Or", "So", "If", "When", "Then",
    "Just", "Also", "Actually", "Really", "Very", "Well",
    "Yes", "No", "Ok", "Okay", "Sure", "Thanks", "Thank",
})


@dataclass(frozen=True, slots=True)
class Entity:
    text: str
    entity_type: str


def extract_entities(text: str) -> list[Entity]:
    """Extract named entities from text using regex heuristics."""
    entities: list[Entity] = []
    seen: set[str] = set()

    for m in _ORG_SUFFIX_RE.finditer(text):
        full = m.group(0).strip()
        if full not in seen:
            entities.append(Entity(text=full, entity_type="organization"))
            seen.add(full)

    for m in _LOCATION_RE.finditer(text):
        name = m.group(1).strip()
        if name not in seen and name not in _STOP_WORDS:
            entities.append(Entity(text=name, entity_type="location"))
            seen.add(name)

    for m in _PROPER_NOUN_RE.finditer(text):
        name = m.group(1).strip()
        if name not in seen and name not in _STOP_WORDS and len(name) > 1:
            entities.append(Entity(text=name, entity_type="proper_noun"))
            seen.add(name)

    return entities
