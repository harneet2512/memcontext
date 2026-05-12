"""Pluggable domain packs.

A pack declares the closed predicate vocabulary, structured sub-slot schemas,
few-shot examples, and (optionally) domain-specific config paths. Packs
live under a configurable directory and are loaded once at startup.

Controlled by env vars:
- ACTIVE_PACK (default: "general") — which pack to load
- SUBSTRATE_PACKS_DIR — override the packs directory (default: ./predicate_packs)

Cached via @lru_cache; call active_pack.cache_clear() in tests to switch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PACKS_DIR = _PACKAGE_DIR / "predicate_packs"


def _packs_dir() -> Path:
    raw = os.environ.get("SUBSTRATE_PACKS_DIR", "").strip()
    return Path(raw) if raw else _DEFAULT_PACKS_DIR


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One canonical prompt-response pair for in-context learning."""

    name: str
    scenario: str
    prior_turns: tuple[tuple[str, str], ...]
    current_turn: tuple[str, str]
    active_claims_summary: str
    expected_output: str


@dataclass(frozen=True, slots=True)
class PredicatePack:
    """A domain pack registered with the extractor.

    Packs that have domain-specific secondary data (e.g. classification tables)
    can set `aux_data_path`. Non-domain packs leave it as None.
    """

    pack_id: str
    predicate_families: frozenset[str]
    sub_slots: dict[str, frozenset[str]] = field(default_factory=dict)
    aux_data_path: Path | None = None
    few_shot_examples: tuple[FewShotExample, ...] = ()
    description: str = ""


def _parse_few_shot(raw: dict) -> FewShotExample:
    prior = tuple(
        (str(t.get("speaker", "")), str(t.get("text", "")))
        for t in raw.get("prior_turns", [])
    )
    cur = raw.get("current_turn", {})
    return FewShotExample(
        name=str(raw.get("name", "")),
        scenario=str(raw.get("scenario", "")),
        prior_turns=prior,
        current_turn=(str(cur.get("speaker", "")), str(cur.get("text", ""))),
        active_claims_summary=str(raw.get("active_claims_summary", "")),
        expected_output=str(raw.get("expected_output", "")),
    )


def load_pack(pack_dir: str | Path) -> PredicatePack:
    """Load a pack from a directory.

    Expected files:
    - predicates.json — pack_id, predicate_families, optional sub_slots, optional description
    - few_shot_examples.json — either {"examples": [...]} or a bare list
    """
    p = Path(pack_dir)
    if not p.is_absolute():
        p = _packs_dir() / pack_dir
    if not p.is_dir():
        raise FileNotFoundError(f"Pack directory not found: {p}")

    predicates_raw = json.loads((p / "predicates.json").read_text(encoding="utf-8"))
    examples_raw = json.loads((p / "few_shot_examples.json").read_text(encoding="utf-8"))

    pack_id = str(predicates_raw.get("pack_id", p.name))
    families = frozenset(predicates_raw["predicate_families"])
    sub_slots_raw = predicates_raw.get("sub_slots", {})
    sub_slots = {str(k): frozenset(v) for k, v in sub_slots_raw.items()}

    # Check for auxiliary data directories
    aux_data_path: Path | None = None
    for subdir_name in sorted(p.iterdir()):
        if subdir_name.is_dir():
            aux_data_path = subdir_name
            break

    example_entries = examples_raw.get("examples", examples_raw) if isinstance(examples_raw, dict) else examples_raw
    examples = tuple(_parse_few_shot(ex) for ex in example_entries)

    return PredicatePack(
        pack_id=pack_id,
        predicate_families=families,
        sub_slots=sub_slots,
        aux_data_path=aux_data_path,
        few_shot_examples=examples,
        description=str(predicates_raw.get("description", "")),
    )


def load_packs(pack_ids: list[str]) -> PredicatePack:
    """Load multiple packs and merge them.

    Merges predicate_families (union), concatenates few_shot_examples,
    merges sub_slots, and creates a composite pack_id.
    """
    if not pack_ids:
        raise ValueError("pack_ids must be non-empty")
    if len(pack_ids) == 1:
        return load_pack(pack_ids[0])

    packs = [load_pack(pid) for pid in pack_ids]
    merged_families = frozenset().union(*(p.predicate_families for p in packs))
    merged_sub_slots: dict[str, frozenset[str]] = {}
    for p in packs:
        merged_sub_slots.update(p.sub_slots)
    merged_examples: tuple[FewShotExample, ...] = ()
    for p in packs:
        merged_examples = merged_examples + p.few_shot_examples
    merged_id = "merged:" + "+".join(p.pack_id for p in packs)
    merged_desc = " | ".join(p.description for p in packs if p.description)

    return PredicatePack(
        pack_id=merged_id,
        predicate_families=merged_families,
        sub_slots=merged_sub_slots,
        few_shot_examples=merged_examples,
        description=merged_desc,
    )


@lru_cache(maxsize=None)
def active_pack() -> PredicatePack:
    """Return the active pack.

    Controlled by env var ACTIVE_PACK (default: "general"). Supports
    comma-separated pack IDs for composition (e.g. "general,developer").
    Cached — tests should call active_pack.cache_clear() after changing
    the env var.
    """
    pack_id = os.environ.get("ACTIVE_PACK", "general")
    if "," in pack_id:
        return load_packs([p.strip() for p in pack_id.split(",")])
    return load_pack(pack_id)
