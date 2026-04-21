"""Memory substrate — claim lifecycle, supersession, projections, retrieval.

A domain-agnostic memory system that stores claims (subject-predicate-value
triples) extracted from conversation turns, tracks how facts evolve over
time via typed supersession edges, maintains active-state projections, and
supports multi-signal retrieval.

Public API:

- Claim, Turn, SupersessionEdge, OutputSentence, EdgeType, ClaimStatus,
  Speaker, OutputSection — the typed data model.
- open_database(path) — initialise a SQLite connection with schema.
- claims.insert_claim, claims.list_active_claims, etc. — CRUD.
- supersession.detect_pass1(...) — deterministic structural supersession.
- supersession_semantic.SemanticSupersession — Pass 2 with pluggable embedder.
- projections.rebuild_active_projection, projections.filtered_projection.
- admission.admit(...) — noise-regex admission filter.
- provenance.* — forward/back-link utilities.
- event_bus.EventBus — synchronous in-memory pub/sub.
- on_new_turn.on_new_turn(...) — orchestrator entry point.
- retrieval.* — embedding-based and hybrid retrieval.
"""
from __future__ import annotations

from memcontext.schema import (
    Claim,
    ClaimStatus,
    EdgeType,
    OutputSection,
    OutputSentence,
    Speaker,
    SupersessionEdge,
    Turn,
    open_database,
)

__all__ = [
    "Claim",
    "ClaimStatus",
    "EdgeType",
    "OutputSection",
    "OutputSentence",
    "Speaker",
    "SupersessionEdge",
    "Turn",
    "open_database",
]
