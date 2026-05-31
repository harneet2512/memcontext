"""The hardcoded demo transcript and its deterministic, model-free seeding.

Scenario: a team states their database, then corrects it three turns later.
The correction is the differentiator — MemContext retains both values, marks the
old one superseded by a *typed* edge, and serves the current value with a
verifiable source span. A summary blob or a top-k vector dump cannot.

Seeding is fully deterministic and uses no model:
  * each turn is ingested through the real ``on_new_turn`` pipeline with a
    per-turn ``PassthroughExtractor`` (the structured-claims path the MCP client
    normally drives);
  * the Postgres -> DynamoDB correction is recorded with the *existing*
    ``write_supersession_edge`` + ``set_claim_status`` primitives — the same
    ones ``handle_memory_correct`` uses.

Why the correction is recorded explicitly rather than auto-detected: Pass-1
structural supersession (``supersession.detect_pass1``) only fires when the old
and new values share content tokens (jaccard >= 0.3) — so a single-slot *update*
like "Postgres 13" -> "Postgres 15" is caught, but lexically-disjoint
replacements like "Postgres" -> "DynamoDB" are left for Pass-2 *semantic*
supersession, which needs an embedding model. This demo takes no model in the
write path, so the cross-value correction is recorded as the explicit typed
event it is. The end state is identical to what the product produces.
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

SESSION_ID = "demo"
QUESTION = "What database do they use?"
DATABASE_SUBJECT = "main_database"

# Map the abstract predicate role used in the transcript to a real predicate in
# each supported pack, so the demo validates against either vocabulary.
_PACK_PREDICATES: dict[str, dict[str, str]] = {
    "developer": {"decision": "decision_made", "convention": "convention_established"},
    "general": {"decision": "user_fact", "convention": "user_fact"},
}

# The transcript. ``claims`` values are surface strings located in ``text`` via
# str.index, giving real character spans. ``corrects`` marks the turn whose
# main_database claim supersedes the earlier one.
TRANSCRIPT: list[dict] = [
    {
        "speaker": "user",
        "text": "We use Postgres for the main database.",
        "claims": [{"subject": DATABASE_SUBJECT, "role": "decision", "value": "Postgres", "confidence": 0.95}],
    },
    {
        "speaker": "user",
        "text": "We deploy on Fridays.",
        "claims": [{"subject": "deploy", "role": "convention", "value": "Fridays", "confidence": 0.9}],
    },
    {
        "speaker": "user",
        "text": "Standups are at 9am.",
        "claims": [],
    },
    {
        "speaker": "user",
        "text": "CI runs on GitHub Actions.",
        "claims": [{"subject": "ci", "role": "decision", "value": "GitHub Actions", "confidence": 0.9}],
    },
    {
        "speaker": "user",
        "text": "Actually we migrated off Postgres, we're on DynamoDB now.",
        "claims": [{"subject": DATABASE_SUBJECT, "role": "decision", "value": "DynamoDB", "confidence": 0.95}],
        "corrects": DATABASE_SUBJECT,
    },
]


def activate_pack(pack: str) -> None:
    """Make *pack* the active vocabulary (sets ACTIVE_PACK, clears the cache).

    NOTE: this mutates ``os.environ["ACTIVE_PACK"]`` process-wide so that
    ``validate_claim``/``brain`` read the demo's vocabulary. Callers that run
    inside a long-lived process (a serving MCP server) should scope it with
    ``pack_active`` instead, which restores the prior pack on exit.
    """
    if pack not in _PACK_PREDICATES:
        raise ValueError(
            f"Unsupported demo pack {pack!r}; use one of {sorted(_PACK_PREDICATES)}"
        )
    os.environ["ACTIVE_PACK"] = pack
    from memcontext.predicate_packs import active_pack

    active_pack.cache_clear()


@contextmanager
def pack_active(pack: str) -> Iterator[None]:
    """Activate *pack* for the duration of the block, then restore the prior pack.

    Use this around the whole demo (seed + reads), so the demo never leaves
    ``ACTIVE_PACK`` mutated in the calling process.
    """
    from memcontext.predicate_packs import active_pack

    prior = os.environ.get("ACTIVE_PACK")
    activate_pack(pack)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("ACTIVE_PACK", None)
        else:
            os.environ["ACTIVE_PACK"] = prior
        active_pack.cache_clear()


def seed_demo(
    conn: sqlite3.Connection,
    *,
    session_id: str = SESSION_ID,
    pack: str = "developer",
) -> dict:
    """Seed the demo transcript and record the typed correction. No model.

    Returns a manifest: session_id, pack, question, the database subject and its
    predicate, a {turn_id: "Turn N"} label map, and the Postgres/DynamoDB claim
    ids.
    """
    from memcontext.claims import get_claim, set_claim_status
    from memcontext.extractors import PassthroughExtractor
    from memcontext.on_new_turn import on_new_turn
    from memcontext.schema import ClaimStatus, EdgeType, Speaker
    from memcontext.supersession import write_supersession_edge

    activate_pack(pack)
    predicates = _PACK_PREDICATES[pack]

    labels: dict[str, str] = {}
    postgres_id: str | None = None
    dynamodb_id: str | None = None

    for ordinal, turn in enumerate(TRANSCRIPT, start=1):
        text = turn["text"]
        claims = []
        for c in turn["claims"]:
            value = c["value"]
            start = text.find(value)
            if start < 0:
                raise ValueError(
                    f"Demo transcript bug: value {value!r} is not a substring of "
                    f"turn text {text!r} (cannot compute a provenance span)."
                )
            claims.append({
                "subject": c["subject"],
                "predicate": predicates[c["role"]],
                "value": value,
                "confidence": c["confidence"],
                "char_start": start,
                "char_end": start + len(value),
            })
        speaker = Speaker.USER if turn["speaker"] == "user" else Speaker.ASSISTANT
        result = on_new_turn(
            conn,
            session_id=session_id,
            speaker=speaker,
            text=text,
            extractor=PassthroughExtractor(claims),
        )
        if result.turn is not None:
            labels[result.turn.turn_id] = f"Turn {ordinal}"
        for claim in result.created_claims:
            if claim.subject == DATABASE_SUBJECT:
                if postgres_id is None:
                    postgres_id = claim.claim_id
                else:
                    dynamodb_id = claim.claim_id

    # Record the explicit typed correction (idempotent: skip if already superseded).
    if postgres_id and dynamodb_id:
        old = get_claim(conn, postgres_id)
        if old is not None and old.status == ClaimStatus.ACTIVE:
            edge = write_supersession_edge(
                conn,
                old_claim_id=postgres_id,
                new_claim_id=dynamodb_id,
                edge_type=EdgeType.USER_CORRECTION,
                identity_score=None,
            )
            set_claim_status(conn, postgres_id, ClaimStatus.SUPERSEDED)
            # Close the temporal window on the superseded claim, exactly as
            # Pass-1 supersession (supersession.detect_pass1) does — so the
            # seeded state is indistinguishable from a real auto-supersession.
            conn.execute(
                "UPDATE claims SET valid_until_ts = ?"
                " WHERE claim_id = ? AND (valid_from_ts IS NULL OR valid_from_ts < ?)",
                (edge.created_ts, postgres_id, edge.created_ts),
            )

    return {
        "session_id": session_id,
        "pack": pack,
        "question": QUESTION,
        "subject": DATABASE_SUBJECT,
        "predicate": predicates["decision"],
        "turn_labels": labels,
        "postgres_claim_id": postgres_id,
        "dynamodb_claim_id": dynamodb_id,
    }
