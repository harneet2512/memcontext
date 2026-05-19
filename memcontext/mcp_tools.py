"""MCP tool handler functions — pure business logic, no MCP protocol dependency.

Each function takes a sqlite3 Connection and keyword arguments, returns a dict.
These are usable from CLI, tests, or the MCP server without importing mcp.
"""
from __future__ import annotations

import json
import sqlite3
import uuid

from memcontext.claims import (
    get_claim,
    get_superseded_by,
    get_turn,
    insert_claim,
    list_active_claims,
    set_claim_status,
)
from memcontext.extractors import PassthroughExtractor, SimpleExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.provenance import span_for_claim
from memcontext.schema import ClaimStatus, EdgeType, Speaker
from memcontext.supersession import write_supersession_edge


def handle_memory_store(
    conn: sqlite3.Connection,
    *,
    text: str,
    speaker: str = "user",
    session_id: str | None = None,
    claims: list[dict] | None = None,
) -> dict:
    sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT

    if claims:
        extractor = PassthroughExtractor(claims)
    else:
        extractor = SimpleExtractor()

    result = on_new_turn(conn, session_id=sid, speaker=sp, text=text, extractor=extractor)

    return {
        "turn_id": result.turn.turn_id if result.turn else None,
        "session_id": sid,
        "admitted": result.admitted,
        "claims_created": len(result.created_claims),
        "claim_ids": [c.claim_id for c in result.created_claims],
        "supersessions": len(result.supersession_edges),
    }


def handle_memory_query(
    conn: sqlite3.Connection,
    *,
    query: str,
    session_id: str = "default",
    top_k: int = 10,
) -> dict:
    active = list_active_claims(conn, session_id)
    if not active:
        return {"claims": [], "total": 0}

    query_tokens = set(query.lower().split())
    scored = []
    for claim in active:
        claim_text = f"{claim.subject} {claim.predicate} {claim.value}".lower()
        claim_tokens = set(claim_text.split())
        overlap = len(query_tokens & claim_tokens)
        score = overlap / max(len(query_tokens), 1)
        scored.append((claim, score))

    scored.sort(key=lambda x: (-x[1], x[0].created_ts))
    top = scored[:top_k]

    return {
        "claims": [
            {
                "claim_id": c.claim_id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "confidence": c.confidence,
                "status": c.status.value,
                "score": round(s, 4),
            }
            for c, s in top
        ],
        "total": len(active),
    }


def handle_memory_trace(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
) -> dict:
    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"error": f"Claim {claim_id} not found"}

    source_turn = get_turn(conn, claim.source_turn_id)
    span = span_for_claim(conn, claim_id)

    # Walk supersession chain
    chain = []
    current_id = claim_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        next_id = get_superseded_by(conn, current_id)
        if next_id:
            chain.append({"from": current_id, "to": next_id})
        current_id = next_id

    return {
        "claim": {
            "claim_id": claim.claim_id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "confidence": claim.confidence,
            "status": claim.status.value,
        },
        "source_turn": {
            "turn_id": source_turn.turn_id,
            "speaker": source_turn.speaker.value,
            "text": source_turn.text,
        } if source_turn else None,
        "char_span": {
            "start": span.char_start,
            "end": span.char_end,
        } if span and span.char_start is not None else None,
        "supersession_chain": chain,
    }


def handle_memory_correct(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    action: str,
    new_value: str | None = None,
) -> dict:
    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"error": f"Claim {claim_id} not found"}

    if action == "dismiss":
        set_claim_status(conn, claim_id, ClaimStatus.DISMISSED)
        return {
            "action": "dismissed",
            "claim_id": claim_id,
            "status": "dismissed",
        }

    if action == "correct":
        if not new_value:
            return {"error": "new_value is required for correction"}

        new_claim = insert_claim(
            conn,
            session_id=claim.session_id,
            subject=claim.subject,
            predicate=claim.predicate,
            value=new_value,
            confidence=1.0,
            source_turn_id=claim.source_turn_id,
        )
        edge = write_supersession_edge(
            conn,
            old_claim_id=claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.USER_CORRECTION,
            identity_score=None,
        )
        set_claim_status(conn, claim_id, ClaimStatus.SUPERSEDED)

        return {
            "action": "corrected",
            "old_claim_id": claim_id,
            "new_claim_id": new_claim.claim_id,
            "edge_id": edge.edge_id,
            "new_value": new_value,
        }

    return {"error": f"Unknown action: {action}"}


def handle_memory_observe(
    conn: sqlite3.Connection,
    *,
    url: str,
    title: str = "",
    accessibility_tree: dict | None = None,
    session_id: str | None = None,
) -> dict:
    """Store browser observation claims from a page snapshot."""
    from datetime import datetime, timezone

    from memcontext.observe.browser import PageSnapshot, observe_page

    sid = session_id or f"observe_{uuid.uuid4().hex[:8]}"
    snapshot = PageSnapshot(
        url=url,
        title=title,
        timestamp=datetime.now(timezone.utc).isoformat(),
        accessibility_tree=accessibility_tree or {},
    )
    result = observe_page(conn, snapshot=snapshot, session_id=sid)
    return {
        "session_id": sid,
        "turn_id": result.turn_id,
        "claims_stored": len(result.claims),
        "claims": [
            {"subject": c.get("subject", ""), "predicate": c.get("predicate", ""), "value": c.get("value", "")}
            for c in result.claims
        ],
        "snapshot_id": snapshot.snapshot_id,
    }
