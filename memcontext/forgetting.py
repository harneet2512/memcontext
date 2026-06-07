"""Cascade-consistent deletion / forgetting (GDPR/HIPAA right-to-be-forgotten).

Unlike supersession (which RETAINS, marking superseded), ``forget()`` HARD-DELETES
the target claims and cascades along the provenance + supersession graph so no
residual content lingers in embeddings, summaries, or derived structures — the
deletion test most RAG systems fail. Every forget writes a verifiable audit row to
``decisions`` first. Deterministic, zero-LLM.

Coverage:
- FK ``ON DELETE CASCADE`` removes claim_embeddings, claim_metadata, claim_entities,
  supersession_edges, event_frame_claims (on claim delete) and turn_embeddings (on
  turn delete).
- JSON-linked derived tables (session_digests, life_events, event_frames) are
  rebuilt on demand, so any row referencing a forgotten claim is deleted (clean
  rebuild). output_sentences are stripped (claim removed; row deleted if no source
  remains).
- A source turn is deleted only when it has NO surviving claims, so episode-only
  Tier-1 turns and turns shared across subjects are preserved.

Known limitation (documented, not hidden): if a forgotten subject shares a turn
with a surviving claim, that turn's raw text is retained (turn-text redaction is
out of scope for v1; the structured claims are fully removed).
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

import structlog

log = structlog.get_logger()

# (table, json_claim_id_column, primary_key) — rebuilt-on-demand summaries
_SUMMARY_TABLES = (
    ("session_digests", "source_claim_ids", "session_id"),
    ("life_events", "claim_ids", "event_id"),
    ("event_frames", "supporting_claim_ids", "event_id"),
)


def _resolve_targets(
    conn: sqlite3.Connection, *, claim_id, subject, session_id, predicate
) -> tuple[list[str], tuple[str, str]]:
    """Resolve the set of claim_ids to forget + the (target_type, target_id) audit key.
    Forgets ALL statuses (active AND superseded) — forgetting means forgetting."""
    if claim_id:
        return [claim_id], ("claim", claim_id)
    if subject:
        rows = conn.execute("SELECT claim_id FROM claims WHERE subject = ?", (subject,)).fetchall()
        return [r[0] for r in rows], ("subject", subject)
    if session_id:
        rows = conn.execute("SELECT claim_id FROM claims WHERE session_id = ?", (session_id,)).fetchall()
        return [r[0] for r in rows], ("session", session_id)
    if predicate:
        rows = conn.execute("SELECT claim_id FROM claims WHERE predicate = ?", (predicate,)).fetchall()
        return [r[0] for r in rows], ("predicate", predicate)
    return [], ("none", "")


def forget(
    conn: sqlite3.Connection,
    *,
    claim_id: str | None = None,
    subject: str | None = None,
    session_id: str | None = None,
    predicate: str | None = None,
    reason: str = "user_request",
) -> dict:
    """Hard-delete the target claims + cascade; audited and verifiable. Returns a
    manifest including the ``decision_id`` proving what was removed."""
    targets, (target_type, target_id) = _resolve_targets(
        conn, claim_id=claim_id, subject=subject, session_id=session_id, predicate=predicate
    )
    if not targets:
        return {"forgotten": 0, "claim_ids": [], "decision_id": None}

    ph = ",".join("?" for _ in targets)
    tset = set(targets)

    # --- verifiable audit: snapshot the targets into the decisions log FIRST ---
    snap_rows = conn.execute(
        f"SELECT claim_id, session_id, subject, predicate, value, text, status"
        f" FROM claims WHERE claim_id IN ({ph})", tuple(targets),
    ).fetchall()
    snapshot = [
        {"claim_id": x[0], "session_id": x[1], "subject": x[2], "predicate": x[3],
         "value": x[4], "text": x[5], "status": x[6]}
        for x in snap_rows
    ]
    sid = snapshot[0]["session_id"] if snapshot else (session_id or "unknown")
    turn_ids = [
        x[0] for x in conn.execute(
            f"SELECT DISTINCT source_turn_id FROM claims WHERE claim_id IN ({ph})", tuple(targets)
        ).fetchall()
    ]
    decision_id = f"dec_{uuid.uuid4().hex[:12]}"
    conn.execute(
        "INSERT INTO decisions (decision_id, session_id, kind, target_type, target_id,"
        " claim_state_snapshot, ts) VALUES (?, ?, 'forget', ?, ?, ?, ?)",
        (decision_id, sid, target_type, target_id,
         json.dumps(snapshot, ensure_ascii=False), time.time_ns()),
    )

    # --- strip JSON-linked derived (no FK) BEFORE the claims vanish ---
    for table, col, pk in _SUMMARY_TABLES:
        for pk_val, ids_json in conn.execute(f"SELECT {pk}, {col} FROM {table}").fetchall():
            try:
                ids = set(json.loads(ids_json or "[]"))
            except (TypeError, json.JSONDecodeError):
                ids = set()
            if ids & tset:
                conn.execute(f"DELETE FROM {table} WHERE {pk} = ?", (pk_val,))
    for sentence_id, scids in conn.execute(
        "SELECT sentence_id, source_claim_ids FROM output_sentences"
    ).fetchall():
        ids = json.loads(scids or "[]")
        kept = [c for c in ids if c not in tset]
        if not kept:
            conn.execute("DELETE FROM output_sentences WHERE sentence_id = ?", (sentence_id,))
        elif len(kept) != len(ids):
            conn.execute(
                "UPDATE output_sentences SET source_claim_ids = ? WHERE sentence_id = ?",
                (json.dumps(kept), sentence_id),
            )

    # --- delete the claims -> FK cascade (embeddings/metadata/entities/edges/frame_claims) ---
    conn.execute(f"DELETE FROM claims WHERE claim_id IN ({ph})", tuple(targets))

    # --- delete now-orphaned source turns (no surviving claims) -> cascades turn_embeddings ---
    for tid in turn_ids:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE source_turn_id = ?", (tid,)
        ).fetchone()[0]
        if remaining == 0:
            conn.execute("DELETE FROM turns WHERE turn_id = ?", (tid,))

    log.info("substrate.forgotten", target_type=target_type, target_id=target_id,
             count=len(targets), decision_id=decision_id, reason=reason)
    return {"forgotten": len(targets), "claim_ids": list(targets), "decision_id": decision_id}
