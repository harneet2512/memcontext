"""Session digests — deterministic, zero LLM."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class SessionDigest:
    session_id: str
    key_facts: list[dict[str, Any]] = field(default_factory=list)  # top 3 by importance
    updates: list[dict[str, Any]] = field(default_factory=list)    # supersession events
    remaining_count: int = 0
    total_claims: int = 0


def build_session_digest(
    conn: sqlite3.Connection, session_id: str,
) -> SessionDigest:
    """Build a digest for a single session.

    1. Key facts — top 3 active claims by importance_score.
    2. Updates — claims that are the *new* side of a supersession edge.
    3. Remaining — everything else.
    """
    digest = SessionDigest(session_id=session_id)

    # All active claims for this session, joined with importance
    rows = conn.execute(
        """
        SELECT c.*, COALESCE(m.importance_score, 0.5) AS importance_score
        FROM claims c
        LEFT JOIN claim_metadata m ON c.claim_id = m.claim_id
        WHERE c.session_id = ?
          AND c.status IN ('active', 'confirmed', 'audited')
        ORDER BY COALESCE(m.importance_score, 0.5) DESC, c.created_ts ASC
        """,
        (session_id,),
    ).fetchall()

    digest.total_claims = len(rows)
    if not rows:
        log.debug("digests.no_claims", session_id=session_id)
        return digest

    # ---- Key facts: top 3 by importance ----
    featured_ids: set[str] = set()
    for r in rows[:3]:
        digest.key_facts.append(
            {
                "claim_id": r["claim_id"],
                "predicate": r["predicate"],
                "value": r["value"],
                # NL form (always present) — lets NL-only facts (empty triple)
                # surface in the digest instead of rendering as garbage.
                "text": r["text"] if "text" in r.keys() else None,
                "importance": r["importance_score"],
            }
        )
        featured_ids.add(r["claim_id"])

    # ---- Updates: claims that are new_claim_id in a supersession edge ----
    update_rows = conn.execute(
        """
        SELECT e.edge_id, e.old_claim_id, e.new_claim_id, e.edge_type,
               c_new.predicate, c_new.value AS new_value, c_new.text AS new_text,
               c_old.value AS old_value
        FROM supersession_edges e
        JOIN claims c_new ON e.new_claim_id = c_new.claim_id
        JOIN claims c_old ON e.old_claim_id = c_old.claim_id
        WHERE c_new.session_id = ?
          AND c_new.status IN ('active', 'confirmed', 'audited')
        ORDER BY e.created_ts DESC
        """,
        (session_id,),
    ).fetchall()

    for ur in update_rows:
        cid = ur["new_claim_id"]
        if cid in featured_ids:
            continue
        digest.updates.append(
            {
                "claim_id": cid,
                "predicate": ur["predicate"],
                "new_value": ur["new_value"],
                "new_text": ur["new_text"] if "new_text" in ur.keys() else None,
                "old_value": ur["old_value"],
                "edge_type": ur["edge_type"],
            }
        )
        featured_ids.add(cid)

    # ---- Remaining ----
    digest.remaining_count = sum(
        1 for r in rows if r["claim_id"] not in featured_ids
    )

    log.info(
        "digests.built",
        session_id=session_id,
        key_facts=len(digest.key_facts),
        updates=len(digest.updates),
        remaining=digest.remaining_count,
        total=digest.total_claims,
    )
    return digest


# ---------------------------------------------------------- persistence ---


def store_digest(conn: sqlite3.Connection, digest: SessionDigest) -> None:
    """Persist a SessionDigest into the ``session_digests`` table."""
    digest_text = format_digest(digest)

    all_claim_ids: list[str] = []
    for kf in digest.key_facts:
        all_claim_ids.append(kf["claim_id"])
    for upd in digest.updates:
        all_claim_ids.append(upd["claim_id"])

    digest_data = json.dumps(
        {
            "session_id": digest.session_id,
            "key_facts": digest.key_facts,
            "updates": digest.updates,
            "remaining_count": digest.remaining_count,
            "total_claims": digest.total_claims,
            "all_claim_ids": all_claim_ids,
        },
        ensure_ascii=False,
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO session_digests
            (session_id, digest_text, digest_data, claim_count, built_at_ts, source_claim_ids)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            digest.session_id,
            digest_text,
            digest_data,
            digest.total_claims,
            time.time_ns(),
            json.dumps(all_claim_ids, ensure_ascii=False),
        ),
    )
    log.info("digests.stored", session_id=digest.session_id)


def load_digest(
    conn: sqlite3.Connection, session_id: str,
) -> SessionDigest | None:
    """Load a SessionDigest from the ``session_digests`` table, or None."""
    row = conn.execute(
        "SELECT * FROM session_digests WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return None

    data = json.loads(row["digest_data"])
    digest = SessionDigest(
        session_id=data["session_id"],
        key_facts=data.get("key_facts", []),
        updates=data.get("updates", []),
        remaining_count=data.get("remaining_count", 0),
        total_claims=data.get("total_claims", row["claim_count"]),
    )
    return digest


def format_digest(digest: SessionDigest) -> str:
    """Format a SessionDigest as human-readable text.

    Example::

        [SESSION sess_abc123] 15 claims
          Key: [user_fact] works at Google (importance: 0.92)
          Key: [user_preference] prefers dark mode (importance: 0.85)
          Update: [user_fact] location: SF (was: Portland, via user_correction)
          + 12 more claims
    """
    header = f"[SESSION {digest.session_id}] {digest.total_claims} claims"
    lines = [header]

    for kf in digest.key_facts:
        # Structured facts render as the triple; NL-only facts (empty triple)
        # render their NL text so they are not dropped from the digest.
        body = f"[{kf['predicate']}] {kf['value']}" if kf.get("predicate") else kf.get("text", "")
        lines.append(f"  Key: {body} (importance: {kf['importance']:.2f})")

    for upd in digest.updates:
        new = f"[{upd['predicate']}] {upd['new_value']}" if upd.get("predicate") else upd.get("new_text", "")
        lines.append(
            f"  Update: {new}"
            f" (was: {upd['old_value']}, via {upd['edge_type']})"
        )

    if digest.remaining_count > 0:
        lines.append(f"  + {digest.remaining_count} more claims")

    return "\n".join(lines)
