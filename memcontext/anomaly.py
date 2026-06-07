"""Embedding-based anomaly detection on writes (EXPERIMENTAL, flag-gated).

OFF by default (set ``MEMCONTEXT_EXPERIMENTAL_ANOMALY=1`` to enable). A write whose
embedding is a strong semantic outlier vs existing memory -- related to nothing
stored -- is a novelty/injection signal. Classified **PLAUSIBLE** (not proven) per
the Research Rule, so it is gated behind a flag and never changes default behavior;
when it fires it RECORDS an auditable anomaly event (it does not block the write).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

import structlog

log = structlog.get_logger()

EXPERIMENTAL_FLAG = "MEMCONTEXT_EXPERIMENTAL_ANOMALY"
# Max cosine similarity to any existing memory below this = anomalous (isolated/novel).
ANOMALY_SIM_THRESHOLD = 0.15


def anomaly_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def is_anomalous(
    new_text: str,
    existing_texts: list[str],
    embedder,
    threshold: float = ANOMALY_SIM_THRESHOLD,
) -> bool:
    """True if ``new_text`` is a strong semantic outlier vs ``existing_texts`` (its
    max cosine to any of them is below ``threshold``). Needs an embedder and at least
    one reference; otherwise returns False (cannot judge -> fail safe)."""
    if embedder is None or not new_text or not existing_texts:
        return False
    from memcontext.retrieval import _cosine_normalised

    vecs = embedder.embed([new_text, *existing_texts])
    new_vec = vecs[0]
    best = max((_cosine_normalised(new_vec, v) for v in vecs[1:]), default=1.0)
    return best < threshold


def record_anomaly(conn: sqlite3.Connection, session_id: str, text: str) -> str:
    """Audit an anomalous write to the decisions log (countable in trust_status)."""
    decision_id = f"dec_{uuid.uuid4().hex[:12]}"
    conn.execute(
        "INSERT INTO decisions (decision_id, session_id, kind, target_type, target_id,"
        " claim_state_snapshot, ts) VALUES (?, ?, 'anomaly_flagged', 'turn', '', ?, ?)",
        (decision_id, session_id, json.dumps({"text": text[:500]}), time.time_ns()),
    )
    log.info("substrate.anomaly_flagged", session_id=session_id)
    return decision_id


def check_write(conn: sqlite3.Connection, session_id: str, text: str, embedder) -> bool:
    """If enabled and a real embedder is present, flag+audit an anomalous write.
    No-op (returns False) when the flag is off or no embedder is available."""
    if not anomaly_enabled() or embedder is None:
        return False
    existing = [
        r[0] for r in conn.execute(
            "SELECT text FROM claims WHERE session_id = ?"
            " AND status IN ('active','confirmed') AND text != ? LIMIT 100",
            (session_id, text),
        ).fetchall()
    ]
    if is_anomalous(text, existing, embedder):
        record_anomaly(conn, session_id, text)
        return True
    return False
