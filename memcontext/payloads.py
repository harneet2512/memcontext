"""Three memory payloads for the same question - the apples-to-apples demo.

Hold the reader constant (the host model) and vary only what the server hands
back for one question:

- ``summary_payload``   - the raw/concatenated transcript blob. MemContext
  ships no summarizer; the host model does any summarizing. No truth-state, no
  provenance.
- ``vector_payload``    - top-k raw statements by cosine similarity, using the
  repo's existing local embedder. Both the old and new fact surface; similarity
  is not recency or truth.
- ``memcontext_payload``- the structured ``brain()`` projection plus an
  ``answer_support`` block: one current value with status, confidence, an exact
  source span, and the prior value linked by a typed supersession edge.

No model call lives here except the repo's existing *local* embedder used by
``vector_payload`` (forced local - no network). Everything else is deterministic.
"""
from __future__ import annotations

import os
import re
import sqlite3

from memcontext.brain import brain
from memcontext.chains import build_chain
from memcontext.claims import find_same_identity_claim, get_claim, get_turn


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _dot(a: list[float], b: list[float]) -> float:
    """Cosine for L2-normalised vectors reduces to the dot product."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _session_turns(conn: sqlite3.Connection, session_id: str) -> list:
    return conn.execute(
        "SELECT turn_id, speaker, text, ts FROM turns"
        " WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    ).fetchall()


# Cached local embedder — the underlying sentence-transformers model is loaded
# once (lazily, on first embed) and reused, so repeated vector_payload calls on
# the live MCP path don't reload the model each time.
_local_embedder = None


def _get_local_embedder():
    """Return a process-wide EmbeddingClient pinned to the local (no-network) path."""
    global _local_embedder
    if _local_embedder is None:
        from memcontext.retrieval import MODAL_URL_ENV, EmbeddingClient

        # Construct with the Modal URL cleared so the client never hits the network.
        prior_modal = os.environ.pop(MODAL_URL_ENV, None)
        try:
            _local_embedder = EmbeddingClient()
        finally:
            if prior_modal is not None:
                os.environ[MODAL_URL_ENV] = prior_modal
    return _local_embedder


def summary_payload(conn: sqlite3.Connection, session_id: str, question: str) -> dict:
    """The raw transcript blob - what a summary/raw-text memory hands back."""
    rows = _session_turns(conn, session_id)
    blob = "\n".join(
        f"Turn {i} ({r['speaker']}): {r['text']}" for i, r in enumerate(rows, 1)
    )
    return {
        "mode": "summary",
        "question": question,
        "payload": blob,
        "fields_present": ["raw_text"],
        "fields_absent": ["current_value", "status", "provenance", "supersession"],
        "note": (
            "Raw transcript blob - no current-value field, no per-fact "
            "provenance, no truth-state. A reader can paraphrase it but cannot "
            "say which value is current or cite where it came from."
        ),
    }


def vector_payload(
    conn: sqlite3.Connection, session_id: str, question: str, top_k: int = 3
) -> dict:
    """Top-k raw statements by cosine similarity (repo's local embedder)."""
    rows = _session_turns(conn, session_id)
    chunks = [{"turn_id": r["turn_id"], "text": r["text"]} for r in rows]
    if not chunks:
        return {"mode": "vector", "question": question, "retrieved": [], "note": "no turns"}

    try:
        vectors = _get_local_embedder().embed([question] + [c["text"] for c in chunks])
    except Exception as exc:  # embedder not installed / unavailable - degrade honestly
        return {
            "mode": "vector",
            "question": question,
            "retrieved": [],
            "error": (
                f"embedder unavailable ({type(exc).__name__}); install "
                "sentence-transformers to run the vector baseline"
            ),
            "note": (
                "The vector payload needs the repo's local embedder. The "
                "brain/trace/summary payloads still run fully offline."
            ),
        }

    qv = vectors[0]
    scored = [
        {
            "text": c["text"],
            "turn_id": c["turn_id"],
            "similarity": round(_dot(qv, v), 4),
        }
        for c, v in zip(chunks, vectors[1:], strict=True)
    ]
    scored.sort(key=lambda x: -x["similarity"])
    return {
        "mode": "vector",
        "question": question,
        "retrieved": scored[:top_k],
        "fields_present": ["text", "similarity"],
        "fields_absent": ["current_value", "status", "provenance", "supersession"],
        "note": (
            "Top-k raw statements by cosine similarity. Both the old and new "
            "fact surface; similarity is not recency or truth. No supersession "
            "edge and no current-value field - the reader sees the two facts "
            "with equal standing and cannot tell which is current."
        ),
    }


def _answer_support(conn: sqlite3.Connection, ws: dict, question: str) -> dict | None:
    """Deterministically pick the active fact most relevant to *question*.

    Transparent lexical overlap between the question tokens and each fact's
    (subject, predicate, value) tokens - no model. Returns the current value
    with provenance and the superseded chain beneath it.
    """
    qtok = _tokens(question)
    best: tuple[str, dict] | None = None
    best_score = 0
    for subject, block in ws.get("subjects", {}).items():
        subject_tokens = set(subject.split("_"))
        for fact in block.get("facts", []):
            cand = qtok & (
                subject_tokens | _tokens(fact["predicate"]) | _tokens(fact["value"])
            )
            if len(cand) > best_score:
                best_score = len(cand)
                best = (subject, fact)

    if best is None or best_score == 0:
        return None

    subject, fact = best
    head = find_same_identity_claim(
        conn, session_id=ws["session_id"], subject=subject, predicate=fact["predicate"]
    )
    superseded: list[dict] = []
    if head is not None:
        for step in build_chain(conn, head.claim_id):
            if step.claim_id == head.claim_id:
                continue
            step_turn = get_turn(conn, step.source_turn_id)
            step_claim = get_claim(conn, step.claim_id)
            quote = (
                step_turn.text[step_claim.char_start : step_claim.char_end]
                if step_turn
                and step_claim
                and step_claim.char_start is not None
                and step_claim.char_end is not None
                else None
            )
            superseded.append({
                "value": step.value,
                "edge_type": step.edge_type,
                "source_turn_id": step.source_turn_id,
                "quote": quote,
            })

    return {
        "subject": subject,
        "predicate": fact["predicate"],
        "current_value": fact["value"],
        "status": fact["status"],
        "confidence": fact["confidence"],
        "provenance": fact["provenance"],
        "superseded": superseded,
    }


def memcontext_payload(conn: sqlite3.Connection, session_id: str, question: str) -> dict:
    """The structured projection - current value + provenance + typed lineage."""
    ws = brain(conn, session_id=session_id)
    return {
        "mode": "memcontext",
        "question": question,
        "world_state": ws,
        "answer_support": _answer_support(conn, ws, question),
        "note": (
            "Structured projection: a single current value per slot with status, "
            "confidence, exact source span, and the prior value linked by a typed "
            "supersession edge. The reader can state the current value and cite "
            "the exact turn and span."
        ),
    }
