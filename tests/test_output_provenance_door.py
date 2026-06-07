"""The memory_output_provenance door wires the previously-unreachable output
provenance functions: insert_output_sentence (write) + sentence_ids_for_claim,
claim_ids_for_turn, turn_id_for_sentence (read).
"""
from __future__ import annotations

import sqlite3

from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_output_provenance
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def test_output_provenance_records_and_traces():
    conn = _conn()
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I use Postgres for the project",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "postgres", "confidence": 0.9}]),
    )
    row = conn.execute("SELECT claim_id, source_turn_id FROM claims").fetchone()
    cid, turn = row["claim_id"], row["source_turn_id"]

    # WRITE: record a generated sentence citing the claim (insert_output_sentence)
    rec = handle_memory_output_provenance(
        conn, session_id="s1",
        record=[{"section": "summary", "text": "You use Postgres.", "source_claim_ids": [cid]}],
    )
    assert rec["recorded"], "output sentence recorded"
    sid = rec["recorded"][0]

    # READ: which sentences cite the claim (sentence_ids_for_claim)
    assert sid in handle_memory_output_provenance(conn, claim_id=cid)["cited_in"]
    # READ: claims from the turn (claim_ids_for_turn)
    assert cid in handle_memory_output_provenance(conn, turn_id=turn)["claims_from_turn"]
    # READ: the source turn of the sentence (turn_id_for_sentence)
    assert handle_memory_output_provenance(conn, sentence_id=sid)["turn_of_sentence"] == turn
