"""GAP-9 (HAR-95): in lexical mode, always-tied channels reward insertion order.

`_rrf_ranks` breaks ties by input position (pinned by test_retrieval.py::
test_rrf_ranks_ties). Several hybrid channels are nearly always tied for every
claim (confidence, usage, frequency, source trust, importance), so each one hands
older claims a better rank. With no embedder, those channels can outweigh BM25: a
claim that matches the query loses to older claims that match nothing. Claims
created in the same millisecond also come back in random-id order, so a claim at
the cut boundary is served in some runs and not others (the HAR-95 state question
served Acme's current status in ~3/4 runs on master).

Sharing ranks between ties fixes this test but re-ranks every query on every
channel (12 other tests move, incl. the pinned tie test), so it needs a ranking
decision and a benchmark re-run, not a drive-by fix.
"""
from __future__ import annotations

import pytest

from memcontext.mcp_tools import handle_memory_store
from memcontext.retrieval import retrieve_hybrid
from memcontext.schema import open_database


@pytest.mark.xfail(strict=True, reason="GAP-9: tied channels rank by insertion order and can outweigh BM25")
def test_a_matching_claim_outranks_older_non_matching_ones():
    conn = open_database(":memory:")
    for i, note in enumerate([
        "moved the quarterly review to next Tuesday",
        "reported a billing discrepancy on the March invoice",
        "wants a demo of the analytics add-on",
        "is migrating their SSO provider next month",
    ]):
        handle_memory_store(conn, text=f"Account {i} {note}.", session_id="s",
                            claims=[{"subject": f"account{i}", "predicate": "observation",
                                     "value": note}])
    handle_memory_store(conn, text="Acme renewal status is confirmed.", session_id="s",
                        claims=[{"subject": "acme", "predicate": "observation",
                                 "value": "renewal status confirmed"}])
    hits = retrieve_hybrid(conn, session_id="s", query="What is Acme's current renewal status?",
                           top_k=5)
    assert hits[0][0].subject == "acme"
