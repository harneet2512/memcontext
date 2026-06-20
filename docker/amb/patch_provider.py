#!/usr/bin/env python3
"""Faithful-wiring patch for the MemContext AMB provider (Finding 3 of the
legitimacy audit).

The frozen provider on `benchmark/amb` calls `on_new_turn(...)` WITHOUT the
`embedder=` and `semantic=` collaborators that master's own real ingest paths
pass (`cli.py:89`, `mcp_tools.py:61`). Omitting them silently disables two
shipping capabilities — Pass-2 semantic supersession and the synchronous
episode-embedding floor — so the benchmark would measure a degraded product.

This patches OUR adapter (provider.py), not any Agent-Memory-Benchmark code, to
wire those collaborators exactly as the product does. It asserts every edit so
the build fails loudly if the upstream provider drifts (no silent no-op).

Usage (run against the exported harness inside the image):
    python patch_provider.py /opt/harness/evals/amb/provider.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# (anchor, replacement, human label) — every anchor MUST be present exactly once.
EDITS = [
    (
        "from memcontext.retrieval import EmbeddingClient, backfill_embeddings, retrieve_hybrid",
        "from memcontext.retrieval import (\n"
        "    EmbeddingClient,\n"
        "    backfill_embeddings,\n"
        "    classify_query_depth,\n"
        "    detect_history_intent,\n"
        "    episode_embedder,\n"
        "    retrieve_hybrid,\n"
        "    retrieve_memory_across,\n"
        "    semantic_supersession,\n"
        ")",
        "import the product's serve-door collaborators: query-aware depth + history-intent + episode_embedder + semantic_supersession + retrieve_memory_across",
    ),
    (
        "        self._extractor = None\n"
        '        self._db_path: str = ":memory:"',
        "        self._extractor = None\n"
        '        self._db_path: str = ":memory:"\n'
        "        # Per-conversation session model: each AMB Document is its OWN\n"
        "        # session (id = amb_{doc.id}), mirroring the product's real\n"
        "        # multi-session store. Track which sessions belong to each user so\n"
        "        # retrieve() can fan out across them via retrieve_memory_across,\n"
        "        # exactly as mcp_tools.handle_memory_query does for a multi-session\n"
        "        # query (mcp_tools.py:156). dedup + insertion order preserved.\n"
        "        self._sessions_by_user: dict[str | None, list[str]] = {}\n"
        "        # Per-session ISO-8601 date captured from the AMB Document.timestamp\n"
        "        # field (longmemeval's per-session haystack_date). The product's\n"
        "        # native harness prefixes each served episode with its date + a\n"
        "        # relative offset to the query date so the reader can do temporal\n"
        "        # reasoning; we mirror that by riding the date INSIDE Document.content\n"
        "        # (AMB's frozen rag.py renders content unchanged).\n"
        "        self._dates_by_session: dict[str, str] = {}",
        "track per-user session ids + per-session ISO date on the provider instance",
    ),
    (
        "        if reset:\n"
        "            self.cleanup()\n"
        "            self._conn = open_database(self._db_path)\n"
        "            self._conn.row_factory = sqlite3.Row",
        "        if reset:\n"
        "            self.cleanup()\n"
        "            self._sessions_by_user = {}\n"
        "            self._dates_by_session = {}\n"
        "            self._conn = open_database(self._db_path)\n"
        "            self._conn.row_factory = sqlite3.Row",
        "reset tracked sessions + per-session dates alongside the connection on prepare(reset=True)",
    ),
    (
        '''        extractor = self._extractor
        all_work: list[tuple[str, str, str]] = []

        first_user_id = documents[0].user_id if documents else "default"
        unified_session = f"amb_{first_user_id}"

        for doc in documents:
            turns = _parse_document_turns(doc)
            for role, text in turns:
                all_work.append((unified_session, role, text))

        extracted: list[tuple[str, str, str, list[dict]]] = []

        def _extract_one(item: tuple[str, str, str]) -> tuple[str, str, str, list[dict]]:
            sid, role, text = item
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            claims = _extract_claims(extractor, sid, sp, text)
            return (sid, role, text, claims)

        _workers = int(os.environ.get("MEMCONTEXT_EXTRACTION_WORKERS", "32"))
        with ThreadPoolExecutor(max_workers=_workers) as pool:
            futures = {pool.submit(_extract_one, w): w for w in all_work}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    extracted.append(fut.result())
                except Exception:
                    w = futures[fut]
                    extracted.append((w[0], w[1], w[2], []))
                if done % 100 == 0:
                    logger.info(f"Extracted {done}/{len(all_work)} turns")

        if done > 0:
            logger.info(f"Extracted {done}/{len(all_work)} turns")

        by_session: dict[str, list[tuple[str, str, list[dict]]]] = {}
        for sid, role, text, claims_data in extracted:
            by_session.setdefault(sid, []).append((role, text, claims_data))

        for sid in sorted(by_session.keys()):
            for role, text, claims_data in by_session[sid]:
                if not claims_data:
                    claims_data = [{
                        "subject": "user" if role == "user" else "assistant",
                        "predicate": "user_fact",
                        "value": text[:500],
                        "confidence": 0.3,
                    }]

                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
                pt = PassthroughExtractor(claims_data)
                on_new_turn(
                    conn,
                    session_id=sid,
                    speaker=sp,
                    text=text,
                    extractor=pt,
                )

        backfill_embeddings(conn, unified_session, client=self._embedding_client)''',
        '''        # FAITHFUL + PARALLEL INGEST. Restores the parallelism the benchmark
        # needs (the serial queue+drain path TIMED OUT CI) WITHOUT losing fidelity:
        # parallel LLM extract WITH set_context (per-worker extractor) + serial
        # deterministic insert. Proven BYTE-IDENTICAL to the serial faithful path
        # (same claims + supersession edges) in tests/test_parallel_faithful_ingest.py.
        import threading
        from concurrent.futures import ThreadPoolExecutor as _Pool
        from memcontext.extraction_queue import InlineQueue
        from memcontext.on_new_turn import run_extraction

        _epi = episode_embedder()
        _sem = semantic_supersession()
        # InlineQueue here only makes on_new_turn DEFER (store + embed, no inline
        # extract); we never drain it -- extraction runs in our pool below.
        _store_q = InlineQueue(conn, extractor=self._extractor, semantic=_sem)

        # Phase 1 -- store every turn (Tier-1: insert + embed; extraction deferred).
        batch_sessions: list[str] = []
        stored: list[Turn] = []
        for doc in documents:
            sid = f"amb_{doc.id}"
            user_sessions = self._sessions_by_user.setdefault(doc.user_id, [])
            if sid not in user_sessions:
                user_sessions.append(sid)
            if sid not in batch_sessions:
                batch_sessions.append(sid)
            _ts = getattr(doc, "timestamp", None)
            if _ts:
                self._dates_by_session[sid] = _ts
            for role, text in _parse_document_turns(doc):
                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
                _r = on_new_turn(
                    conn, session_id=sid, speaker=sp, text=text,
                    extractor=self._extractor, queue=_store_q, embedder=_epi,
                )
                if _r.turn is not None:
                    stored.append(_r.turn)

        # Phase 2 -- pre-fetch each turn's prior-turn context SERIALLY (main thread;
        # SQLite :memory: is not thread-safe, so workers never touch the DB). Same
        # query as run_extraction's set_context -> context identical to the serial path.
        _prior: dict[str, list[Turn]] = {}
        for _t in stored:
            _rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? AND ts < ? ORDER BY ts DESC LIMIT 8",
                (_t.session_id, _t.ts),
            ).fetchall()
            _prior[_t.turn_id] = [
                Turn(turn_id=r["turn_id"], session_id=r["session_id"],
                     speaker=Speaker(r["speaker"]), text=r["text"], ts=r["ts"],
                     asr_confidence=r["asr_confidence"])
                for r in reversed(_rows)
            ]

        # Phase 3 -- PARALLEL extract; each worker its OWN extractor instance so
        # set_context state cannot race. This is the slow LLM call, fanned out;
        # MEMCONTEXT_EXTRACTION_WORKERS sets the width (the parallelism the old pool
        # had, now WITH context). Pure: workers read no DB.
        _tl = threading.local()

        def _worker_extractor():
            ex = getattr(_tl, "ex", None)
            if ex is None:
                ex = _tl.ex = auto_extractor()
            return ex

        def _extract_one(t):
            ex = _worker_extractor()
            if hasattr(ex, "set_context"):
                ex.set_context(_prior[t.turn_id])
            try:
                return (t.turn_id, list(ex(t)))
            except Exception:
                return (t.turn_id, [])

        _workers = int(os.environ.get("MEMCONTEXT_EXTRACTION_WORKERS", "32"))
        if stored:
            with _Pool(max_workers=_workers) as _pool:
                _claims_by_tid = dict(_pool.map(_extract_one, stored))
        else:
            _claims_by_tid = {}

        # Phase 4 -- SERIAL insert in ts order. _Precomputed feeds the already-
        # extracted ExtractedClaim objects (every field preserved -- no dict
        # round-trip) into run_extraction, which runs Pass-1/Pass-2 supersession +
        # projection deterministically, in the same order as the serial drain.
        class _Precomputed:
            is_deferrable = False

            def __init__(self, _claims):
                self._claims = _claims

            def __call__(self, _turn):
                return self._claims

        for _t in sorted(stored, key=lambda x: x.ts):
            run_extraction(
                conn, episode_id=_t.turn_id, session_id=_t.session_id,
                extractor=_Precomputed(_claims_by_tid.get(_t.turn_id, [])), semantic=_sem,
            )

        for _sid in batch_sessions:
            backfill_embeddings(conn, _sid, client=self._embedding_client)''',
        "FAITHFUL + PARALLEL INGEST: parallel extract (per-worker set_context) + serial deterministic insert; byte-identical to serial, restores CI-viable parallelism",
    ),
    (
        "        unified_session = f\"amb_{user_id}\" if user_id else _get_any_session(conn)\n"
        "\n"
        "        results = retrieve_hybrid(\n"
        "            conn,\n"
        "            session_id=unified_session,\n"
        "            query=query,\n"
        "            top_k=k * 5,\n"
        "            embedding_client=self._embedding_client,\n"
        "        )\n"
        "        top = results[:k]\n"
        "\n"
        "        from memcontext.claims import get_turn\n"
        "\n"
        "        result_docs = []\n"
        "        seen_turns: set[str] = set()\n"
        "        for claim, score in top:\n"
        "            if claim.source_turn_id in seen_turns:\n"
        "                continue\n"
        "            seen_turns.add(claim.source_turn_id)\n"
        "\n"
        "            turn = get_turn(conn, claim.source_turn_id)\n"
        "            content = turn.text if turn else claim.value\n"
        "\n"
        "            result_docs.append(Document(\n"
        "                id=claim.claim_id,\n"
        "                content=content,\n"
        "                user_id=user_id,\n"
        "            ))\n"
        "\n"
        "        return result_docs, None",
        "        # REAL SERVE DOOR. mcp_tools.handle_memory_query (mcp_tools.py:156),\n"
        "        # for a query spanning many sessions, fans out across EVERY session\n"
        "        # via retrieve_memory_across — per-session fact+episode fusion merged\n"
        "        # by RANK (RRF). That is the product's actual multi-session door, and\n"
        "        # it surfaces the fact-less EPISODES created by the episode floor\n"
        "        # (ingest change 1) that a facts-only retrieve_hybrid can never reach.\n"
        "        # We replace the old per-session retrieve_hybrid raw-score pooling\n"
        "        # (which silenced whole sessions by raw-score magnitude AND served\n"
        "        # only fact-bearing turns) with a single retrieve_memory_across call.\n"
        "        # Scope STRICTLY to this user's (this unit's) tracked sessions; under\n"
        "        # AMB unit-sequential isolation the DB accumulates every unit, so a\n"
        "        # flatten-all fallback would leak other units' sessions. Only the cold\n"
        "        # _get_any_session floor remains, for a genuinely empty store.\n"
        "        session_ids = list(self._sessions_by_user.get(user_id) or [])\n"
        "        if not session_ids:\n"
        "            session_ids = [_get_any_session(conn)]\n"
        "\n"
        "        from memcontext.claims import get_turn\n"
        "\n"
        "        # LEAK 1 - SERVE-DOOR PARITY (mcp_tools.handle_memory_query): the\n"
        "        # product's multi-session door is QUERY-AWARE before it retrieves. It\n"
        "        # surfaces SUPERSEDED facts for a past-state query (detect_history_\n"
        "        # intent -> 'what was X before' / 'used to') and sizes depth by query\n"
        "        # type (classify_query_depth: temporal 30 / factual 15 / aggregation\n"
        "        # 50). The old bridge hardcoded top_k=50, include_superseded=False, so\n"
        "        # past-state queries were unanswerable and the product's query-typed\n"
        "        # depth never fired. Mirror the door (per_session_k keeps the\n"
        "        # starvation fix; budget = max(top_k, per_session_k*len(sessions))).\n"
        "        _history = detect_history_intent(query)\n"
        "        _, _depth_k = classify_query_depth(query)\n"
        "        hits = retrieve_memory_across(\n"
        "            conn,\n"
        "            session_ids=session_ids,\n"
        "            query=query,\n"
        "            top_k=_depth_k,\n"
        "            rerank_top_k=_depth_k,\n"
        "            include_superseded=_history,\n"
        "            embedding_client=self._embedding_client,\n"
        "        )\n"
        "\n"
        "        result_docs = []\n"
        "        seen: set[str] = set()\n"
        "        for hit, _score in hits:\n"
        "            turn = get_turn(conn, hit.source_turn_id)\n"
        "            # LEAK 2 - SUMMARY-LAYER PARITY: serve the DISTILLED claim for a\n"
        "            # fact hit (hit.text = the resolved, supersession-aware fact the\n"
        "            # product actually serves via handle_memory_query's claim channel)\n"
        "            # and the raw turn for an episode hit (the Tier-1 recall floor for\n"
        "            # when extraction missed). The old bridge resolved EVERY hit back\n"
        "            # to raw turn.text, discarding the distilled facts entirely.\n"
        "            body = hit.text if hit.kind == 'fact' else (turn.text if turn else hit.text)\n"
        "            if body in seen:\n"
        "                continue\n"
        "            seen.add(body)\n"
        "            # TEMPORAL GROUNDING: prefix with the source session's date +\n"
        "            # relative offset to the query date (AMB's frozen rag.py renders\n"
        "            # content verbatim, so the dates reach the reader unchanged).\n"
        "            content = _date_prefix(\n"
        "                self._dates_by_session.get(turn.session_id) if turn else None,\n"
        "                query_timestamp,\n"
        "            ) + body\n"
        "            result_docs.append(Document(\n"
        "                id=hit.id,\n"
        "                content=content,\n"
        "                user_id=user_id,\n"
        "            ))\n"
        "\n"
        "        return result_docs, None",
        "SERVE-DOOR + SUMMARY parity: query-aware depth + history-intent (Leak 1) and distilled facts for fact hits (Leak 2), was hardcoded top_k=50 raw-turn dump",
    ),
    (
        "def _get_any_session(conn: sqlite3.Connection) -> str:",
        "def _date_prefix(session_iso: str | None, query_iso: str | None) -> str:\n"
        "    \"\"\"Build the temporal-grounding prefix the product's native harness adds.\n"
        "\n"
        "    Returns ``'[<YYYY-MM-DD>, ~N days ago]\\n'`` when the session's date is\n"
        "    known, computing N as the day offset from the query timestamp when that is\n"
        "    also known. Returns '' when no session date is available (we NEVER\n"
        "    fabricate a date). Failures degrade to a bare date or empty string.\n"
        "    \"\"\"\n"
        "    if not session_iso:\n"
        "        return \"\"\n"
        "    from datetime import datetime\n"
        "\n"
        "    def _parse(s: str):\n"
        "        try:\n"
        "            return datetime.fromisoformat(s.replace(\"Z\", \"+00:00\"))\n"
        "        except Exception:\n"
        "            return None\n"
        "\n"
        "    sdt = _parse(session_iso)\n"
        "    if sdt is None:\n"
        "        return f\"[{session_iso}]\\n\"\n"
        "    date_str = sdt.strftime(\"%Y-%m-%d\")\n"
        "    qdt = _parse(query_iso) if query_iso else None\n"
        "    if qdt is not None:\n"
        "        try:\n"
        "            days = (qdt - sdt).days\n"
        "            if days >= 0:\n"
        "                return f\"[{date_str}, ~{days} days ago]\\n\"\n"
        "            return f\"[{date_str}, ~{abs(days)} days from now]\\n\"\n"
        "        except Exception:\n"
        "            pass\n"
        "    return f\"[{date_str}]\\n\"\n"
        "\n"
        "\n"
        "def _get_any_session(conn: sqlite3.Connection) -> str:",
        "add _date_prefix helper (temporal grounding: '[<date>, ~N days ago]' inside served content)",
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_provider.py <path-to-provider.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    for anchor, replacement, label in EDITS:
        count = text.count(anchor)
        if count != 1:
            print(
                f"ERROR: anchor for [{label}] found {count}x (expected 1). "
                "The upstream provider drifted — refusing to patch silently.",
                file=sys.stderr,
            )
            return 1
        text = text.replace(anchor, replacement, 1)
        print(f"  patched: {label}")

    path.write_text(text, encoding="utf-8")
    print(f"[patch_provider] faithful-wiring applied to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
