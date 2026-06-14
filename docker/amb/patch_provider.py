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
        "    episode_embedder,\n"
        "    retrieve_hybrid,\n"
        "    retrieve_memory_across,\n"
        "    semantic_supersession,\n"
        ")",
        "import episode_embedder + semantic_supersession + retrieve_memory_across (the product's REAL ingest+serve collaborators)",
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
        "        first_user_id = documents[0].user_id if documents else \"default\"\n"
        "        unified_session = f\"amb_{first_user_id}\"\n"
        "\n"
        "        for doc in documents:\n"
        "            turns = _parse_document_turns(doc)\n"
        "            for role, text in turns:\n"
        "                all_work.append((unified_session, role, text))",
        "        # Per-conversation session model: each AMB Document becomes its OWN\n"
        "        # session (doc.id is a stable per-conversation id), exactly like the\n"
        "        # product, where every ingested conversation is a distinct\n"
        "        # session_id. This is what makes the multi-session machinery\n"
        "        # (cross-session RRF fusion in retrieve_memory_across) real instead\n"
        "        # of collapsing everything into one bag.\n"
        "        batch_sessions: list[str] = []\n"
        "        for doc in documents:\n"
        "            sid = f\"amb_{doc.id}\"\n"
        "            user_sessions = self._sessions_by_user.setdefault(doc.user_id, [])\n"
        "            if sid not in user_sessions:\n"
        "                user_sessions.append(sid)\n"
        "            if sid not in batch_sessions:\n"
        "                batch_sessions.append(sid)\n"
        "            # Capture the per-session date (longmemeval's haystack_date,\n"
        "            # surfaced by the loader as Document.timestamp, ISO-8601). Used\n"
        "            # at serve time to date each episode for temporal reasoning.\n"
        "            _ts = getattr(doc, \"timestamp\", None)\n"
        "            if _ts:\n"
        "                self._dates_by_session[sid] = _ts\n"
        "            turns = _parse_document_turns(doc)\n"
        "            for role, text in turns:\n"
        "                all_work.append((sid, role, text))",
        "per-doc sessions on ingest: id=amb_{doc.id}, tracked per user (dedup, order preserved)",
    ),
    (
        "                except Exception:\n"
        "                    w = futures[fut]\n"
        "                    extracted.append((w[0], w[1], w[2], []))",
        "                except Exception:\n"
        "                    # Mirror the product harness: a FAILED extraction is\n"
        "                    # SKIPPED, not persisted as a zero-claim turn the\n"
        "                    # measured product never creates.\n"
        "                    pass",
        "failed extraction -> skip (do not persist a zero-claim turn) — match product harness",
    ),
    (
        '                if not claims_data:\n'
        '                    claims_data = [{\n'
        '                        "subject": "user" if role == "user" else "assistant",\n'
        '                        "predicate": "user_fact",\n'
        '                        "value": text[:500],\n'
        '                        "confidence": 0.3,\n'
        '                    }]\n'
        '\n'
        '                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT',
        '                # EPISODE FLOOR (product Tier-1): a turn the extractor produced\n'
        '                # NO claims for is STILL ingested via on_new_turn with an empty\n'
        '                # claim set (PassthroughExtractor([])). on_new_turn inserts the\n'
        '                # turn regardless of claim count, so the turn becomes a\n'
        '                # retrievable EPISODE carrying zero facts -- exactly the\n'
        '                # product Tier-1 floor. The old `continue` HARD-DROPPED such turns,\n'
        '                # so a needle turn the extractor missed vanished from the store\n'
        '                # entirely and could never be retrieved. We drop the old raw-text\n'
        '                # text[:500] fallback (that fabricated a junk fact) AND we no\n'
        '                # longer drop the turn -- we keep it as a fact-less episode.\n'
        '                # The on_new_turn call now passes embedder=/semantic= (below),\n'
        '                # so even this fact-less episode is EMBEDDED -- semantic episode\n'
        '                # retrieval works, exactly as the product ingest does.\n'
        '                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT',
        "episode floor: ingest zero-claim turns as fact-less EPISODES (was: hard-drop via continue)",
    ),
    (
        "        for sid in sorted(by_session.keys()):\n"
        "            for role, text, claims_data in by_session[sid]:",
        "        # Faithful ingest: the product's REAL ingest (cli.py:89, mcp_tools.py:61)\n"
        "        # passes BOTH the episode embedder and Pass-2 semantic supersession to\n"
        "        # on_new_turn. Build them ONCE so the bridge ingests EXACTLY as the\n"
        "        # product does -- episodes get embedded (so semantic episode retrieval\n"
        "        # works) and cross-session duplicate/refined claims consolidate. Dropping\n"
        "        # them was measuring a DEGRADED product (episodes blind, Pass-2 off).\n"
        "        _epi = episode_embedder()\n"
        "        _sem = semantic_supersession()\n"
        "        for sid in sorted(by_session.keys()):\n"
        "            for role, text, claims_data in by_session[sid]:",
        "construct the product's episode embedder + Pass-2 supersession once before the ingest loop",
    ),
    (
        "                on_new_turn(\n"
        "                    conn,\n"
        "                    session_id=sid,\n"
        "                    speaker=sp,\n"
        "                    text=text,\n"
        "                    extractor=pt,\n"
        "                )",
        "                on_new_turn(\n"
        "                    conn,\n"
        "                    session_id=sid,\n"
        "                    speaker=sp,\n"
        "                    text=text,\n"
        "                    extractor=pt,\n"
        "                    embedder=_epi,\n"
        "                    semantic=_sem,\n"
        "                )",
        "pass embedder= + semantic= to on_new_turn (match cli.py:89 / mcp_tools.py:61 -- the product's REAL ingest)",
    ),
    (
        "        backfill_embeddings(conn, unified_session, client=self._embedding_client)",
        "        # backfill_embeddings is per-session (retrieval.py:354) — embed each\n"
        "        # session ingested in this batch, not one unified bag. Matches the\n"
        "        # product, where each session's episodes are embedded under its own id.\n"
        "        for _sid in batch_sessions:\n"
        "            backfill_embeddings(conn, _sid, client=self._embedding_client)",
        "backfill embeddings per session (loop over this batch's sessions)",
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
        "        hits = retrieve_memory_across(\n"
        "            conn,\n"
        "            session_ids=session_ids,\n"
        "            query=query,\n"
        "            top_k=50,\n"
        "            embedding_client=self._embedding_client,\n"
        "        )\n"
        "\n"
        "        result_docs = []\n"
        "        seen_turns: set[str] = set()\n"
        "        for hit, _score in hits:\n"
        "            if hit.source_turn_id in seen_turns:\n"
        "                continue\n"
        "            seen_turns.add(hit.source_turn_id)\n"
        "            turn = get_turn(conn, hit.source_turn_id)\n"
        "            content = turn.text if turn else hit.text\n"
        "            # TEMPORAL GROUNDING: prefix the served turn with its session\n"
        "            # date + a relative offset to the query date, mirroring the\n"
        "            # product's native harness. AMB's frozen rag.py renders this\n"
        "            # content verbatim, so the dates reach the reader unchanged.\n"
        "            content = _date_prefix(\n"
        "                self._dates_by_session.get(turn.session_id) if turn else None,\n"
        "                query_timestamp,\n"
        "            ) + content\n"
        "            result_docs.append(Document(\n"
        "                id=hit.id,\n"
        "                content=content,\n"
        "                user_id=user_id,\n"
        "            ))\n"
        "\n"
        "        return result_docs, None",
        "REAL SERVE DOOR: retrieve_memory_across (facts+EPISODES by RRF) + dated turn text (was per-session raw-score pooling)",
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
