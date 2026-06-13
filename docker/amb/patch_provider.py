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
        "    retrieve_memory_across,\n"
        "    semantic_supersession,\n"
        ")",
        "import episode_embedder + semantic_supersession + retrieve_memory_across (production helpers)",
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
        "        self._sessions_by_user: dict[str | None, list[str]] = {}",
        "track per-user session ids on the provider instance (per-conversation session model)",
    ),
    (
        "        if reset:\n"
        "            self.cleanup()\n"
        "            self._conn = open_database(self._db_path)\n"
        "            self._conn.row_factory = sqlite3.Row",
        "        if reset:\n"
        "            self.cleanup()\n"
        "            self._sessions_by_user = {}\n"
        "            self._conn = open_database(self._db_path)\n"
        "            self._conn.row_factory = sqlite3.Row",
        "reset tracked sessions alongside the connection on prepare(reset=True)",
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
        "            turns = _parse_document_turns(doc)\n"
        "            for role, text in turns:\n"
        "                all_work.append((sid, role, text))",
        "per-doc sessions on ingest: id=amb_{doc.id}, tracked per user (dedup, order preserved)",
    ),
    (
        "        for sid in sorted(by_session.keys()):\n"
        "            for role, text, claims_data in by_session[sid]:",
        "        # Faithful wiring: the production ingest path (cli.py / mcp_tools.py)\n"
        "        # supplies the episode embedder and Pass-2 semantic supersession.\n"
        "        _epi = episode_embedder()\n"
        "        _sem = semantic_supersession()\n"
        "        for sid in sorted(by_session.keys()):\n"
        "            for role, text, claims_data in by_session[sid]:",
        "construct production embedder + Pass-2 supersession once before the loop",
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
        "pass embedder= and semantic= to on_new_turn (the two missing capabilities)",
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
        '                # Legitimacy (anti-overfitting rule): NO raw-text fallback.\n'
        '                # A real LLMExtractor returns [] on a miss and contributes\n'
        '                # no claim; injecting {user_fact: text[:500]} would turn\n'
        '                # MemContext into a raw-text RAG baseline — a prohibited\n'
        '                # benchmark hack. on_new_turn still persists the episode\n'
        '                # with zero claims, exactly as the production pipeline does.\n'
        '                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT',
        "remove the prohibited raw-text fallback (miss -> no claim, not text[:500])",
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
        "        # Faithful multi-session serve path. mcp_tools.handle_memory_query\n"
        "        # (mcp_tools.py:156), for a query with no single session, fans out\n"
        "        # across EVERY session via retrieve_memory_across — per-session\n"
        "        # fact+episode fusion merged by RRF rank. We mirror it exactly:\n"
        "        # scope to this user's tracked sessions; if the user is unknown,\n"
        "        # fall back to ALL tracked sessions (flattened), preserving the\n"
        "        # product's cross-session reach. Only if nothing is tracked do we\n"
        "        # keep the original single-session _get_any_session floor so a cold\n"
        "        # retrieve never crashes. No benchmark tuning: same call, same\n"
        "        # args, same default top_k as the product's door.\n"
        "        session_ids = list(self._sessions_by_user.get(user_id) or [])\n"
        "        if not session_ids:\n"
        "            for _sids in self._sessions_by_user.values():\n"
        "                for _s in _sids:\n"
        "                    if _s not in session_ids:\n"
        "                        session_ids.append(_s)\n"
        "        if not session_ids:\n"
        "            session_ids = [_get_any_session(conn)]\n"
        "\n"
        "        hits = retrieve_memory_across(\n"
        "            conn,\n"
        "            session_ids=session_ids,\n"
        "            query=query,\n"
        "            top_k=k,\n"
        "            embedding_client=self._embedding_client,\n"
        "        )\n"
        "\n"
        "        result_docs = [\n"
        "            Document(id=hit.id, content=hit.text, user_id=user_id)\n"
        "            for hit, _score in hits\n"
        "        ]\n"
        "\n"
        "        return result_docs, None",
        "route retrieve() through retrieve_memory_across over the user's tracked sessions (mirrors mcp_tools.handle_memory_query multi-session path)",
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
