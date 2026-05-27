"""MemContext CLI — command-line interface for the memory substrate."""
from __future__ import annotations

import json
import os
import sys

import click


@click.group()
def main() -> None:
    """MemContext — memory and context substrate for AI agents."""


@main.command()
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--pack", default="general", help="Predicate pack(s), comma-separated.")
def init(db: str, pack: str) -> None:
    """Initialize a new MemContext database."""
    os.environ["ACTIVE_PACK"] = pack
    from memcontext.predicate_packs import active_pack

    active_pack.cache_clear()

    from memcontext.schema import open_database

    conn = open_database(db)
    conn.close()
    ap = active_pack()
    click.echo(f"Initialized MemContext database at {os.path.abspath(db)}")
    click.echo(f"Active pack: {ap.pack_id} ({len(ap.predicate_families)} predicates)")


@main.command()
@click.option("--db", default="memcontext.db", help="Database file path.")
def status(db: str) -> None:
    """Show database status."""
    from memcontext.schema import open_database

    try:
        conn = open_database(db)
    except Exception as exc:
        click.echo(f"Error opening database: {exc}", err=True)
        raise SystemExit(1) from exc

    total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    active_claims = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status IN ('active','confirmed','audited')"
    ).fetchone()[0]
    total_turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]

    click.echo(f"Database: {os.path.abspath(db)}")
    click.echo(f"Sessions: {sessions}")
    click.echo(f"Turns: {total_turns}")
    click.echo(f"Claims: {total_claims} total, {active_claims} active")
    conn.close()


@main.command()
@click.argument("text")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--session", default="default", help="Session ID.")
@click.option(
    "--speaker",
    type=click.Choice(["user", "assistant"]),
    default="user",
    help="Speaker role.",
)
def ingest(text: str, db: str, session: str, speaker: str) -> None:
    """Ingest a text turn and extract claims."""
    from memcontext.on_new_turn import on_new_turn
    from memcontext.predicate_packs import active_pack
    from memcontext.schema import Speaker, open_database

    conn = open_database(db)

    from memcontext.extractors import auto_extractor

    extractor = auto_extractor()
    pack = active_pack()

    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT
    result = on_new_turn(
        conn, session_id=session, speaker=sp, text=text, extractor=extractor,
        multi_valued_predicates=pack.multi_valued_predicates,
    )

    if not result.admitted:
        click.echo(f"Turn rejected: {result.admission_reason}")
        conn.close()
        return

    click.echo(f"Turn ingested: {result.turn.turn_id}")
    click.echo(f"Claims created: {len(result.created_claims)}")
    for c in result.created_claims:
        click.echo(f"  [{c.predicate}] {c.subject}: {c.value} (confidence={c.confidence})")
    if result.supersession_edges:
        click.echo(f"Supersessions: {len(result.supersession_edges)}")
    conn.close()


@main.command("query")
@click.argument("query_text")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--session", default="default", help="Session ID.")
@click.option("--top-k", default=10, help="Max results.")
def query_cmd(query_text: str, db: str, session: str, top_k: int) -> None:
    """Query memory for relevant claims."""
    from memcontext.claims import list_active_claims
    from memcontext.retrieval import retrieve_hybrid
    from memcontext.schema import open_database

    conn = open_database(db)
    active = list_active_claims(conn, session)

    if not active:
        click.echo("No active claims found.")
        conn.close()
        return

    results = retrieve_hybrid(
        conn, session_id=session, query=query_text, top_k=top_k,
    )

    if not results:
        results = [(c, 0.0) for c in active[:top_k]]

    click.echo(f"Found {len(results)} claim(s):")
    for claim, score in results:
        out = {
            "claim_id": claim.claim_id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "confidence": claim.confidence,
            "score": round(score, 3),
        }
        click.echo(json.dumps(out))
    conn.close()


@main.command()
@click.argument("url")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--session", default="observe_default", help="Session ID.")
@click.option("--login-email", default=None, help="Email/username for authenticated access.")
@click.option("--login-password", default=None, help="Password for authenticated access.")
@click.option("--login-url", default=None, help="Login page URL if different from target.")
@click.option("--connect-browser", is_flag=True, default=False, help="Attach to running Chrome on port 9222. Inherits all auth sessions.")
def observe(url: str, db: str, session: str, login_email: str | None, login_password: str | None, login_url: str | None, connect_browser: bool) -> None:
    """Observe a live URL — open browser, capture accessibility tree, extract claims."""
    import logging

    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )

    from memcontext.mcp_tools import handle_memory_observe_url
    from memcontext.schema import open_database

    conn = open_database(db)
    click.echo(f"[memcontext] Observing: {url}")

    try:
        result = handle_memory_observe_url(
            conn, url=url, session_id=session,
            login_email=login_email, login_password=login_password,
            login_url=login_url, connect_browser=connect_browser,
        )
    except Exception as exc:
        click.echo(f"[memcontext] Error: {exc}", err=True)
        conn.close()
        raise SystemExit(1) from exc

    click.echo(f"[memcontext] Page title: {result['title']}")
    click.echo(f"[memcontext] Accessibility tree: {result['a11y_nodes']} nodes")
    click.echo(f"[memcontext] DOM hash: {result['dom_hash']}")
    click.echo(f"[memcontext] {result['claims_stored']} claims stored:")

    for c in result["claims"]:
        click.echo(f"  ({c['subject']}, {c['predicate']}, \"{c['value']}\")")

    if result.get("is_revisit"):
        changes = result.get("changes_detected", [])
        if changes:
            click.echo(f"[memcontext] Changes detected: {len(changes)}")
            for ch in changes:
                click.echo(
                    f"  CHANGED: \"{ch['old_value']}\" -> \"{ch['new_value']}\""
                    f"  edge: {ch['edge_type']}"
                )
        else:
            click.echo("[memcontext] Re-visit: no changes detected")

    click.echo(
        f"[memcontext] Provenance: url={url} dom_hash={result['dom_hash']} "
        f"session={result['session_id']}"
    )
    conn.close()


@main.command()
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option(
    "--transport", type=click.Choice(["stdio"]), default="stdio", help="MCP transport."
)
def serve(db: str, transport: str) -> None:
    """Start the MCP server (for Claude Code, Cursor)."""
    try:
        from memcontext.mcp_server import run_server

        run_server(db_path=db, transport=transport)
    except ImportError:
        click.echo(
            "MCP server not available. Install with: pip install memcontext[mcp]", err=True
        )
        raise SystemExit(1)


@main.command("serve-http")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--port", default=8100, help="HTTP port.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--share", is_flag=True, default=False,
              help="Expose via Cloudflare tunnel for remote MCP (ChatGPT, Gemini).")
def serve_http(db: str, port: int, host: str, share: bool) -> None:
    """Start the HTTP API server (for ChatGPT, Gemini, browser extensions, any AI)."""
    from memcontext.http_server import run_server

    click.echo(f"[memcontext] Local MCP ready (stdio)")
    click.echo(f"[memcontext] HTTP API ready: http://localhost:{port}")
    click.echo(f"[memcontext] Database: {db}")

    if share:
        import threading
        try:
            from pycloudflared import try_cloudflare
        except ImportError:
            click.echo(
                "[memcontext] --share requires pycloudflared: python -m pip install pycloudflared",
                err=True,
            )
            raise SystemExit(1)

        def start_tunnel():
            try:
                info = try_cloudflare(port=port)
                url = info.tunnel
                click.echo(f"[memcontext] Remote MCP ready: {url}/mcp/")
                click.echo(f"             Paste this URL into ChatGPT/Gemini ->")
                click.echo(f"             Settings -> Connectors -> Create -> URL: {url}/mcp/")
            except Exception as e:
                click.echo(f"[memcontext] Tunnel failed: {e}", err=True)

        tunnel_thread = threading.Thread(target=start_tunnel, daemon=True)
        tunnel_thread.start()

    run_server(db_path=db, port=port, host=host)


@main.group()
def hooks() -> None:
    """Manage Claude Code ambient hooks."""


@hooks.command()
@click.option("--port", default=8100, help="MemContext HTTP server port.")
@click.option("--project-dir", default=".", help="Project directory containing .claude/")
def install(port: int, project_dir: str) -> None:
    """Install ambient hooks into .claude/settings.json."""
    settings_dir = os.path.join(project_dir, ".claude")
    settings_path = os.path.join(settings_dir, "settings.json")

    os.makedirs(settings_dir, exist_ok=True)

    settings: dict = {}
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            settings = json.load(f)

    base = f"http://localhost:{port}"
    settings["hooks"] = {
        "PostToolUse": [{"matcher": "", "hooks": [
            {"type": "http", "url": f"{base}/api/hooks/post_tool_use", "timeout": 10}
        ]}],
        "UserPromptSubmit": [{"matcher": "", "hooks": [
            {"type": "http", "url": f"{base}/api/hooks/user_prompt_submit", "timeout": 10}
        ]}],
        "PreToolUse": [{"matcher": "", "hooks": [
            {"type": "http", "url": f"{base}/api/hooks/pre_tool_use", "timeout": 5}
        ]}],
        "Stop": [{"matcher": "", "hooks": [
            {"type": "http", "url": f"{base}/api/hooks/stop", "timeout": 5}
        ]}],
    }

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    click.echo(f"Hooks installed in {os.path.abspath(settings_path)}")
    click.echo(f"HTTP server: {base}")
    click.echo("Restart Claude Code to activate. Run 'memcontext serve-http' first.")


@hooks.command()
@click.option("--project-dir", default=".", help="Project directory containing .claude/")
def uninstall(project_dir: str) -> None:
    """Remove ambient hooks from .claude/settings.json."""
    settings_path = os.path.join(project_dir, ".claude", "settings.json")
    if not os.path.exists(settings_path):
        click.echo("No .claude/settings.json found.")
        return

    with open(settings_path) as f:
        settings = json.load(f)

    if "hooks" in settings:
        del settings["hooks"]
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        click.echo("Hooks removed.")
    else:
        click.echo("No hooks configured.")


main.add_command(hooks)


@main.command("storage-stats")
@click.option("--db", default="memcontext.db", help="Database file path.")
def storage_stats(db: str) -> None:
    """Show storage statistics."""
    from memcontext.schema import open_database

    conn = open_database(db)

    active = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status IN ('active','confirmed','audited')"
    ).fetchone()[0]
    with_embeddings = conn.execute(
        "SELECT COUNT(*) FROM claim_embeddings ce"
        " JOIN claims c ON ce.claim_id = c.claim_id"
        " WHERE c.status IN ('active','confirmed','audited')"
    ).fetchone()[0]
    superseded = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status = 'superseded'"
    ).fetchone()[0]
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    digests = conn.execute("SELECT COUNT(*) FROM session_digests").fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM life_events").fetchone()[0]

    click.echo(f"Active claims:     {active} ({with_embeddings} with embeddings)")
    click.echo(f"Superseded claims: {superseded} (no embeddings, provenance preserved)")
    click.echo(f"Turns stored:      {turns}")
    click.echo(f"Profiles cached:   {profiles}")
    click.echo(f"Session digests:   {digests}")
    click.echo(f"Life events:       {events}")
    click.echo(f"Retrieval surface: {active} claims")
    click.echo(f"Provenance depth:  {active + superseded} claims reachable via chain walking")
    conn.close()


@main.command("eval")
@click.option("--suite", required=True, help="Suite: preflight, longmemeval-s, internal")
@click.option("--limit", default=None, type=int, help="Limit number of questions")
@click.option(
    "--reader",
    default="none",
    type=click.Choice(["none", "configured"]),
    help="Reader mode",
)
@click.option("--db", default=":memory:", help="Database path")
@click.option("--dataset", default=None, help="Path to LongMemEval dataset directory")
@click.option("--target-categories", default=None, help="Comma-separated category filter")
def eval_cmd(
    suite: str,
    limit: int | None,
    reader: str,
    db: str,
    dataset: str | None,
    target_categories: str | None,
) -> None:
    """Run evaluation suite."""
    if suite == "internal":
        from evals.runner import print_results, run_suite

        suites_dir = os.path.join(os.path.dirname(__file__), "..", "evals", "suites")
        for name in ["extraction", "retrieval", "supersession"]:
            path = os.path.join(suites_dir, f"{name}.json")
            if os.path.exists(path):
                click.echo(f"\n--- {name} ---")
                results = run_suite(path)
                print_results(results)
    elif suite in ("longmemeval-s", "preflight"):
        from evals.longmemeval import run_preflight

        if not dataset:
            click.echo("Error: --dataset PATH required for longmemeval-s", err=True)
            raise SystemExit(1)
        cats = (
            [c.strip() for c in target_categories.split(",")]
            if target_categories
            else None
        )
        result = run_preflight(
            dataset_path=dataset,
            limit=limit or 5,
            reader=reader,
            target_categories=cats,
        )
        # Print compact summary, not full JSON with all prompts
        summary = {k: v for k, v in result.items() if k != "questions"}
        click.echo(json.dumps(summary, indent=2, default=str))
        click.echo(f"\nPer-question details:")
        for q in result["questions"]:
            score_str = f" score={q['score']:.3f}" if "score" in q else ""
            correct_str = f" {'CORRECT' if q.get('correct') else 'WRONG'}" if "correct" in q else ""
            click.echo(
                f"  [{q['category']}] {q['question_id']}: "
                f"{q['num_claims_retrieved']} claims{score_str}{correct_str}"
            )
            if q.get("predicted_answer"):
                click.echo(f"    predicted: {q['predicted_answer'][:100]}")
                click.echo(f"    gold:      {q['gold_answer'][:100]}")
    else:
        click.echo(f"Unknown suite: {suite}", err=True)
        raise SystemExit(1)
