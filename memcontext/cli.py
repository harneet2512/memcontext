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
    from memcontext.schema import Speaker, open_database

    conn = open_database(db)

    from memcontext.extractors import auto_extractor
    from memcontext.retrieval import episode_embedder

    extractor = auto_extractor()

    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT
    result = on_new_turn(
        conn, session_id=session, speaker=sp, text=text, extractor=extractor,
        embedder=episode_embedder(),
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
    """Query memory — unified two-tier retrieval (facts + episodes), the same
    path the MCP/HTTP door serves (was facts-only via retrieve_hybrid)."""
    from memcontext.retrieval import retrieve_memory
    from memcontext.schema import open_database

    conn = open_database(db)
    hits = retrieve_memory(conn, session_id=session, query=query_text, top_k=top_k)

    if not hits:
        click.echo("No relevant memory found.")
        conn.close()
        return

    click.echo(f"Found {len(hits)} memory item(s):")
    for hit, score in hits:
        click.echo(json.dumps({
            "kind": hit.kind,
            "id": hit.id,
            "text": hit.text,
            "source_turn_id": hit.source_turn_id,
            "score": round(score, 3),
        }))
    conn.close()


@main.command()
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--session", default="default", help="Session ID.")
@click.option("--pack", default="general", help="Predicate pack(s) the db was seeded with (controls the gaps vocabulary).")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of the formatted view.")
def brain(db: str, session: str, pack: str, as_json: bool) -> None:
    """Show the deterministic world-state projection (by subject, with provenance + gaps)."""
    os.environ["ACTIVE_PACK"] = pack
    from memcontext.predicate_packs import active_pack

    active_pack.cache_clear()

    from memcontext.brain import brain as brain_fn
    from memcontext.schema import open_database

    conn = open_database(db)
    ws = brain_fn(conn, session_id=session)
    conn.close()

    if as_json:
        click.echo(json.dumps(ws, indent=2))
    else:
        from memcontext.trace_view import format_world_state

        click.echo(format_world_state(ws))


@main.command()
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--session", default="default", help="Session ID.")
@click.option("--subject", required=True, help="Subject to trace.")
@click.option("--predicate", required=True, help="Predicate to trace.")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of the table.")
def trace(db: str, session: str, subject: str, predicate: str, as_json: bool) -> None:
    """Render the supersession lineage for a subject+predicate (active on top)."""
    from memcontext.mcp_tools import handle_memory_trace
    from memcontext.schema import open_database

    conn = open_database(db)
    result = handle_memory_trace(
        conn, session_id=session, subject=subject, predicate=predicate
    )
    conn.close()

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        from memcontext.trace_view import render_trace_table

        click.echo(render_trace_table(result))


@main.command()
@click.option("--db", default="memcontext_demo.db", help="Demo database file (recreated each run).")
@click.option(
    "--pack",
    type=click.Choice(["developer", "general"]),
    default="developer",
    help="Predicate vocabulary for the demo (controls the gaps report).",
)
def demo(db: str, pack: str) -> None:
    """Run the 'one corrected fact, three memories' differentiator demo."""
    from demo.run_demo import run

    run(db=db, pack=pack)


@main.command()
@click.argument("url")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--session", default="observe_default", help="Session ID.")
@click.option("--login-email", default=None, help="Email/username for form login. Prefer --connect-browser.")
@click.option("--login-url", default=None, help="Login page URL if different from target.")
@click.option("--connect-browser", is_flag=True, default=False, help="PREFERRED: attach to running Chrome on port 9222. Inherits all auth sessions, no credentials.")
@click.option("--allow-password-login", is_flag=True, default=False, help="Opt in to password login. Password is read from MEMCONTEXT_OBSERVE_PASSWORD or prompted (never a CLI arg).")
def observe(url: str, db: str, session: str, login_email: str | None, login_url: str | None, connect_browser: bool, allow_password_login: bool) -> None:
    """Observe a live URL — open browser, capture accessibility tree, extract claims."""
    import logging

    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )

    from memcontext.mcp_tools import handle_memory_observe_url
    from memcontext.schema import open_database

    # Password is never a CLI arg (would leak in the process list / shell history).
    # Read it from the environment, or prompt without echo when opted in.
    login_password: str | None = None
    if allow_password_login:
        login_password = os.environ.get("MEMCONTEXT_OBSERVE_PASSWORD") or None
        if login_password is None and login_email:
            import getpass
            login_password = getpass.getpass("Login password (hidden): ") or None

    conn = open_database(db)
    click.echo(f"[memcontext] Observing: {url}")

    try:
        result = handle_memory_observe_url(
            conn, url=url, session_id=session,
            login_email=login_email, login_password=login_password,
            login_url=login_url, connect_browser=connect_browser,
            allow_password_login=allow_password_login,
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


@main.command("mcp-config")
@click.option(
    "--client",
    type=click.Choice(["claude", "codex", "both"]),
    default="both",
    help="Which client config to emit.",
)
@click.option("--db", default="memcontext.db", help="Database path (emitted absolute).")
def mcp_config(client: str, db: str) -> None:
    """Print ready-to-paste MCP client config to attach this server.

    Uses a PATH-independent launch: `<python> -m memcontext.mcp_server --db <path>`,
    so it works whether or not the `memcontext` console script is on PATH.
    """
    py = sys.executable
    db_abs = os.path.abspath(db)
    launch_args = ["-m", "memcontext.mcp_server", "--db", db_abs]

    if client in ("claude", "both"):
        cfg = {"mcpServers": {"memcontext": {"command": py, "args": launch_args}}}
        click.echo("# Claude Code - add to .mcp.json (project) or ~/.claude.json (user):")
        click.echo(json.dumps(cfg, indent=2))
        click.echo("")

    if client in ("codex", "both"):
        # TOML literal (single-quoted) strings so Windows backslash paths need no escaping.
        args_toml = ", ".join(f"'{a}'" for a in launch_args)
        click.echo("# Codex - add to ~/.codex/config.toml:")
        click.echo("[mcp_servers.memcontext]")
        click.echo(f"command = '{py}'")
        click.echo(f"args = [{args_toml}]")
        click.echo("")


@main.command()
@click.option("--client", type=click.Choice(["claude", "codex", "both"]), default="both",
              help="Which client(s) to attach to.")
@click.option("--db", default="memcontext.db", help="Database file path (stored absolute).")
@click.option("--user", is_flag=True, default=False,
              help="Also write user config (~/.claude.json), attaching in every project.")
@click.option("--project-dir", default=".", help="Project dir for Claude Code .mcp.json.")
def attach(client: str, db: str, user: bool, project_dir: str) -> None:
    """Attach MemContext to your AI client(s) — writes the config, no manual editing.

    Claude Code → project .mcp.json (and ~/.claude.json with --user); Codex →
    ~/.codex/config.toml. Idempotent (re-run is a no-op), backs up to *.bak, and
    never touches other servers.
    """
    from memcontext import client_config as cc

    py = sys.executable
    db_abs = os.path.abspath(db)
    targets = []
    if client in ("claude", "both"):
        targets.append(("Claude Code (project)", cc.claude_project_path(project_dir), cc.attach_claude))
        if user:
            targets.append(("Claude Code (user)", cc.claude_user_path(), cc.attach_claude))
    if client in ("codex", "both"):
        targets.append(("Codex", cc.codex_path(), cc.attach_codex))

    for label, path, fn in targets:
        changed = fn(path, py, db_abs)
        click.echo(f"[memcontext] {label}: {'attached' if changed else 'already up to date'}  ({path})")
    click.echo(f"[memcontext] database: {db_abs}")
    click.echo("[memcontext] Restart your client(s) to load the memcontext tools.")


@main.command()
@click.option("--client", type=click.Choice(["claude", "codex", "both"]), default="both",
              help="Which client config to detach from.")
@click.option("--db", default="memcontext.db", help="DB path (only deleted with --purge).")
@click.option("--user", is_flag=True, default=False,
              help="Also remove from user config (~/.claude.json).")
@click.option("--purge", is_flag=True, default=False,
              help="Also DELETE the database file (irreversible). Off by default.")
@click.option("--project-dir", default=".", help="Project dir for .mcp.json / .claude/ hooks.")
def uninstall(client: str, db: str, user: bool, purge: bool, project_dir: str) -> None:
    """Remove MemContext from agent configs. Preserves your data unless --purge."""
    from memcontext import client_config as cc

    removed: list[str] = []
    if client in ("claude", "both"):
        if cc.detach_claude(cc.claude_project_path(project_dir)):
            removed.append(str(cc.claude_project_path(project_dir)))
        if user and cc.detach_claude(cc.claude_user_path()):
            removed.append(str(cc.claude_user_path()))
    if client in ("codex", "both"):
        if cc.detach_codex(cc.codex_path()):
            removed.append(str(cc.codex_path()))

    # Strip MemContext ambient hooks (urls under /api/hooks/) from .claude/settings.json.
    settings_path = os.path.join(project_dir, ".claude", "settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
        except (json.JSONDecodeError, OSError):
            settings = {}
        hooks_cfg = settings.get("hooks")
        if isinstance(hooks_cfg, dict):
            changed = False
            for event in list(hooks_cfg):
                groups = [
                    g for g in hooks_cfg[event]
                    if not any("/api/hooks/" in (h.get("url", "")) for h in g.get("hooks", []))
                ]
                if groups != hooks_cfg[event]:
                    changed = True
                    if groups:
                        hooks_cfg[event] = groups
                    else:
                        del hooks_cfg[event]
            if changed:
                if not hooks_cfg:
                    settings.pop("hooks", None)
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(settings, f, indent=2)
                removed.append(settings_path + " (hooks)")

    for r in removed:
        click.echo(f"[memcontext] detached from {r}")
    if not removed:
        click.echo("[memcontext] no MemContext config entries found.")

    db_abs = os.path.abspath(db)
    if purge:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_abs + suffix)
            except OSError:
                pass
        click.echo(f"[memcontext] purged database {db_abs}")
    elif os.path.exists(db_abs):
        click.echo(f"[memcontext] your data is preserved at {db_abs} (use --purge to delete).")


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


@main.command("reindex-importance")
@click.option("--db", default="memcontext.db", help="Database path")
def reindex_importance_cmd(db: str) -> None:
    """Recompute importance scores for all active claims (run after backfills).

    Wires importance.recompute_all_importance, which previously had no caller.
    """
    from memcontext.importance import recompute_all_importance
    from memcontext.schema import open_database

    conn = open_database(db)
    n = recompute_all_importance(conn)
    conn.commit()
    click.echo(f"Recomputed importance for {n} claim(s).")
    conn.close()


@main.command("prune-memory")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--threshold", default=0.35, type=float,
              help="Demote claims with utility below this (0-1)")
@click.option("--min-age-days", default=30.0, type=float,
              help="Only demote claims older than this many days")
def prune_memory_cmd(db: str, threshold: float, min_age_days: float) -> None:
    """Demote low-utility, old claims out of active retrieval (utility-weighted
    retention). Reversible, never deletes — bounds the active set / token cost.
    """
    from memcontext.retention import demote_low_utility
    from memcontext.schema import open_database

    conn = open_database(db)
    n = demote_low_utility(conn, threshold=threshold, min_age_days=min_age_days)
    conn.commit()
    click.echo(f"Demoted {n} low-utility claim(s) out of active retrieval.")
    conn.close()


def cli() -> None:
    """Console-script entry point: run the CLI, surfacing DB errors cleanly.

    A locked or corrupt database raises `DatabaseUnavailableError` from any
    command; catch it here so the user sees a one-line message, not a traceback.
    """
    from memcontext.schema import DatabaseUnavailableError

    try:
        main()
    except DatabaseUnavailableError as exc:
        click.echo(f"[memcontext] {exc}", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
