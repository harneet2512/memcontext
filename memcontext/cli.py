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

    from memcontext.retrieval import semantic_enabled
    click.echo(f"Database: {os.path.abspath(db)}")
    click.echo(
        f"Semantic memory: {'ON (embeddings)' if semantic_enabled() else 'OFF -- lexical-only (BM25)'}"
    )
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
@click.option("--namespace", default="default", help="Tenant namespace (isolation scope).")
def ingest(text: str, db: str, session: str, speaker: str, namespace: str) -> None:
    """Ingest a text turn and extract claims."""
    from memcontext.on_new_turn import on_new_turn
    from memcontext.schema import Speaker, open_database

    conn = open_database(db)

    from memcontext.extractors import auto_extractor
    from memcontext.retrieval import episode_embedder, semantic_supersession

    extractor = auto_extractor()

    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT
    result = on_new_turn(
        conn, session_id=session, speaker=sp, text=text, extractor=extractor,
        embedder=episode_embedder(), semantic=semantic_supersession(),
        namespace=namespace,
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
@click.option("--namespace", default=None, help="Restrict to a tenant namespace (isolation).")
def query_cmd(query_text: str, db: str, session: str, top_k: int, namespace: str | None) -> None:
    """Query memory — unified two-tier retrieval (facts + episodes), the same
    path the MCP/HTTP door serves (was facts-only via retrieve_hybrid)."""
    from memcontext.mcp_tools import _session_in_namespace
    from memcontext.retrieval import retrieve_memory
    from memcontext.schema import open_database

    conn = open_database(db)
    if namespace is not None and not _session_in_namespace(conn, session, namespace):
        click.echo("Access denied: session not in namespace.")
        conn.close()
        return
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
    "--transport", type=click.Choice(["stdio", "http"]), default="stdio",
    help="MCP transport: stdio (local: Claude Code/Cursor) or http (Streamable HTTP for"
    " a remote client like the claude.ai web app, via a tunnel).",
)
@click.option("--host", default="127.0.0.1", help="HTTP bind host (transport=http).")
@click.option("--port", default=8765, help="HTTP port (transport=http).")
@click.option(
    "--token", default=None,
    help="Bearer token required on http (or set env MEMCONTEXT_MCP_TOKEN). Set this"
    " before exposing the endpoint on a public tunnel.",
)
@click.option(
    "--oauth", is_flag=True, default=False,
    help="Serve OAuth 2.1 on http (the claude.ai 'add connector -> log in' flow)"
    " instead of a bearer token. Requires --public-url and --oauth-password.",
)
@click.option(
    "--public-url", default=None,
    help="OAuth issuer: the https URL the client sees (your tunnel/domain).",
)
@click.option(
    "--oauth-password", default=None,
    help="Login-gate password for OAuth (or set env MEMCONTEXT_OAUTH_PASSWORD).",
)
def serve(db: str, transport: str, host: str, port: int, token: str | None,
          oauth: bool, public_url: str | None, oauth_password: str | None) -> None:
    """Start the MCP server (stdio for Claude Code/Cursor; http[+oauth] for claude.ai-web)."""
    import os

    from memcontext.retrieval import enforce_semantic_policy, semantic_enabled

    click.echo(
        f"[memcontext] Semantic memory: {'ON' if semantic_enabled() else 'OFF (degraded lexical-only)'}"
    )
    enforce_semantic_policy()  # loud warning, or raises under MEMCONTEXT_REQUIRE_EMBEDDINGS=1
    token = token or os.environ.get("MEMCONTEXT_MCP_TOKEN")
    oauth_password = oauth_password or os.environ.get("MEMCONTEXT_OAUTH_PASSWORD")
    try:
        from memcontext.mcp_server import run_server

        run_server(db_path=db, transport=transport, host=host, port=port, token=token,
                   oauth=oauth, public_url=public_url, oauth_password=oauth_password)
    except ImportError:
        click.echo(
            "MCP server not available. Install with: pip install memcontext[mcp]", err=True
        )
        raise SystemExit(1)


@main.command()
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--port", default=8765, help="Local port behind the tunnel.")
@click.option(
    "--password", default=None,
    help="Login password for the OAuth gate. First run generates one and saves it"
    " next to the DB; later runs reuse it.",
)
def share(db: str, port: int, password: str | None) -> None:
    """Connect your LOCAL brain to web apps (claude.ai, ChatGPT) — one command.

    Your memory stays in the local SQLite file on this machine. This command dials
    an OUTBOUND tunnel (nothing inbound is opened, nothing is hosted anywhere) and
    serves MCP + an OAuth login at the printed URL. Paste that URL into
    claude.ai -> Settings -> Connectors, log in with the printed password, done.
    Stop the process and your brain is offline again.
    """
    import json as _json
    import secrets as _secrets
    from pathlib import Path

    from memcontext.retrieval import enforce_semantic_policy, semantic_enabled

    click.echo(
        f"[memcontext] Semantic memory: {'ON' if semantic_enabled() else 'OFF (degraded lexical-only)'}"
    )
    enforce_semantic_policy()

    # Stable per-brain password: generate once, reuse forever (sits next to the DB).
    cfg_path = Path(db).expanduser().resolve().with_suffix(".share.json")
    if password is None:
        if cfg_path.exists():
            password = _json.loads(cfg_path.read_text(encoding="utf-8"))["password"]
        else:
            password = _secrets.token_urlsafe(9)
            cfg_path.write_text(_json.dumps({"password": password}), encoding="utf-8")
            click.echo(f"[memcontext] Generated login password (saved to {cfg_path.name})")

    try:
        from pycloudflared import try_cloudflare
    except ImportError:
        click.echo("[memcontext] share requires pycloudflared:"
                   " python -m pip install pycloudflared", err=True)
        raise SystemExit(1)

    # Tunnel FIRST: the OAuth issuer must be the public URL, so resolve it before
    # the server starts. The tunnel is outbound-only; data stays in the local DB.
    click.echo("[memcontext] Opening tunnel (outbound only) ...")
    public_url = try_cloudflare(port=port, verbose=False).tunnel

    click.echo("")
    click.echo("=" * 62)
    click.echo("  Your brain is connected. In claude.ai / ChatGPT add:")
    click.echo(f"    Connector URL : {public_url}/mcp")
    click.echo(f"    Login password: {password}")
    click.echo("  Memory stays local in: " + str(Path(db).expanduser().resolve()))
    click.echo("  NOTE: this URL changes on restart (quick tunnel); re-add the")
    click.echo("  connector after a restart. Ctrl+C disconnects your brain.")
    click.echo("=" * 62)
    click.echo("")

    from memcontext.mcp_server import run_server

    run_server(db_path=db, transport="http", host="127.0.0.1", port=port,
               oauth=True, public_url=public_url, oauth_password=password)


@main.command("serve-http")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--port", default=8100, help="HTTP port.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--share", is_flag=True, default=False,
              help="Expose via Cloudflare tunnel for remote MCP (ChatGPT, Gemini).")
def serve_http(db: str, port: int, host: str, share: bool) -> None:
    """Start the HTTP API server (for ChatGPT, Gemini, browser extensions, any AI)."""
    from memcontext.http_server import run_server
    from memcontext.retrieval import enforce_semantic_policy, semantic_enabled

    click.echo(
        f"[memcontext] Semantic memory: {'ON' if semantic_enabled() else 'OFF (degraded lexical-only)'}"
    )
    enforce_semantic_policy()
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


@main.command("consolidate")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--min-sessions", default=3, type=int,
              help="Graduate facts recurring across at least this many sessions")
def consolidate_cmd(db: str, min_sessions: int) -> None:
    """Graduate cross-session-recurring facts into durable consolidated facts
    (episodic -> semantic). Deterministic, zero-LLM; never deletes.
    """
    from memcontext.consolidate import consolidate_facts
    from memcontext.schema import open_database

    conn = open_database(db)
    n = consolidate_facts(conn, min_sessions=min_sessions)
    conn.commit()
    click.echo(f"Consolidated {n} cross-session fact(s).")
    conn.close()


@main.command("working-context")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--session", default="default", help="Session id")
@click.option("--token-budget", default=2000, type=int,
              help="Token budget for the assembled working set")
def working_context_cmd(db: str, session: str, token_budget: int) -> None:
    """Assemble the task-relevant memory for a session within a token budget
    (working context), cued by recent turns instead of all active memory.
    """
    from memcontext.schema import open_database
    from memcontext.working_context import build_working_context

    conn = open_database(db)
    ctx = build_working_context(conn, session, token_budget=token_budget)
    click.echo(json.dumps({
        "session_id": ctx.session_id,
        "salient_entities": ctx.salient_entities,
        "included": ctx.included,
        "total_active": ctx.total_active,
        "tokens_used": ctx.tokens_used,
        "token_budget": ctx.token_budget,
        "excluded_for_budget": ctx.excluded_for_budget,
        "facts": [{"kind": h.kind, "text": h.text, "score": round(s, 3)}
                  for h, s in ctx.facts],
    }))
    conn.close()


@main.command("procedures")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--min-sessions", default=2, type=int,
              help="Minimum distinct sessions for a sequence to count as a procedure")
def procedures_cmd(db: str, min_sessions: int) -> None:
    """Detect recurring procedures across sessions (EXPERIMENTAL; enable with
    MEMCONTEXT_EXPERIMENTAL_PROCEDURAL=1).
    """
    from memcontext.mcp_tools import handle_memory_procedures
    from memcontext.schema import open_database

    conn = open_database(db)
    click.echo(json.dumps(handle_memory_procedures(conn, min_sessions=min_sessions)))
    conn.close()


@main.command("reindex-embeddings")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--session", default=None, help="Session id (default: all sessions)")
def reindex_embeddings_cmd(db: str, session: str | None) -> None:
    """Backfill missing embeddings (claims, episodes, event frames) -- useful after
    enabling embeddings on a DB that was ingested while they were off.
    """
    from memcontext.retrieval import (
        backfill_embeddings,
        backfill_episode_embeddings,
        backfill_event_frame_embeddings,
        episode_embedder,
    )
    from memcontext.schema import open_database

    conn = open_database(db)
    client = episode_embedder()
    if client is None:
        click.echo("Embeddings are disabled (MEMCONTEXT_EMBED_EPISODES=0); nothing to backfill.")
        conn.close()
        return
    if session:
        sessions = [session]
    else:
        sessions = [r[0] for r in conn.execute("SELECT DISTINCT session_id FROM turns").fetchall()]
    claims = episodes = frames = 0
    for sid in sessions:
        claims += backfill_embeddings(conn, sid, client=client)
        episodes += backfill_episode_embeddings(conn, sid, client=client)
        frames += backfill_event_frame_embeddings(conn, sid, client=client)
    conn.commit()
    click.echo(
        f"Backfilled embeddings: {claims} claim(s), {episodes} episode(s), "
        f"{frames} event frame(s) across {len(sessions)} session(s)."
    )
    conn.close()


@main.command("forget")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--claim-id", default=None, help="Forget a single claim")
@click.option("--subject", default=None, help="Forget everything about a subject")
@click.option("--session", default=None, help="Forget an entire session")
@click.option("--predicate", default=None, help="Forget all claims with a predicate")
@click.option("--reason", default="user_request", help="Audit reason")
def forget_cmd(db, claim_id, subject, session, predicate, reason):
    """Right-to-be-forgotten: cascade-consistent hard delete (no residual), audited
    to the decisions log. Specify exactly one of --claim-id/--subject/--session/--predicate.
    """
    from memcontext.forgetting import forget
    from memcontext.schema import open_database

    conn = open_database(db)
    res = forget(conn, claim_id=claim_id, subject=subject,
                 session_id=session, predicate=predicate, reason=reason)
    conn.commit()
    click.echo(json.dumps(res))
    conn.close()


@main.command("trust-status")
@click.option("--db", default="memcontext.db", help="Database path")
def trust_status_cmd(db):
    """Trust/governance observability: source-trust distribution, contradiction
    rate, forgetting + drift audit, tenant distribution, and a staleness proxy.
    Measures whether the trust layer is working, not just recall."""
    from memcontext.schema import open_database
    from memcontext.trust_report import trust_status

    conn = open_database(db)
    click.echo(json.dumps(trust_status(conn), indent=2))
    conn.close()


@main.command("grant")
@click.option("--db", default="memcontext.db", help="Database path")
@click.option("--principal", required=True, help="Principal name")
@click.option("--namespace", required=True, help="Namespace the token is scoped to")
@click.option("--token", default=None, help="Token to grant (generated if omitted)")
@click.option("--read-only", is_flag=True, default=False, help="Grant read-only access")
def grant_cmd(db, principal, namespace, token, read_only):
    """Grant a principal a scoped access token (namespace + read/write permission).
    The token is stored hashed; the plaintext is printed once, here."""
    import secrets as _secrets

    from memcontext.authz import register_principal
    from memcontext.schema import open_database

    tok = token or _secrets.token_urlsafe(32)
    conn = open_database(db)
    register_principal(conn, token=tok, principal=principal, namespace=namespace,
                       can_write=not read_only)
    conn.commit()
    conn.close()
    click.echo(json.dumps({
        "principal": principal, "namespace": namespace,
        "can_write": not read_only, "token": tok,
    }))


@main.group()
def tools() -> None:
    """Activation layer: ingest, embed, and discover tools from the registry."""


@tools.command("ingest")
@click.argument("json_file")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--server", default="mcp", help="Parent MCP server name/id.")
def tools_ingest(json_file: str, db: str, server: str) -> None:
    """Ingest an MCP tools/list JSON payload into the tool registry."""
    import json as _json
    from pathlib import Path

    from memcontext.schema import open_database
    from memcontext.tool_registry import ingest_mcp_tools_list

    conn = open_database(db)
    data = _json.loads(Path(json_file).read_text(encoding="utf-8"))
    n = ingest_mcp_tools_list(conn, data, server=server)
    conn.commit()
    conn.close()
    click.echo(f"Ingested {n} tools from {json_file} (server={server}).")


@tools.command("reindex")
@click.option("--db", default="memcontext.db", help="Database file path.")
def tools_reindex(db: str) -> None:
    """Embed registry tools not yet embedded (uses the local embedder)."""
    from memcontext.retrieval import EmbeddingClient
    from memcontext.schema import open_database
    from memcontext.tool_registry import embed_tools

    conn = open_database(db)
    n = embed_tools(conn, embedder=EmbeddingClient())
    conn.commit()
    conn.close()
    click.echo(f"Embedded {n} tools.")


@tools.command("discover")
@click.argument("query_text")
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option("--top-k", default=10, help="Max tools to return.")
@click.option("--use-memory", is_flag=True, default=False, help="Condition on user memory.")
@click.option("--session", "sessions", multiple=True, help="Memory session id (repeatable).")
def tools_discover(
    query_text: str, db: str, top_k: int, use_memory: bool, sessions: tuple[str, ...]
) -> None:
    """Return the curated top-K tools for a query (query-only by default)."""
    from memcontext.retrieval import EmbeddingClient
    from memcontext.schema import open_database
    from memcontext.tool_activation import discover_tools

    conn = open_database(db)
    results = discover_tools(
        conn, query=query_text, session_ids=list(sessions), top_k=top_k,
        use_memory=use_memory, embedder=EmbeddingClient(),
    )
    conn.close()
    if not results:
        click.echo("No tools found (is the registry populated? run `tools ingest`).")
        return
    used = any(r.used_memory for r in results)
    click.echo(f"Top {len(results)} tools (used_memory={used}):")
    for r in results:
        click.echo(json.dumps({"tool_id": r.tool_id, "name": r.name, "score": round(r.score, 4)}))


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
