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

    try:
        from memcontext.extractors import SimpleExtractor

        extractor = SimpleExtractor()
    except ImportError:
        from memcontext.on_new_turn import ExtractedClaim
        from memcontext.predicate_packs import active_pack

        families = active_pack().predicate_families
        predicate = "user_fact" if "user_fact" in families else next(iter(families))

        def extractor(turn):  # type: ignore[misc]
            return [
                ExtractedClaim(subject="user", predicate=predicate, value=turn.text, confidence=0.5)
            ]

    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT
    result = on_new_turn(conn, session_id=session, speaker=sp, text=text, extractor=extractor)

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
    from memcontext.schema import open_database

    conn = open_database(db)
    active = list_active_claims(conn, session)

    if not active:
        click.echo("No active claims found.")
        conn.close()
        return

    query_tokens = set(query_text.lower().split())
    scored = []
    for claim in active:
        claim_text = f"{claim.subject} {claim.predicate} {claim.value}".lower()
        claim_tokens = set(claim_text.split())
        overlap = len(query_tokens & claim_tokens)
        if overlap > 0:
            scored.append((claim, overlap / max(len(query_tokens), 1)))

    scored.sort(key=lambda x: -x[1])
    results = scored[:top_k]

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
@click.option("--db", default="memcontext.db", help="Database file path.")
@click.option(
    "--transport", type=click.Choice(["stdio"]), default="stdio", help="MCP transport."
)
def serve(db: str, transport: str) -> None:
    """Start the MCP server."""
    try:
        from memcontext.mcp_server import run_server

        run_server(db_path=db, transport=transport)
    except ImportError:
        click.echo(
            "MCP server not available. Install with: pip install memcontext[mcp]", err=True
        )
        raise SystemExit(1)
