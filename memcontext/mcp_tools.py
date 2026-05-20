"""MCP tool handler functions — pure business logic, no MCP protocol dependency.

Each function takes a sqlite3 Connection and keyword arguments, returns a dict.
These are usable from CLI, tests, or the MCP server without importing mcp.
"""
from __future__ import annotations

import json
import sqlite3
import uuid

from memcontext.claims import (
    get_claim,
    get_superseded_by,
    get_turn,
    insert_claim,
    list_active_claims,
    row_to_claim,
    set_claim_status,
)
from memcontext.extractors import PassthroughExtractor, auto_extractor
from memcontext.on_new_turn import on_new_turn
from memcontext.provenance import span_for_claim
from memcontext.schema import ClaimStatus, EdgeType, Speaker
from memcontext.supersession import write_supersession_edge


def handle_memory_store(
    conn: sqlite3.Connection,
    *,
    text: str,
    speaker: str = "user",
    session_id: str | None = None,
    claims: list[dict] | None = None,
) -> dict:
    sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT

    if claims:
        extractor = PassthroughExtractor(claims)
    else:
        extractor = auto_extractor()

    result = on_new_turn(conn, session_id=sid, speaker=sp, text=text, extractor=extractor)

    return {
        "turn_id": result.turn.turn_id if result.turn else None,
        "session_id": sid,
        "admitted": result.admitted,
        "claims_created": len(result.created_claims),
        "claim_ids": [c.claim_id for c in result.created_claims],
        "supersessions": len(result.supersession_edges),
    }


def handle_memory_query(
    conn: sqlite3.Connection,
    *,
    query: str,
    session_id: str | None = None,
    top_k: int = 10,
) -> dict:
    if session_id:
        active = list_active_claims(conn, session_id)
    else:
        rows = conn.execute(
            "SELECT * FROM claims"
            " WHERE status IN ('active','confirmed','audited')"
            " ORDER BY created_ts ASC",
        ).fetchall()
        active = [row_to_claim(r) for r in rows]
    if not active:
        return {"claims": [], "total": 0}

    query_tokens = set(query.lower().split())
    scored = []
    for claim in active:
        claim_text = f"{claim.subject} {claim.predicate} {claim.value}".lower()
        claim_tokens = set(claim_text.split())
        overlap = len(query_tokens & claim_tokens)
        score = overlap / max(len(query_tokens), 1)
        scored.append((claim, score))

    scored.sort(key=lambda x: (-x[1], x[0].created_ts))
    top = scored[:top_k]

    return {
        "claims": [
            {
                "claim_id": c.claim_id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "confidence": c.confidence,
                "status": c.status.value,
                "score": round(s, 4),
            }
            for c, s in top
        ],
        "total": len(active),
    }


def handle_memory_trace(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
) -> dict:
    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"error": f"Claim {claim_id} not found"}

    source_turn = get_turn(conn, claim.source_turn_id)
    span = span_for_claim(conn, claim_id)

    # Walk supersession chain
    chain = []
    current_id = claim_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        next_id = get_superseded_by(conn, current_id)
        if next_id:
            chain.append({"from": current_id, "to": next_id})
        current_id = next_id

    return {
        "claim": {
            "claim_id": claim.claim_id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "confidence": claim.confidence,
            "status": claim.status.value,
        },
        "source_turn": {
            "turn_id": source_turn.turn_id,
            "speaker": source_turn.speaker.value,
            "text": source_turn.text,
        } if source_turn else None,
        "char_span": {
            "start": span.char_start,
            "end": span.char_end,
        } if span and span.char_start is not None else None,
        "supersession_chain": chain,
    }


def handle_memory_correct(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    action: str,
    new_value: str | None = None,
) -> dict:
    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"error": f"Claim {claim_id} not found"}

    if action == "dismiss":
        set_claim_status(conn, claim_id, ClaimStatus.DISMISSED)
        return {
            "action": "dismissed",
            "claim_id": claim_id,
            "status": "dismissed",
        }

    if action == "correct":
        if not new_value:
            return {"error": "new_value is required for correction"}

        new_claim = insert_claim(
            conn,
            session_id=claim.session_id,
            subject=claim.subject,
            predicate=claim.predicate,
            value=new_value,
            confidence=1.0,
            source_turn_id=claim.source_turn_id,
        )
        edge = write_supersession_edge(
            conn,
            old_claim_id=claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.USER_CORRECTION,
            identity_score=None,
        )
        set_claim_status(conn, claim_id, ClaimStatus.SUPERSEDED)

        return {
            "action": "corrected",
            "old_claim_id": claim_id,
            "new_claim_id": new_claim.claim_id,
            "edge_id": edge.edge_id,
            "new_value": new_value,
        }

    return {"error": f"Unknown action: {action}"}


def handle_memory_observe(
    conn: sqlite3.Connection,
    *,
    url: str,
    title: str = "",
    accessibility_tree: dict | None = None,
    session_id: str | None = None,
) -> dict:
    """Store browser observation claims from a page snapshot."""
    from datetime import datetime, timezone

    from memcontext.observe.browser import PageSnapshot, observe_page

    sid = session_id or f"observe_{uuid.uuid4().hex[:8]}"
    snapshot = PageSnapshot(
        url=url,
        title=title,
        timestamp=datetime.now(timezone.utc).isoformat(),
        accessibility_tree=accessibility_tree or {},
    )
    result = observe_page(conn, snapshot=snapshot, session_id=sid)
    return {
        "session_id": sid,
        "turn_id": result.turn_id,
        "claims_stored": len(result.claims),
        "claims": [
            {"subject": c.get("subject", ""), "predicate": c.get("predicate", ""), "value": c.get("value", "")}
            for c in result.claims
        ],
        "snapshot_id": snapshot.snapshot_id,
    }


def _capture_page(
    url: str,
    *,
    login_email: str | None = None,
    login_password: str | None = None,
    login_url: str | None = None,
    connect_browser: bool = False,
) -> tuple[str, dict, str]:
    """Capture a page's a11y tree + DOM hash. Returns (title, tree, hash).

    Three modes:
    - connect_browser=True: attach to the user's running Chrome
      (started with --remote-debugging-port=9222). Reads the page the
      user can see — inherits all auth sessions, cookies, SSO, 2FA.
    - login_email/password: launch headless, fill login form, then read.
    - neither: launch headless, read the page as-is.
    """
    import hashlib

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        if connect_browser:
            source = p.chromium.connect_over_cdp("http://localhost:9222")
            cookies = source.contexts[0].cookies()
            source.close()

            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            context.add_cookies(cookies)
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            owns_browser = True
        else:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            owns_browser = True

            if login_email and login_password:
                target = login_url or url
                page.goto(target, wait_until="networkidle", timeout=30000)
                email_field = page.locator(
                    "input[type='email'], input[name='email'], "
                    "input[autocomplete='email'], input[autocomplete='username']"
                ).first
                password_field = page.locator("input[type='password']").first
                email_field.fill(login_email)
                password_field.fill(login_password)
                page.locator(
                    "button[type='submit'], button:has-text('Sign in'), "
                    "button:has-text('Log in'), button:has-text('Login')"
                ).first.click()
                page.wait_for_load_state("networkidle", timeout=15000)
                if login_url and login_url != url:
                    page.goto(url, wait_until="networkidle", timeout=30000)
            else:
                page.goto(url, wait_until="networkidle", timeout=30000)

        title = page.title()

        cdp = page.context.new_cdp_session(page)
        ax_result = cdp.send("Accessibility.getFullAXTree")
        a11y = _cdp_to_tree(ax_result.get("nodes", []))

        content = page.content()
        dom_hash = hashlib.sha256(content.encode()).hexdigest()
        if owns_browser:
            browser.close()
        return title, a11y, dom_hash


def _cdp_to_tree(nodes: list[dict]) -> dict:
    """Convert CDP flat node list to nested {role, name, value, children} tree."""
    if not nodes:
        return {}

    by_id: dict[str, dict] = {}
    for node in nodes:
        if node.get("ignored"):
            by_id[node["nodeId"]] = {"_skip": True, "_child_ids": node.get("childIds", [])}
            continue
        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "")
        value_obj = node.get("value", {})
        value = value_obj.get("value", "") if isinstance(value_obj, dict) else ""
        by_id[node["nodeId"]] = {
            "role": role,
            "name": name,
            "value": str(value) if value else "",
            "children": [],
            "_child_ids": node.get("childIds", []),
        }

    def _build(nid: str) -> dict | None:
        entry = by_id.get(nid)
        if not entry:
            return None
        child_ids = entry.pop("_child_ids", [])
        if entry.get("_skip"):
            # Ignored node: promote its children
            kids = []
            for cid in child_ids:
                kid = _build(cid)
                if kid:
                    kids.append(kid)
            return {"role": "none", "name": "", "children": kids} if kids else None
        for cid in child_ids:
            kid = _build(cid)
            if kid:
                if kid.get("role") == "none" and kid.get("children"):
                    entry["children"].extend(kid["children"])
                else:
                    entry["children"].append(kid)
        return entry

    root = _build(nodes[0]["nodeId"])
    return root or {}


def handle_memory_observe_url(
    conn: sqlite3.Connection,
    *,
    url: str,
    session_id: str | None = None,
    login_email: str | None = None,
    login_password: str | None = None,
    login_url: str | None = None,
    connect_browser: bool = False,
) -> dict:
    """Observe a live URL, capture a11y tree, extract and store claims.

    Auth modes (pick one):
    - connect_browser=True — attach to user's running Chrome on port 9222.
      Inherits all sessions: SSO, 2FA, OAuth, saved passwords. No
      credentials needed. This is how CUAs work on Cloud PCs.
    - login_email/password — launch headless, fill login form, then read.
    - neither — launch headless, read the page unauthenticated.

    If the URL was previously observed in the same session, supersession
    fires automatically for changed values.
    """
    from datetime import datetime, timezone

    from memcontext.observe.browser import PageSnapshot, observe_page
    from memcontext.observe.extractors import _url_to_subject

    sid = session_id or "observe_default"
    title, a11y_tree, dom_hash = _capture_page(
        url,
        login_email=login_email,
        login_password=login_password,
        login_url=login_url,
        connect_browser=connect_browser,
    )

    url_subject = _url_to_subject(url)
    prev_count = conn.execute(
        "SELECT COUNT(*) FROM claims"
        " WHERE session_id = ? AND subject = ?"
        " AND status IN ('active','confirmed','audited')",
        (sid, url_subject),
    ).fetchone()[0]
    is_revisit = prev_count > 0

    snapshot = PageSnapshot(
        url=url,
        title=title,
        timestamp=datetime.now(timezone.utc).isoformat(),
        accessibility_tree=a11y_tree,
        dom_hash=dom_hash,
    )
    result = observe_page(conn, snapshot=snapshot, session_id=sid)

    supersessions = conn.execute(
        "SELECT e.old_claim_id, e.new_claim_id, e.edge_type,"
        "       c_old.value AS old_value, c_new.value AS new_value"
        " FROM supersession_edges e"
        " JOIN claims c_old ON e.old_claim_id = c_old.claim_id"
        " JOIN claims c_new ON e.new_claim_id = c_new.claim_id"
        " WHERE c_new.source_turn_id = ?",
        (result.turn_id,),
    ).fetchall() if result.turn_id else []

    resp = {
        "session_id": sid,
        "url": url,
        "title": title,
        "dom_hash": dom_hash[:12],
        "a11y_nodes": _count_a11y_nodes(a11y_tree),
        "claims_stored": len(result.claims),
        "claims": [
            {"subject": c.get("subject", ""), "predicate": c.get("predicate", ""), "value": c.get("value", "")}
            for c in result.claims
        ],
        "snapshot_id": snapshot.snapshot_id,
        "is_revisit": is_revisit,
    }

    if supersessions:
        resp["changes_detected"] = [
            {
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "edge_type": row["edge_type"],
            }
            for row in supersessions
        ]
        resp["supersessions"] = len(supersessions)

    return resp


def _count_a11y_nodes(tree: dict) -> int:
    """Count nodes in an accessibility tree."""
    if not isinstance(tree, dict):
        return 0
    count = 1
    for child in tree.get("children", []):
        count += _count_a11y_nodes(child)
    return count
