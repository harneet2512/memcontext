"""Smart profiles — deterministic, zero LLM."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProfileLine:
    text: str
    claim_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    volatility: str  # "stable" | "evolving" | "volatile"
    chain_summary: str  # empty if no chain


@dataclass(slots=True)
class SmartProfile:
    subject: str
    lines: list[ProfileLine] = field(default_factory=list)
    total_facts: int = 0
    total_sessions: int = 0
    total_updates: int = 0
    most_volatile: str = ""
    most_stable: str = ""


# ---------------------------------------------------------------- helpers ---


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 tokens per word."""
    return len(text.split()) * 4


def _volatility_label(count: int) -> str:
    """Map supersession edge count to a volatility label."""
    if count == 0:
        return "stable"
    if count <= 2:
        return "evolving"
    return "volatile"


def _get_volatility_map(
    conn: sqlite3.Connection, subject: str,
) -> dict[str, int]:
    """Count supersession edges per (subject, predicate) pair.

    Returns {predicate: edge_count}.
    """
    rows = conn.execute(
        """
        SELECT c.predicate, COUNT(e.edge_id) AS edge_count
        FROM claims c
        JOIN supersession_edges e
            ON c.claim_id = e.old_claim_id OR c.claim_id = e.new_claim_id
        WHERE c.subject = ?
        GROUP BY c.predicate
        """,
        (subject,),
    ).fetchall()
    result: dict[str, int] = {}
    for r in rows:
        result[r["predicate"]] = r["edge_count"]
    return result


def _chain_summary_for_claim(
    conn: sqlite3.Connection, claim_id: str,
) -> str:
    """Build a short chain summary string, or empty if no predecessors."""
    from memcontext.chains import build_chain, format_chain

    chain = build_chain(conn, claim_id)
    if len(chain) <= 1:
        return ""
    return format_chain(chain)


# ------------------------------------------------------------- core logic ---


def build_smart_profile(
    conn: sqlite3.Connection,
    subject: str,
    *,
    max_tokens: int = 500,
) -> SmartProfile:
    """Build a tiered profile for *subject* under a token budget.

    Tiers (added in order until budget exhausted):
      1. High-importance stable claims (importance >= 0.8, stable)
      2. Evolving claims with chain context
      3. User-preference claims
      4. Life events
      5. Volatile claims with change counts
    """
    profile = SmartProfile(subject=subject)
    used_tokens = 0

    # ---- Gather active claims with metadata ----
    rows = conn.execute(
        """
        SELECT c.*, COALESCE(m.importance_score, 0.5) AS importance_score
        FROM claims c
        LEFT JOIN claim_metadata m ON c.claim_id = m.claim_id
        WHERE c.subject = ?
          AND c.status IN ('active', 'confirmed', 'audited')
        ORDER BY COALESCE(m.importance_score, 0.5) DESC, c.created_ts ASC
        """,
        (subject,),
    ).fetchall()

    if not rows:
        log.debug("profiles.no_claims", subject=subject)
        return profile

    volatility_map = _get_volatility_map(conn, subject)

    # Build lookup structures
    claims_by_predicate: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        claims_by_predicate.setdefault(r["predicate"], []).append(r)

    used_claim_ids: set[str] = set()

    def _add_line(text: str, claim_ids: tuple[str, ...],
                  source_turn_ids: tuple[str, ...], volatility: str,
                  chain_summary: str) -> bool:
        """Add a line if budget allows. Returns True if added."""
        nonlocal used_tokens
        tokens = _estimate_tokens(text)
        if used_tokens + tokens > max_tokens:
            return False
        profile.lines.append(
            ProfileLine(
                text=text,
                claim_ids=claim_ids,
                source_turn_ids=source_turn_ids,
                volatility=volatility,
                chain_summary=chain_summary,
            )
        )
        used_tokens += tokens
        used_claim_ids.update(claim_ids)
        return True

    # ---- Tier 1: High-importance stable claims ----
    for r in rows:
        if r["claim_id"] in used_claim_ids:
            continue
        importance = r["importance_score"]
        vol_count = volatility_map.get(r["predicate"], 0)
        vol = _volatility_label(vol_count)
        if importance >= 0.8 and vol == "stable":
            text = f"[{r['predicate']}] {r['value']}"
            if not _add_line(
                text,
                (r["claim_id"],),
                (r["source_turn_id"],),
                vol,
                "",
            ):
                break

    # ---- Tier 2: Evolving claims with chain context ----
    if used_tokens < max_tokens:
        for r in rows:
            if r["claim_id"] in used_claim_ids:
                continue
            vol_count = volatility_map.get(r["predicate"], 0)
            vol = _volatility_label(vol_count)
            if vol != "evolving":
                continue
            chain_sum = _chain_summary_for_claim(conn, r["claim_id"])
            # Find the previous value from the chain
            from memcontext.claims import get_supersession_chain
            predecessors = get_supersession_chain(conn, r["claim_id"])
            if predecessors:
                prev_value = predecessors[-1][0].value
                text = f"[{r['predicate']}] {r['value']} (previously: {prev_value})"
            else:
                text = f"[{r['predicate']}] {r['value']}"
            if not _add_line(
                text,
                (r["claim_id"],),
                (r["source_turn_id"],),
                vol,
                chain_sum,
            ):
                break

    # ---- Tier 3: User-preference claims ----
    if used_tokens < max_tokens:
        pref_rows = [
            r for r in rows
            if r["predicate"] == "user_preference" and r["claim_id"] not in used_claim_ids
        ]
        for r in pref_rows:
            vol_count = volatility_map.get(r["predicate"], 0)
            vol = _volatility_label(vol_count)
            text = f"[{r['predicate']}] {r['value']}"
            if not _add_line(
                text,
                (r["claim_id"],),
                (r["source_turn_id"],),
                vol,
                "",
            ):
                break

    # ---- Tier 4: Life events ----
    if used_tokens < max_tokens:
        life_rows = conn.execute(
            """
            SELECT * FROM life_events
            WHERE subject = ?
            ORDER BY significance DESC, timestamp_start ASC
            """,
            (subject,),
        ).fetchall()
        for le in life_rows:
            text = f"[life_event] {le['summary_text']}"
            claim_ids_list = [
                cid.strip()
                for cid in le["claim_ids"].split(",")
                if cid.strip()
            ]
            if not _add_line(
                text,
                tuple(claim_ids_list),
                (),  # life events don't have direct source_turn_ids
                "stable",
                "",
            ):
                break

    # ---- Tier 5: Volatile predicates with change counts ----
    if used_tokens < max_tokens:
        for r in rows:
            if r["claim_id"] in used_claim_ids:
                continue
            vol_count = volatility_map.get(r["predicate"], 0)
            vol = _volatility_label(vol_count)
            if vol != "volatile":
                continue
            text = f"[{r['predicate']}] {r['value']} (changed {vol_count} times)"
            chain_sum = _chain_summary_for_claim(conn, r["claim_id"])
            if not _add_line(
                text,
                (r["claim_id"],),
                (r["source_turn_id"],),
                vol,
                chain_sum,
            ):
                break

    # ---- Footer stats ----
    profile.total_facts = len(rows)

    session_row = conn.execute(
        """
        SELECT COUNT(DISTINCT session_id) AS cnt
        FROM claims
        WHERE subject = ? AND status IN ('active', 'confirmed', 'audited')
        """,
        (subject,),
    ).fetchone()
    profile.total_sessions = session_row["cnt"] if session_row else 0

    update_row = conn.execute(
        """
        SELECT COUNT(e.edge_id) AS cnt
        FROM supersession_edges e
        JOIN claims c ON e.old_claim_id = c.claim_id OR e.new_claim_id = c.claim_id
        WHERE c.subject = ?
        """,
        (subject,),
    ).fetchone()
    profile.total_updates = update_row["cnt"] if update_row else 0

    # Most volatile / most stable predicates
    if volatility_map:
        profile.most_volatile = max(volatility_map, key=volatility_map.get)  # type: ignore[arg-type]
        profile.most_stable = min(volatility_map, key=volatility_map.get)  # type: ignore[arg-type]
    elif rows:
        # All predicates are stable (no supersession edges)
        profile.most_stable = rows[0]["predicate"]

    log.info(
        "profiles.built",
        subject=subject,
        lines=len(profile.lines),
        total_facts=profile.total_facts,
        tokens_used=used_tokens,
    )
    return profile


# ---------------------------------------------------------- persistence ---


def store_profile(conn: sqlite3.Connection, profile: SmartProfile) -> None:
    """Persist a SmartProfile into the ``profiles`` table."""
    profile_text = format_profile(profile)
    profile_data = json.dumps(
        {
            "subject": profile.subject,
            "lines": [
                {
                    "text": line.text,
                    "claim_ids": list(line.claim_ids),
                    "source_turn_ids": list(line.source_turn_ids),
                    "volatility": line.volatility,
                    "chain_summary": line.chain_summary,
                }
                for line in profile.lines
            ],
            "total_facts": profile.total_facts,
            "total_sessions": profile.total_sessions,
            "total_updates": profile.total_updates,
            "most_volatile": profile.most_volatile,
            "most_stable": profile.most_stable,
        },
        ensure_ascii=False,
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO profiles
            (subject, profile_text, profile_data, claim_count, session_count, built_at_ts)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            profile.subject,
            profile_text,
            profile_data,
            profile.total_facts,
            profile.total_sessions,
            time.time_ns(),
        ),
    )
    log.info("profiles.stored", subject=profile.subject)


def load_profile(conn: sqlite3.Connection, subject: str) -> SmartProfile | None:
    """Load a SmartProfile from the ``profiles`` table, or None."""
    row = conn.execute(
        "SELECT * FROM profiles WHERE subject = ?", (subject,)
    ).fetchone()
    if row is None:
        return None

    data = json.loads(row["profile_data"])
    profile = SmartProfile(
        subject=data["subject"],
        total_facts=data.get("total_facts", row["claim_count"]),
        total_sessions=data.get("total_sessions", row["session_count"]),
        total_updates=data.get("total_updates", 0),
        most_volatile=data.get("most_volatile", ""),
        most_stable=data.get("most_stable", ""),
    )
    for line_data in data.get("lines", []):
        profile.lines.append(
            ProfileLine(
                text=line_data["text"],
                claim_ids=tuple(line_data.get("claim_ids", ())),
                source_turn_ids=tuple(line_data.get("source_turn_ids", ())),
                volatility=line_data.get("volatility", "stable"),
                chain_summary=line_data.get("chain_summary", ""),
            )
        )
    return profile


def format_profile(profile: SmartProfile) -> str:
    """Format a SmartProfile as a readable text block.

    Example::

        [PROFILE] user — 47 facts, 12 sessions
          [user_fact] Name: Sarah Chen
          [user_fact] Location: San Francisco (previously: Portland)
          ...
        Footer: 47 facts, 12 sessions, 8 updates
    """
    header = (
        f"[PROFILE] {profile.subject}"
        f" — {profile.total_facts} facts, {profile.total_sessions} sessions"
    )
    body_lines = [f"  {line.text}" for line in profile.lines]
    footer = (
        f"Footer: {profile.total_facts} facts,"
        f" {profile.total_sessions} sessions,"
        f" {profile.total_updates} updates"
    )
    parts = [header]
    if body_lines:
        parts.extend(body_lines)
    parts.append(footer)
    return "\n".join(parts)
