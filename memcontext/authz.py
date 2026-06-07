"""Per-principal access control — bind a bearer token to a principal scoped to a
namespace + read/write permission (GOVERNANCE_AUDIT D).

Tokens are stored HASHED (sha256), never in plaintext, so a leaked database does
not leak usable credentials. A request's token resolves to a `Principal`; the HTTP
transport then bounds that caller to its namespace (substrate isolation from P5)
and gates mutations on `can_write`. Deterministic, zero-LLM.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    name: str
    namespace: str
    can_write: bool


def hash_token(token: str) -> str:
    """sha256 of the bearer token — what we store and compare against."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register_principal(
    conn: sqlite3.Connection,
    *,
    token: str,
    principal: str,
    namespace: str,
    can_write: bool = True,
) -> None:
    """Grant (or update) a principal a scoped access token."""
    conn.execute(
        "INSERT OR REPLACE INTO principals"
        " (token_hash, principal, namespace, can_write, created_ts)"
        " VALUES (?, ?, ?, ?, ?)",
        (hash_token(token), principal, namespace, 1 if can_write else 0, time.time_ns()),
    )


def resolve_principal(conn: sqlite3.Connection, token: str | None) -> Principal | None:
    """Resolve a bearer token to its Principal, or None if unknown."""
    if not token:
        return None
    row = conn.execute(
        "SELECT principal, namespace, can_write FROM principals WHERE token_hash = ?",
        (hash_token(token),),
    ).fetchone()
    if row is None:
        return None
    return Principal(name=row[0], namespace=row[1], can_write=bool(row[2]))


def any_principals(conn: sqlite3.Connection) -> bool:
    """True once at least one principal is registered (enables per-principal authz;
    until then the single shared token applies, for backward compatibility)."""
    return conn.execute("SELECT 1 FROM principals LIMIT 1").fetchone() is not None
