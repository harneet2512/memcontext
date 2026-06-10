"""OAuth 2.1 for the MemContext MCP HTTP server.

Gives remote clients (claude.ai web) the Membase-style "paste URL → log in" flow:
metadata discovery + Dynamic Client Registration (RFC 7591) + PKCE authorization-code
+ token, all served by the MCP SDK's spec-compliant routes. We implement only the
provider logic and a password-gated login page — the gate is what makes OAuth
meaningful on a public tunnel (without it, anyone with the URL completes the flow).

Single-user, in-memory: one shared password authenticates the human; clients register
dynamically; tokens live in memory (a restart invalidates them — the client just
re-runs the flow). For multi-tenant hosting, swap the in-memory stores for a DB.

Zero crypto hand-rolled: PKCE verification, metadata, and the OAuth endpoints come
from the `mcp` SDK; we only mint/track codes and tokens.
"""
from __future__ import annotations

import secrets
import time

import structlog
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.server.auth.routes import (
    build_resource_metadata_url,
    create_auth_routes,
    create_protected_resource_routes,
)
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Mount, Route

log = structlog.get_logger(__name__)

_SCOPE = "memory"
_CODE_TTL = 300       # seconds an auth code is valid
_TOKEN_TTL = 3600     # access-token lifetime

_LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>MemContext — authorize</title>
<style>body{{font-family:system-ui;max-width:24rem;margin:5rem auto;padding:1rem}}
input,button{{font-size:1rem;padding:.5rem;width:100%;box-sizing:border-box;margin:.3rem 0}}
.err{{color:#b00}}</style></head><body>
<h2>MemContext</h2><p>Authorize this client to access your memory.</p>
<p class="err">{err}</p>
<form method="post" action="/memcontext/login">
<input type="hidden" name="rid" value="{rid}">
<input type="password" name="password" placeholder="MemContext password" autofocus>
<button type="submit">Authorize</button></form></body></html>"""


class MemContextOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Single-password OAuth provider; also serves as the TokenVerifier.

    Durable state (registered clients, access + refresh tokens) persists to a
    SQLite *sidecar* (``state_path``) so a restart does NOT log every client out —
    required for a stable connector URL to feel permanent. Short-lived state
    (pending logins, 5-minute auth codes) stays in memory: a restart mid-login
    just means redoing the login. With ``state_path=None`` everything is in-memory
    (tests, throwaway runs).
    """

    def __init__(self, *, public_url: str, password: str, state_path: str | None = None) -> None:
        import sqlite3

        self._public = public_url.rstrip("/")
        self._password = password
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending: dict[str, tuple[str, AuthorizationParams]] = {}
        self._codes: dict[str, AuthorizationCode] = {}
        self._access: dict[str, AccessToken] = {}
        self._refresh: dict[str, tuple[str, list[str]]] = {}
        self._db: sqlite3.Connection | None = None
        if state_path:
            self._db = sqlite3.connect(state_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(
                "CREATE TABLE IF NOT EXISTS oauth_clients ("
                " client_id TEXT PRIMARY KEY, client_json TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS oauth_access ("
                " token TEXT PRIMARY KEY, client_id TEXT NOT NULL, scopes TEXT NOT NULL,"
                " expires_at INTEGER, resource TEXT);"
                "CREATE TABLE IF NOT EXISTS oauth_refresh ("
                " token TEXT PRIMARY KEY, client_id TEXT NOT NULL, scopes TEXT NOT NULL);"
            )
            self._db.commit()

    # --- sidecar persistence helpers (no-ops when state_path is None) ---
    def _persist_client(self, c: OAuthClientInformationFull) -> None:
        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO oauth_clients VALUES (?, ?)",
                (c.client_id, c.model_dump_json()),
            )
            self._db.commit()

    def _load_client_db(self, client_id: str) -> OAuthClientInformationFull | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT client_json FROM oauth_clients WHERE client_id = ?", (client_id,)
        ).fetchone()
        return OAuthClientInformationFull.model_validate_json(row[0]) if row else None

    def _persist_access(self, at: AccessToken) -> None:
        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO oauth_access VALUES (?, ?, ?, ?, ?)",
                (at.token, at.client_id, " ".join(at.scopes), at.expires_at, at.resource),
            )
            self._db.commit()

    def _load_access_db(self, token: str) -> AccessToken | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT token, client_id, scopes, expires_at, resource"
            " FROM oauth_access WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        return AccessToken(token=row[0], client_id=row[1], scopes=row[2].split(),
                           expires_at=row[3], resource=row[4])

    def _persist_refresh(self, token: str, client_id: str, scopes: list[str]) -> None:
        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO oauth_refresh VALUES (?, ?, ?)",
                (token, client_id, " ".join(scopes)),
            )
            self._db.commit()

    def _load_refresh_db(self, token: str) -> tuple[str, list[str]] | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT client_id, scopes FROM oauth_refresh WHERE token = ?", (token,)
        ).fetchone()
        return (row[0], row[1].split()) if row else None

    def _delete_token_db(self, token: str) -> None:
        if self._db is not None:
            self._db.execute("DELETE FROM oauth_access WHERE token = ?", (token,))
            self._db.execute("DELETE FROM oauth_refresh WHERE token = ?", (token,))
            self._db.commit()

    # --- client registration (DCR) ---
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        c = self._clients.get(client_id) or self._load_client_db(client_id)
        if c is not None:
            self._clients[client_id] = c
        return c

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info
        self._persist_client(client_info)

    # --- authorization ---
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Hand off to our password-gated login page; return its URL for the 302."""
        rid = secrets.token_urlsafe(16)
        self._pending[rid] = (client.client_id, params)
        return f"{self._public}/memcontext/login?rid={rid}"

    def password_ok(self, password: str) -> bool:
        return secrets.compare_digest(password, self._password)

    def complete_login(self, rid: str) -> str | None:
        """After a correct password: mint an auth code, return the client redirect URL."""
        if rid not in self._pending:
            return None
        client_id, params = self._pending.pop(rid)
        code = secrets.token_urlsafe(24)
        self._codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [_SCOPE],
            expires_at=time.time() + _CODE_TTL,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        sep = "&" if "?" in str(params.redirect_uri) else "?"
        url = f"{params.redirect_uri}{sep}code={code}"
        if params.state:
            url += f"&state={params.state}"
        return url

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        ac = self._codes.get(authorization_code)
        if ac and ac.client_id == client.client_id and ac.expires_at > time.time():
            return ac
        return None

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        self._codes.pop(authorization_code.code, None)
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        scopes = list(authorization_code.scopes)
        at = AccessToken(
            token=access, client_id=client.client_id, scopes=scopes,
            expires_at=int(time.time() + _TOKEN_TTL), resource=authorization_code.resource,
        )
        self._access[access] = at
        self._refresh[refresh] = (client.client_id, scopes)
        self._persist_access(at)
        self._persist_refresh(refresh, client.client_id, scopes)
        return OAuthToken(
            access_token=access, token_type="Bearer", expires_in=_TOKEN_TTL,
            scope=" ".join(scopes) or None, refresh_token=refresh,
        )

    # --- refresh ---
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        rec = self._refresh.get(refresh_token) or self._load_refresh_db(refresh_token)
        if rec and rec[0] == client.client_id:
            return RefreshToken(token=refresh_token, client_id=client.client_id,
                                scopes=rec[1], expires_at=None)
        return None

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        self._refresh.pop(refresh_token.token, None)
        self._delete_token_db(refresh_token.token)
        sc = scopes or refresh_token.scopes
        access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(32)
        at = AccessToken(
            token=access, client_id=client.client_id, scopes=sc,
            expires_at=int(time.time() + _TOKEN_TTL), resource=None,
        )
        self._access[access] = at
        self._refresh[new_refresh] = (client.client_id, sc)
        self._persist_access(at)
        self._persist_refresh(new_refresh, client.client_id, sc)
        return OAuthToken(
            access_token=access, token_type="Bearer", expires_in=_TOKEN_TTL,
            scope=" ".join(sc) or None, refresh_token=new_refresh,
        )

    # --- token verification (Resource Server + TokenVerifier) ---
    async def load_access_token(self, token: str) -> AccessToken | None:
        at = self._access.get(token) or self._load_access_db(token)
        if at and (at.expires_at is None or at.expires_at > time.time()):
            self._access[token] = at
            return at
        return None

    async def verify_token(self, token: str) -> AccessToken | None:
        return await self.load_access_token(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        tok = getattr(token, "token", None)
        if tok:
            self._access.pop(tok, None)
            self._refresh.pop(tok, None)
            self._delete_token_db(tok)


def build_oauth_http_app(*, db_path: str, public_url: str, password: str):
    """Starlette app: OAuth (metadata/DCR/authorize/token) + login page + protected /mcp."""
    from memcontext.mcp_server import create_http_app

    public = public_url.rstrip("/")
    issuer = AnyHttpUrl(public)
    resource_url = AnyHttpUrl(f"{public}/mcp")
    # Durable OAuth state lives in a sidecar next to the brain DB (never inside it):
    # clients + tokens survive restarts, so the user logs in once, not per restart.
    state_path = None if db_path == ":memory:" else f"{db_path}.oauth.db"
    provider = MemContextOAuthProvider(
        public_url=public, password=password, state_path=state_path
    )

    reg = ClientRegistrationOptions(
        enabled=True, valid_scopes=[_SCOPE], default_scopes=[_SCOPE]
    )
    auth_routes = create_auth_routes(provider, issuer, client_registration_options=reg)
    resource_routes = create_protected_resource_routes(
        resource_url, [issuer], scopes_supported=[_SCOPE], resource_name="MemContext memory"
    )

    async def login_get(request: Request) -> HTMLResponse:
        rid = request.query_params.get("rid", "")
        return HTMLResponse(_LOGIN_HTML.format(rid=rid, err=""))

    # Brute-force guard: the login page is the ONLY thing standing between a public
    # URL and the brain, so failed attempts back off exponentially per client IP
    # (5 free tries, then 30s, 60s, 120s, ... lockout). In-memory; resets on restart.
    _fails: dict[str, tuple[int, float]] = {}
    _FREE_TRIES, _LOCK_BASE = 5, 30.0

    async def login_post(request: Request):
        ip = request.client.host if request.client else "?"
        n, locked_until = _fails.get(ip, (0, 0.0))
        if time.time() < locked_until:
            return HTMLResponse("too many attempts — try again later", status_code=429)
        form = await request.form()
        rid = str(form.get("rid", ""))
        password_in = str(form.get("password", ""))

        def _count_failure() -> None:
            # EVERY failed attempt counts (bad rid included) — an attacker can mint
            # valid rids via DCR+authorize, so only counting bad passwords would
            # leave the password brute-forceable through fresh rids.
            nonlocal n
            n += 1
            lock = _LOCK_BASE * (2 ** (n - _FREE_TRIES)) if n >= _FREE_TRIES else 0.0
            _fails[ip] = (n, time.time() + lock)
            log.warning("oauth.login_failed", ip=ip, consecutive=n)

        if rid not in provider._pending:
            _count_failure()
            return HTMLResponse("invalid or expired authorization request", status_code=400)
        if not provider.password_ok(password_in):
            _count_failure()
            return HTMLResponse(_LOGIN_HTML.format(rid=rid, err="Wrong password."), status_code=401)
        _fails.pop(ip, None)
        target = provider.complete_login(rid)
        if not target:
            return HTMLResponse("authorization request expired", status_code=400)
        return RedirectResponse(target, status_code=302)

    login_routes = [
        Route("/memcontext/login", login_get, methods=["GET"]),
        Route("/memcontext/login", login_post, methods=["POST"]),
    ]

    protected_mcp = RequireAuthMiddleware(
        create_http_app(db_path),
        required_scopes=[_SCOPE],
        resource_metadata_url=build_resource_metadata_url(resource_url),
    )

    # The bare connector URL ".../mcp" 307-redirects to ".../mcp/" (Starlette Mount
    # semantics); standard HTTP/MCP clients follow it preserving POST + auth headers.
    return Starlette(
        routes=[*auth_routes, *resource_routes, *login_routes, Mount("/mcp", app=protected_mcp)],
        middleware=[
            Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(provider)),
            Middleware(AuthContextMiddleware),
        ],
    )
