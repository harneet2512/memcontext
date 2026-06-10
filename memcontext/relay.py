"""MemContext relay — the invisible plumbing behind `memcontext share`.

The two-line contract: `pip install memcontext` -> you own a brain (a local file);
`memcontext share` -> you get a link that never changes. The user never sees what
carries the bytes. This module is that carrier:

- RELAY SERVER (runs on our box, one for all users): a dumb forwarder. Brains dial
  OUT to it over WebSocket; web clients (claude.ai, ChatGPT) hit
  ``https://<brain_id>.<relay-domain>/mcp`` and the relay pipes the request down the
  brain's connection. It stores NOTHING — an in-memory routing table and bytes in
  flight. Laptop offline -> 503 "brain offline".

- BRAIN CONNECTOR (runs inside `share` on the user's machine): generates an Ed25519
  keypair next to the DB on first run. ``brain_id = sha256(public_key)`` — the URL
  is *derived from the key*, so the same key always yields the same URL (the "never
  changes" guarantee) and the relay can verify ownership by signature alone: no
  accounts, no signup, and a relay that cannot be tricked into giving your URL to
  someone else (they'd have to forge an Ed25519 signature).

Identity is self-certifying; the relay is dumb and replaceable.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_FRAME_LIMIT = 10 * 1024 * 1024  # 10 MB per message — far above MCP payloads
_REQUEST_TIMEOUT = 60.0


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


# ----------------------------------------------------------------- identity ---


def load_or_create_identity(key_path: str | Path):
    """Ed25519 keypair for this brain; created on first use, reused forever.

    Returns (private_key, brain_id). brain_id = first 16 hex chars of
    sha256(raw public key) — stable for the life of the key file, which is the
    stable-URL guarantee.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    p = Path(key_path)
    if p.exists():
        key = serialization.load_pem_private_key(p.read_bytes(), password=None)
    else:
        key = Ed25519PrivateKey.generate()
        p.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    pub_raw = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    brain_id = hashlib.sha256(pub_raw).hexdigest()[:16]
    return key, brain_id


# ------------------------------------------------------------- relay server ---


def create_relay_app():
    """The public forwarder. Routes by brain_id; stores nothing.

    Brain registration: WS /register — server sends a nonce; brain replies with
    {brain_id, pubkey, sig(nonce)}; server checks the signature AND that
    sha256(pubkey) == brain_id, so only the keyholder can claim a URL.

    Client traffic: ``/b/<brain_id>/<path>`` (and, in production, Host-header
    subdomain routing — same table) is framed as JSON and piped down the brain's
    socket; the response frame is piped back.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket, WebSocketDisconnect

    # brain_id -> (websocket, send_lock, pending {req_id -> Future})
    table: dict[str, tuple] = {}

    async def register(ws: WebSocket) -> None:
        await ws.accept()
        nonce = secrets.token_bytes(32)
        await ws.send_text(json.dumps({"nonce": _b64(nonce)}))
        try:
            hello = json.loads(await ws.receive_text())
            pub_raw = _unb64(hello["pubkey"])
            brain_id = hello["brain_id"]
            if hashlib.sha256(pub_raw).hexdigest()[:16] != brain_id:
                await ws.close(code=4403)
                return
            Ed25519PublicKey.from_public_bytes(pub_raw).verify(_unb64(hello["sig"]), nonce)
        except Exception:  # bad hello / bad signature — not our brain
            await ws.close(code=4403)
            return

        send_lock = asyncio.Lock()
        pending: dict[str, asyncio.Future] = {}
        table[brain_id] = (ws, send_lock, pending)
        log.info("relay.brain_online", brain_id=brain_id)
        try:
            while True:
                frame = json.loads(await ws.receive_text())
                fut = pending.pop(frame.get("id", ""), None)
                if fut is not None and not fut.done():
                    fut.set_result(frame)
        except WebSocketDisconnect:
            pass
        finally:
            if table.get(brain_id, (None,))[0] is ws:
                del table[brain_id]
            for fut in pending.values():
                if not fut.done():
                    fut.cancel()
            log.info("relay.brain_offline", brain_id=brain_id)

    import re as _re

    _HEX16 = _re.compile(r"^[0-9a-f]{16}$")

    async def _forward(brain_id: str, path: str, request: Request) -> Response:
        entry = table.get(brain_id)
        if entry is None:
            return Response(
                json.dumps({"error": "brain offline"}), status_code=503,
                media_type="application/json",
            )
        ws, send_lock, pending = entry
        req_id = secrets.token_hex(8)
        body = await request.body()
        frame = {
            "id": req_id,
            "method": request.method,
            "path": "/" + path,
            "query": request.url.query,
            "headers": {k: v for k, v in request.headers.items()
                        if k.lower() not in ("host", "content-length")},
            "body": _b64(body),
        }
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        pending[req_id] = fut
        async with send_lock:
            await ws.send_text(json.dumps(frame))
        try:
            resp = await asyncio.wait_for(fut, timeout=_REQUEST_TIMEOUT)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pending.pop(req_id, None)
            return Response(
                json.dumps({"error": "brain timeout"}), status_code=504,
                media_type="application/json",
            )
        headers = {k: v for k, v in resp.get("headers", {}).items()
                   if k.lower() not in ("content-length", "transfer-encoding")}
        return Response(_unb64(resp.get("body", "")), status_code=resp.get("status", 502),
                        headers=headers)

    _METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

    async def forward_by_path(request: Request) -> Response:
        # Dev/testing route: /b/<brain_id>/<path>.
        return await _forward(
            request.path_params["brain_id"], request.path_params.get("path", ""), request
        )

    async def forward_by_host(request: Request) -> Response:
        # Production route: https://<brain_id>.<relay-domain>/<path> — subdomain
        # routing keeps the brain's OAuth issuer a clean origin (required for
        # /.well-known discovery, which always resolves against the URL root).
        host = request.headers.get("host", "").split(":")[0]
        label = host.split(".", 1)[0]
        if not _HEX16.match(label):
            return Response(json.dumps({"error": "unknown brain"}), status_code=404,
                            media_type="application/json")
        return await _forward(label, request.path_params.get("path", ""), request)

    async def health(_request: Request) -> Response:
        return Response(json.dumps({"online_brains": len(table)}),
                        media_type="application/json")

    return Starlette(routes=[
        WebSocketRoute("/register", register),
        Route("/healthz", health),
        Route("/b/{brain_id}/{path:path}", forward_by_path, methods=_METHODS),
        Route("/{path:path}", forward_by_host, methods=_METHODS),
    ])


# ---------------------------------------------------------- brain connector ---


class BrainConnector:
    """Runs inside `memcontext share`: dials OUT to the relay, answers forwarded
    requests against the local server, reconnects forever with backoff."""

    def __init__(self, *, key_path: str | Path, relay_ws_url: str, local_port: int) -> None:
        self._key, self.brain_id = load_or_create_identity(key_path)
        self._relay = relay_ws_url.rstrip("/") + "/register"
        self._local = f"http://127.0.0.1:{local_port}"
        self._stop = asyncio.Event()
        self._ws = None  # the live socket, so stop() can actually sever it

    async def _handle(self, ws, send_lock: asyncio.Lock, frame: dict) -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT,
                                         follow_redirects=True) as client:
                url = self._local + frame["path"]
                if frame.get("query"):
                    url += "?" + frame["query"]
                r = await client.request(
                    frame["method"], url, headers=frame.get("headers") or {},
                    content=_unb64(frame.get("body", "")),
                )
            resp = {"id": frame["id"], "status": r.status_code,
                    "headers": dict(r.headers), "body": _b64(r.content)}
        except Exception as exc:  # local server down / unreachable
            resp = {"id": frame["id"], "status": 502,
                    "headers": {"content-type": "application/json"},
                    "body": _b64(json.dumps({"error": type(exc).__name__}).encode())}
        async with send_lock:
            await ws.send(json.dumps(resp))

    async def run(self) -> None:
        """Connect-serve-reconnect loop. Returns only when stop() is called."""
        import websockets

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self._relay, max_size=_FRAME_LIMIT) as ws:
                    challenge = json.loads(await ws.recv())
                    nonce = _unb64(challenge["nonce"])
                    from cryptography.hazmat.primitives import serialization
                    pub_raw = self._key.public_key().public_bytes(
                        serialization.Encoding.Raw, serialization.PublicFormat.Raw
                    )
                    await ws.send(json.dumps({
                        "brain_id": self.brain_id,
                        "pubkey": _b64(pub_raw),
                        "sig": _b64(self._key.sign(nonce)),
                    }))
                    log.info("relay.connected", brain_id=self.brain_id)
                    backoff = 1.0
                    send_lock = asyncio.Lock()
                    self._ws = ws
                    async for raw in ws:
                        frame = json.loads(raw)
                        asyncio.get_event_loop().create_task(
                            self._handle(ws, send_lock, frame)
                        )
            except asyncio.CancelledError:
                return
            except Exception as exc:  # relay unreachable / dropped — retry
                log.warning("relay.reconnect", error=type(exc).__name__, wait=backoff)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)

    def stop(self) -> None:
        """Disconnect and stay down. Must run on the connector's event loop (use
        ``loop.call_soon_threadsafe(conn.stop)`` from other threads). Closing the
        live socket matters: a connected ``run()`` blocks in the websocket recv
        loop and would never observe the stop event on its own."""
        self._stop.set()
        if self._ws is not None:
            asyncio.ensure_future(self._ws.close())
