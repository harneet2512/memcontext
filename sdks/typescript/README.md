# @memcontext/client

TypeScript client for the MemContext HTTP API. Zero dependencies — uses the built-in `fetch` available in Node 18+.

**Experimental:** covers the four core REST endpoints (`store`, `query`, `trace`, `status`). The full surface — corrections, governance, digests — is over MCP (`mcp_tools.py` handlers) or the Python API.

## Install

```bash
npm install @memcontext/client
```

## Quickstart

```typescript
import { MemContextClient } from "@memcontext/client";

// Every /api/* route requires a bearer token — the one printed by
// `memcontext serve-http` on startup, or MEMCONTEXT_HTTP_TOKEN.
const mc = new MemContextClient("http://localhost:8100", {
  token: process.env.MEMCONTEXT_HTTP_TOKEN,
});

// Store a conversational turn
const stored = await mc.store({
  text: "My favorite programming language is TypeScript.",
  speaker: "user",
  session_id: "demo",
});
console.log(`Stored turn ${stored.turn_id}, ${stored.claims_created} claims created`);

// Query memory
const results = await mc.query({ query: "favorite programming language" });
for (const claim of results.claims) {
  console.log(`${claim.subject} ${claim.predicate}: ${claim.value} (${claim.confidence})`);
}

// Check status
const info = await mc.status();
console.log(`${info.active_claims} active claims across ${info.sessions} sessions`);
```

## API

### `new MemContextClient(baseUrl?, options?)`

Create a client. Defaults to `http://localhost:8100`. Pass `{ token }` for the
bearer token required by all `/api/*` routes.

### `store(req: StoreRequest): Promise<StoreResponse>`

Store a conversational turn with optional explicit claims.

### `query(req: QueryRequest): Promise<QueryResponse>`

Query memory with a natural-language string.

### `trace(claimId: string): Promise<TraceResponse>`

Trace the provenance chain of a claim — full supersession lineage with
per-step source turns and character-span quotes.

### `status(): Promise<StatusResponse>`

Get database statistics (total claims, active claims, sessions, turns).
