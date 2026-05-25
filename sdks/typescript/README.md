# @memcontext/client

TypeScript client for the MemContext memory API. Zero dependencies -- uses the built-in `fetch` available in Node 18+.

## Install

```bash
npm install @memcontext/client
```

## Quickstart

```typescript
import { MemContextClient } from "@memcontext/client";

const mc = new MemContextClient("http://localhost:8100");

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

### `new MemContextClient(baseUrl?)`

Create a client. Defaults to `http://localhost:8100`.

### `store(req: StoreRequest): Promise<StoreResponse>`

Store a conversational turn with optional explicit claims.

### `query(req: QueryRequest): Promise<QueryResponse>`

Query memory with a natural-language string.

### `trace(claimId: string): Promise<TraceResponse>`

Trace the provenance chain of a claim.

### `correct(req: CorrectRequest): Promise<unknown>`

Dismiss or correct an existing claim.

### `observe(req: ObserveRequest): Promise<unknown>`

Observe a URL and extract claims from its content.

### `status(): Promise<StatusResponse>`

Get database statistics (total claims, active claims, sessions, turns).
