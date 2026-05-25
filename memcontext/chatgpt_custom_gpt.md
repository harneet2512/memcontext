# MemContext — ChatGPT Custom GPT Setup

## 1. Start the HTTP API

```bash
# Terminal 1: Start MemContext HTTP API
memcontext init --db memcontext.db
memcontext serve-http --db memcontext.db --port 8100

# Terminal 2: Expose via ngrok (so ChatGPT can reach it)
ngrok http 8100
# Copy the https://xxxx.ngrok-free.app URL
```

## 2. Create Custom GPT

Go to https://chatgpt.com/gpts/editor

**Name:** MemContext Memory

**Description:** Your universal AI memory. Store and recall information across all your AI tools — Claude, ChatGPT, Gemini, and any app.

**Instructions:**
```
You are connected to MemContext, a universal memory layer. You can store and query the user's memory.

When the user says "remember this" or shares information to save, call the memory_store action.
When the user asks a question that might be answered from memory, call memory_query first.
Always tell the user what you stored or found, including the source/provenance.

You are one of many AIs connected to this memory. The user may have stored information from Claude Code, Cursor, browser observations, or other tools. Treat all memory equally regardless of source.
```

**Actions → Import OpenAPI schema:**

```yaml
openapi: 3.1.0
info:
  title: MemContext Memory API
  version: 0.1.0
  description: Universal AI memory layer — store, query, and trace structured claims with provenance.
servers:
  - url: https://YOUR-NGROK-URL-HERE
paths:
  /api/memory/store:
    post:
      operationId: memoryStore
      summary: Store information in memory with structured claims
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [text]
              properties:
                text:
                  type: string
                  description: The text to remember
                claims:
                  type: array
                  description: Structured claims extracted from the text
                  items:
                    type: object
                    required: [subject, predicate, value]
                    properties:
                      subject:
                        type: string
                        description: The entity this fact is about
                      predicate:
                        type: string
                        enum: [user_fact, user_preference, user_event, user_relationship, context, action, observation, metadata]
                      value:
                        type: string
                        description: The fact itself
                      confidence:
                        type: number
                        default: 0.9
      responses:
        "200":
          description: Memory stored successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  claims_created:
                    type: integer
                  session_id:
                    type: string
                  supersessions:
                    type: integer
  /api/memory/query:
    post:
      operationId: memoryQuery
      summary: Search memory for relevant claims
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [query]
              properties:
                query:
                  type: string
                  description: What to search for in memory
                top_k:
                  type: integer
                  default: 10
      responses:
        "200":
          description: Memory search results
          content:
            application/json:
              schema:
                type: object
                properties:
                  claims:
                    type: array
                    items:
                      type: object
                      properties:
                        subject:
                          type: string
                        predicate:
                          type: string
                        value:
                          type: string
                        confidence:
                          type: number
                        score:
                          type: number
                  total:
                    type: integer
  /api/memory/trace:
    post:
      operationId: memoryTrace
      summary: Trace a claim back to its source and supersession history
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [claim_id]
              properties:
                claim_id:
                  type: string
      responses:
        "200":
          description: Full provenance trace
  /api/memory/status:
    get:
      operationId: memoryStatus
      summary: Get memory statistics
      responses:
        "200":
          description: Current memory status
          content:
            application/json:
              schema:
                type: object
                properties:
                  total_claims:
                    type: integer
                  active_claims:
                    type: integer
                  sessions:
                    type: integer
```

## 3. Test

In ChatGPT with your Custom GPT:
- "Remember that our Q4 OKR is to launch AI search by Nov 15"
- "What do I know about Q4 OKRs?"

Then in Claude Code (connected via MCP to the same memcontext.db):
- "What are our Q4 OKRs?"
- Same answer. Same memory. Two different AIs.
