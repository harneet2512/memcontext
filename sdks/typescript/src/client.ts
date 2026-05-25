import type {
  StoreRequest,
  StoreResponse,
  QueryRequest,
  QueryResponse,
  TraceResponse,
  CorrectRequest,
  ObserveRequest,
} from "./types.js";

export class MemContextClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8100") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`MemContext API error ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  }

  private async get<T>(path: string): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`MemContext API error ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  }

  /**
   * Store a conversational turn and optional explicit claims.
   */
  async store(req: StoreRequest): Promise<StoreResponse> {
    return this.post<StoreResponse>("/api/memory/store", req);
  }

  /**
   * Query memory for claims matching a natural-language query.
   */
  async query(req: QueryRequest): Promise<QueryResponse> {
    return this.post<QueryResponse>("/api/memory/query", req);
  }

  /**
   * Trace the provenance of a specific claim by ID.
   */
  async trace(claimId: string): Promise<TraceResponse> {
    return this.post<TraceResponse>("/api/memory/trace", {
      claim_id: claimId,
    });
  }

  /**
   * Correct or dismiss an existing claim.
   */
  async correct(req: CorrectRequest): Promise<unknown> {
    return this.post("/api/memory/correct", req);
  }

  /**
   * Observe a URL and extract claims from its content.
   */
  async observe(req: ObserveRequest): Promise<unknown> {
    return this.post("/api/memory/observe", req);
  }

  /**
   * Get memory database status (total claims, active claims, sessions, turns).
   */
  async status(): Promise<{
    total_claims: number;
    active_claims: number;
    sessions: number;
    turns: number;
  }> {
    return this.get("/api/memory/status");
  }
}
