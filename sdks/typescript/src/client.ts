import type {
  StoreRequest,
  StoreResponse,
  QueryRequest,
  QueryResponse,
  TraceResponse,
  StatusResponse,
} from "./types.js";

export interface MemContextClientOptions {
  /** Bearer token for the HTTP API. Every /api/* route requires it —
   *  the value printed by `memcontext serve-http` or MEMCONTEXT_HTTP_TOKEN. */
  token?: string;
}

export class MemContextClient {
  private baseUrl: string;
  private token?: string;

  constructor(baseUrl: string = "http://localhost:8100", options: MemContextClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.token = options.token;
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    const h: Record<string, string> = { ...extra };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`MemContext API error ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  }

  private async get<T>(path: string): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, { headers: this.headers() });
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
   * Get memory database status (total claims, active claims, sessions, turns).
   */
  async status(): Promise<StatusResponse> {
    return this.get("/api/memory/status");
  }
}
