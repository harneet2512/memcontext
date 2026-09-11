export interface Claim {
  claim_id: string;
  subject: string;
  predicate: string;
  value: string;
  confidence: number;
  status: string;
  score?: number;
}

export interface StoreRequest {
  text: string;
  speaker?: "user" | "assistant";
  session_id?: string;
  claims?: Array<{
    subject: string;
    predicate: string;
    value: string;
    confidence: number;
  }>;
}

export interface StoreResponse {
  turn_id: string;
  session_id: string;
  admitted: boolean;
  claims_created: number;
  claim_ids: string[];
  supersessions: number;
}

export interface QueryRequest {
  query: string;
  session_id?: string;
  top_k?: number;
}

export interface QueryResponse {
  claims: Claim[];
  episodes: unknown[];
  total: number;
}

export interface TraceLineageStep {
  claim_id: string;
  value: string;
  status: string;
  edge_type: string | null;
  confidence: number | null;
  source_turn_id: string;
  speaker: string | null;
  text: string | null;
  char_start: number | null;
  char_end: number | null;
  quote: string | null;
}

export interface TraceResponse {
  subject: string;
  predicate: string;
  claim: Claim | null;
  source_turn: {
    turn_id: string;
    text: string;
    speaker: string;
  } | null;
  char_span: { start: number; end: number } | null;
  /** Newest-first full supersession lineage with provenance quotes. */
  lineage: TraceLineageStep[];
  supersession_chain: Array<{ from: string; to: string }>;
}

export interface StatusResponse {
  total_claims: number;
  active_claims: number;
  sessions: number;
  turns: number;
}
