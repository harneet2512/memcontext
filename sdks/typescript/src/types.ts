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
  total: number;
}

export interface TraceResponse {
  claim: Claim | null;
  source_turn: {
    turn_id: string;
    text: string;
    speaker: string;
  } | null;
  supersession_chain: Array<{ from: string; to: string }>;
}

export interface CorrectRequest {
  claim_id: string;
  action: "dismiss" | "correct";
  new_value?: string;
}

export interface ObserveRequest {
  url: string;
  session_id?: string;
}
