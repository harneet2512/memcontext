# Retrieval Architecture

MemContext retrieval serves current, provenance-backed memory through two
layers: raw conversational episodes and extracted structured facts. The benchmark
is only a diagnostic instrument; this document describes the product retrieval
surface.

## Ingest Path

Every accepted turn is stored first as an episode. When an embedder is supplied,
the episode is embedded synchronously as the Tier-1 floor, so retrieval still has
the original turn even when extraction produces no facts.

Fact extraction is optional and may run inline or through a queue. Extracted
claims become Tier-2 facts linked back to their source episode. Supersession then
marks older facts inactive when a newer claim replaces them, and provenance keeps
forward/back links from facts to the original turn span. Embeddings live in
sidecar tables for both claims and turns; schema and stored text remain separate.

## Retrieval Tiers

`retrieve_hybrid` ranks facts within one session using reciprocal-rank fusion
(RRF) over semantic, entity, temporal, BM25, confidence, frequency, importance,
usage, and source-trust signals. It filters to active facts unless a history
query asks for superseded facts.

`retrieve_episodes` ranks raw turns within one session using semantic, BM25,
entity-overlap, and recency signals. It degrades to lexical ranking if episode
embeddings are unavailable, but the episode floor still lets the original turn
carry recall when facts are absent or incomplete.

`retrieve_memory` fuses the fact and episode rankings for one session. Facts get
a slight tie advantage because they are distilled, but episodes remain in the
ranking so the system can answer from the original context.

`retrieve_memory_across` runs `retrieve_memory` independently for each session
and fuses those per-session rankings by rank, not raw score. Cross-session raw
scores are not comparable: a long session can produce larger BM25 or semantic
magnitudes than a short session that actually contains the answer. RRF preserves
breadth without score calibration.

## Cross-Session Depth

The cross-session fusion now uses a reserve/overflow policy. Each queried
session reserves its top `per_session_k` hits (`DEFAULT_PER_SESSION_K = 3`), and
the remaining hits share overflow budget. The final budget is:

```python
min(max(top_k, len(reserved)), MAX_ACROSS_HITS)
```

This keeps normal few-session queries inside the usual `top_k` envelope when the
per-session guarantee already fits. When the number of sessions outnumbers
`top_k`, the budget grows enough to keep per-session depth and avoid starving
rank-2+ evidence inside a session. `MAX_ACROSS_HITS = 300` bounds pathological
fan-out.

This fix is classified as PROVEN for the observed recall-starvation failure: the
same data and embeddings measured gold-turn recall improving from 33% to 72%
when per-session depth was preserved. It is not a benchmark-answer hack and does
not depend on LongMemEval labels, examples, or answer keys.

## Known Gap

Assistant-recall and preference questions can still underperform even after
cross-session depth is preserved. The remaining issue is within-session ranking:
the user query often resembles the user's earlier question more than the
assistant's answer, recommendation, or implicit preference evidence. A future
fix should measure that failure directly and classify it separately; until then,
assistant/preference ranking changes are PLAUSIBLE, not proven.
