"""OFFLINE PROOF (master fusion): a FLAT retrieval channel dilutes a needle that a
real channel ranks #1 — because `_rrf_ranks` breaks ties by INDEX, so a flat
channel emits ranks 1..N in claim order instead of one shared rank.

This proves (a) the failure mode EXISTS on master's actual fusion primitives, and
(b) the deterministic root-cause fix — tie-aware ranking (equal scores -> equal
rank) — makes a flat channel NEUTRAL and recovers the needle. No embedder, no DB:
it exercises the exact `_rrf_ranks` + RRF_K math master's `retrieve_hybrid` uses.
"""
from __future__ import annotations

from memcontext.retrieval import _rrf_ranks, RRF_K  # _rrf_ranks = the SHIPPED tie-aware fix


def _rrf_ranks_OLD(scores: list[float]) -> list[int]:
    """The OLD strict-index ranking (the bug): ties broken by index -> a flat channel
    emits ranks 1..N in claim order, injecting index dilution."""
    indexed = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank_minus_1, idx in enumerate(indexed):
        ranks[idx] = rank_minus_1 + 1
    return ranks


def _fuse(rank_fn, sem_scores, ent_scores, *, w_sem=1.0, w_ent=1.0):
    sem = rank_fn(sem_scores)
    ent = rank_fn(ent_scores)
    out = []
    for i in range(len(sem_scores)):
        s = w_sem / (RRF_K + sem[i]) + w_ent / (RRF_K + ent[i])
        out.append((i, s))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


# Corpus: 12 claims. Claim 7 is the NEEDLE — the ENTITY channel ranks it the clear
# #1 (score 9.0, everything else ~0). The SEMANTIC channel is FLAT (all 0.80: a
# degenerate/abstaining embedder that "has no opinion"). The needle sits late in the
# index (i=7) so the index-order tie-break in `_rrf_ranks` puts it mid-pack on the
# flat semantic channel — exactly the dilution.
N = 12
NEEDLE = 7
ent_scores = [0.0] * N
ent_scores[NEEDLE] = 9.0                       # unique strong entity hit
ent_scores[2] = 0.3; ent_scores[10] = 0.2      # weak noise
sem_scores = [0.80] * N                         # FLAT semantic channel (no signal)

print(f"corpus N={N}, needle=claim#{NEEDLE} (unique entity hit 9.0; semantic flat 0.80 everywhere)\n")

print("--- ranks emitted for the FLAT semantic channel (1 = neutral/tied; >1 = dilution) ---")
sem_old = _rrf_ranks_OLD(sem_scores)
sem_new = _rrf_ranks(sem_scores)            # the SHIPPED fix
print(f"OLD strict-index : {sem_old}  -> {len(set(sem_old))} distinct ranks")
print(f"NEW tie-aware    : {sem_new}  -> {len(set(sem_new))} distinct ranks\n")

old = _fuse(_rrf_ranks_OLD, sem_scores, ent_scores)
new = _fuse(_rrf_ranks, sem_scores, ent_scores)
old_pos = [i for i, _ in old].index(NEEDLE) + 1
new_pos = [i for i, _ in new].index(NEEDLE) + 1

print(f"OLD _rrf_ranks (index tie-break): needle ranks #{old_pos} of {N}   top3={[i for i,_ in old[:3]]}")
print(f"NEW _rrf_ranks (shipped fix):     needle ranks #{new_pos} of {N}   top3={[i for i,_ in new[:3]]}\n")

assert len(set(sem_old)) > 1, "OLD must emit index-ordered ranks (the bug)"
assert len(set(sem_new)) == 1, "NEW must make a flat channel one tied rank (neutral)"
assert old_pos != 1, f"OLD must BURY the needle (got #{old_pos})"
assert new_pos == 1, f"NEW must recover the needle to #1 (got #{new_pos})"
print(f"PROVEN: OLD strict-index ranking buried the entity-needle to #{old_pos} via a flat")
print(f"        channel's index noise; the shipped tie-aware _rrf_ranks makes that channel")
print(f"        neutral (1 rank) and recovers the needle to #{new_pos}. Deterministic, general.")
