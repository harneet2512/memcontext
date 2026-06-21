#!/usr/bin/env python3
"""Build-time patch: tie-aware (competition) RRF rank fusion for the product's
``_rrf_ranks`` (memcontext/retrieval.py @ PRODUCT_REF).

Diagnosis (from results/proof_fusion_master.py on master): the shipping
``_rrf_ranks`` strict-orders by ``(-score, index)`` and assigns EVERY item a
distinct rank ``1..n``. When a channel is FLAT or degenerate (all scores equal —
e.g. a temporal/entity channel that finds no signal for the query), the strict
order falls back to INDEX order, so that channel injects index-ordered noise into
the fused score and can BURY an item another channel ranked #1. Measured: a flat
channel buried an entity-needle to #4; tie-aware ranking restores it to #1.

Fix: competition ranking — tied scores share the same (lowest) rank, so a flat
channel assigns every item the SAME rank and contributes a CONSTANT to the fused
score (stays NEUTRAL) instead of index noise. Distinct scores are byte-identical
to the old strict ordering, so ONLY genuine ties change — exactly the
degenerate-channel case. Ties are matched with a tolerance (``math.isclose`` with
``_RRF_TIE_REL_TOL`` / ``_RRF_TIE_ABS_TOL``) so floating-point dust between
structurally-identical scores collapses to ONE rank; the tie group anchors on the
FIRST score of the current rank (not the immediate predecessor) so a slow drift of
many dust-apart values cannot transitively fuse into one oversized tie.

This is the master deepening (tie-aware ``_rrf_ranks``) ported onto PRODUCT_REF's
retrieval.py as a build-time patch, mirroring patch_retrieval.py /
patch_supersession.py — the benchmark archives PRODUCT_REF, not master, so the
product under test must carry the deepening as a patch. ``math`` is already
imported at module top in PRODUCT_REF, so no import edit is needed. Deterministic,
zero generative-LLM — pure ranking arithmetic on the existing channel scores.

Asserts the anchor matches exactly once; fails the build loudly on drift.

Usage: python patch_fusion.py /opt/.../memcontext/retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Marker for idempotence (a distinctive token from the replacement body).
_MARKER = "Competition ranking: tied scores share the same"

# The full strict-ordering ``_rrf_ranks`` def as it exists in PRODUCT_REF
# (73c34f6). Anchored on the whole function body so the rewrite is unambiguous
# and refuses to apply if the product's fusion drifted.
ANCHOR = (
    "def _rrf_ranks(scores: list[float]) -> list[int]:\n"
    "    indexed = sorted(range(len(scores)), key=lambda i: (-scores[i], i))\n"
    "    ranks = [0] * len(scores)\n"
    "    for rank_minus_1, idx in enumerate(indexed):\n"
    "        ranks[idx] = rank_minus_1 + 1\n"
    "    return ranks"
)

REPLACEMENT = (
    "# Tolerances for tie detection in _rrf_ranks. Scores within this window are\n"
    "# treated as ONE rank (floating-point dust between structurally-identical\n"
    "# channel scores is not signal); anything beyond is a real, strictly-ordered\n"
    "# gap. Scale-invariant (rel_tol) with an absolute floor, deterministic.\n"
    "_RRF_TIE_REL_TOL = 1e-9\n"
    "_RRF_TIE_ABS_TOL = 1e-12\n"
    "\n"
    "\n"
    "def _rrf_ranks(scores: list[float]) -> list[int]:\n"
    "    # Competition ranking: tied scores share the same (lowest) rank. A FLAT or\n"
    "    # degenerate channel (all-equal scores) therefore assigns every item the SAME\n"
    "    # rank and contributes a CONSTANT to the fused score -- staying NEUTRAL instead\n"
    "    # of injecting index-ordered noise that buries an item another channel ranked\n"
    "    # #1. Distinct scores are unaffected (identical to the old strict ordering), so\n"
    "    # only genuine ties change -- exactly the degenerate-channel case. Proven:\n"
    "    # results/proof_fusion_master.py (flat channel buried an entity-needle to #4;\n"
    "    # tie-aware ranking restores it to #1).\n"
    "    #\n"
    "    # Ties are matched with a tolerance (see _RRF_TIE_*), so floating-point dust\n"
    "    # between structurally-identical scores collapses to ONE rank instead of being\n"
    "    # ranked as if it were signal. The tie group is anchored to the FIRST score of\n"
    "    # the current rank (the representative) and each subsequent score is compared\n"
    "    # against that anchor -- not its immediate predecessor -- so a slow drift of\n"
    "    # many dust-apart values cannot transitively fuse into one oversized tie.\n"
    "    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))\n"
    "    ranks = [0] * len(scores)\n"
    "    anchor: float | None = None\n"
    "    rank = 0\n"
    "    for pos, idx in enumerate(order, start=1):\n"
    "        s = scores[idx]\n"
    "        if anchor is None or not math.isclose(\n"
    "            s, anchor, rel_tol=_RRF_TIE_REL_TOL, abs_tol=_RRF_TIE_ABS_TOL\n"
    "        ):\n"
    "            rank = pos\n"
    "            anchor = s\n"
    "        ranks[idx] = rank\n"
    "    return ranks"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_fusion.py <path-to memcontext/retrieval.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if _MARKER in text:
        print("[patch_fusion] already applied")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(
            f"ERROR: _rrf_ranks anchor found {n}x (expected 1). retrieval.py "
            "drifted from PRODUCT_REF — refusing to patch.",
            file=sys.stderr,
        )
        return 1
    path.write_text(text.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"[patch_fusion] tie-aware RRF rank fusion applied to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
