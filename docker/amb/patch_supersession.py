#!/usr/bin/env python3
"""Build-time patch: scope Pass-1 supersession to the NAMESPACE, not the session.

Diagnosis (measured offline): the product's Pass-1 supersession keys on
``session_id``, so a value update that lands in a different session (the
LongMemEval knowledge-update regime, and any real personal-brain user across
sessions) never supersedes the old value -> stale facts stay active. Measured
0/6 cross-session vs 2/6 same-session.

Fix: scope Pass-1 to the new claim's NAMESPACE (its source turn's
``turns.namespace``) instead of its single session. Within one tenant/namespace,
a newer value supersedes the older one regardless of which session it came from;
across namespaces nothing is compared (tenant isolation preserved). The adapter
ingests each AMB question under ``namespace=<question_id>`` so this is
cross-session WITHIN a question and never leaks across questions.

Scope is deliberately Pass-1 ONLY (its matching is strict: cardinality /
attribute-slot / quantity-correction with non-numeric content identical /
Jaccard >= 2 shared content tokens), so widening to namespace has a low
false-positive risk. Pass-2 (semantic) is LEFT session-scoped: cross-namespace
Pass-2 would compare a new claim against every same-predicate claim (hundreds of
generic ``user_fact``s) and could falsely merge distinct facts.

Asserts the anchor matches exactly once; fails the build loudly on drift.

Usage: python patch_supersession.py /opt/.../memcontext/supersession.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = (
    "    rows = conn.execute(\n"
    "        \"SELECT * FROM claims WHERE session_id = ? AND subject = ? AND predicate = ?\"\n"
    "        \" AND status IN ('active','confirmed') AND claim_id != ?\"\n"
    "        \" AND source_turn_id != ?\"\n"
    "        \" ORDER BY created_ts DESC\",\n"
    "        (\n"
    "            new_claim.session_id,\n"
    "            new_claim.subject,\n"
    "            new_claim.predicate,\n"
    "            new_claim.claim_id,\n"
    "            new_claim.source_turn_id,\n"
    "        ),\n"
    "    ).fetchall()"
)

REPLACEMENT = (
    "    # NAMESPACE-scoped (was session-scoped): supersede a value across all of a\n"
    "    # tenant's sessions, not just the one it was first stated in. The new claim's\n"
    "    # namespace is its source turn's namespace; candidates are claims whose source\n"
    "    # turn is in the SAME namespace. Tenant isolation preserved (no cross-namespace).\n"
    "    _ns_row = conn.execute(\n"
    "        \"SELECT namespace FROM turns WHERE turn_id = ?\", (new_claim.source_turn_id,)\n"
    "    ).fetchone()\n"
    "    _ns = _ns_row[0] if _ns_row else \"default\"\n"
    "    rows = conn.execute(\n"
    "        \"SELECT * FROM claims WHERE source_turn_id IN\"\n"
    "        \" (SELECT turn_id FROM turns WHERE namespace = ?)\"\n"
    "        \" AND subject = ? AND predicate = ?\"\n"
    "        \" AND status IN ('active','confirmed') AND claim_id != ?\"\n"
    "        \" AND source_turn_id != ?\"\n"
    "        \" ORDER BY created_ts DESC\",\n"
    "        (\n"
    "            _ns,\n"
    "            new_claim.subject,\n"
    "            new_claim.predicate,\n"
    "            new_claim.claim_id,\n"
    "            new_claim.source_turn_id,\n"
    "        ),\n"
    "    ).fetchall()"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_supersession.py <path-to memcontext/supersession.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if "NAMESPACE-scoped (was session-scoped)" in text:
        print("[patch_supersession] already applied")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(
            f"ERROR: Pass-1 query anchor found {n}x (expected 1). supersession.py "
            "drifted from PRODUCT_REF — refusing to patch.",
            file=sys.stderr,
        )
        return 1
    path.write_text(text.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"[patch_supersession] Pass-1 scoped to namespace in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
