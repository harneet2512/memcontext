"""Instance-preserving enumeration — embedding-cluster distinct-count.

COUNTING deepening (design §2). Two collapse points in the product destroy
count signal:

  * ``projections.claims_grouped_by_subject_predicate`` keeps only the newest
    active claim per ``(subject, predicate)`` (newest-wins), so "how many times
    did X happen" reads as 1.
  * the freq channel in ``retrieval.retrieve_hybrid`` counts raw active rows per
    ``(subject, predicate)``, which OVER-counts near-duplicate phrasings
    ("ate sushi" + "had sushi for lunch" = 2 occurrences read as 2 when they are
    one) and UNDER-counts after supersession (superseded rows are dropped).

``count_distinct_instances`` replaces both with a deterministic, embedding-based
agglomerative single-pass clustering over ALL instances (active AND superseded),
calibrated against the LIVE embedder's cosine distribution (no E5 literal as the
operating point), with a temporal guard so distinct dated occurrences never
merge even at byte-identical value.

Deterministic. Embedding-based (the smart signal we are allowed). No hardcoded
predicate lists, no benchmark coupling. Temporal-preserving.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from memcontext.schema import Claim
from memcontext.claims import row_to_claim
from memcontext.supersession_semantic import DEFAULT_COSINE_THRESHOLD, cosine


def _as_session_ids(session_id: "str | Sequence[str]") -> list[str]:
    """Normalise the session scope to a de-duplicated, order-stable id list.

    ADDITIVE generalisation for the multi-session store (the product keeps one
    session per ingested document, so a namespace's instances for a slot are
    spread across many sessions). A bare ``str`` is the legacy single-session
    scope and behaves byte-identically to the original; a sequence counts across
    all of the listed sessions as ONE instance set (the cross-session distinct
    count an aggregation query actually needs). No clustering logic changes.
    """
    if isinstance(session_id, str):
        return [session_id]
    seen: set[str] = set()
    out: list[str] = []
    for sid in session_id:
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out

# Statuses that represent a real instance of an occurrence. Superseded rows are
# INCLUDED on purpose: "how many times did X happen" must see retired instances,
# not only the surviving newest one.
_ENUMERATION_STATUSES = ("active", "superseded", "confirmed", "audited")


@dataclass(frozen=True)
class Cluster:
    """One distinct occurrence: its representative value + member claim ids."""

    representative: str
    value_normalised: str
    member_claim_ids: tuple[str, ...]
    event_ts_set: tuple[int, ...]


@dataclass(frozen=True)
class EnumerationResult:
    distinct_count: int
    clusters: tuple[Cluster, ...]
    t_dup: float  # the data-driven near-dup threshold actually used


class _EmbedderProto:
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        ...


def _norm_value(c: Claim) -> str:
    return (c.value_normalised or c.value or "").strip().lower()


def _instance_text(c: Claim) -> str:
    """Value-bearing text for the dedup embedding — value INCLUDED.

    Counting distinguishes occurrences by *what* happened, so unlike Pass-2's
    identity_text (value-excluded), enumeration embeds the value.
    """
    return f"{c.subject} {c.predicate} {c.value}".strip()


def _derive_t_dup(pairwise: list[float]) -> float:
    """Data-driven near-dup threshold via the LARGEST GAP in the sorted pairwise
    cosine distribution (deterministic 1-D separation, a.k.a. maximum-gap cut).

    Why not a fixed percentile: in a corpus of K distinct things each with a few
    paraphrases, the cross-thing pairs vastly outnumber the within-thing pairs
    (for 5 kits x 3 phrasings: 90 cross vs 15 within). A high percentile is
    dominated by the cross band and lands ABOVE the real within band, severing
    genuine paraphrases. The maximum-gap cut instead finds the natural valley
    between the two bands — the embedder's own separation — with no fixed number.

    Concretely: sort the pairwise cosines, find the widest adjacent gap whose
    upper edge is above the global floor (so we never cut inside the noisy low
    band), and place T_dup at the MIDPOINT of that gap. The floor
    (DEFAULT_COSINE_THRESHOLD) only bounds WHERE we look for the valley; it is
    not the operating point. Falls back to the floor when there is no signal.
    Env override for trial reproducibility.
    """
    override = os.environ.get("MEMCONTEXT_ENUM_TDUP", "").strip()
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    if not pairwise:
        return DEFAULT_COSINE_THRESHOLD
    s = sorted(pairwise)
    if len(s) < 2:
        # one pair: split iff it is below the floor; otherwise merge.
        return DEFAULT_COSINE_THRESHOLD
    # Search the widest adjacent gap. Restrict the gap's UPPER edge to the upper
    # half of the range so we separate the high (within) band from the low
    # (cross) band, not two noise sub-bands at the bottom.
    lo, hi = s[0], s[-1]
    span = hi - lo
    if span <= 1e-9:
        return DEFAULT_COSINE_THRESHOLD  # degenerate: all equal
    best_mid = DEFAULT_COSINE_THRESHOLD
    best_gap = -1.0
    midpoint_of_range = lo + span * 0.5
    for i in range(len(s) - 1):
        a, b = s[i], s[i + 1]
        gap = b - a
        # Only consider valleys whose top edge reaches into the upper region;
        # this keeps the cut between the cross band and the within band.
        if b < midpoint_of_range:
            continue
        if gap > best_gap:
            best_gap = gap
            best_mid = (a + b) / 2.0
    # If no qualifying valley (e.g. one band only), fall back to the floor.
    return best_mid if best_gap > 0.0 else DEFAULT_COSINE_THRESHOLD


def _load_instances(
    conn: sqlite3.Connection,
    session_id: "str | Sequence[str]",
    subject: str,
    predicate: str,
) -> list[Claim]:
    sids = _as_session_ids(session_id)
    if not sids:
        return []
    status_ph = ",".join("?" for _ in _ENUMERATION_STATUSES)
    sid_ph = ",".join("?" for _ in sids)
    rows = conn.execute(
        f"""
        SELECT * FROM claims
        WHERE session_id IN ({sid_ph}) AND subject = ? AND predicate = ?
          AND status IN ({status_ph})
        ORDER BY created_ts ASC, claim_id ASC
        """,
        (*sids, subject, predicate, *_ENUMERATION_STATUSES),
    ).fetchall()
    return [row_to_claim(r) for r in rows]


def count_distinct_instances(
    conn: sqlite3.Connection,
    session_id: "str | Sequence[str]",
    subject: str,
    predicate: str,
    embedder: _EmbedderProto,
) -> EnumerationResult:
    """Count DISTINCT occurrences for a (session(s), subject, predicate).

    ``session_id`` may be a single id (legacy, byte-identical behaviour) or a
    sequence of ids — the latter counts distinct instances ACROSS those sessions
    as one set, which is what a namespace-wide aggregation query needs.

    Stage A (exact): collapse identical normalized values with zero embed cost.
    Stage B (near-dup): embed each surviving representative with the LIVE
        embedder; merge two clusters iff cosine >= data-driven T_dup.
    Temporal guard: never merge two claims that both carry an event_ts and
        differ — distinct dated occurrences stay distinct even at identical value.

    Deterministic: instances processed in (created_ts ASC, claim_id ASC); a
    claim joins the first existing cluster it matches, else opens a new one.
    Instance-preserving: each cluster keeps its member claim_ids.
    """
    instances = _load_instances(conn, session_id, subject, predicate)
    if not instances:
        return EnumerationResult(0, (), DEFAULT_COSINE_THRESHOLD)

    # --- Stage A: exact normalized-value buckets (order-stable) ---------------
    # CRITICAL: the temporal guard applies HERE too, not only in Stage B. Two
    # claims with the SAME normalized value but DIFFERENT event_ts are distinct
    # dated occurrences ("ran a 5K" twice) and must NOT share a bucket — else the
    # exact-collapse would silently fuse distinct events before embeddings run.
    # Bucket key is therefore (normalized_value, event_ts); a None event_ts is its
    # own slot (undated facts collapse on value as before).
    exact_order: list[tuple[str, int | None]] = []
    exact_members: dict[tuple[str, int | None], list[Claim]] = {}
    for c in instances:
        key = (_norm_value(c), c.event_ts)
        if key not in exact_members:
            exact_members[key] = []
            exact_order.append(key)
        exact_members[key].append(c)

    # Representative per exact bucket = its earliest claim (stable sort already).
    reps: list[Claim] = [exact_members[k][0] for k in exact_order]

    # --- Embed representatives once (the only embed pass) ---------------------
    rep_vecs = embedder.embed([_instance_text(c) for c in reps])

    # --- Derive T_dup from the in-set pairwise distribution -------------------
    pairwise: list[float] = []
    for i in range(len(rep_vecs)):
        for j in range(i + 1, len(rep_vecs)):
            pairwise.append(cosine(rep_vecs[i], rep_vecs[j]))
    t_dup = _derive_t_dup(pairwise)

    # --- Stage B: transitive merge over representatives (connected components) -
    # An edge i-j exists iff cosine >= T_dup AND the temporal guard permits it.
    # Clustering is the connected components of that graph via union-find. This is
    # transitive: paraphrases that link through an intermediate still co-cluster,
    # fixing the single-anchor severing of drifted paraphrases. Deterministic
    # given the fixed (created_ts, claim_id) sort of `reps`.
    n = len(reps)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            # attach higher index under lower for stable, order-independent roots
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if cosine(rep_vecs[i], rep_vecs[j]) < t_dup:
                continue
            # Temporal guard: never link two reps that both carry differing event_ts.
            if _temporal_block(reps[j], [reps[i]]):
                continue
            _union(i, j)

    # group rep indices by component root, preserving first-seen order
    root_order: list[int] = []
    by_root: dict[int, list[int]] = {}
    for i in range(n):
        r = _find(i)
        if r not in by_root:
            by_root[r] = []
            root_order.append(r)
        by_root[r].append(i)
    cluster_rep_indices: list[list[int]] = [by_root[r] for r in root_order]

    # --- Temporal split pass (defense in depth) -------------------------------
    # Pairwise blocking above can still leave a component holding two reps with
    # DIFFERING event_ts if they were linked through a None-event_ts intermediate
    # (transitivity bridge). Distinct dated occurrences must never share a final
    # cluster, so split any component into one sub-cluster per distinct non-None
    # event_ts; None-event_ts reps attach to the earliest sub-cluster (or form
    # their own if the component is entirely undated). Deterministic.
    split_clusters: list[list[int]] = []
    for rep_idxs in cluster_rep_indices:
        dated: dict[int, list[int]] = {}
        undated: list[int] = []
        for ri in rep_idxs:
            ev = reps[ri].event_ts
            if ev is None:
                undated.append(ri)
            else:
                dated.setdefault(ev, []).append(ri)
        if not dated:
            split_clusters.append(rep_idxs)
            continue
        # one sub-cluster per distinct event_ts (ascending, deterministic)
        sub_by_ev = [dated[ev] for ev in sorted(dated)]
        # undated reps ride with the earliest dated sub-cluster (they carry no
        # contradicting date, so they do not create a new occurrence)
        sub_by_ev[0] = sub_by_ev[0] + undated
        split_clusters.extend(sub_by_ev)
    cluster_rep_indices = split_clusters

    # --- Materialize clusters (expand exact buckets back to member claims) ----
    clusters: list[Cluster] = []
    for rep_idxs in cluster_rep_indices:
        members: list[Claim] = []
        for ri in rep_idxs:
            key = exact_order[ri]
            members.extend(exact_members[key])
        members.sort(key=lambda c: (c.created_ts, c.claim_id))
        ev = tuple(sorted({c.event_ts for c in members if c.event_ts is not None}))
        clusters.append(
            Cluster(
                representative=reps[rep_idxs[0]].value,
                value_normalised=_norm_value(reps[rep_idxs[0]]),
                member_claim_ids=tuple(c.claim_id for c in members),
                event_ts_set=ev,
            )
        )

    return EnumerationResult(len(clusters), tuple(clusters), t_dup)


def _temporal_block(candidate: Claim, cluster_members: list[Claim]) -> bool:
    """True if merging `candidate` into the cluster would fuse two distinct dated
    events. Distinct event_ts => distinct occurrence, regardless of value.
    """
    if candidate.event_ts is None:
        return False
    for m in cluster_members:
        if m.event_ts is not None and m.event_ts != candidate.event_ts:
            return True
    return False


def enumerate_retrieved(
    conn: sqlite3.Connection,
    session_id: "str | Sequence[str]",
    retrieved_claims: list[dict],
    embedder: _EmbedderProto,
) -> EnumerationResult | None:
    """Count distinct occurrences for the DOMINANT slot in a retrieved fact set.

    Serve-side orchestration for aggregation ("how many", "count", "list all")
    queries. The target ``(subject, predicate)`` is derived from the retrieved
    claims THEMSELVES — the most-represented slot in the result set — never from
    a hardcoded predicate list and never from parsing the query string. This
    keeps it domain-general: whatever the user is actually counting is whatever
    dominates the retrieval.

    ``retrieved_claims`` is the served fact list (each item carries at least
    ``subject`` and ``predicate`` keys). Returns ``None`` when there is no
    usable slot (empty input, or no item carries a non-empty subject+predicate),
    so the caller can simply skip attaching an enumeration block.

    The full instance set for the chosen slot is loaded from the store (active
    AND superseded) inside ``count_distinct_instances`` — the retrieved subset
    only selects WHICH slot to count, it does not bound the count.
    """
    if not retrieved_claims:
        return None

    # Tally slots present in the retrieved set, preserving first-seen order so
    # ties break deterministically toward the earliest-ranked retrieved slot.
    order: list[tuple[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for item in retrieved_claims:
        subject = (item.get("subject") or "").strip()
        predicate = (item.get("predicate") or "").strip()
        if not subject or not predicate:
            continue
        slot = (subject, predicate)
        if slot not in counts:
            counts[slot] = 0
            order.append(slot)
        counts[slot] += 1

    if not order:
        return None

    # Dominant slot = highest retrieved frequency; first-seen order breaks ties.
    dominant = max(order, key=lambda s: counts[s])
    subject, predicate = dominant
    return count_distinct_instances(conn, session_id, subject, predicate, embedder)
