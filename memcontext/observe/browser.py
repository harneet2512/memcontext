"""Browser observation engine using Playwright accessibility tree.

Captures page state as an accessibility snapshot, extracts structured
claims from the tree, and stores provenance linking claims to their
screen-state source.

All Playwright imports are lazy -- this module can be imported without
Playwright installed, but observe() requires it at runtime.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class PageSnapshot:
    """Captured state of a browser page."""

    url: str
    title: str
    timestamp: str  # ISO 8601
    accessibility_tree: dict  # Playwright a11y snapshot
    screenshot_hash: str | None = None  # SHA256 of screenshot bytes if captured
    dom_hash: str | None = None  # SHA256 of page content

    @property
    def snapshot_id(self) -> str:
        """Deterministic ID from URL + timestamp."""
        raw = f"{self.url}:{self.timestamp}"
        return f"snap_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


@dataclass
class ObservationResult:
    """Result of observing a page."""

    snapshot: PageSnapshot
    claims: list[dict]  # extracted claim dicts
    session_id: str
    turn_id: str | None = None
    errors: list[str] = field(default_factory=list)


async def capture_snapshot(page) -> PageSnapshot:
    """Capture a PageSnapshot from a Playwright page object.

    Args:
        page: A Playwright Page object (async API).

    Returns:
        PageSnapshot with URL, title, timestamp, and accessibility tree.
    """
    url = page.url
    title = await page.title()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Capture accessibility tree
    a11y_tree = await page.accessibility.snapshot() or {}

    # Compute content hash
    content = await page.content()
    dom_hash = hashlib.sha256(content.encode()).hexdigest()

    return PageSnapshot(
        url=url,
        title=title,
        timestamp=timestamp,
        accessibility_tree=a11y_tree,
        dom_hash=dom_hash,
    )


def observe_page(
    conn,
    *,
    snapshot: PageSnapshot,
    session_id: str,
    extractor=None,
) -> ObservationResult:
    """Extract claims from a page snapshot and store them.

    Uses AccessibilityTreeExtractor by default, or a custom extractor.
    Stores claims via on_new_turn with provenance metadata.
    """
    from memcontext.extractors import PassthroughExtractor
    from memcontext.observe.extractors import AccessibilityTreeExtractor
    from memcontext.on_new_turn import on_new_turn
    from memcontext.predicate_packs import active_pack
    from memcontext.schema import Speaker

    ext = extractor or AccessibilityTreeExtractor()
    claims = ext.extract(snapshot)

    if not claims:
        return ObservationResult(
            snapshot=snapshot,
            claims=[],
            session_id=session_id,
        )

    # Build observation text for the turn
    observation_text = f"[Observed: {snapshot.url}] {snapshot.title}"
    for c in claims:
        observation_text += f"\n- {c.get('subject', 'unknown')}: {c.get('value', '')}"

    # Store via on_new_turn with PassthroughExtractor
    pack = active_pack()
    pt = PassthroughExtractor(claims)
    result = on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.ASSISTANT,
        text=observation_text,
        extractor=pt,
        multi_valued_predicates=pack.multi_valued_predicates,
    )

    return ObservationResult(
        snapshot=snapshot,
        claims=claims,
        session_id=session_id,
        turn_id=result.turn.turn_id if result.turn else None,
    )


async def observe(
    conn,
    *,
    page,
    session_id: str,
    extractor=None,
) -> ObservationResult:
    """Full observation pipeline: capture snapshot -> extract -> store.

    Args:
        conn: SQLite connection
        page: Playwright Page object
        session_id: Session to store claims in
        extractor: Optional custom extractor
    """
    snapshot = await capture_snapshot(page)
    return observe_page(
        conn, snapshot=snapshot, session_id=session_id, extractor=extractor
    )
