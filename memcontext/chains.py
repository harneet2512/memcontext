"""Supersession chain summaries — deterministic, zero LLM."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ChainStep:
    value: str
    edge_type: str
    timestamp: int  # nanoseconds
    claim_id: str
    source_turn_id: str


def build_chain(conn: sqlite3.Connection, claim_id: str) -> list[ChainStep]:
    """Build a full supersession chain ending at *claim_id*.

    Uses ``get_supersession_chain`` to walk backwards through predecessors,
    then appends the current (active) claim as the final step with
    ``edge_type="active"``.

    Returns steps in chronological order (oldest first).
    """
    from memcontext.claims import get_claim, get_supersession_chain

    predecessors = get_supersession_chain(conn, claim_id)

    steps: list[ChainStep] = []
    for pred_claim, edge_type in predecessors:
        steps.append(
            ChainStep(
                value=pred_claim.value,
                edge_type=edge_type,
                timestamp=pred_claim.created_ts,
                claim_id=pred_claim.claim_id,
                source_turn_id=pred_claim.source_turn_id,
            )
        )

    current = get_claim(conn, claim_id)
    if current is not None:
        steps.append(
            ChainStep(
                value=current.value,
                edge_type="active",
                timestamp=current.created_ts,
                claim_id=current.claim_id,
                source_turn_id=current.source_turn_id,
            )
        )

    return steps


def format_chain(chain: list[ChainStep]) -> str:
    """Format a supersession chain as human-readable text.

    Example output::

        [2024-01-15] "dark mode" (SUPERSEDED via user_correction)
        [2024-03-20] "light mode" (ACTIVE)
    """
    if not chain:
        return ""

    lines: list[str] = []
    for step in chain:
        ts_seconds = step.timestamp / 1e9
        date_str = datetime.fromtimestamp(ts_seconds, tz=UTC).strftime("%Y-%m-%d")

        if step.edge_type == "active":
            status_part = "(ACTIVE)"
        else:
            status_part = f"(SUPERSEDED via {step.edge_type})"

        lines.append(f'[{date_str}] "{step.value}" {status_part}')

    return "\n".join(lines)
