"""The validation task: one multi-session software-project scenario with planted
correction points and cross-session persistence.

Facts are stated, some are later corrected, and probes (asked after the
corrections) check whether the reader answers with the CURRENT value. Every fact
carries structured claims so Block B can seed MemContext deterministically; a
correction turn carries a ``correction`` so the harness drives MemContext's
typed correction (handle_memory_correct) instead of a plain store.
"""
from __future__ import annotations

from dataclasses import dataclass

PACK = "general"
SESSION_ID = "validation"


@dataclass(frozen=True)
class Turn:
    session: int
    speaker: str           # "user" | "assistant"
    text: str
    # Structured claims for Block B passthrough store (subject, predicate, value).
    claims: tuple[dict, ...] = ()
    # If set, this turn CORRECTS an existing fact: {"subject","predicate","new_value"}.
    correction: dict | None = None


@dataclass(frozen=True)
class Probe:
    after_turn: int        # fire after this many turns have been ingested
    subject: str
    predicate: str
    question: str
    current_value: str               # the correct answer at this checkpoint
    stale_values: tuple[str, ...]    # superseded values that must NOT be used


def _fact(subject: str, value: str) -> tuple[dict, ...]:
    return ({"subject": subject, "predicate": "user_fact", "value": value, "confidence": 0.95},)


# ── The task: project "Apollo", four sessions ────────────────────────────────
TURNS: list[Turn] = [
    # Session 1 — initial facts
    Turn(1, "user", "We're building Apollo, a payments service. The main database is PostgreSQL.",
         claims=_fact("apollo_database", "PostgreSQL")),
    Turn(1, "user", "Our team lead is Alice.", claims=_fact("apollo_lead", "Alice")),
    Turn(1, "user", "We deploy every Friday.", claims=_fact("apollo_deploy", "Friday")),
    # Session 2 — more facts (persist across sessions, never corrected)
    Turn(2, "user", "Apollo's API is REST-based.", claims=_fact("apollo_api", "REST")),
    Turn(2, "user", "We use Stripe for card processing.", claims=_fact("apollo_payments", "Stripe")),
    # Session 3 — corrections
    Turn(3, "user", "Update: we migrated Apollo off Postgres to DynamoDB last sprint.",
         correction={"subject": "apollo_database", "predicate": "user_fact", "new_value": "DynamoDB"}),
    Turn(3, "user", "Alice moved teams; Bob is the lead now.",
         correction={"subject": "apollo_lead", "predicate": "user_fact", "new_value": "Bob"}),
    # Session 4 — another correction + a probe-only session
    Turn(4, "user", "After the incident we switched to daily deploys.",
         correction={"subject": "apollo_deploy", "predicate": "user_fact", "new_value": "daily"}),
    Turn(4, "user", "Reminder: card processing still goes through Stripe."),
]

PROBES: list[Probe] = [
    # After session 3's corrections
    Probe(7, "apollo_database", "user_fact", "What database does Apollo use?",
          current_value="DynamoDB", stale_values=("PostgreSQL", "Postgres")),
    Probe(7, "apollo_lead", "user_fact", "Who is Apollo's team lead?",
          current_value="Bob", stale_values=("Alice",)),
    # After session 4
    Probe(9, "apollo_deploy", "user_fact", "How often does the team deploy?",
          current_value="daily", stale_values=("Friday",)),
    # Cross-session persistence, never corrected
    Probe(9, "apollo_payments", "user_fact", "What payment processor does Apollo use?",
          current_value="Stripe", stale_values=()),
    # Re-probe a correction to test durable supersession
    Probe(9, "apollo_database", "user_fact", "Which database is Apollo on now?",
          current_value="DynamoDB", stale_values=("PostgreSQL", "Postgres")),
]
