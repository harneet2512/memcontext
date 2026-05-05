"""Category-specific answer prompts for LongMemEval.

These prompts are the key gap vs OMEGA (~95.4%). The current system uses a
single generic prompt; category-specific prompts should improve accuracy
especially for preference and temporal questions.
"""
from __future__ import annotations

PROMPTS: dict[str, str] = {
    "single_session_user_fact": (
        "Based on the following memory claims about the user from a single "
        "conversation, answer the question directly and concisely.\n\n"
        "Claims:\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
    "single_session_preference": (
        "Based on the following memory claims, state the user's preference "
        "directly. Do NOT give advice or recommendations. Only state what "
        "the user prefers.\n\n"
        "Claims:\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
    "cross_session_preference": (
        "The following claims span multiple conversations with the user. "
        "When preferences conflict, use the most recent one (highest "
        "timestamp). State the user's current preference directly.\n\n"
        "Claims:\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
    "cross_session_user_fact": (
        "The following claims span multiple conversations. Answer the "
        "question using only information from these claims.\n\n"
        "Claims:\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
    "temporal_ordering": (
        "The following claims have temporal information. Answer the "
        "question about the order or timing of events.\n\n"
        "Claims (ordered by time):\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
    "knowledge_update": (
        "The following claims may include superseded information. Use only "
        "the most recent active claim for each fact. Answer with the "
        "current/updated value.\n\n"
        "Claims:\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
    "abstention": (
        "Based on the following memory claims, answer the question. If the "
        "information is not mentioned in any claim, respond with "
        "'Not mentioned in memory.'\n\n"
        "Claims:\n{claims}\n\nQuestion: {question}\nAnswer:"
    ),
}


def get_prompt(category: str, claims_text: str, question: str) -> str:
    """Format a category-specific prompt. Falls back to generic if unknown."""
    template = PROMPTS.get(category, PROMPTS["single_session_user_fact"])
    return template.format(claims=claims_text, question=question)


def format_claims_for_prompt(claims: list[dict]) -> str:
    """Format claim dicts into a readable string for the prompt."""
    lines = []
    for i, c in enumerate(claims, 1):
        line = (
            f"{i}. [{c.get('predicate', 'unknown')}] "
            f"{c.get('subject', 'unknown')}: {c.get('value', '')}"
        )
        if c.get("confidence"):
            line += f" (confidence: {c['confidence']})"
        lines.append(line)
    return "\n".join(lines) if lines else "(no claims)"
