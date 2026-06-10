"""Category-specific answer prompts for LongMemEval.

These prompts are the key gap vs OMEGA (~95.4%). The current system uses a
single generic prompt; category-specific prompts should improve accuracy
especially for preference and temporal questions.
"""
from __future__ import annotations

PROMPTS: dict[str, str] = {
    "single_session_user_fact": (
        "You are given a question and memory items from a single conversation.\n\n"
        "Step 1 — Notes: For each item, extract any facts about the user relevant to the question. Skip irrelevant items.\n"
        "Step 2 — Reasoning: Using only your notes, reason toward the answer.\n"
        "Step 3 — Answer: State the answer directly and concisely.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
    "single_session_preference": (
        "You are given a question and memory items from a conversation. "
        "The user's preferences may be IMPLICIT — inferred from their behavior, choices, experiences, and context, "
        "not necessarily stated as 'I prefer X.'\n\n"
        "Step 1 — Evidence: For each item, extract any clues about user preferences: "
        "tools they use, brands they own, activities they enjoy, past experiences they liked, "
        "styles they gravitate toward, constraints they work within. Include specific names and details.\n"
        "Step 2 — Synthesize: Based on the evidence, describe what the user would prefer. "
        "Reference their specific tools, experiences, and interests by name. "
        "Include what they would NOT prefer if there is evidence of dislikes.\n"
        "Step 3 — Answer: State the user's preference as a complete description. "
        "Be specific — mention product names, genres, styles, and constraints from the evidence.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Evidence:"
    ),
    "cross_session_preference": (
        "You are given a question and memory items spanning multiple conversations.\n\n"
        "Step 1 — Notes: For each item, extract any user preferences. Note the date of each.\n"
        "Step 2 — Reasoning: If preferences conflict, the most recent one is current. Do NOT give advice.\n"
        "Step 3 — Answer: State the user's current preference.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
    "cross_session_user_fact": (
        "You are given a question and memory items spanning multiple conversations.\n\n"
        "Step 1 — List: For each item, extract EVERY distinct instance relevant to the question. "
        "Number each instance separately (e.g., 'Plant 1: succulent from garden center, Plant 2: fern from farmer's market'). "
        "Include dates and sources. Do not skip any mention.\n"
        "Step 2 — Aggregate: Count, sum, or compute as the question requires. Show your arithmetic. "
        "If counting, verify your count matches the numbered list. If summing money, list each amount.\n"
        "Step 3 — Answer: State the final number or result.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — List:"
    ),
    "temporal_ordering": (
        "You are given a question about timing or order of events, and memory items with dates.\n\n"
        "Step 1 — Notes: For each item, extract events and their dates/times. Skip irrelevant items.\n"
        "Step 2 — Reasoning: Order the events chronologically and reason about the timing question.\n"
        "Step 3 — Answer: State the answer about timing or order.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
    "knowledge_update": (
        "You are given a question and memory items that may include outdated information.\n\n"
        "Step 1 — Notes: For each item, extract the fact and its date. Note if it updates a previous fact.\n"
        "Step 2 — Reasoning: Identify the most recent value for each fact. Discard outdated information.\n"
        "Step 3 — Answer: State the current/updated answer.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
    "abstention": (
        "You are given a question and memory items from past conversations.\n\n"
        "Step 1 — Notes: For each item, check if it contains information relevant to the question.\n"
        "Step 2 — Reasoning: Determine if the question can be answered from the available information.\n"
        "Step 3 — Answer: If answerable, state the answer. If not, say the information is not available.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
    "single_session_assistant": (
        "You are given a question about what the assistant said or recommended, and memory items from a single conversation.\n\n"
        "Step 1 — Notes: For each item, extract what the assistant said, recommended, or provided. Skip user-only items.\n"
        "Step 2 — Reasoning: Identify the specific assistant action or recommendation asked about.\n"
        "Step 3 — Answer: State what the assistant said or recommended.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
    "default": (
        "You are given a question and memory items from past conversations.\n\n"
        "Step 1 — Notes: For each item, extract only information relevant to the question. Skip irrelevant items.\n"
        "Step 2 — Reasoning: Using only your notes, reason step by step toward the answer.\n"
        "Step 3 — Answer: State the answer concisely.\n\n"
        "Memory items:\n{claims}\n\nQuestion: {question}\n\nStep 1 — Notes:"
    ),
}


# Maps LongMemEval dataset category names to prompt template keys.
# The dataset uses hyphenated names; our prompts use underscored keys.
CATEGORY_MAP: dict[str, str] = {
    # Dataset name → prompt key
    "single-session-user": "single_session_user_fact",
    "single-session-assistant": "single_session_assistant",
    "single-session-preference": "single_session_preference",
    "temporal-reasoning": "temporal_ordering",
    "knowledge-update": "knowledge_update",
    "multi-session": "cross_session_user_fact",
    "abstention": "abstention",
    # Internal / underscore variants (direct match)
    "single_session_user_fact": "single_session_user_fact",
    "single_session_assistant": "single_session_assistant",
    "single_session_preference": "single_session_preference",
    "cross_session_preference": "cross_session_preference",
    "cross_session_user_fact": "cross_session_user_fact",
    "temporal_ordering": "temporal_ordering",
    "knowledge_update": "knowledge_update",
}


def get_prompt(category: str, claims_text: str, question: str) -> str:
    """Format a category-specific prompt. Falls back to default if unknown."""
    # Resolve via CATEGORY_MAP first, then direct lookup, then default
    prompt_key = CATEGORY_MAP.get(category, category)
    template = PROMPTS.get(prompt_key, PROMPTS["default"])
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
