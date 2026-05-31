"""Two-block validation experiment for MemContext.

Measures whether a host model answers with the CURRENT value of a fact (vs a
SUPERSEDED/stale one) across a multi-session task with planted correction points:

  * Block A — host tool with native memory only, MemContext NOT attached.
  * Block B — same task, MemContext attached over MCP.

The reader is the host model in both blocks; the only variable is MemContext's
presence. The scorer rewards current-state use and penalizes reliance on
invalidated facts (forgetting-aware memory accuracy).

This package is a SCAFFOLD: `validation.run` seeds the task and captures each
probe's projection state; the host model fills in answers; `validation.score`
scores them automatically. Nothing here calls an external API.
"""
