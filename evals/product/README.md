# Product evals

Diagnostic evals for MemContext's actual promises. They exist to find and fix
weaknesses, not to produce a leaderboard number: every eval reports *which
category failed and why*, and every fixed failure becomes a regression test in
`tests/`.

MemContext promises three things. Each eval below checks one of them.

1. **Current, not stale.** When a fact changes, the agent gets the new value, and
   the old one is kept as history, never served as current.
2. **Right state decisions.** A new fact replaces, adds to, or conflicts with what
   is already known, correctly and deterministically, within the right scope and
   trust boundaries.
3. **It works where the user is.** A fresh Claude Code session gets the current
   project decision without the user repeating it, and never acts on a stale one.

| Eval | Promise | Question it answers | Real failure it guards against | Cost |
|---|---|---|---|---|
| `stale_exposure.py` | 1 | After a change, how often does anything served (facts, raw turns, hook injection) still carry the old value? | Raw turns re-served the superseded value in 19/20 queries. | seconds |
| `supersession_matrix.py` | 2 | For update / additive / history / duplicate / negation / cross-session / cross-namespace / trust-direction cases, is the final state right, in deterministic and semantic mode, and where do the two disagree? | Assistant claims overriding user facts; decisions changed in a later session never retiring the old one; concurrent writes superseding each other. | seconds (deterministic), ~1 min (semantic) |
| `canary_real_embedder.py` | 2 | Is semantic memory actually on when the product says it is? | A type-checker fix made the local embedder raise on every call for 19 days while CI and `status` said ON. | < 2 min, needs the model |
| `claude_code_recall.py` | 3 | Across real, separate `claude -p` sessions: current-decision recall, stale answers, stale *actions* (code written with an old decision), and whether the answer came from hook injection or a tool call. | Hooks rejected with 401 on every call; a fresh session answering `DB_ENGINE = 'unknown'`. | a few dollars per full run |

## Rules

- **Dev vs held-out.** Datasets are split. Look at and debug with `*_dev.json`
  only. `*_heldout.json` is run once per change, never used to tune prompts,
  thresholds, weights or labels. (The Anti-Overfitting Rule applies to our own
  evals too.)
- **Label from intent, before running.** Expected outcomes come from the
  documented product behaviour, not from what the code currently does. Genuinely
  ambiguous cases are marked `ambiguous` and excluded from accuracy; known
  unsupported behaviour is marked `expected_limitation`.
- **Diagnose, then fix the deepest layer.** A failing case is classified with
  LIPI (Logic, Implementation, Integration, Plumbing) before any fix. The fix
  lands with a failing-first regression test in `tests/`.
- **Report n and limitations with every number.** These are small, hand-written
  sets. They diagnose; they do not rank systems.
- **Results are committed** under `results/` with the commit sha and timestamp,
  so any number quoted can be traced to the code that produced it.

## Running

From the repo root, with the dev environment installed (`pip install -e ".[dev]"`,
plus `.[embeddings]` for semantic mode and the canary):

```bash
python -m evals.product.stale_exposure
python -m evals.product.supersession_matrix --mode both --split dev
python -m evals.product.canary_real_embedder
python -m evals.product.claude_code_recall --model haiku --max-cost-usd 5
```

These are not part of the CI test gate; `tests/` is. Run them before a release,
after changing supersession/retrieval/hooks, and before quoting any number.
