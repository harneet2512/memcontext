# LIPI — Four-Avenue Bug Diagnosis

A method for finding the *real* cause of a bug instead of the first plausible one.

Bugs compound across layers. The most common diagnostic failure is stopping at the first
thing that looks wrong — you fix it, ship, and either the symptom returns (because the real
cause was deeper) or a second bug that was hiding behind the first surfaces later. LIPI
prevents that: when something breaks, check **all four layers** a bug can live in before you
decide what to change. The diagnosis is complete only when all four are accounted for.

---

## The four avenues

**1. Logic — is the approach itself right?**
The code does exactly what you told it to, but what you told it to do is wrong.
Inverted condition, `<` where you needed `<=`, wrong sort key, off-by-one at a boundary,
a threshold/weight/constant that doesn't match the spec, a correct algorithm pointed at the
wrong question. *Stepped through on paper, would it give the right answer?*

**2. Implementation — does the code do what you meant?**
The approach is right; the code doesn't carry it out faithfully.
Swallowed exception returning a default instead of failing loudly, an unreachable branch,
the wrong variable, a missing `await`, a stale cached value, a silent type coercion, a null
deref, a resource left open. *The gap between the comment and the lines under it.*

**3. Integration — do the pieces actually fit together?**
Each component works alone but they don't compose.
Module A hands B data in a shape B doesn't expect; two code paths (new vs legacy, fast vs
fallback) where the fix is on the one that *isn't* running; a caller and callee drifted out
of sync after a rename; version skew; something registered/wired but never actually invoked
on the live path.

**4. Plumbing — does the data make it end to end?**
The wiring is correct but the data doesn't travel.
It was never written to the store; the query filters out the row you needed; a path/key is
normalized one way here and another there so they never match; config doesn't survive a
boundary (request, restart, worker); a connection is read-only where it needs to write; a
field is dropped in serialization; a channel truncates or swallows what it was meant to
carry. *Don't assume the data is there — go look.*

---

## How to apply it

1. **Start from a real symptom** — a failing test, an error line, a reproducible
   misbehavior. Diagnosis without a concrete failure is guessing.
2. **Walk all four, and state what you find** — even the clean ones: *what you checked,
   what you found, broken or clean.* Clearing a layer out loud is what stops tunnel vision.
3. **Don't stop when one layer explains it.** The first cause is often a *consequence* of a
   deeper one, and independent bugs hide behind each other. Keep going.
4. **Fix the deepest layer that explains the failure** — patching the symptom leaves the
   cause live.
5. **Re-check the other three after the fix.** The most common regression is a fix in one
   layer quietly breaking another. You're not done until you've confirmed it didn't.
6. **If you split a hard bug, each person checks all four layers** for their piece — never
   "you take logic, I'll take plumbing." Bugs cross layers; that split guarantees a blind
   spot.

---

## Template

```
Symptom:   <the exact failure — test, log line, repro>

Logic:           <checked / found> — broken | clean
Implementation:  <checked / found> — broken | clean
Integration:     <checked / found> — broken | clean
Plumbing:        <checked / found> — broken | clean

Root cause:  <which layer, with file:line>
Fix:         <the change>
Re-checked:  <other three layers still hold>
```

---

**One line:** four layers, check all four — the bug you find first is rarely the only one,
and sometimes isn't even the real one.
