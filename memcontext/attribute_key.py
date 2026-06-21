"""Deterministic attribute-slot derivation for coarse-predicate identity.

FRACTURE B fix. Supersession (Pass-1 candidate match + Pass-2), projection
collapse, and enumeration all key on ``(subject, predicate)``. But the live
extractor (extractors.py, the ``personal_assistant``/general packs) emits ONE
coarse predicate for most personal facts — ``user_fact`` — so the identity key
degenerates to ``(user, user_fact)`` for residence, employer, hobby, allergy,
office, commute … *everything*. Under that single key:

  * Pass-1's single-valued branch is dead (``user_fact`` is not single_valued),
    and the multi-valued jaccard branch can mis-fuse two unrelated facts that
    happen to share a stray token.
  * ``projections.claims_grouped_by_subject_predicate`` (newest-wins) would fuse
    ALL the user's facts into one row.
  * ``enumeration`` would count the entire personal corpus as one slot.

This module derives a SECOND, deterministic discriminator from the VALUE — the
attribute *slot* the value describes — so identity becomes
``(subject, predicate, attribute)`` without touching the schema or the extractor.

Determinism + generality (NO LLM, NO benchmark coupling, NO domain predicate
lists):

  * ``attribute_key`` reads the slot off the VALUE's surface form using two
    domain-agnostic English patterns: an explicit ``label: value`` prefix, and a
    leading generic relation verb phrase ("lives in", "works at", "is allergic
    to", "likes", …). Both are properties of English, not of any dataset.
  * It returns ``""`` whenever it finds no slot signal — and callers treat an
    empty attribute as "no opinion", so behaviour is byte-identical to today
    when the value carries no derivable slot (the no-regression guarantee).

The split rule (``attributes_conflict``) only ever SPLITS — and only when BOTH
sides carry a non-empty, DIFFERING attribute. If either side is empty it abstains.
This makes the change strictly additive: it can prevent a false fuse, but it can
never create one, and it never blocks a supersession the product fires today
unless the two values demonstrably name different slots.
"""
from __future__ import annotations

import re

# Generic English relation heads. These are NOT predicate names and NOT domain
# vocabulary — they are ordinary verbs/relations a speaker uses to attach a value
# to an attribute slot. The relation phrase itself becomes the slot token, so
# "lives in Boston" and "moved to Denver" both resolve to the residence slot
# ("reside"), while "works at Acme" resolves to the employer slot ("work"). The
# map normalises surface variants onto a canonical slot token.
#
# Ordered longest-first within each group so multi-word triggers win over their
# prefixes ("is allergic to" before "is").
_RELATION_SLOTS: tuple[tuple[str, str], ...] = (
    # residence
    ("lives in", "reside"), ("live in", "reside"), ("living in", "reside"),
    ("moved to", "reside"), ("relocated to", "reside"), ("relocating to", "reside"),
    ("resides in", "reside"), ("reside in", "reside"), ("based in", "reside"),
    # employer
    ("works at", "work"), ("work at", "work"), ("working at", "work"),
    ("employed at", "work"), ("employed by", "work"), ("works for", "work"),
    ("work for", "work"),
    # allergy / medical avoidance
    ("is allergic to", "allergic"), ("allergic to", "allergic"),
    ("is allergic", "allergic"),
    # likes / preferences — ALL liking/preference verbs map to ONE slot
    # ('prefer'). A user restating a preference with a different verb
    # ("likes coffee" -> "prefers tea") is an UPDATE of the same role, not a
    # distinct fact, so these must NOT conflict. (Polarity — like vs dislike — is
    # NOT a slot distinction here: "likes X" then "dislikes X" is still an update
    # about X; value-level negation handling lives elsewhere.)
    ("prefers", "prefer"), ("prefer", "prefer"),
    ("likes", "prefer"), ("like", "prefer"), ("loves", "prefer"),
    ("enjoys", "prefer"), ("dislikes", "prefer"), ("hates", "prefer"),
    # generic "has a" possession ("has a dog", "owns a car")
    ("owns", "own"), ("has a", "have"), ("have a", "have"),
)


def _norm_label(label: str) -> str:
    """Normalise an explicit ``label:`` prefix to a stable slot token.

    Lowercase, keep alphanumerics, collapse separators to single underscores,
    strip a few leading determiners ("my", "the", "a") that add no slot
    information. Deterministic.
    """
    toks = [t for t in re.findall(r"[a-z0-9]+", label.lower())
            if t not in {"my", "the", "a", "an"}]
    return "_".join(toks)


def attribute_key(value: str) -> str:
    """Derive a deterministic attribute-slot token from a claim value.

    Returns ``""`` when no slot can be read off the surface form (the
    no-opinion / no-regression case). Never raises.

    Priority:
      1. Explicit ``label: value`` prefix  -> normalised label
         ("home city: Toronto" -> "home_city", "employer: Acme" -> "employer").
      2. Leading generic relation verb      -> canonical relation slot
         ("lives in NYC" -> "reside", "works at Acme" -> "work").
      3. otherwise                          -> "".

    Zero-LLM, zero embeddings, domain-agnostic.
    """
    if not value:
        return ""
    v = value.strip()

    # 1) Explicit "label: value". Require the label to be short (a slot name, not
    #    a sentence with an incidental colon) — at most 4 tokens before the colon.
    colon = v.find(":")
    if 0 < colon <= 60:
        label = v[:colon]
        if len(re.findall(r"[a-z0-9]+", label.lower())) <= 4:
            key = _norm_label(label)
            if key:
                return key

    # 2) Leading generic relation phrase, matched on word-token boundaries so
    #    "works at" never matches inside "frameworks at". We scan the value's
    #    token stream and accept the FIRST relation trigger whose token sequence
    #    appears as a contiguous run within the first few tokens of the value —
    #    relations normally lead the value the extractor emits ("works at Acme").
    vtoks = re.findall(r"[a-z0-9]+", v.lower())
    if vtoks:
        head = vtoks[:6]  # the relation lives at the head of the value
        for trigger, slot in _RELATION_SLOTS:
            ptoks = trigger.split()
            n = len(ptoks)
            if any(head[i : i + n] == ptoks for i in range(len(head) - n + 1)):
                return slot

    return ""


def attributes_conflict(value_a: str, value_b: str) -> bool:
    """True iff the two values demonstrably name DIFFERENT attribute slots.

    Abstains (returns False) whenever either value yields an empty attribute, so
    the discriminator can only ever PREVENT a false fuse — never create one, and
    never block a supersession the product fires today on values with no
    derivable slot. This is the no-regression contract.
    """
    ka = attribute_key(value_a)
    kb = attribute_key(value_b)
    if not ka or not kb:
        return False
    return ka != kb
