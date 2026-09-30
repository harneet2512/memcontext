"""Human-readable renderers for the world-state and supersession lineage.

Presentation only — these consume the dicts produced by ``brain()`` and
``handle_memory_trace`` and never touch the database or a model. ASCII output
so it renders cleanly on a Windows console.
"""
from __future__ import annotations


def _span_part(char_start: int | None, char_end: int | None) -> str:
    """``"  span [s:e]"`` when a char span exists, else ``""`` (omit the noise)."""
    if char_start is None or char_end is None:
        return ""
    return f"  span [{char_start}:{char_end}]"


def format_world_state(ws: dict) -> str:
    """Render the brain() projection grouped by subject, with provenance + gaps."""
    lines: list[str] = []
    lines.append(
        f"WORLD STATE  .  session={ws.get('session_id')}  .  pack={ws.get('pack')}"
    )
    subjects = ws.get("subjects", {})
    if not subjects:
        lines.append("  (no active claims)")
        return "\n".join(lines)

    for subject in subjects:
        block = subjects[subject]
        lines.append("")
        lines.append(f"> {subject}")
        for fact in block.get("facts", []):
            prov = fact.get("provenance", {})
            quote = prov.get("quote")
            quote_str = f' "{quote}"' if quote else ""
            lines.append(
                f"    {fact['predicate']:<22} = {fact['value']}"
                f"   [{fact['status'].upper()}]   conf {fact['confidence']:.2f}"
            )
            lines.append(
                f"        source: turn {prov.get('source_turn_id')}"
                f"{_span_part(prov.get('char_start'), prov.get('char_end'))}{quote_str}"
            )
        gaps = block.get("gaps", [])
        if gaps:
            lines.append(f"    gaps (no active claim): {', '.join(gaps)}")
    return "\n".join(lines)


def render_trace_table(trace: dict) -> str:
    """Render a supersession lineage: active claim on top, superseded beneath.

    Consumes the ``lineage`` produced by ``handle_memory_trace`` — a list of
    rows ordered newest-first, each with value, status, typed edge, source turn,
    and span quote.
    """
    if trace.get("error"):
        return f"TRACE error: {trace['error']}"

    subject = trace.get("subject")
    predicate = trace.get("predicate")
    header = "TRACE"
    if subject is not None and predicate is not None:
        header = f"TRACE  .  {subject} / {predicate}"

    lineage = trace.get("lineage", [])
    if not lineage:
        return f"{header}\n  (no claim found)"

    lines: list[str] = [header, ""]
    for row in lineage:
        status = str(row.get("status", "")).upper()
        # NL-only facts (out-of-vocab predicate) have no structured value.
        value = row.get("value") or row.get("fact") or ""
        edge = row.get("edge_type", "")
        if status == "ACTIVE" or edge == "active":
            lines.append(f"  ACTIVE      {value}")
        else:
            # The edge is the typed reason this value was replaced.
            lines.append(f"  SUPERSEDED  {value}   <-- {edge}")
        speaker = row.get("speaker", "")
        speaker_str = f" ({speaker})" if speaker else ""
        quote = row.get("quote")
        quote_str = f'  "{quote}"' if quote else ""
        conf = row.get("confidence")
        conf_str = f"conf {conf:.2f}  " if isinstance(conf, (int, float)) else ""
        lines.append(
            f"              {conf_str}turn {row.get('source_turn_id')}{speaker_str}"
            f"{_span_part(row.get('char_start'), row.get('char_end'))}{quote_str}"
        )
    return "\n".join(lines)
