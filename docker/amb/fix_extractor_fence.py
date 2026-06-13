#!/usr/bin/env python3
"""Strip markdown code fences from the extractor LLM response before json.loads.

MiniMax-M3 (and other gateway-routed models) wrap their JSON in a ```json ... ```
fence EVEN with response_format=json_object set, which makes the extractor's
`json.loads(content)` raise — and the AMB provider swallows that to []. Result:
zero claims, silently. This adds a `_strip_code_fence` helper and applies it at
the single choke point `_split_entities`, so every fenced response parses.

Overlay-applied to the INSTALLED memcontext (fast iteration). Fold into the
product (master memcontext/extractors.py) for the canonical artifact.

Usage: python fix_extractor_fence.py <path-to-installed memcontext/extractors.py>
"""
import sys
import pathlib

NL = chr(10)

HELPER = NL.join([
    "def _strip_code_fence(content: str) -> str:",
    '    """Strip a leading/trailing markdown code fence. MiniMax-M3 and other',
    "    gateway-routed models wrap JSON in a fence even with",
    '    response_format=json_object, which would break json.loads."""',
    "    s = content.strip()",
    '    if s.startswith("```"):',
    "        nl = s.find(chr(10))",
    "        if nl != -1:",
    "            s = s[nl + 1:]",
    "        s = s.rstrip()",
    '        if s.endswith("```"):',
    "            s = s[:-3]",
    "    return s.strip()",
    "",
    "",
    "def _split_entities(content: str) -> tuple[str, str]:",
])

EDITS = [
    # 1) insert the helper just before _split_entities
    ("def _split_entities(content: str) -> tuple[str, str]:", HELPER),
    # 2) strip the fence at the top of _split_entities body
    (
        "    try:" + NL + "        parsed = json.loads(content)",
        "    content = _strip_code_fence(content)" + NL
        + "    try:" + NL + "        parsed = json.loads(content)",
    ),
]


def main() -> int:
    ext = pathlib.Path(sys.argv[1])
    text = ext.read_text(encoding="utf-8")
    if "_strip_code_fence" in text:
        print("[fix_extractor_fence] already applied"); return 0
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            print(f"ERROR: anchor found {n}x (expected 1): {old[:40]!r}", file=sys.stderr)
            return 1
        text = text.replace(old, new, 1)
    ext.write_text(text, encoding="utf-8")
    print(f"[fix_extractor_fence] fence-strip applied to {ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
