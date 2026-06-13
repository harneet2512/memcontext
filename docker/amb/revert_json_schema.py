#!/usr/bin/env python3
"""Revert install.py's json_object downgrade back to AMB's NATIVE strict
json_schema — so the benchmark's answer/judge parsing is pristine.

install.py changes openai.py's response_format from strict json_schema to
json_object for gateway compatibility. But strict json_schema is verified to
work through both gateways here (OpenRouter gpt-oss-120b, TokenRouter
gemini-3-flash-preview) AND it forces the model to return ALL required fields —
so the answer (`answer`/`reasoning`) and judge (`correct`/`reason`) responses are
always complete and AMB's native rag.py/judge.py parse them WITHOUT modification.
This restores the strict schema, leaving the benchmark untouched on the json path.

Usage: python revert_json_schema.py <path-to AMB llm/openai.py>
"""
import sys
import pathlib

NEW = 'response_format={"type": "json_object"},'
OLD = (
    'response_format={\n'
    '                        "type": "json_schema",\n'
    '                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},\n'
    '                    },'
)


def main() -> int:
    oai = pathlib.Path(sys.argv[1])
    text = oai.read_text(encoding="utf-8")
    if OLD in text:
        print("[revert_json_schema] already native json_schema"); return 0
    n = text.count(NEW)
    if n != 1:
        print(f"ERROR: json_object anchor found {n}x (expected 1)", file=sys.stderr)
        return 1
    oai.write_text(text.replace(NEW, OLD, 1), encoding="utf-8")
    print("[revert_json_schema] openai.py -> native strict json_schema (benchmark untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
