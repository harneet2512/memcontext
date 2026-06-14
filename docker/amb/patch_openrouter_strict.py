#!/usr/bin/env python3
"""Transport fix: make OpenRouter ENFORCE the strict json_schema for the answer
role (gpt-oss-120b). Same model the official benchmark uses (groq backend) — we
route it via OpenRouter, and OpenRouter only guarantees structured output if you
explicitly require it. Without this, OpenRouter may pick a provider that ignores
`strict`, so gpt-oss-120b's reasoning channel bleeds into message.content (the
"We must answer the user's question..." leak) and json.loads(content) can return
a bare float -> rag.py's data["answer"] raises 'float' object is not subscriptable
and the whole shard dies.

OpenRouter's `provider.require_parameters=true` forces it to only route to
providers that honour every request parameter — including response_format
json_schema. So the answer comes back as a clean {"answer","reasoning"} object,
exactly as the groq backend returns it. Transport only: no scoring/prompt/judge
change. Scoped to OpenRouter clients (base_url contains 'openrouter') so the
TokenRouter judge call is completely unaffected.

Run AFTER revert_json_schema.py (which restores the json_schema response_format
this anchors on).

Usage: python patch_openrouter_strict.py <path-to AMB llm/openai.py>
"""
import sys
import pathlib

ANCHOR = (
    "                response = self._client.chat.completions.create(\n"
    "                    model=self._model,\n"
    '                    messages=[{"role": "user", "content": prompt}],\n'
    "                    response_format={\n"
    '                        "type": "json_schema",\n'
    '                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},\n'
    "                    },\n"
    "                )"
)

REPLACEMENT = (
    "                # OpenRouter only enforces strict json_schema when explicitly\n"
    "                # required; without this gpt-oss-120b's reasoning bleeds into\n"
    "                # content and json.loads can return a bare float (shard crash).\n"
    "                # Scoped to OpenRouter so the TokenRouter judge is untouched.\n"
    "                _extra = {}\n"
    '                _burl = str(getattr(self._client, "base_url", "") or "")\n'
    '                if "openrouter" in _burl:\n'
    '                    _extra = {"extra_body": {"provider": {"require_parameters": True}}}\n'
    "                response = self._client.chat.completions.create(\n"
    "                    model=self._model,\n"
    '                    messages=[{"role": "user", "content": prompt}],\n'
    "                    response_format={\n"
    '                        "type": "json_schema",\n'
    '                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},\n'
    "                    },\n"
    "                    **_extra,\n"
    "                )"
)


def main() -> int:
    oai = pathlib.Path(sys.argv[1])
    text = oai.read_text(encoding="utf-8")
    if "require_parameters" in text:
        print("[patch_openrouter_strict] already applied")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(
            f"ERROR: create() anchor found {n}x (expected 1) — openai.py drifted "
            "or revert_json_schema did not run first.",
            file=sys.stderr,
        )
        return 1
    oai.write_text(text.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print("[patch_openrouter_strict] OpenRouter require_parameters enforced (answer role; transport only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
