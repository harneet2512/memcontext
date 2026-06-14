#!/usr/bin/env python3
"""Two transport-robustness fixes to the AMB answer/judge backend (llm/openai.py).
Transport only — no scoring/prompt/judge change. Run AFTER revert_json_schema.py.

(1) OpenRouter STRICT enforcement. Same model the official benchmark uses (groq
    backend); we route it via OpenRouter, which only guarantees structured output
    if you require it. Without `provider.require_parameters=true`, OpenRouter may
    pick a provider that ignores `strict`, so gpt-oss-120b's reasoning bleeds into
    message.content. Scoped to OpenRouter clients (base_url contains 'openrouter')
    so the TokenRouter judge call is untouched.

(2) NON-DICT GUARD. Even so, a reasoning-bleed response can make content a bare
    scalar, and json.loads() returns e.g. a float; rag.py then does data["answer"]
    -> 'float' object is not subscriptable, which kills the ENTIRE shard (3/6
    shards died this way in trial12). Validate the parse is a dict and, if not,
    treat it as a transient and retry within the existing loop instead of crashing.

Usage: python patch_openrouter_strict.py <path-to AMB llm/openai.py>
"""
import sys
import pathlib

EDITS = [
    # (1) require_parameters on the OpenRouter (answer) create() call only.
    (
        (
            "                response = self._client.chat.completions.create(\n"
            "                    model=self._model,\n"
            '                    messages=[{"role": "user", "content": prompt}],\n'
            "                    response_format={\n"
            '                        "type": "json_schema",\n'
            '                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},\n'
            "                    },\n"
            "                )"
        ),
        (
            "                # OpenRouter only enforces strict json_schema when explicitly\n"
            "                # required; without this gpt-oss-120b's reasoning bleeds into\n"
            "                # content. Scoped to OpenRouter so the TokenRouter judge is untouched.\n"
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
        ),
        "OpenRouter require_parameters (answer role; transport only)",
    ),
    # (2) non-dict guard: retry instead of letting rag.py subscript a float.
    (
        (
            "                text = response.choices[0].message.content\n"
            "                return json.loads(text)"
        ),
        (
            "                text = response.choices[0].message.content\n"
            "                result = json.loads(text)\n"
            "                if isinstance(result, dict):\n"
            "                    return result\n"
            "                # Non-dict (reasoning bleed -> bare scalar). Without this,\n"
            '                # rag.py data["answer"] subscripts a float and kills the shard.\n'
            "                # Treat as transient and retry inside this loop.\n"
            '                last_exc = ValueError(f"non-dict LLM response: {text[:120]!r}")\n'
            "                if attempt < _MAX_RETRIES - 1:\n"
            "                    time.sleep(delay)\n"
            "                    delay *= 2\n"
            "                    continue\n"
            "                raise last_exc"
        ),
        "non-dict guard: retry instead of crashing the shard on data['answer']",
    ),
]


def main() -> int:
    oai = pathlib.Path(sys.argv[1])
    text = oai.read_text(encoding="utf-8")
    if "require_parameters" in text and "non-dict LLM response" in text:
        print("[patch_openrouter_strict] already applied")
        return 0
    for anchor, replacement, label in EDITS:
        n = text.count(anchor)
        if n != 1:
            print(
                f"ERROR: anchor for [{label}] found {n}x (expected 1) — openai.py "
                "drifted or revert_json_schema did not run first.",
                file=sys.stderr,
            )
            return 1
        text = text.replace(anchor, replacement, 1)
        print(f"  patched: {label}")
    oai.write_text(text, encoding="utf-8")
    print("[patch_openrouter_strict] OpenRouter strict + non-dict guard applied (transport only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
