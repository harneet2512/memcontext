"""Install the MemContext provider into a local AMB checkout.

Run from inside the agent-memory-benchmark/ directory:

    python ../memcontext/evals/amb/install.py

Does three things, idempotently:
  1. Copies provider.py -> src/memory_bench/memory/memcontext.py
  2. Registers MemContextProvider in src/memory_bench/memory/__init__.py
  3. Patches src/memory_bench/llm/openai.py to use json_object mode
     (so AMB's answer/judge work through OpenRouter-hosted models)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMB = Path.cwd()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    mem_dir = AMB / "src" / "memory_bench" / "memory"
    if not mem_dir.is_dir():
        fail(f"Run this from the agent-memory-benchmark/ root. Not found: {mem_dir}")

    # 1. Copy provider
    dst = mem_dir / "memcontext.py"
    shutil.copy(HERE / "provider.py", dst)
    print(f"  copied provider -> {dst}")

    # 2. Register in __init__.py
    init = mem_dir / "__init__.py"
    text = init.read_text(encoding="utf-8")
    if "MemContextProvider" not in text:
        text = text.replace(
            "from .base import MemoryProvider",
            "from .base import MemoryProvider\nfrom .memcontext import MemContextProvider",
            1,
        )
        # add to REGISTRY dict
        text = text.replace(
            "REGISTRY: dict[str, type[MemoryProvider]] = {",
            'REGISTRY: dict[str, type[MemoryProvider]] = {\n    "memcontext": MemContextProvider,',
            1,
        )
        init.write_text(text, encoding="utf-8")
        print("  registered memcontext in __init__.py")
    else:
        print("  memcontext already registered")

    # 3. Patch openai.py for json_object mode
    oai = AMB / "src" / "memory_bench" / "llm" / "openai.py"
    otext = oai.read_text(encoding="utf-8")
    if "json_object" not in otext:
        otext = otext.replace(
            'response_format={\n                        "type": "json_schema",\n                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},\n                    },',
            'response_format={"type": "json_object"},',
        )
        oai.write_text(otext, encoding="utf-8")
        print("  patched openai.py -> json_object mode")
    else:
        print("  openai.py already patched")

    print("\nInstall complete. Set keys in .env, then run via run.sh")


if __name__ == "__main__":
    main()
