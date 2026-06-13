#!/usr/bin/env python3
"""Enumerate every (dataset, split, category) shard across the whole AMB suite,
for the sharded GHA matrix. Prints a single line `MATRIX=<json-array>`.

Uses omb's own API: DATASET_REGISTRY + Dataset.splits + Dataset.categories(split).
Datasets that fail to load (e.g. need an external data path) are skipped, not fatal.
Set ONLY=<dataset> to restrict to one dataset.
"""
import json
import os

shards: list[dict] = []
# The registry is `REGISTRY` in memory_bench.dataset (cli.py aliases it
# DATASET_REGISTRY). Try both so we're robust to either name.
DATASET_REGISTRY = {}
for _attr in ("REGISTRY", "DATASET_REGISTRY"):
    try:
        import importlib
        DATASET_REGISTRY = getattr(importlib.import_module("memory_bench.dataset"), _attr)
        break
    except Exception:
        continue
if not DATASET_REGISTRY:
    print("skip-all: could not find dataset registry (REGISTRY/DATASET_REGISTRY)")

only = os.environ.get("ONLY", "").strip()

for name in list(DATASET_REGISTRY):
    if only and name != only:
        continue
    try:
        obj = DATASET_REGISTRY[name]
        ds = obj() if isinstance(obj, type) else obj
        for split in (getattr(ds, "splits", []) or []):
            try:
                cats = ds.categories(split)
            except Exception:
                cats = None
            if cats:
                for c in cats:
                    shards.append({"dataset": name, "split": split, "category": str(c)})
            else:
                shards.append({"dataset": name, "split": split, "category": ""})
    except Exception as e:
        print("skip", name, repr(e)[:160])

print("MATRIX=" + json.dumps(shards))
