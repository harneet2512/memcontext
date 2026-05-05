"""Browser observation sub-package -- the memory layer that can see.

Captures browser application state via Playwright's accessibility tree,
extracts structured claims, and stores them with screen-state provenance.
Re-visit flows detect changes and trigger supersession.

Requires playwright: pip install memcontext[browser]
"""
from __future__ import annotations
