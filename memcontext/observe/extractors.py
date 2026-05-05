"""Claim extractors for browser observation.

AccessibilityTreeExtractor: parses Playwright accessibility tree into claims.
DOMExtractor: fallback for pages with poor accessibility markup.

Both are local-only, no external API calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservedField:
    """A field extracted from page observation."""

    name: str
    value: str
    role: str  # a11y role: "heading", "textbox", "link", etc.
    source: str  # "a11y" or "dom"


class AccessibilityTreeExtractor:
    """Extract structured claims from a Playwright accessibility tree snapshot.

    Walks the a11y tree and identifies:
    - Page title -> observation claim
    - Headings -> context claims
    - Form fields with labels -> observation claims
    - Links -> context claims
    - Text content -> observation claims
    """

    def extract(self, snapshot) -> list[dict]:
        """Extract claims from a PageSnapshot."""
        claims: list[dict] = []
        tree = (
            snapshot.accessibility_tree
            if hasattr(snapshot, "accessibility_tree")
            else {}
        )
        url = snapshot.url if hasattr(snapshot, "url") else "unknown"
        title = snapshot.title if hasattr(snapshot, "title") else ""

        # Page-level claim
        if title:
            claims.append(
                {
                    "subject": _url_to_subject(url),
                    "predicate": "observation",
                    "value": f"page title: {title}",
                    "confidence": 0.95,
                }
            )

        # Walk tree
        self._walk_node(tree, claims, url, depth=0)

        return claims

    def _walk_node(
        self, node: dict, claims: list[dict], url: str, depth: int
    ) -> None:
        if not isinstance(node, dict):
            return
        if depth > 20:  # prevent infinite recursion
            return

        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")

        subject = _url_to_subject(url)

        if role == "heading" and name:
            claims.append(
                {
                    "subject": subject,
                    "predicate": "observation",
                    "value": f"heading: {name}",
                    "confidence": 0.9,
                }
            )

        elif role in ("textbox", "combobox", "searchbox") and name:
            claims.append(
                {
                    "subject": subject,
                    "predicate": "observation",
                    "value": (
                        f"field '{name}': {value}" if value else f"field: {name}"
                    ),
                    "confidence": 0.8,
                }
            )

        elif role == "link" and name and len(name) > 3:
            claims.append(
                {
                    "subject": subject,
                    "predicate": "observation",
                    "value": f"link: {name}",
                    "confidence": 0.7,
                }
            )

        elif role in ("text", "paragraph", "article") and name and len(name) > 10:
            # Only capture meaningful text content
            truncated = name[:200] if len(name) > 200 else name
            claims.append(
                {
                    "subject": subject,
                    "predicate": "observation",
                    "value": truncated,
                    "confidence": 0.6,
                }
            )

        # Recurse into children
        for child in node.get("children", []):
            self._walk_node(child, claims, url, depth + 1)


class DOMExtractor:
    """Fallback extractor using raw DOM text content.

    Used when accessibility tree is empty or unhelpful.
    Extracts text from common structural elements.
    """

    def extract_from_text(
        self, url: str, title: str, text_blocks: list[str]
    ) -> list[dict]:
        """Extract claims from raw text blocks."""
        claims: list[dict] = []
        subject = _url_to_subject(url)

        if title:
            claims.append(
                {
                    "subject": subject,
                    "predicate": "observation",
                    "value": f"page title: {title}",
                    "confidence": 0.95,
                }
            )

        for block in text_blocks[:20]:  # limit to 20 blocks
            block = block.strip()
            if len(block) > 10:
                truncated = block[:200] if len(block) > 200 else block
                claims.append(
                    {
                        "subject": subject,
                        "predicate": "observation",
                        "value": truncated,
                        "confidence": 0.5,
                    }
                )

        return claims


def _url_to_subject(url: str) -> str:
    """Convert URL to a readable subject identifier."""
    # Extract domain + path, stripping protocol and query params
    clean = re.sub(r"^https?://", "", url)
    clean = clean.split("?")[0].split("#")[0]
    clean = clean.rstrip("/")
    return clean or "unknown_page"
