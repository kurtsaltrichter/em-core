"""
Universe loader for em-core.

Single source of truth for the list of tickers the TripWire scanner watches
and the EM Dashboard offers in its dropdowns. The raw list lives in
``em_core/data/universe.txt`` as one ticker per line with ``#`` section
headers; this module parses it into a flat tuple plus a per-section mapping.

The section headers in ``universe.txt`` follow the pattern::

    # ─── SECTION NAME ──────────────────────────

Any line starting with ``#`` is treated as a comment (new section if it
matches that pattern), blank lines are ignored, and everything else is taken
as a ticker (upper-cased, stripped).
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Iterable

_SECTION_RE = re.compile(r"^#\s*[─\-]+\s*(.+?)\s*[─\-]+\s*$")


def _iter_raw_lines() -> Iterable[str]:
    """Yield lines from the packaged ``data/universe.txt``."""
    with resources.files("em_core.data").joinpath("universe.txt").open(
        "r", encoding="utf-8"
    ) as fh:
        for line in fh:
            yield line.rstrip("\n")


def _parse() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    tickers: list[str] = []
    sections: dict[str, list[str]] = {}
    current = "UNCATEGORIZED"
    for line in _iter_raw_lines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = _SECTION_RE.match(stripped)
            if m:
                current = m.group(1).strip().upper()
                sections.setdefault(current, [])
            continue
        t = stripped.upper()
        tickers.append(t)
        sections.setdefault(current, []).append(t)
    # dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    frozen_sections = {k: tuple(v) for k, v in sections.items() if v}
    return tuple(unique), frozen_sections


@lru_cache(maxsize=1)
def _cached() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    return _parse()


def load_universe() -> tuple[str, ...]:
    """Return the full deduped ticker universe as a tuple of upper-case symbols."""
    return _cached()[0]


def load_sections() -> dict[str, tuple[str, ...]]:
    """Return ``{section_name: (ticker, ...)}`` parsed from ``universe.txt``."""
    return dict(_cached()[1])


def section(name: str) -> tuple[str, ...]:
    """Return the tickers in a single section (case-insensitive)."""
    return load_sections().get(name.strip().upper(), ())


def is_in_universe(ticker: str) -> bool:
    """Case-insensitive membership test."""
    return ticker.strip().upper() in set(load_universe())


def reload() -> None:
    """Clear the cache. Useful in tests after patching the data file."""
    _cached.cache_clear()


__all__ = [
    "load_universe",
    "load_sections",
    "section",
    "is_in_universe",
    "reload",
]
