"""The one place that turns a Claude Code transcript into API responses.

A transcript is JSONL, one entry per line, but a line is NOT an API call.
Claude Code writes a separate line per content block of a single assistant
response — a ``thinking`` line, a ``text`` line, a ``tool_use`` line — and
every one of them repeats the full ``message.usage``. Counting lines
therefore inflates token totals two- to three-fold.

This module owns that knowledge. Three parsers used to each carry their own
copy of the "is this a completed assistant message" predicate, and the
de-duplication fix reached only one of them, so the MCP server and the
session-reconstruction path kept reporting inflated numbers long after the
ingest path was correct. Anything that reads a transcript should go through
:func:`iter_api_responses` or :func:`coalesce_assistant_entries` rather than
re-deriving this.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SYNTHETIC_MODEL = "synthetic"

# A response is identified by message.id; requestId is the fallback for
# transcripts that predate it.
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def load_entries(path: Path) -> list[dict]:
    """Load every JSONL entry from a transcript, skipping malformed lines."""
    entries: list[dict] = []
    if not Path(path).exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def is_completed_assistant(msg: dict) -> bool:
    """True when an assistant message represents a finished API response.

    Streaming chunks have no ``stop_reason``; synthetic messages are locally
    generated and were never billed; a zero ``output_tokens`` means nothing
    was produced.
    """
    if msg.get("model") == SYNTHETIC_MODEL:
        return False
    if msg.get("stop_reason") is None:
        return False
    usage = msg.get("usage", {})
    if not isinstance(usage, dict):
        return False
    return int(usage.get("output_tokens", 0) or 0) > 0


def response_key(entry: dict) -> str | None:
    """Identify which API response an assistant entry belongs to.

    All lines of one response share ``message.id`` and ``requestId``. None
    means neither is present, so the line cannot be matched to a sibling and
    has to stand on its own.
    """
    msg = entry.get("message", {})
    message_id = msg.get("id") if isinstance(msg, dict) else None
    if message_id:
        return str(message_id)
    request_id = entry.get("requestId")
    return str(request_id) if request_id else None


def coalesce_assistant_entries(entries: list[dict]) -> list[dict]:
    """Fold transcript lines describing the same API response into one.

    Completed assistant lines sharing a response key are merged into the
    first of them, concatenating their content blocks in order; the usage is
    counted once. Everything else — user entries, streaming chunks, unknown
    types — passes through untouched and in order, so callers that rely on
    entry ordering or index tool_use blocks are unaffected.
    """
    merged: dict[str, dict] = {}
    result: list[dict] = []

    for entry in entries:
        msg = entry.get("message", {})
        if entry.get("type") != "assistant" or not isinstance(msg, dict) or not is_completed_assistant(msg):
            result.append(entry)
            continue

        key = response_key(entry)
        if key is None:
            result.append(entry)
            continue

        first = merged.get(key)
        if first is None:
            content = msg.get("content", [])
            copy = dict(entry)
            copy["message"] = {**msg, "content": list(content) if isinstance(content, list) else content}
            merged[key] = copy
            result.append(copy)
            continue

        # Same response, another content block — keep the block, drop the usage.
        extra = msg.get("content", [])
        target = first["message"].get("content")
        if isinstance(target, list) and isinstance(extra, list):
            target.extend(extra)

    return result


def iter_api_responses(path: Path) -> Iterator[dict]:
    """Yield one entry per real API response, in transcript order."""
    for entry in coalesce_assistant_entries(load_entries(path)):
        msg = entry.get("message", {})
        if entry.get("type") == "assistant" and isinstance(msg, dict) and is_completed_assistant(msg):
            yield entry


def ephemeral_1h_tokens(usage: dict[str, Any]) -> int:
    """Cache-creation tokens written with the 1-hour TTL.

    ``cache_creation_input_tokens`` is the total across both TTLs; the nested
    ``cache_creation`` object breaks it down. A 1-hour write bills at 2x base
    input against 1.25x for the 5-minute one, so the split has to survive
    into the cost model. Older transcripts omit the object, in which case
    everything is treated as a 5-minute write.
    """
    breakdown = usage.get("cache_creation")
    if not isinstance(breakdown, dict):
        return 0
    try:
        return int(breakdown.get("ephemeral_1h_input_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0
