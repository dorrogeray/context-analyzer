"""Parse a Claude Code transcript into per-API-call token events.

Thin adapter over :mod:`context_tracker.transcript_reader`, which owns the
de-duplication of the multiple transcript lines a single API response is
written as. This module only maps those responses onto ``ApiTurnEvent``.
"""

from __future__ import annotations

from pathlib import Path

from context_tracker.models import ApiTurnEvent
from context_tracker.transcript_reader import SYNTHETIC_MODEL, iter_api_responses

__all__ = ["SYNTHETIC_MODEL", "parse_transcript"]


def parse_transcript(transcript_path: Path) -> list[ApiTurnEvent]:
    """Extract one ApiTurnEvent per API call from a transcript.

    One event per API *response*, not per transcript line: Claude Code
    writes a line per content block and repeats the usage on each, so
    counting lines inflated every token total by 1.6-3x.
    """
    events: list[ApiTurnEvent] = []

    for turn_number, entry in enumerate(iter_api_responses(transcript_path), start=1):
        message = entry["message"]
        usage = message.get("usage", {})
        events.append(
            ApiTurnEvent(
                session_id=entry.get("sessionId", "unknown"),
                turn_number=turn_number,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                model=message.get("model", "unknown"),
                stop_reason=message.get("stop_reason"),
            )
        )

    return events
