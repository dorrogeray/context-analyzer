"""Repro for the inflated session-cost bug.

Sessions whose real spend is in the hundreds of dollars are reported at
several thousand. Four independent defects compound:

1. Claude Code writes ONE transcript line per content block of a single API
   response (a ``thinking`` line, a ``text`` line, a ``tool_use`` line), each
   repeating the full ``message.usage``. The parser treats every line as its
   own API call, so usage is summed 2-3x.
2. ``ingest.ingest_session`` looks for ``churn[0]["model"]``, which the
   parser never emits — so the model is never recorded and every session is
   priced at fixed Opus rates regardless of which model actually ran.
3. The cache-read rate is 0.125x input; Anthropic charges 0.1x.
4. ``stats.compute_stats`` aggregates across agents, but Codex sessions are
   deliberately stored with ``total_cost_usd = 0.0``.

Every test here asserts the CORRECT behaviour, so they fail until the bugs
are fixed. The existing fixtures miss defect 1 because their synthetic
assistant entries carry no ``message.id`` / ``requestId`` and never split a
response across lines — the shape that occurs in every real transcript.
"""

from __future__ import annotations

import json

from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from context_tracker.db import (
    AGENT_CLAUDE_CODE,
    AGENT_CODEX,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.ingest import ingest_session
from context_tracker.stats import compute_stats

# Usage of the one API response the repro transcripts are built around.
_USAGE = {
    "input_tokens": 4,
    "cache_creation_input_tokens": 30_000,
    "cache_read_input_tokens": 120_000,
    "output_tokens": 900,
}


# ---------------------------------------------------------------------------
# Helpers — transcript builders that model the REAL on-disk shape
# ---------------------------------------------------------------------------


def _user_entry(content, uuid="u1"):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"content": content},
    }


def _assistant_line(content_blocks, usage, uuid, message_id, request_id, parent="u1"):
    """One transcript LINE of an assistant response.

    A single API response may be written as several of these, all sharing
    ``message.id`` / ``requestId`` and carrying identical ``usage``.
    """
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "requestId": request_id,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {
            "id": message_id,
            "model": "claude-opus-4-6",
            "stop_reason": "tool_use",
            "content": content_blocks,
            "usage": usage,
        },
    }


def _split_response_entries():
    """One user turn + ONE API response split across three lines.

    This is what Claude Code actually writes when a response contains a
    thinking block, a text block and a tool call.
    """
    common = {
        "usage": _USAGE,
        "message_id": "msg_01SplitAcrossLines",
        "request_id": "req_01SplitAcrossLines",
    }
    return [
        _user_entry("Find the bug"),
        _assistant_line([{"type": "thinking", "thinking": "hmm"}], uuid="a1", **common),
        _assistant_line([{"type": "text", "text": "Looking now."}], uuid="a2", **common),
        _assistant_line(
            [{"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file_path": "/a.py"}}],
            uuid="a3",
            **common,
        ),
    ]


def _write_transcript(projects_dir, session_id, entries):
    path = projects_dir / "test-project" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def _ingest(tmp_path, session_id, entries):
    projects_dir = tmp_path / "projects"
    _write_transcript(projects_dir, session_id, entries)
    return ingest_session(
        session_id,
        trace_dir=tmp_path / "traces",
        db_path=tmp_path / "analyzer.db",
        projects_dir=projects_dir,
    )


# ---------------------------------------------------------------------------
# Bug 1 — one response counted once per content block
# ---------------------------------------------------------------------------


def test_split_response_is_one_api_call(tmp_path):
    """Three lines sharing a message.id are ONE API call, not three."""
    path = _write_transcript(tmp_path, "sess-split", _split_response_entries())
    _blocks, churn = parse_transcript_to_blocks(path)

    assert len(churn) == 1, f"expected 1 API call, parser produced {len(churn)}"


def test_split_response_counts_usage_once(tmp_path):
    """Usage repeated on each line must not be summed once per line."""
    path = _write_transcript(tmp_path, "sess-split", _split_response_entries())
    _blocks, churn = parse_transcript_to_blocks(path)

    assert sum(c["cache_read"] for c in churn) == _USAGE["cache_read_input_tokens"]
    assert sum(c["cache_creation"] for c in churn) == _USAGE["cache_creation_input_tokens"]
    assert sum(c["output"] for c in churn) == _USAGE["output_tokens"]
    assert sum(c["input"] for c in churn) == _USAGE["input_tokens"]


def test_split_response_does_not_inflate_session_totals(tmp_path):
    """End to end: the DB row must hold one response worth of tokens."""
    rec = _ingest(tmp_path, "sess-split", _split_response_entries())

    assert rec is not None
    assert rec.total_api_calls == 1
    assert rec.total_cache_read == _USAGE["cache_read_input_tokens"]
    assert rec.total_output_tokens == _USAGE["output_tokens"]


def test_split_response_does_not_inflate_peak_context(tmp_path):
    """Peak context is a single call's resident set, never a sum of lines."""
    rec = _ingest(tmp_path, "sess-split", _split_response_entries())
    resident = (
        _USAGE["input_tokens"]
        + _USAGE["cache_read_input_tokens"]
        + _USAGE["cache_creation_input_tokens"]
    )

    assert rec is not None
    assert rec.peak_context_tokens == resident


# ---------------------------------------------------------------------------
# Bug 2 — the model is never recorded, so rates are always Opus
# ---------------------------------------------------------------------------


def test_session_records_the_model_that_ran(tmp_path):
    """``message.model`` is in every transcript line; it must reach the DB.

    ``ingest_session`` reads ``churn[0]["model"]``, but the parser emits only
    turn/input/output/cache_read/cache_creation — the branch is dead and the
    column stays NULL, which is why every session is priced as Opus.
    """
    rec = _ingest(tmp_path, "sess-split", _split_response_entries())

    assert rec is not None
    assert rec.model == "claude-opus-4-6"


def test_sonnet_session_is_not_priced_as_opus(tmp_path):
    """A Sonnet session must be cheaper than the same tokens on Opus."""
    entries = _split_response_entries()
    for entry in entries:
        if entry["type"] == "assistant":
            entry["message"]["model"] = "claude-sonnet-4-6"
    sonnet = _ingest(tmp_path / "sonnet", "sess-sonnet", entries)
    opus = _ingest(tmp_path / "opus", "sess-opus", _split_response_entries())

    assert sonnet is not None and opus is not None
    assert sonnet.total_cost_usd < opus.total_cost_usd


# ---------------------------------------------------------------------------
# Bug 3 — cache reads are billed at 0.1x input, not 0.125x
# ---------------------------------------------------------------------------


def test_cache_read_rate_is_one_tenth_of_input():
    """Anthropic prices cache reads at 0.1x base input, for every model."""
    from context_tracker.analysis.config import PRICING

    for model, rates in PRICING.items():
        assert rates["cache_read"] == rates["input"] / 10, (
            f"{model}: cache_read {rates['cache_read']} is "
            f"{rates['cache_read'] / rates['input']:.3f}x input, expected 0.1x"
        )


def test_cache_write_rate_is_one_and_a_quarter_input():
    """The 5-minute cache write is 1.25x base input."""
    from context_tracker.analysis.config import PRICING

    for model, rates in PRICING.items():
        assert rates["cache_create"] == rates["input"] * 1.25, f"{model}: cache_create is off"


# ---------------------------------------------------------------------------
# Bug 4 — stats mixes priced and unpriced agents
# ---------------------------------------------------------------------------


def _stats_db(tmp_path):
    db = get_session_factory(get_engine(tmp_path / "stats.db"))()
    # A modest Claude session, priced by ingest.
    db.add(
        SessionRecord(
            session_id="cc-1",
            agent=AGENT_CLAUDE_CODE,
            started_at="2026-07-28T09:00:00Z",
            total_api_calls=70,
            peak_context_tokens=150_000,
            total_cache_read=4_000_000,
            total_cache_creation=250_000,
            total_input_tokens=200,
            total_output_tokens=30_000,
            total_cost_usd=14.44,
            source_mtime=0.0,
        )
    )
    # A much larger Codex session — stored at $0 by ingest_codex_session.
    db.add(
        SessionRecord(
            session_id="cx-1",
            agent=AGENT_CODEX,
            started_at="2026-07-29T09:00:00Z",
            model="gpt-5-codex",
            total_api_calls=260,
            peak_context_tokens=240_000,
            total_cache_read=30_000_000,
            total_cache_creation=0,
            total_input_tokens=1_100_000,
            total_output_tokens=400_000,
            total_cost_usd=0.0,
            source_mtime=0.0,
        )
    )
    db.commit()
    return db


def test_total_spend_is_not_silently_missing_codex(tmp_path):
    """Codex rows contribute $0, so "Total spend" is not total spend.

    Either price Codex sessions or exclude unpriced agents from the money
    lines — summing both makes the card wrong by however much Codex ran.
    """
    db = _stats_db(tmp_path)
    card = compute_stats(db)
    priced = [
        r for r in db.query(SessionRecord) if float(r.total_cost_usd or 0.0) > 0.0
    ]

    assert card.total_sessions == len(priced), (
        f"card sums {card.total_sessions} sessions but only {len(priced)} carry a price"
    )


def test_most_expensive_session_is_not_decided_by_missing_prices(tmp_path):
    """The top-spend session must not be picked from a $0-priced pool.

    ``cx-1`` burns 8x the tokens of ``cc-1`` but is stored at $0, so the
    ranking can only ever return the Claude session.
    """
    db = _stats_db(tmp_path)
    card = compute_stats(db)
    top_tokens = max(
        int(r.total_cache_read or 0) + int(r.total_input_tokens or 0)
        for r in db.query(SessionRecord)
    )
    top_row = next(
        r for r in db.query(SessionRecord)
        if int(r.total_cache_read or 0) + int(r.total_input_tokens or 0) == top_tokens
    )

    assert card.top_session_cost_usd >= float(top_row.total_cost_usd or 0.0)
    assert float(top_row.total_cost_usd or 0.0) > 0.0, (
        f"largest session ({top_row.agent}, {top_tokens:,} prompt tokens) is priced at $0"
    )
