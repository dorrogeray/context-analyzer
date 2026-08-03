"""Regression tests for the inflated session-cost bug.

Sessions whose real spend was in the hundreds of dollars were reported at
several thousand. Four independent defects compounded:

1. Claude Code writes ONE transcript line per content block of a single API
   response (a ``thinking`` line, a ``text`` line, a ``tool_use`` line), each
   repeating the full ``message.usage``. The parser treated every line as its
   own API call, so usage was summed 2-3x.
2. ``ingest.ingest_session`` looked for ``churn[0]["model"]``, which the
   parser never emitted — so the model was never recorded and every session
   was priced at fixed Opus rates regardless of which model actually ran.
3. The cache-read rate was 0.125x input; Anthropic charges 0.1x.
4. ``stats.compute_stats`` aggregated across agents, but Codex sessions are
   deliberately stored with ``total_cost_usd = 0.0``.

The pre-existing fixtures missed defect 1 because their synthetic assistant
entries carry no ``message.id`` / ``requestId`` and never split a response
across lines — the shape that occurs in every real transcript. The builders
here model that shape deliberately.
"""

from __future__ import annotations

import json

import pytest

from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from context_tracker.db import (
    AGENT_CLAUDE_CODE,
    AGENT_CODEX,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.ingest import ingest_session
from context_tracker.stats import UNPRICED_AGENTS, compute_stats, render_card

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


def test_spend_covers_only_the_sessions_that_carry_a_price(tmp_path):
    """Codex rows are stored at $0, so they must not dilute the spend figures.

    The repo declines to invent OpenAI rates, which is the right call — so
    the fix is to scope the money lines to priced agents and report the
    remainder separately, not to sum a priced and an unpriced population.
    """
    db = _stats_db(tmp_path)
    card = compute_stats(db)

    assert card.total_sessions == 2  # both are still analyzed
    assert card.priced_sessions == 1
    assert card.unpriced_sessions == 1
    assert card.total_spend_usd == pytest.approx(14.44)


def test_unpriced_sessions_are_disclosed_on_the_card(tmp_path):
    """A spend number covering a subset has to say so."""
    db = _stats_db(tmp_path)
    rendered = render_card(compute_stats(db))

    assert "1 priced session" in rendered
    assert "no rate table" in rendered


def test_most_expensive_session_is_not_decided_by_missing_prices(tmp_path):
    """The top-spend session must be picked from the priced pool only.

    ``cx-1`` burns 8x the tokens of ``cc-1`` but is unpriced, so ranking
    across both would be decided by which agent happens to have rates.
    """
    db = _stats_db(tmp_path)
    card = compute_stats(db)
    priced = [r for r in db.query(SessionRecord) if str(r.agent) not in UNPRICED_AGENTS]

    assert card.top_session_cost_usd == pytest.approx(
        max(float(r.total_cost_usd or 0.0) for r in priced)
    )
    assert card.top_session_peak_context == 150_000  # cc-1, not the Codex row
