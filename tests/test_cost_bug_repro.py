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
import sqlite3
from pathlib import Path

import pytest

import context_tracker

from context_tracker.analysis.config import (
    PRICING,
    cost_breakdown,
    context_window_for,
    cost_of_call,
    normalize_model,
)
from context_tracker.ccscope.parse_transcript import parse_transcript_to_blocks
from context_tracker.analysis.report import _residency, _resident_cost
from context_tracker.db import (
    AGENT_CLAUDE_CODE,
    AGENT_CODEX,
    BlockRecord,
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


# ---------------------------------------------------------------------------
# Bug 5 — the 1h cache TTL bills at 2x, not 1.25x
# ---------------------------------------------------------------------------


def _usage_1h(total_creation, ttl_1h):
    """Usage where part of cache creation used the 1-hour TTL."""
    return {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": total_creation,
        "output_tokens": 10,
        "cache_creation": {
            "ephemeral_5m_input_tokens": total_creation - ttl_1h,
            "ephemeral_1h_input_tokens": ttl_1h,
        },
    }


def _one_response(usage, session="s"):
    return [
        _user_entry("go"),
        _assistant_line(
            [{"type": "text", "text": "ok"}],
            usage=usage,
            uuid="a1",
            message_id=f"msg_{session}",
            request_id=f"req_{session}",
        ),
    ]


def test_one_hour_cache_writes_cost_twice_base_input():
    """A 1h write is 2x base input; a 5m write is 1.25x."""
    base = PRICING["claude-opus-5"]["input"]

    assert cost_of_call("claude-opus-5", cache_creation=1_000_000) == pytest.approx(base * 1.25)
    assert cost_of_call(
        "claude-opus-5", cache_creation=1_000_000, cache_creation_1h=1_000_000
    ) == pytest.approx(base * 2.0)


def test_mixed_ttl_prices_each_portion_at_its_own_rate():
    """cache_creation is the total; the 1h portion is carved out of it."""
    cost = cost_of_call(
        "claude-opus-5", cache_creation=1_000_000, cache_creation_1h=400_000
    )
    base = PRICING["claude-opus-5"]["input"]
    expected = (600_000 * base * 1.25 + 400_000 * base * 2.0) / 1_000_000

    assert cost == pytest.approx(expected)


def test_one_hour_portion_never_exceeds_the_total():
    """A malformed split must not be able to inflate the cost."""
    sane = cost_of_call("claude-opus-5", cache_creation=1000, cache_creation_1h=1000)

    assert cost_of_call("claude-opus-5", cache_creation=1000, cache_creation_1h=9999) == sane
    assert cost_of_call("claude-opus-5", cache_creation=1000, cache_creation_1h=-5) == cost_of_call(
        "claude-opus-5", cache_creation=1000
    )


def test_ingest_records_and_prices_the_ttl_split(tmp_path):
    """End to end: the 1h portion reaches the DB and the recorded cost."""
    rec = _ingest(tmp_path, "sess-ttl", _one_response(_usage_1h(100_000, 100_000)))
    five_min = _ingest(
        tmp_path / "b", "sess-5m", _one_response(_usage_1h(100_000, 0), session="b")
    )

    assert rec is not None and five_min is not None
    assert rec.total_cache_creation == 100_000
    assert rec.total_cache_creation_1h == 100_000
    assert five_min.total_cache_creation_1h == 0
    # 2x vs 1.25x on the cache-creation term, which is all this session has
    # beyond a trivial output charge.
    assert rec.total_cost_usd > five_min.total_cost_usd


def test_transcripts_without_the_ttl_breakdown_still_ingest(tmp_path):
    """Older transcripts omit usage.cache_creation — treat it all as 5m."""
    usage = dict(_USAGE)
    usage.pop("cache_creation", None)
    rec = _ingest(tmp_path, "sess-old", _one_response(usage))

    assert rec is not None
    assert rec.total_cache_creation_1h == 0


def test_existing_databases_gain_the_ttl_columns(tmp_path):
    """A DB written before these columns existed must migrate, not crash."""
    db_path = tmp_path / "old.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, agent TEXT, "
        "total_cache_creation INTEGER, total_cost_usd REAL, source_mtime REAL)"
    )
    con.execute("CREATE TABLE api_calls (id INTEGER PRIMARY KEY, session_id TEXT, cache_creation INTEGER)")
    con.commit()
    con.close()

    get_engine(db_path)  # runs _migrate_schema

    con = sqlite3.connect(db_path)
    sess_cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
    call_cols = {r[1] for r in con.execute("PRAGMA table_info(api_calls)")}
    con.close()

    assert "total_cache_creation_1h" in sess_cols
    assert "cache_creation_1h" in call_cols


# ---------------------------------------------------------------------------
# Bug 6 — dated model IDs missed the table and fell back to Opus rates
# ---------------------------------------------------------------------------


def test_dated_model_ids_resolve_to_their_alias():
    """Transcripts report dated snapshots; those must not miss the table."""
    assert normalize_model("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert normalize_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"
    assert normalize_model("claude-opus-4-1-20250805") == "claude-opus-4-1"


def test_context_window_suffix_survives_normalization():
    """The [1m] variant is a distinct key and must not be stripped away."""
    assert normalize_model("claude-opus-4-6[1m]") == "claude-opus-4-6[1m]"
    assert normalize_model("claude-sonnet-4-5-20250929[1m]") == "claude-sonnet-4-5[1m]"


def test_unknown_models_still_fall_back():
    """Genuinely unknown models fall back rather than raising."""
    assert normalize_model("<synthetic>") == "_default"
    assert normalize_model("") == "_default"
    assert normalize_model(None) == "_default"


def test_dated_haiku_is_not_billed_as_opus():
    """The bug: a dated Haiku ID was priced at the Opus default (5x)."""
    dated = cost_of_call("claude-haiku-4-5-20251001", input_tokens=1_000_000)
    alias = cost_of_call("claude-haiku-4-5", input_tokens=1_000_000)

    assert dated == alias == pytest.approx(1.0)
    assert dated < cost_of_call("_default", input_tokens=1_000_000)


def test_session_with_a_dated_model_is_priced_correctly(tmp_path):
    """End to end through ingest, not just the lookup helper."""
    entries = _split_response_entries()
    for entry in entries:
        if entry["type"] == "assistant":
            entry["message"]["model"] = "claude-haiku-4-5-20251001"
    rec = _ingest(tmp_path, "sess-dated", entries)
    expected = cost_of_call(
        "claude-haiku-4-5",
        input_tokens=_USAGE["input_tokens"],
        output_tokens=_USAGE["output_tokens"],
        cache_read=_USAGE["cache_read_input_tokens"],
        cache_creation=_USAGE["cache_creation_input_tokens"],
    )

    assert rec is not None
    assert rec.total_cost_usd == pytest.approx(round(expected, 4))


# ---------------------------------------------------------------------------
# Bug 7 — waste priced at the input rate, ignoring residency
# ---------------------------------------------------------------------------


def test_residency_counts_re_sends_not_calls_present():
    """enter 5 / exit 10 means five re-transmissions."""
    block = BlockRecord(session_id="s", block_id="b", block_type="tool_result", enter_turn=5, exit_turn=10)

    assert _residency(block, end_turn=99) == 5


def test_residency_of_a_block_that_never_left_runs_to_session_end():
    """A NULL exit_turn means the block survived the whole session."""
    block = BlockRecord(session_id="s", block_id="b", block_type="tool_result", enter_turn=2, exit_turn=None)

    assert _residency(block, end_turn=12) == 10


def test_residency_is_zero_when_a_block_enters_and_leaves_on_one_call():
    """Never re-sent means no carry cost — zero, not a floor of one."""
    block = BlockRecord(session_id="s", block_id="b", block_type="tool_result", enter_turn=7, exit_turn=7)

    assert _residency(block, end_turn=99) == 0


def test_waste_is_priced_as_cache_traffic_not_fresh_input():
    """The old model charged the input rate; carrying context is 0.1x that."""
    rates = PRICING["claude-opus-5"]
    # One 1M-token block, re-sent 3 times: one cache write plus three reads.
    cost = _resident_cost(1_000_000, 3_000_000, "claude-opus-5")
    expected = (1_000_000 * rates["cache_create"] + 3_000_000 * rates["cache_read"]) / 1_000_000

    assert cost == pytest.approx(expected)
    # And well under what the input rate would have charged for the tokens.
    assert cost < 1_000_000 * rates["input"] / 1_000_000 * 2


def test_longer_residency_costs_more_for_the_same_tokens():
    """The whole point: cost scales with how long waste stays resident."""
    brief = _resident_cost(100_000, 100_000, "claude-opus-5")
    lingering = _resident_cost(100_000, 5_000_000, "claude-opus-5")

    assert lingering > brief


# ---------------------------------------------------------------------------
# Bug 8 — stale context-window table skewed utilization
# ---------------------------------------------------------------------------


def test_current_models_have_their_real_context_window():
    """Current models ship a 1M window; the table said 200K."""
    assert context_window_for("claude-opus-5", 200_000) == 1_000_000
    assert context_window_for("claude-sonnet-5", 200_000) == 1_000_000
    assert context_window_for("claude-opus-4-6", 200_000) == 1_000_000


def test_haiku_keeps_its_smaller_window():
    """Not every current model is 1M — Haiku 4.5 is 200K."""
    assert context_window_for("claude-haiku-4-5", 1_000_000) == 200_000


def test_context_window_lookup_normalizes_dated_ids():
    """Same normalization as pricing, so dated IDs don't take the fallback."""
    assert context_window_for("claude-opus-5-20260101", 12345) == 1_000_000
    assert context_window_for("claude-opus-4-6[1m]", 12345) == 1_000_000


def test_unknown_models_take_the_supplied_default():
    assert context_window_for("who-knows", 200_000) == 200_000
    assert context_window_for(None, 200_000) == 200_000


def test_utilization_is_not_overstated_fivefold(tmp_path):
    """A 500K peak on a 1M model is 50% utilized, not 250%."""
    window = context_window_for("claude-opus-5", 200_000)

    assert 500_000 / window == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Bug 9 — force=True could not re-ingest an existing session
# ---------------------------------------------------------------------------


def test_force_reingest_replaces_an_existing_row(tmp_path):
    """force=True must recompute, not collide with the row already there.

    The delete lived inside `if existing and not force`, so a forced
    re-ingest skipped it and the insert hit the primary key. This is the
    only way to recost sessions after a pricing fix, and it raised.
    """
    entries = _split_response_entries()
    first = _ingest(tmp_path, "sess-force", entries)
    assert first is not None

    db_path = tmp_path / "analyzer.db"
    con = sqlite3.connect(db_path)
    con.execute("UPDATE sessions SET total_cost_usd = 999.99")
    con.commit()
    con.close()

    again = ingest_session(
        "sess-force",
        trace_dir=tmp_path / "traces",
        db_path=db_path,
        projects_dir=tmp_path / "projects",
        force=True,
    )

    assert again is not None
    assert again.total_cost_usd != pytest.approx(999.99)
    assert again.total_cost_usd == pytest.approx(first.total_cost_usd)


def test_unforced_reingest_leaves_an_up_to_date_row_alone(tmp_path):
    """The idempotence that force overrides still holds without it.

    Re-ingests against the *same* transcript file — writing it again would
    bump source_mtime and legitimately trigger a re-ingest.
    """
    _ingest(tmp_path, "sess-idem", _split_response_entries())
    db_path = tmp_path / "analyzer.db"
    con = sqlite3.connect(db_path)
    con.execute("UPDATE sessions SET total_cost_usd = 999.99")
    con.commit()
    con.close()

    again = ingest_session(
        "sess-idem",
        trace_dir=tmp_path / "traces",
        db_path=db_path,
        projects_dir=tmp_path / "projects",
    )

    assert again is not None
    assert again.total_cost_usd == pytest.approx(999.99)


# ---------------------------------------------------------------------------
# Bug 10 — the dashboard priced sessions in JS against its own rate table
# ---------------------------------------------------------------------------


def test_no_rate_literals_survive_in_the_frontend():
    """The dashboard must not carry a second copy of the pricing table.

    It had `const PRICING = {input: 15/1e6, ...}` and computed the scorecard
    client-side, so the homepage and the session dropdown disagreed by 3.3x.
    """
    static_dir = Path(context_tracker.__file__).parent / "static"
    offenders = []
    for page in static_dir.glob("*.html"):
        text = page.read_text(encoding="utf-8")
        for literal in ("1.875", "18.75", "15 / 1e6", "75 / 1e6", "PRICING"):
            if literal in text:
                offenders.append(f"{page.name}: {literal}")

    assert not offenders, f"pricing moved back into the frontend: {offenders}"


def test_breakdown_components_sum_to_the_total():
    calls = [
        {"model": "claude-opus-5", "input": 100, "output": 1000, "cache_read": 500_000},
        {"model": "claude-sonnet-5", "cache_creation": 10_000, "cache_creation_1h": 4_000},
    ]
    b = cost_breakdown(calls)

    assert b["total"] == pytest.approx(b["input"] + b["output"] + b["cache_read"] + b["cache_create"])


def test_breakdown_prices_each_call_at_its_own_model():
    """A mixed-model session is not collapsed onto one rate."""
    opus = cost_breakdown([{"model": "claude-opus-5", "input": 1_000_000}])
    haiku = cost_breakdown([{"model": "claude-haiku-4-5", "input": 1_000_000}])
    both = cost_breakdown(
        [
            {"model": "claude-opus-5", "input": 1_000_000},
            {"model": "claude-haiku-4-5", "input": 1_000_000},
        ]
    )

    assert both["total"] == pytest.approx(opus["total"] + haiku["total"])
    assert opus["total"] > haiku["total"]


def test_dashboard_cost_matches_the_stored_session_cost(tmp_path):
    """The scorecard figure and the dropdown figure must be the same number."""
    rec = _ingest(tmp_path, "sess-ui", _split_response_entries())
    _blocks, churn = parse_transcript_to_blocks(
        tmp_path / "projects" / "test-project" / "sess-ui.jsonl"
    )

    assert rec is not None
    assert cost_breakdown(churn)["total"] == pytest.approx(rec.total_cost_usd, abs=1e-4)
