# Handoff: cost accuracy and parser unification

You are picking up work on `context-analyzer` (package `context-tracker`), on the
branch `claude/context-analyzer-cost-bug-972502`. Read this before changing
anything that touches token counting, pricing, or transcript parsing. It exists
because the facts below are expensive to rediscover and were each learned by
shipping a wrong number first.

## What this tool is

It reconstructs Claude Code (and OpenAI Codex CLI) sessions from files on disk
and reports what they cost and where context was wasted. Surfaces: a FastAPI web
dashboard, an MCP server, a CLI, and hook-driven real-time nudges.

**The SQLite database is derived data.** The sources of truth are transcripts
under `~/.claude/projects/**/<session-id>.jsonl` and hook events under
`~/.claude/context-trace/`. Every DB row can be rebuilt from them. This is why
`context-tracker purge --reingest` is a recoverable operation and why you should
reach for it freely when debugging.

## The one domain fact that causes the most damage

**A transcript line is not an API call.** Claude Code writes a *separate line per
content block* of a single assistant response — one for `thinking`, one for
`text`, one for `tool_use` — and **every one repeats the full `message.usage`**.
All lines of a response share `message.id` and `requestId`.

Counting lines therefore inflates every token total by 1.6–3x. Measured on a real
transcript: 107 completed assistant lines, 64 actual API responses.

`src/context_tracker/transcript_reader.py` owns this. Anything that reads a
transcript must go through `iter_api_responses()` or
`coalesce_assistant_entries()`. Do not write another JSONL walk. A guard test
(`tests/test_cost_bug_repro.py::test_only_one_module_defines_the_completed_assistant_predicate`)
fails the build if any module outside `transcript_reader` grows its own
completed-assistant check.

Related shapes worth knowing:

- `cache_creation_input_tokens` is the **total** across both cache TTLs. The
  nested `usage.cache_creation` object splits it into
  `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`. A 1h write bills at
  **2x** base input, a 5m write at **1.25x**. Claude Code uses the 1h TTL
  heavily, so ignoring the split under-counts materially.
- Model strings arrive bare (`claude-opus-5`) or dated
  (`claude-sonnet-4-5-20250929`), sometimes with a `[1m]` context-window suffix.
  Always resolve through `analysis.config.normalize_model()`.
- `enter_turn` / `exit_turn` on blocks are **API-call indices**, not conversation
  turns. `exit_turn IS NULL` means the block survived to session end.

## Invariants this branch established

Each is enforced by a test that fails the build, because the repo's failure mode
is duplicated knowledge, and fixing copy N+1 has a poor track record here.

1. **One transcript reader.** `transcript_reader.py`. Five sites used to each
   carry their own copy of the parsing rules; the de-duplication fix reached one
   of them and the rest kept reporting inflated numbers for weeks.
2. **One pricing implementation.** `analysis/config.py` — `cost_of_call()` for a
   single call, `cost_breakdown()` for a session, `rates_for_model()` for rates.
   Rates are *derived* from a per-model base price
   (`cache_read = 0.1x input`, `cache_create = 1.25x`, `1h = 2.0x`) rather than
   written out per model, because a hand-copied table had cache reads at 0.125x
   on every row.
3. **No rate literals in the frontend.** The dashboard used to recompute cost in
   JavaScript against its own table and disagreed with the rest of the tool by
   3.3x. Cost is now computed server-side and rendered.
   `test_no_rate_literals_survive_in_the_frontend` guards `src/**/static/*.html`.
4. **Tests never touch real user data.** `tests/conftest.py` redirects `HOME` to
   a temp dir **at import time** — this ordering is load-bearing, because
   `DEFAULT_DB_PATH` is computed from `Path.home()` at import and bound into
   function defaults, so a monkeypatch fixture runs too late. Call sites also
   pass `db_path` explicitly, guarded by `tests/test_test_isolation.py`.

## Distinctions that are easy to get wrong

**Occupancy vs volume.** Context percentage is *peak resident context on a single
call*, over that model's window. It is not the sum of token counts across calls —
that is cumulative volume, grows without bound, and produced a literal "4622% of
context" reading that made `mcp_should_clear` recommend clearing every session.
Use `SessionRecord.peak_context_tokens` or a `max()` over calls, and
`analysis.config.context_window_for(model, default)`.

**Waste is cache traffic, not fresh input.** Tokens sitting in the context window
are re-sent on each subsequent call and served from the prompt cache, so they
bill at the cache-read rate — a tenth of input. Cost scales with *token-calls*
(tokens x calls re-sent), not tokens. See `analysis/report.py::_resident_cost`
and `_residency`.

**Unpriced is not zero.** Codex sessions store `total_cost_usd = 0.0` because
this repo has no OpenAI rate table and deliberately refuses to invent one. That
is correct — but `$0` must never flow into an aggregate as if it were measured.
`stats.UNPRICED_AGENTS` scopes the money lines; the dashboard renders "n/a"; the
`cost` command exits non-zero.

## Working habits that paid off here

- **Run it against a real transcript.** Every significant finding on this branch
  came from executing code against `~/.claude/projects/`, not from reading it.
  Reasoning about this codebase is unreliable; the data is not.
- **Drive the UI in a real browser.** Chromium is preinstalled at
  `/opt/pw-browsers/chromium`. Two dashboard bugs looked fine in the diff and
  only appeared on page load.
- **Write the guard test instead of the grep.** Two false "this is clean"
  conclusions on this branch came from searches that could not see the answer —
  `--include=*.py` missed a pricing table in an `.html` file, and a truncating
  `head -4` made a live module look dead. Tests do not truncate.
- **Suspect the tests.** They encoded the bugs: assertions on `1.875`, on stale
  Opus rates, and fixtures whose synthetic transcripts never split a response
  across lines. A green suite here meant "matches what we wrote."
- Re-ingest is idempotent on the transcript's mtime, so **nothing recomputes on
  its own** after a pricing change. Use `context-tracker reingest` (defaults to
  forced) or `purge --reingest`.

## Known gaps

Listed by size of the number they affect. None are started.

**Subagent spend is invisible.** Subagent transcripts are separate
`agent-*.jsonl` files, so their tokens never enter the parent's churn, and
`SubagentRecord` has no cost column. Task-heavy sessions under-report, possibly
by a lot. This is blocked on a product decision, not on implementation: rolling
subagent cost into the parent's `total_cost_usd` changes the meaning of every
existing session-cost figure. Ask before building.

**The nudge context window is a fixed 1M.** `nudge_config.NUDGE_DEFAULTS` sets
`context_window` as a constant rather than resolving per model, so a Haiku
session (200K) under-warns by 5x. `context_window_for()` already exists; it is
config plumbing, and the value is user-overridable, which is why it was left.

**Bulk re-ingest only discovers UUID-named transcripts.**
`storage.list_sessions` filters the projects directory on `_UUID_RE`. Real
Claude Code sessions are UUIDs so this is fine in practice, but a renamed or
hand-written transcript is invisible to `reingest` with no arguments while
`reingest <id>` on the same file works. An asymmetry, not obviously a bug.

**No batch-API discount modelling.** Batch requests bill at 50%. Nothing
represents this. Almost certainly irrelevant for Claude Code transcripts; noted
for completeness.

**No `CLAUDE.md`.** This file is a stopgap. The transcript-line-per-block fact in
particular deserves to be somewhere an agent reads by default.

**Test fixtures beyond the ones already fixed.** `db_path` isolation is enforced
for `create_app()`. Other entry points that default to `Path.home()` paths are
not individually guarded — the conftest `HOME` redirect covers them, but the
guard test only understands `create_app`.

## Orientation

```
src/context_tracker/
  transcript_reader.py     the ONE transcript parser primitive
  analysis/config.py       the ONE pricing + context-window table
  ingest.py                transcripts -> SQLite (Claude Code and Codex paths)
  data_cli.py              cost / reingest / purge
  server.py                MCP server + CLI dispatch
  dashboard.py             FastAPI app
  static/dashboard-v3.html the dashboard (renders server-computed cost only)
  ccscope/                 blocks + churn model, `ccscope build` static output
tests/test_cost_bug_repro.py   regression tests for everything above
docs/HANDOFF-cost-accuracy.md  this file
```

```bash
make install-dev                      # editable install into .venv
.venv/bin/python -m pytest tests/ -q  # 760 passing
.venv/bin/python -m ruff check src/ tests/
uv tool install --force .             # install the CLI globally from the checkout
```

The branch is green and pushed. No PR has been opened.
