"""CLI commands over the local analyzer database: cost, reingest, purge.

The database is *derived* data. Transcripts under ~/.claude/projects and hook
events under the trace directory are the source of truth, and everything here
can be rebuilt from them — which is what makes purge a safe operation rather
than a destructive one.

Claude Code's own `/cost` covers the current session only. `cost` here reads
the transcripts, so it works for any session id, and prices through
:mod:`context_tracker.analysis.config` like every other surface.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from context_tracker.analysis.config import cost_breakdown, normalize_model
from context_tracker.db import (
    AGENT_CODEX,
    DEFAULT_DB_PATH,
    SessionRecord,
    get_engine,
    get_session_factory,
)
from context_tracker.storage import DEFAULT_TRACE_DIR

_TIERS = ("input", "output", "cache_read", "cache_create")


def _resolve_session(session_id: str, db_path: Path) -> SessionRecord | None:
    """Look a session up by full id or unambiguous prefix."""
    factory = get_session_factory(get_engine(db_path))
    with factory() as db:
        rec = db.get(SessionRecord, session_id)
        if rec is not None:
            return rec
        matches = [r for r in db.query(SessionRecord) if str(r.session_id).startswith(session_id)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            ids = ", ".join(str(m.session_id)[:12] for m in matches[:5])
            print(f"error: '{session_id}' matches {len(matches)} sessions ({ids}...)", file=sys.stderr)
            raise SystemExit(2)
    return None


def run_cost(
    session_id: str,
    as_json: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    projects_dir: Path | None = None,
) -> int:
    """Print the cost breakdown for one session."""
    from context_tracker.ccscope.reconcile import reconcile

    rec = _resolve_session(session_id, db_path)
    resolved = str(rec.session_id) if rec is not None else session_id

    if rec is not None and str(rec.agent or "") == AGENT_CODEX:
        print(f"{resolved}: no rate table for agent '{rec.agent}' — not priced", file=sys.stderr)
        return 1

    try:
        _blocks, churn, _subagents = reconcile(resolved, projects_dir=projects_dir, trace_dir=trace_dir)
    except FileNotFoundError:
        print(f"error: no transcript found for session '{session_id}'", file=sys.stderr)
        return 1

    if not churn:
        print(f"error: session '{resolved}' has no API calls", file=sys.stderr)
        return 1

    breakdown = cost_breakdown(churn)
    models = sorted({normalize_model(c.get("model")) for c in churn})
    stored = float(rec.total_cost_usd) if rec is not None and rec.total_cost_usd is not None else None

    if as_json:
        print(
            json.dumps(
                {
                    "session_id": resolved,
                    "models": models,
                    "api_calls": len(churn),
                    "cost": breakdown,
                    "stored_cost_usd": stored,
                },
                indent=2,
            )
        )
        return 0

    total = breakdown["total"]
    print(f"Session {resolved}")
    print(f"  model{'s' if len(models) > 1 else ''}: {', '.join(models)}")
    print(f"  API calls: {len(churn):,}")
    print()
    width = max(len(t) for t in _TIERS)
    for tier in sorted(_TIERS, key=lambda t: breakdown[t], reverse=True):
        amount = breakdown[tier]
        share = (amount / total * 100) if total else 0.0
        print(f"  {tier.replace('_', ' '):<{width}}  ${amount:>10,.4f}  {share:>5.1f}%")
    print(f"  {'total':<{width}}  ${total:>10,.4f}")

    if stored is not None and abs(stored - total) > 0.01:
        print()
        print(
            f"  note: the database has ${stored:,.2f} for this session. It was ingested\n"
            f"        before the current pricing; run 'context-tracker reingest {resolved[:8]}'."
        )
    return 0


def run_reingest(
    session_id: str | None = None,
    force: bool = True,
    db_path: Path = DEFAULT_DB_PATH,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    projects_dir: Path | None = None,
) -> int:
    """Re-ingest one session, or every session when none is named.

    Defaults to force because the reason to run this is a code change, and
    ingest is otherwise idempotent on the transcript's mtime — an unforced
    run would leave every stale row exactly as it was.
    """
    from context_tracker.ingest import ingest_all, ingest_session

    if session_id:
        rec = _resolve_session(session_id, db_path)
        resolved = str(rec.session_id) if rec is not None else session_id
        result = ingest_session(
            resolved,
            trace_dir=trace_dir,
            db_path=db_path,
            force=force,
            projects_dir=projects_dir,
        )
        if result is None:
            print(f"error: no transcript found for session '{session_id}'", file=sys.stderr)
            return 1
        print(f"Re-ingested {resolved} — ${float(result.total_cost_usd or 0.0):,.2f}")
        return 0

    ingested = ingest_all(trace_dir=trace_dir, db_path=db_path, force=force, projects_dir=projects_dir)
    print(f"Re-ingested {len(ingested):,} session(s)")
    return 0


def _db_files(db_path: Path) -> list[Path]:
    """The database and its SQLite sidecars, whichever exist."""
    candidates = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
    return [p for p in candidates if p.exists()]


def _describe(db_path: Path) -> tuple[int, int]:
    """(session count, bytes on disk) for an existing database."""
    files = _db_files(db_path)
    size = sum(p.stat().st_size for p in files)
    if not db_path.exists():
        return 0, size
    try:
        factory = get_session_factory(get_engine(db_path))
        with factory() as db:
            return db.query(SessionRecord).count(), size
    except Exception:
        return 0, size


def run_purge(
    yes: bool = False,
    reingest: bool = False,
    include_scratch: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    projects_dir: Path | None = None,
    confirm: Callable[[str], str] | None = None,
) -> int:
    """Delete the analyzer database and recreate it empty.

    Safe because the database is derived: every row can be rebuilt from the
    transcripts with `reingest`. Still asks first, because rebuilding a large
    corpus is slow and the caller may not realise what they are dropping.
    """
    from context_tracker.headroom_audit import KEEP_DB_DIR

    sessions, size = _describe(db_path)
    if not db_path.exists():
        print(f"No database at {db_path} — nothing to purge.")
    else:
        print(f"About to delete {db_path}")
        print(f"  {sessions:,} session(s), {size / 1024:,.0f} KB")
    if include_scratch and KEEP_DB_DIR.exists():
        print(f"  and the audit scratch directory {KEEP_DB_DIR}")
    print("Transcripts and hook events are untouched; 'context-tracker reingest' rebuilds this.")

    if not yes:
        ask = confirm if confirm is not None else input
        if not sys.stdin.isatty() and confirm is None:
            print("error: refusing to purge without a terminal — pass --yes to confirm.", file=sys.stderr)
            return 2
        try:
            answer = ask("Type 'purge' to confirm: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return 1
        if answer.strip() != "purge":
            print("Aborted.", file=sys.stderr)
            return 1

    for path in _db_files(db_path):
        path.unlink()
    if include_scratch and KEEP_DB_DIR.exists():
        shutil.rmtree(KEEP_DB_DIR)

    # Recreate immediately so the next reader finds a valid empty schema
    # rather than racing to create one.
    get_engine(db_path)
    print(f"Recreated empty database at {db_path}")

    if reingest:
        return run_reingest(None, force=True, db_path=db_path, trace_dir=trace_dir, projects_dir=projects_dir)
    return 0
