"""The test suite must never touch the developer's real data.

context_tracker resolves its data paths from Path.home() at import time, so
a test that omits db_path silently writes to ~/.context-analyzer/analyzer.db.
That happened: the suite left a fixture session named "test-session-123" in
the real database, where it showed up in `context-tracker stats` and the
dashboard's session list.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from context_tracker.db import DEFAULT_DB_DIR, DEFAULT_DB_PATH
from context_tracker.storage import DEFAULT_TRACE_DIR

TESTS_DIR = Path(__file__).parent


def test_default_paths_resolve_under_the_redirected_home(test_home):
    """conftest moves HOME before context_tracker is imported."""
    assert str(DEFAULT_DB_PATH).startswith(str(test_home))
    assert str(DEFAULT_DB_DIR).startswith(str(test_home))
    assert str(DEFAULT_TRACE_DIR).startswith(str(test_home))


def test_the_redirected_home_is_a_temporary_directory(test_home):
    assert test_home.exists()
    assert "context-analyzer-tests-" in test_home.name


_SCANNED = sorted(p for p in TESTS_DIR.glob("test_*.py") if p.name != Path(__file__).name)


@pytest.mark.parametrize("path", _SCANNED, ids=lambda p: p.name)
def test_every_create_app_call_passes_a_db_path(path):
    """Belt to the conftest's braces — keep the intent visible at call sites."""
    source = path.read_text(encoding="utf-8")
    offenders = []

    for match in re.finditer(r"create_app\(", source):
        index = match.end()
        depth = 1
        while depth and index < len(source):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
            index += 1
        if "db_path" not in source[match.start() : index]:
            offenders.append(source[: match.start()].count("\n") + 1)

    assert not offenders, f"{path.name}: create_app() without db_path at line(s) {offenders}"
