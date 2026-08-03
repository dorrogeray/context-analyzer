"""Shared test fixtures.

The HOME redirection below runs at import time, before any test module
imports context_tracker, and that ordering is load-bearing — see the comment
on _TEST_HOME.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Redirect HOME before context_tracker is imported anywhere.
#
# DEFAULT_DB_PATH, DEFAULT_TRACE_DIR and friends are computed from Path.home()
# at import time and then bound into function defaults (create_app(...,
# db_path: Path = DEFAULT_DB_PATH)). Monkeypatching the module attribute in a
# fixture is therefore too late: the default argument already holds the old
# value. Moving HOME first is the only interception point that covers every
# call site, including ones added later that forget to pass a path.
#
# Without this, a test that omits db_path writes to the developer's real
# ~/.context-analyzer/analyzer.db — the suite used to leave a fixture session
# named "test-session-123" in it, which then showed up in `context-tracker
# stats` and the dashboard's session list.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="context-analyzer-tests-"))
os.environ["HOME"] = str(_TEST_HOME)
os.environ.pop("XDG_CONFIG_HOME", None)
os.environ.pop("XDG_DATA_HOME", None)


@pytest.fixture
def test_home() -> Path:
    """The redirected HOME, for tests that want to assert against it."""
    return _TEST_HOME
