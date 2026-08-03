"""Configurable thresholds for staleness detection and health scoring.

All defaults are labeled as uncalibrated — to be tuned against real sessions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StalenessConfig:
    decay_window: int = 10
    resource_window: int = 10
    reference_scan_window: int = 15
    task_boundary_time_gap: int = 10  # Minutes
    task_boundary_overlap: float = 0.2
    min_prompt_length_for_boundary: int = 20


@dataclass
class HealthConfig:
    model_context_window: int = 200_000
    weight_dead_weight: float = 0.35
    weight_utilization: float = 0.25
    weight_cache: float = 0.15
    weight_output_inflation: float = 0.10
    weight_repeated: float = 0.10
    weight_errors: float = 0.05
    threshold_healthy: float = 0.3
    threshold_degrading: float = 0.5
    threshold_recommend_new: float = 0.7
    repeated_read_warning: int = 3
    repeated_read_critical: int = 5
    repeated_read_rolling_window: int = 20
    edit_churn_window: int = 5
    error_spike_multiplier: float = 2.0
    output_inflation_multiplier: float = 1.5
    cache_trend_window: int = 10


# Context window (input tokens) by model. Current models ship a 1M window as
# the default rather than an opt-in variant, so the "[1m]" keys below are
# aliases kept for transcripts that still report the suffixed form.
MODEL_CONTEXT_WINDOWS = {
    "claude-fable-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5": 200_000,
    # Older models, 200K unless the 1M variant was requested explicitly.
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-0": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-opus-4-0": 200_000,
}

# A "[1m]" suffix always means the 1M-context variant.
MODEL_CONTEXT_WINDOWS.update({f"{name}[1m]": 1_000_000 for name in list(MODEL_CONTEXT_WINDOWS)})


def context_window_for(model: str | None, default: int) -> int:
    """Context window for a reported model string, or ``default`` if unknown.

    Goes through the same normalization as pricing, so a dated snapshot
    (``claude-sonnet-4-5-20250929``) resolves instead of silently taking the
    fallback.
    """
    name = (model or "").strip()
    if name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[name]
    normalized = normalize_model(name)
    return MODEL_CONTEXT_WINDOWS.get(normalized, default)


# Cache rates are fixed multiples of a model's base input price, so they are
# derived rather than written out per model — the previous hand-copied table
# had cache_read at 0.125x input on every entry instead of 0.1x.
CACHE_READ_MULTIPLIER = 0.1
CACHE_CREATE_MULTIPLIER = 1.25  # 5-minute TTL
CACHE_CREATE_1H_MULTIPLIER = 2.0  # 1-hour TTL

# Base input/output price per million tokens, by model.
MODEL_BASE_RATES = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Older models still present in archived transcripts.
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-0": (3.0, 15.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4-0": (15.0, 75.0),
}

# The 1M-context variants carry no long-context premium.
MODEL_BASE_RATES.update({f"{name}[1m]": rates for name, rates in list(MODEL_BASE_RATES.items())})

# Unknown models are priced as current Opus — the Claude Code default.
MODEL_BASE_RATES["_default"] = MODEL_BASE_RATES["claude-opus-5"]


def _rates(input_price: float, output_price: float) -> dict[str, float]:
    # Rounded because input_price * 0.1 is not bit-identical to input_price / 10
    # in binary floating point, and callers do compare these against the ratio.
    return {
        "input": input_price,
        "output": output_price,
        "cache_read": round(input_price * CACHE_READ_MULTIPLIER, 6),
        "cache_create": round(input_price * CACHE_CREATE_MULTIPLIER, 6),
        "cache_create_1h": round(input_price * CACHE_CREATE_1H_MULTIPLIER, 6),
    }


# Pricing per million tokens
PRICING = {name: _rates(*rates) for name, rates in MODEL_BASE_RATES.items()}


_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def normalize_model(model: str | None) -> str:
    """Map a reported model string onto a pricing-table key.

    Transcripts report either a bare alias (``claude-opus-5``) or a dated
    snapshot (``claude-sonnet-4-5-20250929``); the context-window variant
    adds a ``[1m]`` suffix. Without normalization every dated ID missed the
    table and silently fell back to ``_default``, billing a Haiku session at
    Opus rates.

    Returns the matching key, or ``_default`` when the model is genuinely
    unknown (a retired model we have no rates for, or an empty string).
    """
    name = (model or "").strip()
    if not name:
        return "_default"
    if name in PRICING:
        return name

    suffix = ""
    if name.endswith("]") and "[" in name:
        base, _, rest = name.rpartition("[")
        name, suffix = base, f"[{rest}"
    else:
        base = name

    stripped = _DATE_SUFFIX_RE.sub("", name)
    for candidate in (f"{stripped}{suffix}", stripped, f"{base}{suffix}", base):
        if candidate in PRICING:
            return candidate
    return "_default"


def rates_for_model(model: str | None) -> dict[str, float]:
    """Per-Mtok rates for a reported model string."""
    return PRICING[normalize_model(model)]


def cost_of_call(
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
    cache_creation_1h: int = 0,
) -> float:
    """Cost in USD of one API call, priced at the given model's rates.

    Unknown or missing models fall back to ``_default``.

    ``cache_creation`` is the total of both cache TTLs, matching what the
    API reports as ``cache_creation_input_tokens``. ``cache_creation_1h`` is
    the portion of it written with the 1-hour TTL, which bills at 2x base
    input instead of 1.25x; the remainder is priced as a 5-minute write.
    Callers that don't know the split pass the total and get 5-minute
    pricing, which is what the API charges when no 1h TTL is requested.
    """
    rates = rates_for_model(model)
    ttl_1h = max(0, min(int(cache_creation_1h), int(cache_creation)))
    ttl_5m = int(cache_creation) - ttl_1h
    return (
        int(input_tokens) * rates["input"]
        + int(output_tokens) * rates["output"]
        + int(cache_read) * rates["cache_read"]
        + ttl_5m * rates["cache_create"]
        + ttl_1h * rates["cache_create_1h"]
    ) / 1_000_000


def cost_breakdown(calls: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Total cost of a sequence of API calls, split by billing component.

    Each call is priced at its own model. Returns ``total`` plus a per-tier
    breakdown, so callers that want to show what drove the spend do not have
    to re-derive it — and therefore do not need a second copy of the rates.
    The dashboard used to compute exactly this in JavaScript against its own
    hardcoded table, which drifted from the server the moment rates changed.
    """
    out = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_create": 0.0}
    for call in calls:
        model = call.get("model")
        out["input"] += cost_of_call(model, input_tokens=call.get("input", 0) or 0)
        out["output"] += cost_of_call(model, output_tokens=call.get("output", 0) or 0)
        out["cache_read"] += cost_of_call(model, cache_read=call.get("cache_read", 0) or 0)
        out["cache_create"] += cost_of_call(
            model,
            cache_creation=call.get("cache_creation", 0) or 0,
            cache_creation_1h=call.get("cache_creation_1h", 0) or 0,
        )
    out["total"] = sum(out.values())
    return out


def load_config(
    config_path: Path | None = None,
) -> tuple[StalenessConfig, HealthConfig]:
    """Load config from JSON file, falling back to defaults."""
    staleness = StalenessConfig()
    health = HealthConfig()

    if config_path is None:
        config_path = Path.home() / ".claude" / "context-analyzer.json"

    if not config_path.exists():
        return staleness, health

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return staleness, health

    staleness_data = data.get("staleness", {})
    for key, value in staleness_data.items():
        if hasattr(staleness, key):
            setattr(staleness, key, value)

    health_data = data.get("health", {})
    for key, value in health_data.items():
        if hasattr(health, key):
            setattr(health, key, value)

    return staleness, health
