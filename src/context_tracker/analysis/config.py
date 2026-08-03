"""Configurable thresholds for staleness detection and health scoring.

All defaults are labeled as uncalibrated — to be tuned against real sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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


MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 200_000,
    "claude-opus-4-6[1m]": 1_000_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
}

# Cache rates are fixed multiples of a model's base input price, so they are
# derived rather than written out per model — the previous hand-copied table
# had cache_read at 0.125x input on every entry instead of 0.1x.
CACHE_READ_MULTIPLIER = 0.1
CACHE_CREATE_MULTIPLIER = 1.25  # 5-minute TTL; a 1-hour write costs 2x base

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
    }


# Pricing per million tokens
PRICING = {name: _rates(*rates) for name, rates in MODEL_BASE_RATES.items()}


def cost_of_call(
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> float:
    """Cost in USD of one API call, priced at the given model's rates.

    Unknown or missing models fall back to ``_default``. Cache creation is
    priced at the 5-minute write rate; the transcript records a single
    cache_creation figure, so 1-hour writes (2x base) are under-counted.
    """
    rates = PRICING.get(model or "", PRICING["_default"])
    return (
        int(input_tokens) * rates["input"]
        + int(output_tokens) * rates["output"]
        + int(cache_read) * rates["cache_read"]
        + int(cache_creation) * rates["cache_create"]
    ) / 1_000_000


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
