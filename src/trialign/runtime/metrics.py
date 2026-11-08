"""Run-level settings validation and explicitly priced API usage estimates."""

from __future__ import annotations

import math

RATE_FIELDS = ("input_per_million", "cached_input_per_million", "output_per_million")


def pricing_settings(raw=None):
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("Pricing must contain judge and critic token rates.")
    result = {"currency": "USD", "source": "user-entered rates", "rates": {}}
    for role in ["judge", "critic"]:
        fields = (raw or {}).get(role, {})
        if not isinstance(fields, dict):
            raise ValueError("Each model's pricing must be an object.")
        rates = {}
        for field in RATE_FIELDS:
            value = fields.get(field)
            if value in (None, ""):
                rates[field] = None
                continue
            if isinstance(value, bool):
                raise ValueError("Token rates must be nonnegative finite numbers.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Token rates must be nonnegative finite numbers.") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError("Token rates must be nonnegative finite numbers.")
            rates[field] = number
        if any(v is not None for v in rates.values()) and any(
            rates[k] is None for k in ("input_per_million", "output_per_million")
        ):
            raise ValueError("Provide both input and output prices for each priced model.")
        result["rates"][role] = rates
    return result


def estimate_api_cost(ledger, pricing, calls):
    total = 0.0
    reasons = set()
    for entry in ledger:
        rates = pricing["rates"][entry["agent"]]
        tokens = entry.get("usage")
        if tokens is None:
            reasons.add("Some API requests did not return usage.")
            continue
        if tokens.get("cache_write_tokens", 0):
            reasons.add("Separate cache-write pricing is not configured.")
            continue
        cached = tokens.get("cached_input_tokens")
        if cached is None:
            reasons.add("Cached-input usage was not reported.")
            continue
        if rates["input_per_million"] is None or rates["output_per_million"] is None:
            reasons.add("Input/output pricing is missing for a used model.")
            continue
        if cached and rates["cached_input_per_million"] is None:
            reasons.add("Cached-input pricing is missing for a used model.")
            continue
        total += (
            (tokens["input_tokens"] - cached) * rates["input_per_million"]
            + cached * (rates["cached_input_per_million"] or 0)
            + tokens["output_tokens"] * rates["output_per_million"]
        ) / 1_000_000
    if len(ledger) != calls:
        reasons.add("Some model calls have no recorded usage ledger entry.")
    return {
        "currency": "USD",
        "estimated_api_cost_usd": round(total, 10) if not reasons else None,
        "known_api_cost_usd": round(total, 10),
        "status": "estimated" if not reasons else "unavailable_or_partial",
        "reasons": sorted(reasons),
        "scope": (
            "Model-token API charges only; excludes local compute and other fees. Not an invoice."
        ),
    }
