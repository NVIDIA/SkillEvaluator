# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission control and accounting for bounded Tier 3 runs."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BudgetConfigurationError(ValueError):
    """Raised when a run budget or price file is invalid."""


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None


@dataclass(frozen=True)
class PriceBook:
    prices: dict[str, Price]

    @classmethod
    def load(cls, path: Path) -> PriceBook:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetConfigurationError(f"{path}: invalid price file: {exc}") from exc
        if not isinstance(raw, dict) or not raw:
            raise BudgetConfigurationError(f"{path}: price file must be a non-empty object")
        prices: dict[str, Price] = {}
        for model, value in raw.items():
            if not isinstance(model, str) or not isinstance(value, dict):
                raise BudgetConfigurationError(f"{path}: each model price must be an object")
            try:
                input_price = float(value["input_per_mtok"])
                output_price = float(value["output_per_mtok"])
                cached_input_price = float(value.get("cached_input_per_mtok", input_price))
            except (KeyError, TypeError, ValueError) as exc:
                raise BudgetConfigurationError(
                    f"{path}: {model} must provide input_per_mtok and output_per_mtok, with optional cached_input_per_mtok"
                ) from exc
            if (
                not math.isfinite(input_price)
                or not math.isfinite(output_price)
                or not math.isfinite(cached_input_price)
                or input_price < 0
                or output_price < 0
                or cached_input_price < 0
            ):
                raise BudgetConfigurationError(f"{path}: {model} prices must be finite and non-negative")
            prices[model] = Price(input_price, output_price, cached_input_price)
        return cls(prices)

    def price_for(self, model: str) -> Price:
        try:
            return self.prices[model]
        except KeyError as exc:
            raise BudgetConfigurationError(f"No price is supplied for run model {model!r}") from exc


def planned_attempts(
    cases: int,
    attempts: int,
    agents: int,
    *,
    arms: int = 2,
    stop_on_pass: bool = False,
) -> int:
    """Return the upper bound used for fail-fast admission checks."""
    if stop_on_pass:
        return cases * attempts * arms * agents
    return cases * attempts * arms * agents


def _usage_number(usage: dict[str, Any] | None, *keys: str) -> int | float | None:
    if not isinstance(usage, dict):
        return None
    values = [usage[key] for key in keys if isinstance(usage.get(key), (int, float))]
    values = [value for value in values if not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0]
    return max(values) if values else None


@dataclass
class BudgetController:
    max_agent_tokens: int | None = None
    max_judge_calls: int | None = None
    max_attempts_total: int | None = None
    max_cost_usd: float | None = None
    price_book: PriceBook | None = None
    planned: int = 0
    used_agent_tokens: int | None = None
    used_judge_calls: int | None = None
    used_cost_usd: float | None = None
    exhausted_limit: str | None = None
    not_run: list[dict[str, Any]] = field(default_factory=list)
    _started: int = 0
    _finished: int = 0
    _agent_usage_seen: bool = False
    _judge_usage_seen: bool = False
    _cost_usage_seen: bool = False
    unmeasured_agent_attempts: int = 0
    unmeasured_judge_attempts: int = 0
    unmeasured_cost_attempts: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_agent_tokens", "max_judge_calls", "max_attempts_total"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise BudgetConfigurationError(f"{name} must be a non-negative integer")
        if self.max_cost_usd is not None:
            if not math.isfinite(self.max_cost_usd) or self.max_cost_usd < 0:
                raise BudgetConfigurationError("max_cost_usd must be finite and non-negative")
            if self.price_book is None:
                raise BudgetConfigurationError("max_cost_usd requires a user-supplied price file")

    def admit(self, *, agent: str, arm: str, case_id: str, attempt: int) -> bool:
        return self.admit_many([{"agent": agent, "arm": arm, "case_id": case_id, "attempt": attempt}])

    def admit_many(self, attempts: list[dict[str, Any]]) -> bool:
        """Atomically admit a logical unit, or record every member as not run."""
        if not attempts:
            return True
        with self._lock:
            if self.exhausted_limit is not None:
                self.not_run.extend(
                    self._not_run(item["agent"], item["arm"], item["case_id"], item["attempt"]) for item in attempts
                )
                return False
            if self.max_attempts_total is not None and self._started + len(attempts) > self.max_attempts_total:
                self.exhausted_limit = "attempts"
                self.not_run.extend(
                    self._not_run(item["agent"], item["arm"], item["case_id"], item["attempt"]) for item in attempts
                )
                return False
            if self.max_agent_tokens == 0:
                self.exhausted_limit = "agent_tokens"
            elif self.max_judge_calls == 0:
                self.exhausted_limit = "judge_calls"
            elif self.max_cost_usd == 0:
                self.exhausted_limit = "cost"
            if self.exhausted_limit is not None:
                self.not_run.extend(
                    self._not_run(item["agent"], item["arm"], item["case_id"], item["attempt"]) for item in attempts
                )
                return False
            if self._finished and self.max_agent_tokens is not None and self.unmeasured_agent_attempts:
                self.exhausted_limit = "agent_tokens"
            elif self._finished and self.max_judge_calls is not None and self.unmeasured_judge_attempts:
                self.exhausted_limit = "judge_calls"
            elif self._finished and self.max_cost_usd is not None and self.unmeasured_cost_attempts:
                self.exhausted_limit = "cost"
            if self.exhausted_limit is not None:
                self.not_run.extend(
                    self._not_run(item["agent"], item["arm"], item["case_id"], item["attempt"]) for item in attempts
                )
                return False
            if (
                self.max_agent_tokens is not None
                and self.used_agent_tokens is not None
                and self.used_agent_tokens >= self.max_agent_tokens
            ):
                self.exhausted_limit = "agent_tokens"
            elif (
                self.max_judge_calls is not None
                and self.used_judge_calls is not None
                and self.used_judge_calls >= self.max_judge_calls
            ):
                self.exhausted_limit = "judge_calls"
            elif (
                self.max_cost_usd is not None
                and self.used_cost_usd is not None
                and self.used_cost_usd >= self.max_cost_usd
            ):
                self.exhausted_limit = "cost"
            if self.exhausted_limit is not None:
                self.not_run.extend(
                    self._not_run(item["agent"], item["arm"], item["case_id"], item["attempt"]) for item in attempts
                )
                return False
            self._started += len(attempts)
            return True

    def record_usage(self, usage: dict[str, Any] | None, *, model: str | None = None) -> None:
        with self._lock:
            input_tokens = _usage_number(usage, "input_tokens", "prompt_tokens", "total_prompt_tokens")
            output_tokens = _usage_number(usage, "output_tokens", "completion_tokens", "total_completion_tokens")
            tokens = _usage_number(usage, "total_tokens")
            if tokens is None and input_tokens is not None and output_tokens is not None:
                tokens = input_tokens + output_tokens
            if tokens is None:
                self.unmeasured_agent_attempts += 1
            else:
                self.used_agent_tokens = (self.used_agent_tokens or 0) + int(tokens)
                self._agent_usage_seen = True
            judge_calls = _usage_number(usage, "judge_calls", "llm_call_count")
            if judge_calls is None:
                self.unmeasured_judge_attempts += 1
            else:
                self.used_judge_calls = (self.used_judge_calls or 0) + int(judge_calls)
                self._judge_usage_seen = True
            cost = None
            if self.price_book is not None and model and input_tokens is not None and output_tokens is not None:
                price = self.price_book.price_for(model)
                cached_tokens = _usage_number(usage, "cached_input_tokens", "total_cached_tokens", "cached_tokens") or 0
                cached_tokens = min(int(cached_tokens), int(input_tokens))
                cached_price = price.cached_input_per_mtok
                if cached_price is None:
                    cached_price = price.input_per_mtok
                uncached_tokens = int(input_tokens) - cached_tokens
                cost = (
                    uncached_tokens * price.input_per_mtok
                    + cached_tokens * cached_price
                    + output_tokens * price.output_per_mtok
                ) / 1_000_000
            if cost is None:
                cost = _usage_number(usage, "cost_usd", "total_cost_usd")
            if cost is None:
                self.unmeasured_cost_attempts += 1
            else:
                self.used_cost_usd = (self.used_cost_usd or 0.0) + float(cost)
                self._cost_usage_seen = True
            self._finished += 1

    def mark_remaining(self, attempts: list[dict[str, Any]]) -> None:
        with self._lock:
            for item in attempts:
                self.not_run.append(dict(item, status="not_run", reason="budget_exhausted"))

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            metadata = {
                "limits": {
                    "max_agent_tokens": self.max_agent_tokens,
                    "max_judge_calls": self.max_judge_calls,
                    "max_attempts_total": self.max_attempts_total,
                    "max_cost_usd": self.max_cost_usd,
                },
                "planned_attempts": self.planned,
                "started_attempts": self._started,
                "finished_attempts": self._finished,
                "unmeasured_attempts": {
                    "agent_tokens": self.unmeasured_agent_attempts,
                    "judge_calls": self.unmeasured_judge_attempts,
                    "cost_usd": self.unmeasured_cost_attempts,
                },
                "used": {
                    "agent_tokens": self.used_agent_tokens if self.used_agent_tokens is not None else "not measured",
                    "judge_calls": self.used_judge_calls if self.used_judge_calls is not None else "not measured",
                    "cost_usd": self.used_cost_usd if self.used_cost_usd is not None else "not measured",
                },
                "status": "exhausted" if self.exhausted_limit else "ok",
                "partial": bool(self.not_run),
                "exhausted_limit": self.exhausted_limit,
                "attempts_not_run": len(self.not_run),
                "not_run": list(self.not_run),
            }
            metadata["observed_overruns"] = _budget_observations(metadata)[0]
            return metadata

    @staticmethod
    def _not_run(agent: str, arm: str, case_id: str, attempt: int) -> dict[str, Any]:
        return {
            "agent": agent,
            "arm": arm,
            "case_id": case_id,
            "attempt": attempt,
            "status": "not_run",
            "reason": "budget_exhausted",
        }


_BUDGET_LIMITS = {
    "agent_tokens": ("max_agent_tokens", "agent tokens"),
    "judge_calls": ("max_judge_calls", "judge calls"),
    "cost_usd": ("max_cost_usd", "cost"),
    "attempts": ("max_attempts_total", "attempts"),
}


def _measured_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value if isinstance(value, int) or math.isfinite(value) else None


def _budget_observations(metadata: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    """Derive reporting facts, including for legacy metadata, without changing admission."""
    limits = metadata.get("limits", {})
    if not isinstance(limits, dict):
        return {}, ["limits invalid"], True
    used = metadata.get("used")
    used = used if isinstance(used, dict) else {}
    unmeasured = metadata.get("unmeasured_attempts", {})
    overruns: dict[str, Any] = {}
    incomplete = []
    configured = False
    for key, (limit_key, label) in _BUDGET_LIMITS.items():
        raw_cap = limits.get(limit_key)
        if raw_cap is None:
            continue
        configured = True
        cap = _measured_number(raw_cap)
        if cap is None:
            incomplete.append(f"{label} limit invalid")
            continue
        value = _measured_number(metadata.get("started_attempts") if key == "attempts" else used.get(key))
        if value is None:
            incomplete.append(f"{label} not measured")
        else:
            if value > cap:
                overruns[key] = {"used": value, "limit": cap, "excess": value - cap}
            if key != "attempts":
                missing = _measured_number(unmeasured.get(key, 0)) if isinstance(unmeasured, dict) else None
                if missing is None:
                    incomplete.append(f"{label} measurement completeness unknown")
                elif missing > 0:
                    incomplete.append(f"{label} usage partially measured")
    return overruns, incomplete, configured


def _budget_number(value: Any, *, cost: bool = False, missing: str = "not measured") -> str:
    number = _measured_number(value)
    if number is None:
        return missing
    if cost:
        return f"${number:,.2f}" if number == 0 or number >= 0.01 else f"${number:.6g}"
    return f"{number:,}"


def budget_line(metadata: dict[str, Any]) -> str:
    # Recompute from usage and limits: cached observations may be absent or stale.
    overruns, incomplete, configured = _budget_observations(metadata)
    observations = "; ".join(
        f"{_BUDGET_LIMITS[key][1]} {_budget_number(row['used'], cost=key == 'cost_usd')} "
        f"of {_budget_number(row['limit'], cost=key == 'cost_usd')}"
        for key, row in overruns.items()
    )
    not_run = _measured_number(metadata.get("attempts_not_run"))
    if not_run is None:
        rows = metadata.get("not_run")
        not_run = len(rows) if isinstance(rows, list) else 0
    if metadata.get("status") == "exhausted":
        limit = metadata.get("exhausted_limit")
        key = {"cost": "cost_usd"}.get(limit, limit) if isinstance(limit, str) else None
        limit_key, label = _BUDGET_LIMITS.get(key, ("", "budget"))
        used = metadata.get("used")
        used = used if isinstance(used, dict) else {}
        limits = metadata.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        value = metadata.get("started_attempts") if key == "attempts" else used.get(key)
        shown = _budget_number(value, cost=key == "cost_usd")
        cap = _budget_number(limits.get(limit_key), cost=key == "cost_usd", missing="not configured")
        line = f"Budget exhausted ({label} {shown} of {cap})"
    elif observations:
        line = f"Budget overrun observed ({observations})"
    elif incomplete:
        line = "Budget: configured limits not fully verified"
    else:
        line = "Budget: within configured limits" if configured else "Budget: no limits configured"
    if not_run:
        planned = _budget_number(metadata.get("planned_attempts"), missing="unknown")
        line += f": {not_run:,} of {planned} attempts not run; results are partial"
    if observations and metadata.get("status") == "exhausted":
        line += f"; overrun observed ({observations})"
    if incomplete:
        if observations or metadata.get("status") == "exhausted":
            line += "; configured limits not fully verified"
        line += f" ({'; '.join(incomplete)})"
    return line
