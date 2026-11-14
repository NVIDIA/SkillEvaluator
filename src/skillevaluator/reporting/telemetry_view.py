# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, read-only presentation of recorded native Tier 3 telemetry.

This adapter reads only named accounting fields. It does not walk trials, sum
counters, calculate timing, or decide whether two conditions are comparable.
The canonical payload and the newer public resources presentation stay separate.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping
from html import escape
from itertools import islice
from typing import Any

from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.rich_markup import strip_terminal_controls

_MAX_AGENTS = 32
_MAX_CASES_PER_AGENT = 64
_MAX_SCOPES = 128
_MAX_TEXT_CHARS = 240
_MAX_TEXT_INPUT_CHARS = 16384
_MAX_TEXT_ITEMS = 8
_SECTION_BYTES = 100 * 1024
_FAMILY_BYTES = 92 * 1024
_CHART_LABEL_CHARS = 80
_MAX_NUMBER = 2**53 - 1
_TOKEN_FIELDS = (
    "prompt_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "efficiency_tokens",
)
_LATENCY_FIELDS = ("llm_s", "tool_exec_s", "llm_ms_per_token")
_CONDITIONS = (("with_skill", "With skill"), ("baseline", "Without skill"))
_TIMING_NAMES = {"atif": "native (ATIF)", "step_timestamps": "inferred (timestamps)"}


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, *, signed: bool = False, integer: bool = False) -> int | float | None:
    if type(value) not in (int, float) or (integer and type(value) is not int):
        return None
    # Compare before converting: even a maliciously large Python int is bounded.
    if not -_MAX_NUMBER <= value <= _MAX_NUMBER or (not signed and value < 0):
        return None
    return value if math.isfinite(value) else None


def _display(value: object, *, signed: bool = False, integer: bool = False) -> str:
    number = _number(value, signed=signed, integer=integer)
    if number is None:
        return "N/A"
    return format(number, "+," if signed else ",")


class _Text:
    """Limit copied text before encoding; never stringify arbitrary objects."""

    def __init__(self) -> None:
        self.shortened = False

    def __call__(self, value: object, default: str = "") -> str:
        if not isinstance(value, str):
            return default
        if len(value) > _MAX_TEXT_INPUT_CHARS:
            self.shortened = True
            return "[text exceeds inspection bound]"
        # Inspect the complete bounded field before shortening: a cut API key
        # or authorization header may no longer match its redaction pattern.
        value = redact_sensitive_text(strip_terminal_controls(value))
        value = "".join(
            "\ufffd"
            if unicodedata.category(char) == "Cs"
            else f"\\u{ord(char):04x}"
            if unicodedata.category(char) == "Cf"
            else char
            for char in value
        )
        if len(value) > _MAX_TEXT_CHARS:
            self.shortened = True
        return value[:_MAX_TEXT_CHARS] + ("…" if len(value) > _MAX_TEXT_CHARS else "")

    def items(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items = [self(item) for item in value[:_MAX_TEXT_ITEMS] if isinstance(item, str)]
        if len(value) > _MAX_TEXT_ITEMS:
            self.shortened = True
            items.append("Additional recorded items omitted; see JSON.")
        return items


def _coverage(raw: object, text: _Text) -> tuple[str, list[str]]:
    coverage = _mapping(raw)
    status = text(coverage.get("status"), "unknown")
    counts = [_number(coverage.get(key), integer=True) for key in ("measured_attempts", "expected_attempts")]
    summary = f"{counts[0] if counts[0] is not None else '?'}/{counts[1] if counts[1] is not None else '?'} · {status}"
    notes = []
    for key, label in (
        ("observed_attempts", "observed"),
        ("partial_attempts", "partial"),
        ("unscored_attempts", "unscored"),
    ):
        if (value := _number(coverage.get(key), integer=True)) is not None:
            notes.append(f"{value:,} {label}")
    notes.extend(text.items(coverage.get("reason_codes")))
    return summary, notes


def _scope_rows(scope: Mapping, *, case: str, latency: bool, runtime: Mapping, text: _Text) -> list[dict]:
    fields = _LATENCY_FIELDS if latency else _TOKEN_FIELDS
    rows = []
    for condition, label in _CONDITIONS:
        arm = _mapping(scope.get(condition))
        totals = _mapping(arm.get("totals"))
        coverage = _mapping(arm.get("coverage"))
        # Native token totals need positive measured coverage. Missing coverage
        # never turns an aggregate default zero into an observed measurement.
        measured = _number(coverage.get("measured_attempts"), integer=True)
        has_counts = latency or (measured is not None and measured > 0 and coverage.get("status") != "not_run")
        values = [_display(totals.get(key) if has_counts else None, integer=not latency) for key in fields]
        if latency:
            sources = text.items(coverage.get("llm_timing_sources"))
            values.append(", ".join(_TIMING_NAMES.get(source, source) for source in sources) or "—")
            arm_runtime = _mapping(runtime.get("with_skill" if condition == "with_skill" else "without_skill"))
            values.append(_display(arm_runtime.get("duration_ms")) if runtime else "—")
        summary, notes = _coverage(coverage, text)
        rows.append({"case": case, "condition": label, "values": values, "coverage": summary, "notes": notes})

    delta = _mapping(scope.get("delta"))
    status = text(delta.get("status"), "unknown")
    # Native 'partial' may record an absolute change with no percent (zero
    # baseline), or a tool-only change. Preserve it without creating a value.
    comparable = status in ("available", "partial")
    values = []
    percentages = {
        "total_tokens": "total_tokens_change_percent",
        "llm_s": "llm_s_change_percent",
        "llm_ms_per_token": "llm_ms_per_token_change_percent",
    }
    for key in fields:
        value = delta.get(key) if comparable else None
        cell = _display(value, signed=True, integer=not latency)
        percent = _number(delta.get(percentages.get(key, "")), signed=True) if comparable else None
        if cell != "N/A" and percent is not None:
            cell += f" ({percent:+.2f}%)"
        values.append(cell)
    if latency:
        values.extend(("—", "—"))
    notes = [text(delta.get("reason"))] if delta.get("reason") else []
    notes.extend(text.items(delta.get("reason_codes")))
    rows.append(
        {"case": case, "condition": "Change (with - without)", "values": values, "coverage": status, "notes": notes}
    )
    return rows


def _sections(raw: object, *, latency: bool) -> list[dict[str, Any]]:
    usage = _mapping(raw)
    text = _Text()
    sections: list[dict[str, Any]] = []
    scopes = 0
    byte_count = 0
    omitted = 0

    def append(scope: Mapping, case: str, section: dict, runtime: Mapping | None = None) -> bool:
        nonlocal scopes, byte_count
        if scopes >= _MAX_SCOPES:
            return False
        rows = _scope_rows(scope, case=case, latency=latency, runtime=runtime or {}, text=text)
        # Charge escaped text as well: a '<' costs four bytes in the table,
        # while its JSON string needs only one. This also bounds HTML output.
        size = len(escape(json.dumps(rows, ensure_ascii=True)).encode("utf-8"))
        if byte_count + size > _SECTION_BYTES:
            return False
        scopes += 1
        byte_count += size
        section["rows"].extend(rows)
        return True

    dataset = usage.get("dataset")
    if isinstance(dataset, Mapping) and dataset:
        section = {"label": "Dataset", "rows": [], "notes": []}
        append(dataset, "All cases", section)
        sections.append(section)

    agents = _mapping(usage.get("agents"))
    for name, agent in islice(agents.items(), _MAX_AGENTS):
        if not isinstance(name, str) or not isinstance(agent, Mapping):
            continue
        section = {"label": text(name), "rows": [], "notes": []}
        cases = agent.get("cases")
        cases = cases if isinstance(cases, list) else []
        if not append(agent, "All cases", section, _mapping(agent.get("arm_runtime_ms"))):
            omitted += 1 + len(cases)
            continue
        for index, case in enumerate(cases[:_MAX_CASES_PER_AGENT]):
            if not isinstance(case, Mapping):
                continue
            label = text(case.get("entry_id"), "unknown")
            if case.get("case_id_in_dataset") is False:
                label += " (unverified case ID)"
                section["notes"] = ["Unverified case IDs are measured attempts, not verified dataset coverage."]
            if not append(case, label, section):
                omitted += min(len(cases), _MAX_CASES_PER_AGENT) - index
                break
        omitted += max(0, len(cases) - _MAX_CASES_PER_AGENT)
        sections.append(section)
    if not sections:
        return []
    omitted_agents = max(0, len(agents) - _MAX_AGENTS)
    first = sections[0]
    first["source"] = text(usage.get("source"), "unknown")
    first["schema_version"] = text(usage.get("schema_version"), "unknown")
    if omitted:
        first["notes"].append(f"{omitted:,} scope/case row group(s) omitted by display limits; see JSON.")
    if omitted_agents:
        first["notes"].append(f"{omitted_agents:,} agent(s) omitted by display limits; see JSON.")
    if dropped := _number(usage.get("case_rows_omitted"), integer=True):
        first["notes"].append(
            f"{dropped:,} case row(s) omitted before this display projection; retained JSON may also be incomplete."
        )
    if usage.get("baseline_reused") is True:
        first["notes"].append("Historical baseline: baseline measurements were recorded in an earlier run.")
    if text.shortened:
        first["notes"].append(
            "Display text shortened or recorded list items omitted; see retained JSON for full recorded text."
        )
    # Include labels, provenance and omission notes in the family byte bound,
    # not just row values. Astral Unicode takes twelve bytes in ASCII JSON.
    if len(json.dumps(sections).encode("utf-8")) > _FAMILY_BYTES:
        first["notes"].append("Additional scope/case row groups omitted by display byte limits; see JSON.")
        while len(json.dumps(sections).encode("utf-8")) > _FAMILY_BYTES:
            if len(sections[-1]["rows"]) > 3:
                del sections[-1]["rows"][-3:]
            elif len(sections) > 1:
                sections.pop()
            else:
                # Even a maximal single scope is smaller than the budget;
                # retain the qualification if malformed input ever reaches it.
                first["rows"].clear()
                break
    return sections


def _token_chart(raw: object) -> dict[str, Any] | None:
    agents = _mapping(_mapping(raw).get("agents"))
    labels: list[str] = []
    fields = ("uncached_prompt_tokens", "cached_tokens", "completion_tokens")
    datasets: list[dict[str, Any]] = [
        {"label": label, "data": []} for label in ("Uncached prompt", "Cached prompt", "Completion")
    ]
    text = _Text()
    for name, agent in islice(agents.items(), _MAX_AGENTS):
        if not isinstance(name, str) or not isinstance(agent, Mapping):
            continue
        for condition, label in _CONDITIONS:
            arm = _mapping(agent.get(condition))
            coverage = _mapping(arm.get("coverage"))
            measured = _number(coverage.get("measured_attempts"), integer=True)
            if measured is None or measured == 0 or coverage.get("status") == "not_run":
                continue
            totals = _mapping(arm.get("totals"))
            values = [_number(totals.get(key), integer=True) for key in fields]
            if all(value is None for value in values):
                continue
            name_label = text(name)[:_CHART_LABEL_CHARS] + ("…" if len(name) > _CHART_LABEL_CHARS else "")
            labels.append(f"{name_label} · {label}" + (" (partial)" if coverage.get("status") != "complete" else ""))
            for dataset, value in zip(datasets, values, strict=True):
                dataset["data"].append(value)
    return {"labels": labels, "datasets": datasets} if labels else None


def build_telemetry_view(payload: object) -> dict[str, Any]:
    """Project native recorded telemetry with bounded traversal, text and bytes.

    At most 32 agents, 64 cases/agent and 128 scope groups per metric family
    are considered. Each family (including metadata) is capped at 92 KiB
    encoded JSON; metadata/chart bounds keep the complete view below 256 KiB. Unknown
    measurements stay unavailable, and all chart/table lists are newly owned.
    """
    payload = _mapping(payload)
    tokens = _sections(payload.get("token_usage"), latency=False)
    latency = _sections(payload.get("latency_metrics"), latency=True)
    return {
        "token_sections": tokens,
        "latency_sections": latency,
        "token_chart": _token_chart(payload.get("token_usage")),
        "has_tokens": bool(tokens),
        "has_latency": bool(latency),
    }
