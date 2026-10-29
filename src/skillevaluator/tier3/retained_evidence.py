# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, attributable retained observations for refinement and skill editing."""

from __future__ import annotations

import json
import re
from typing import Any

from skillevaluator.tier3.eval_core.atif_helpers import (
    _tool_call_observation,
    get_final_response,
    iter_tool_calls,
)
from skillevaluator.tier3.eval_core.llm_judge import _configured_secret_values, _redact_configured_credentials
from skillevaluator.utils.redaction import redact_sensitive_data, redact_sensitive_text

SUMMARY_LIMIT = 32000
_DIAGNOSTIC_LINE = re.compile(
    r"\b(?:FAIL(?:ED|URE)?|ERROR|PASS(?:ED)?|Traceback|AssertionError|Exception|exit_code|exit status|exit\s+-?\d+)\b",
    re.IGNORECASE,
)
_FAILURE_LINE = re.compile(r"\b(?:FAIL(?:ED|URE)?|ERROR|Traceback|AssertionError|Exception)\b", re.IGNORECASE)


def _excerpt(text: str, limit: int) -> str:
    """Keep exact numbered diagnostic lines, plus both ends; never synthesize a return."""
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    selected: dict[int, str] = {}
    remaining = limit - 100
    # A long listing or stack must not hide a failure in its middle. Source line
    # numbers preserve the order even when selected excerpts are non-contiguous.
    diagnostic = [i for i, line in enumerate(lines) if _DIAGNOSTIC_LINE.search(line)]
    # Positive output can be arbitrarily noisy. Select failures first, but render
    # all selected excerpts in their original order so recovery stays observable.
    diagnostic.sort(key=lambda index: not bool(_FAILURE_LINE.search(lines[index])))
    candidates = [*diagnostic, 0, len(lines) - 1]
    for index in dict.fromkeys(candidates):
        if remaining < 40:
            break
        line = lines[index]
        cap = min(1200, remaining - 20)
        rendered = f"L{index + 1}: {line[:cap]}" + (" [line clipped]" if len(line) > cap else "")
        selected[index] = rendered
        remaining -= len(rendered) + 1
    prefix = f"[clipped; exact excerpts from {len(lines)} lines; gaps are omitted]\n"
    return (prefix + "\n".join(selected[index] for index in sorted(selected)))[:limit]


def _row_budgets(sizes: list[int], available: int) -> list[int]:
    """Share unused space from short complete rows among the remaining rows."""
    budgets = [0] * len(sizes)
    order = sorted(range(len(sizes)), key=sizes.__getitem__)
    for remaining, index in zip(range(len(sizes), 0, -1), order, strict=True):
        budgets[index] = min(sizes[index], available // remaining)
        available -= budgets[index]
    return budgets


def summarize_trajectory(trajectory: dict[str, Any], *, include_final_response: bool = True) -> str:
    """Separate tool observations from claims with explicit bounds and provenance."""
    # Redact the entire value before clipping so a clipped credential is never
    # made unrecognizable to the redactor. Copying also leaves raw evidence intact.
    copied = json.loads(_redact_configured_credentials(
        json.dumps(redact_sensitive_data(trajectory), ensure_ascii=True),
        extra_secret_values=tuple(json.dumps(value)[1:-1] for value in _configured_secret_values()),
    ))
    raw_steps = copied.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    valid_steps = [(index, step) for index, step in enumerate(steps) if isinstance(step, dict)]
    calls = [
        (index, step, call)
        for index, step in valid_steps if step.get("source") == "agent"
        for _, call in iter_tool_calls({"steps": [step]})
    ]
    # Every shown call gets an independent allowance; excess calls are disclosed,
    # retaining both early investigation and late recovery in chronological order.
    chosen = calls if len(calls) <= 64 else calls[:32] + calls[-32:]
    normalized = []
    for index, step, call in chosen:
        identity = f"step_index={index} step={step.get('step_id', '?')} call={call.get('tool_call_id', '?')}"
        arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=True)
        command = f"Agent called: {call.get('function_name', '?')} {arguments}"
        observed = _tool_call_observation(step, call) or "[no attributable observation text]"
        if len(step.get("tool_calls") or []) > 1 and any(
            isinstance(result, dict) and not result.get("source_call_id")
            for result in (step.get("observation") or {}).get("results", [])
        ):
            observed = "[unattributed step-level observation; does not prove which call completed]\n" + observed
        # Redact normalized observations as well: normalization can render nested
        # fields as plain text, and escaped strings may become recognizable here.
        identity, command, observed = (
            _redact_configured_credentials(redact_sensitive_text(text)).encode("utf-8", errors="replace").decode("utf-8")
            for text in (identity, command, observed)
        )
        normalized.append((identity, command, observed))

    def header(clipped: int) -> str:
        return (
            "EXECUTION TOOL EVIDENCE (chronological recorded calls and returns; not a correctness verdict)\n"
            f"Coverage: {len(chosen)}/{len(calls)} calls shown; {len(calls) - len(chosen)} omitted; {clipped} clipped.\n"
            "Coverage describes this retained trajectory only; missing source events remain unknown. "
            "A compound command exit status is not an individual test exit status. "
            "Read log text can describe an earlier run; inspect its command before inferring execution.\n"
            "The tool trace alone cannot establish exhaustive file preservation. "
            "Consult separately supplied declared artifact evidence, if available.\n"
        )

    final = get_final_response({"steps": [step for _, step in valid_steps]}) if include_final_response else ""
    final = _redact_configured_credentials(redact_sensitive_text(final)).encode("utf-8", errors="replace").decode("utf-8")
    claims = "\n\nFINAL RESPONSE (unverified agent claims)\n" + _excerpt(final, 1300) if final else ""
    rows = []
    clipped = 0
    complete = [f"{identity}\n{command}\nTool returned: {observed}" for identity, command, observed in normalized]
    separators = 2 * max(0, len(chosen) - 1)
    if len(header(0)) + sum(map(len, complete)) + separators + len(claims) <= SUMMARY_LIMIT:
        return header(0) + "\n\n".join(complete) + claims
    available = SUMMARY_LIMIT - len(header(len(chosen))) - separators - len(claims)
    budgets = _row_budgets([len(row) for row in complete], available)
    for (identity, command, observed), full_row, row_budget in zip(normalized, complete, budgets, strict=True):
        if len(full_row) <= row_budget:
            rows.append(full_row)
            continue
        shown_identity = _excerpt(identity, min(160, row_budget // 4))
        shown_command = _excerpt(command, max(120, row_budget // 3))
        shown_return = _excerpt(observed, max(120, row_budget - len(shown_identity) - len(shown_command) - 40))
        clipped += int((identity, command, observed) != (shown_identity, shown_command, shown_return))
        rows.append(f"{shown_identity}\n{shown_command}\nTool returned: {shown_return}")
    return (header(clipped) + "\n\n".join(rows) + claims)[:SUMMARY_LIMIT]
