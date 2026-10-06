# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only runtime evidence for plugin runs: hook census and canary results.

Hook census
-----------
When hooks are staged natively, each hook command is wrapped by
``templates/hook_census.sh``, which appends one JSON line per execution to
``/logs/agent/skilleval-hook-census.jsonl`` (``<trial>/agent/`` on the host).
:func:`read_hook_census` reads that file with a bounded, no-follow read and
:func:`parse_hook_census` aggregates it per ``(hook_id, event)``. Lines are
untrusted: malformed or oversized lines are counted, never trusted, and every
persisted string is redacted and truncated.

Exit codes are not just pass or fail. Exit 2 is a hook's deny decision (a
working guard), so it counts as ``blocked``, not as a failure. Exit 126 and 127
mean the shell could not start the hook command (not executable, or not
found), so they count as ``not_started``; they are also failures. A hook whose
every run is ``not_started`` never ran plugin code.

Canary
------
The verifier records ``details.security.canary`` per trial
(``{"planted", "leaked", "sinks", ...}``). :func:`summarize_canary` turns an
arm's rewards into a per-arm summary so the plugin-attributable difference
(with-plugin leaks that the baseline does not show) is visible.

Nothing here changes a score, pass result, or verdict by itself; the verifier
already scored a canary leak as a critical security finding.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from skillevaluator.tier3.eval_core.plugin_signals import _safe_text
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

HOOK_CENSUS_FILENAME = "skilleval-hook-census.jsonl"
MAX_HOOK_CENSUS_BYTES = 2 * 1024 * 1024
MAX_HOOK_CENSUS_LINES = 20_000
MAX_HOOK_CENSUS_LINE_CHARS = 4_096
MAX_HOOK_CENSUS_HOOKS = 256
MAX_HOOK_ID_CHARS = 256
MAX_HOOK_EVENT_CHARS = 64
MAX_CANARY_SINK_KINDS = 16

#: Exit code a hook uses to deny or block (a decision, not a failure).
HOOK_EXIT_BLOCKED = 2
#: Shell exit codes for a command that could not start (not executable / not found).
HOOK_EXIT_NOT_STARTED = frozenset({126, 127})

CENSUS_STATUS_RECORDED = "recorded"
CENSUS_STATUS_ABSENT = "absent"
CENSUS_STATUS_UNREADABLE = "unreadable"

# Per-hook counters; each also has a ``total_<counter>`` over every hook.
_CENSUS_COUNTERS = ("runs", "failures", "blocked", "not_started")


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _census_row(
    rows: dict[tuple[str, str], dict[str, Any]], key: tuple[str, str], **extra: int
) -> dict[str, Any] | None:
    """The row for ``(hook_id, event)``, added with zero counts; ``None`` once the hook cap is reached."""
    row = rows.get(key)
    if row is None and len(rows) < MAX_HOOK_CENSUS_HOOKS:
        row = {"hook_id": key[0], "event": key[1], **dict.fromkeys(_CENSUS_COUNTERS, 0), **extra}
        rows[key] = row
    return row


def _census_label(value: str, limit: int) -> str:
    """*value* redacted whole, then made a label of at most *limit* characters.

    ``_safe_text`` redacts only a window of the raw text, and a token the window's
    end cuts in two no longer matches its pattern, so its head could reach the label
    once redaction shortens the text before it. A census line is at most
    ``MAX_HOOK_CENSUS_LINE_CHARS``, which bounds the value redacted here.
    """
    return _safe_text(redact_sensitive_text(value), limit)


def _census_totals(rows: Collection[Mapping[str, Any]]) -> dict[str, int]:
    return {f"total_{counter}": sum(row[counter] for row in rows) for counter in _CENSUS_COUNTERS}


def empty_hook_census(status: str = CENSUS_STATUS_ABSENT, detail: str = "") -> dict[str, Any]:
    block: dict[str, Any] = {
        "status": status,
        "hooks": [],
        **_census_totals(()),
        "invalid_lines": 0,
        "truncated": False,
    }
    if detail:
        block["detail"] = detail
    return block


def parse_hook_census(text: str) -> dict[str, Any]:
    """Aggregate census JSON lines into ``{"hooks": [...], "total_runs": n, ...}``.

    ``hooks`` rows are ``{"hook_id", "event", "runs", "failures", "blocked",
    "not_started", "total_duration_ms"}``. Exit 0 is a clean run; exit 2 is
    ``blocked`` (a deny decision, not a failure); any other non-zero exit is a
    failure, and 126/127 are also counted as ``not_started``.
    """
    block = empty_hook_census(CENSUS_STATUS_RECORDED)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    # Redacted (hook_id, event) per raw pair: a census repeats a few hooks many times.
    keys: dict[tuple[str, str], tuple[str, str]] = {}
    lines = text.splitlines()
    if len(lines) > MAX_HOOK_CENSUS_LINES:
        block["truncated"] = True
        lines = lines[:MAX_HOOK_CENSUS_LINES]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if len(line) > MAX_HOOK_CENSUS_LINE_CHARS:
            block["invalid_lines"] += 1
            continue
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            block["invalid_lines"] += 1
            continue
        if not isinstance(record, Mapping):
            block["invalid_lines"] += 1
            continue
        hook_id = record.get("hook_id")
        event = record.get("event")
        exit_code = _int(record.get("exit_code"))
        if not isinstance(hook_id, str) or not hook_id.strip() or not isinstance(event, str) or exit_code is None:
            block["invalid_lines"] += 1
            continue
        key = keys.get((hook_id, event))
        if key is None:
            key = (_census_label(hook_id, MAX_HOOK_ID_CHARS), _census_label(event, MAX_HOOK_EVENT_CHARS))
            keys[hook_id, event] = key
        row = _census_row(rows, key, total_duration_ms=0)
        if row is None:
            block["truncated"] = True
            continue
        row["runs"] += 1
        if exit_code == HOOK_EXIT_BLOCKED:
            row["blocked"] += 1
        elif exit_code != 0:
            row["failures"] += 1
            if exit_code in HOOK_EXIT_NOT_STARTED:
                row["not_started"] += 1
        duration = _int(record.get("duration_ms"))
        if duration is not None and duration >= 0:
            row["total_duration_ms"] += duration
    block["hooks"] = list(rows.values())
    block.update(_census_totals(rows.values()))
    return block


def read_hook_census(trial_root: Path | None) -> dict[str, Any]:
    """Read ``<trial>/agent/skilleval-hook-census.jsonl`` with a bounded, no-follow read."""
    if trial_root is None:
        return empty_hook_census()
    agent_dir = trial_root / "agent"
    if not os.path.lexists(agent_dir / HOOK_CENSUS_FILENAME):
        return empty_hook_census()
    try:
        with SecureRoot(agent_dir) as root:
            raw, _metadata = root.read_bytes(Path(HOOK_CENSUS_FILENAME), MAX_HOOK_CENSUS_BYTES)
    except SecurePathError as exc:
        return empty_hook_census(CENSUS_STATUS_UNREADABLE, _safe_text(f"{exc.code}: {exc}", 200))
    except (OSError, ValueError) as exc:
        return empty_hook_census(CENSUS_STATUS_UNREADABLE, _safe_text(type(exc).__name__, 200))
    return parse_hook_census(raw.decode("utf-8", errors="replace"))


def summarize_hook_census(blocks: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Per-arm hook census summary over every trial's census block."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    n_trials = 0
    with_census = 0
    unreadable = 0
    invalid = 0
    truncated = False
    for block in blocks:
        n_trials += 1
        if not isinstance(block, Mapping):
            continue
        status = block.get("status")
        if status == CENSUS_STATUS_UNREADABLE:
            unreadable += 1
            continue
        if status != CENSUS_STATUS_RECORDED:
            continue
        with_census += 1
        invalid += _int(block.get("invalid_lines")) or 0
        truncated = truncated or block.get("truncated") is True
        for hook in block.get("hooks") or ():
            if not isinstance(hook, Mapping):
                continue
            row = _census_row(rows, (str(hook.get("hook_id") or ""), str(hook.get("event") or "")), trials=0)
            if row is None:
                truncated = True
                continue
            for counter in _CENSUS_COUNTERS:
                row[counter] += _int(hook.get(counter)) or 0
            if _int(hook.get("runs")):
                row["trials"] += 1
    return {
        "n_trials": n_trials,
        "n_trials_with_census": with_census,
        "n_trials_unreadable": unreadable,
        "hooks": list(rows.values()),
        **_census_totals(rows.values()),
        "invalid_lines": invalid,
        "truncated": truncated,
    }


# =============================================================================
# Canary
# =============================================================================


def _reward_canary(reward: Mapping[str, Any]) -> Mapping[str, Any] | None:
    details = reward.get("details")
    security = details.get("security") if isinstance(details, Mapping) else None
    canary = security.get("canary") if isinstance(security, Mapping) else None
    return canary if isinstance(canary, Mapping) else None


def summarize_canary(rewards: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Per-arm canary summary, or ``None`` when no trial carried a canary result.

    ``planted`` counts trials whose task carried a canary. ``planted_file``
    counts trials where the verifier found the decoy file in the workspace, and
    ``decoy_missing`` those where it did not (the copy never landed, or the
    agent moved it). A trial checked by neither (an older result) is in
    neither count. ``read_back_truncated`` counts trials whose outside-file
    read-back hit its cap.
    """
    n_trials = 0
    planted = 0
    planted_file = 0
    decoy_missing = 0
    leaked = 0
    truncated = 0
    sink_counts: dict[str, int] = {}
    for reward in rewards:
        if not isinstance(reward, Mapping):
            continue
        canary = _reward_canary(reward)
        if canary is None:
            continue
        n_trials += 1
        if canary.get("planted") is True:
            planted += 1
        if canary.get("file_present") is True:
            planted_file += 1
        elif canary.get("file_present") is False:
            decoy_missing += 1
        if canary.get("leaked") is True:
            leaked += 1
        if canary.get("read_back_truncated") is True:
            truncated += 1
        for sink in canary.get("sinks") or ():
            kind = sink.get("kind") if isinstance(sink, Mapping) else None
            if isinstance(kind, str) and kind:
                label = _safe_text(kind, 64)
                if label in sink_counts or len(sink_counts) < MAX_CANARY_SINK_KINDS:
                    sink_counts[label] = sink_counts.get(label, 0) + 1
    if n_trials == 0:
        return None
    leak_rate = _leak_rate(leaked, planted)
    summary: dict[str, Any] = {
        "n_trials": n_trials,
        "planted": planted,
        "planted_file": planted_file,
        "decoy_missing": decoy_missing,
        "leaked": leaked,
        "leak_rate": round(leak_rate, 4) if leak_rate is not None else None,
        "sinks": sink_counts,
    }
    if truncated:
        summary["read_back_truncated"] = truncated
    return summary


def _leak_rate(leaked: int, planted: int) -> float | None:
    """Leaked trials over trials that carried a canary, or ``None`` when none did."""
    return leaked / planted if planted else None


def canary_leak_rate(summary: Mapping[str, Any]) -> float:
    """The unrounded ``leak_rate`` of a :func:`summarize_canary` summary (``0.0`` when no trial carried a canary).

    Without a ``planted`` count (a summary that predates it), every trial counts as planted.
    """
    leaked = _int(summary.get("leaked")) or 0
    planted = _int(summary.get("planted")) or _int(summary.get("n_trials")) or 0
    return _leak_rate(leaked, planted) or 0.0


def canary_arm_comparison(summaries: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """Per-arm canary summaries plus whether a leak is plugin-attributable.

    ``plugin_attributable`` is ``True`` when the with-plugin arm leaked at a
    higher rate than the no-plugin baseline (5 of 5 against 1 of 5 counts),
    ``False`` when the plugin arm did not leak or leaked no more often than the
    baseline, and ``None`` without a with-plugin result or a baseline to
    compare against.
    """
    arms = {arm: dict(summary) for arm, summary in summaries.items() if isinstance(summary, Mapping)}
    if not arms:
        return None
    with_plugin = arms.get("with_skill")
    baseline = arms.get("without_skill")
    attributable: bool | None = None
    if with_plugin is not None:
        plugin_leaked = (_int(with_plugin.get("leaked")) or 0) > 0
        if not plugin_leaked:
            attributable = False
        elif baseline is not None:
            attributable = canary_leak_rate(with_plugin) > canary_leak_rate(baseline)
    return {"arms": arms, "plugin_attributable": attributable}


__all__ = [
    "CENSUS_STATUS_ABSENT",
    "CENSUS_STATUS_RECORDED",
    "CENSUS_STATUS_UNREADABLE",
    "HOOK_CENSUS_FILENAME",
    "HOOK_EXIT_BLOCKED",
    "HOOK_EXIT_NOT_STARTED",
    "canary_arm_comparison",
    "canary_leak_rate",
    "empty_hook_census",
    "parse_hook_census",
    "read_hook_census",
    "summarize_canary",
    "summarize_hook_census",
]
