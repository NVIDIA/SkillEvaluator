# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

if __package__:
    from skillevaluator.tier3.eval_core import script_invocation as _script_invocation
    from skillevaluator.tier3.eval_core.checks import (
        _command_text,
        _exit_status,
        _is_execution_action,
        _tool_call_shell,
    )
else:
    import script_invocation as _script_invocation
    from checks import _command_text, _exit_status, _is_execution_action, _tool_call_shell

_cmd_executes_script = _script_invocation._cmd_executes_script
_SHELL_SEPARATORS = _script_invocation._SHELL_SEPARATORS

SUPPORTED_CHECK_TYPES = (
    "script_invoked",
    "command_succeeded",
    "file_exists",
    "output_matches",
    "output_contains",
)


def validate_declarative_checks(value: object) -> list[str]:
    if not isinstance(value, list):
        return ["checks must be a list"]
    errors = []
    for index, check in enumerate(value):
        where = f"checks[{index}]"
        if not isinstance(check, dict):
            errors.append(f"{where} must be an object")
            continue
        check_type = check.get("type")
        if check_type not in SUPPORTED_CHECK_TYPES:
            supported = ", ".join(SUPPORTED_CHECK_TYPES)
            errors.append(f"{where} has unknown type {check_type!r}; supported types: {supported}")
            continue
        field = {
            "script_invoked": "script",
            "command_succeeded": "script",
            "file_exists": "path",
            "output_matches": "pattern",
            "output_contains": "text",
        }[check_type]
        extra_fields = set(check) - {"type", field}
        if extra_fields:
            errors.append(f"{where} has unsupported field(s): {', '.join(sorted(extra_fields))}")
        expected = check.get(field)
        if not isinstance(expected, str) or not expected.strip():
            errors.append(f"{where}.{field} must be a non-empty string")
            continue
        if check_type == "file_exists":
            path = Path(expected)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"{where}.path must be relative and cannot traverse parent directories")
        elif check_type == "output_matches":
            try:
                re.compile(expected)
            except re.error as exc:
                errors.append(f"{where}.pattern is not a valid regular expression: {exc}")
    return errors


def _invocations(tool_calls: list[dict[str, Any]], script: str) -> list[dict[str, Any]]:
    calls = []
    for call in tool_calls:
        if not _is_execution_action(str(call.get("action", ""))):
            continue
        command = _command_text(call)
        shell, modelled = _tool_call_shell(call)
        if modelled:
            verdict = _cmd_executes_script(command, script, _shell=shell)
        else:
            verdict = None if script in command else False
        if verdict is True:
            calls.append(call)
    return calls


def _script_exit_status(call: dict[str, Any], script: str) -> int | None:
    status = _exit_status(call)
    if status is None:
        return None
    command = _command_text(call)
    tokens = _script_invocation._shell_tokens(command)
    if any(
        token
        in {
            "(",
            ")",
            "{",
            "}",
            "if",
            "then",
            "else",
            "elif",
            "fi",
            "for",
            "while",
            "until",
            "select",
            "do",
            "done",
            "case",
            "esac",
        }
        or re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", token)
        for token in tokens
    ):
        return None
    segments: list[list[str]] = [[]]
    separators: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if not segments[-1]:
                return None
            separators.append(token)
            segments.append([])
        else:
            segments[-1].append(token)
    if not segments[-1]:
        return None
    shell, modelled = _tool_call_shell(call)
    if not modelled:
        return None
    invocations = [
        index
        for index, segment in enumerate(segments)
        if _cmd_executes_script(" ".join(segment), script, _shell=shell) is True
    ]
    if not invocations:
        return None
    last_invocation = invocations[-1]
    following = separators[last_invocation:]
    if any(separator != "&&" for separator in following):
        return None
    if last_invocation < len(segments) - 1 and status != 0:
        return None
    return status


def _file_exists(path: str, roots: list[Path]) -> bool | None:
    checked_root = False
    relative = Path(path)
    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            continue
        checked_root = True
        try:
            candidate = (resolved_root / relative).resolve(strict=True)
        except OSError:
            continue
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if candidate.is_file():
            return True
    return False if checked_root else None


def evaluate_declarative_checks(
    checks: object,
    *,
    tool_calls: list[dict[str, Any]],
    final_response: str,
    workspace_roots: list[Path],
) -> list[dict[str, Any]]:
    results = []
    if not isinstance(checks, list):
        return results
    for check in checks:
        if not isinstance(check, dict):
            results.append({"type": None, "passed": None, "reason": "Declarative check must be an object"})
            continue
        check_type = check.get("type")
        if not isinstance(check_type, str) or check_type not in SUPPORTED_CHECK_TYPES:
            supported = ", ".join(SUPPORTED_CHECK_TYPES)
            results.append(
                {
                    "type": check_type,
                    "passed": None,
                    "reason": f"Unknown check type {check_type!r}; supported types: {supported}",
                }
            )
            continue
        expected = check.get({"script_invoked": "script", "command_succeeded": "script"}.get(check_type, ""))
        if check_type == "script_invoked":
            passed = bool(_invocations(tool_calls, str(expected)))
            reason = (
                f"Invoked {expected}"
                if passed
                else f"Expected {expected} to be invoked, but no invocation was recorded"
            )
        elif check_type == "command_succeeded":
            invocations = _invocations(tool_calls, str(expected))
            statuses = [_script_exit_status(call, str(expected)) for call in invocations]
            if not invocations:
                passed = False
                reason = f"Expected {expected} to exit successfully, but it was not invoked"
            elif 0 in statuses:
                passed = True
                reason = f"Expected {expected} to exit successfully; it exited 0"
            elif any(status is None for status in statuses):
                passed = None
                if any(_exit_status(call) is not None for call in invocations):
                    reason = (
                        f"Expected {expected} to exit successfully, but shell command status could not be attributed"
                    )
                else:
                    reason = f"Expected {expected} to exit successfully, but exit status was not recorded"
            else:
                passed = False
                status = statuses[-1]
                reason = f"Expected {expected} to exit successfully; it was invoked but exited {status}"
        elif check_type == "file_exists":
            expected = check.get("path", "")
            passed = _file_exists(str(expected), workspace_roots)
            reason = (
                f"Expected file {expected} to exist"
                if passed is not False
                else f"Expected file {expected} to exist, but it was not found"
            )
            if passed is None:
                reason = f"Could not check whether file {expected} exists because workspace artifacts are unavailable"
        elif check_type == "output_contains":
            expected = check.get("text", "")
            passed = str(expected) in final_response
            reason = (
                f"Final response contains {expected!r}"
                if passed
                else f"Expected final response to contain {expected!r}"
            )
        elif check_type == "output_matches":
            expected = check.get("pattern", "")
            try:
                passed = re.search(str(expected), final_response) is not None
            except re.error as exc:
                passed = None
                reason = f"Could not check output pattern because it is not a valid regular expression: {exc}"
            else:
                reason = (
                    f"Final response matches /{expected}/"
                    if passed
                    else f"Expected final response to match /{expected}/"
                )
        else:
            raise AssertionError(f"Unhandled declarative check type: {check_type}")
        results.append({"type": check_type, "passed": passed, "reason": reason, "expected": expected})
    return results


def mark_no_skill_script_checks(
    checks: object,
    results: list[dict[str, Any]],
    *,
    has_skill: object,
    skill_metrics_on: bool,
    evaluated_skill: str,
) -> list[dict[str, Any]]:
    if has_skill is not False or skill_metrics_on or not evaluated_skill or not isinstance(checks, list):
        return results
    skill_root = (Path("/workspace/skills") / evaluated_skill).resolve(strict=False)
    for check, result in zip(checks, results):  # noqa: B905 -- zip(strict=) needs Python 3.10
        if (
            not isinstance(check, dict)
            or check.get("type") not in {"script_invoked", "command_succeeded"}
            or not isinstance(check.get("script"), str)
        ):
            continue
        script_path = Path(check["script"])
        candidates = (
            [script_path] if script_path.is_absolute() else [Path.cwd() / script_path, skill_root / script_path]
        )
        for candidate in candidates:
            try:
                candidate.resolve(strict=False).relative_to(skill_root)
            except ValueError:
                continue
            result["passed"] = None
            result["reason"] = "skill script; not applicable without the skill"
            break
    return results


def combine_behavior_check(
    judge_result: dict[str, Any],
    declarative_results: list[dict[str, Any]],
    judged_count: int,
) -> dict[str, Any]:
    checked = [item for item in declarative_results if item.get("passed") is not None]
    not_checked = [item for item in declarative_results if item.get("passed") is None]
    judge_score = judge_result.get("score")
    has_judge_error = judge_result.get("status") == "error"
    denominator = judged_count + len(checked)
    if has_judge_error:
        score = None
    elif denominator:
        judged_total = float(judge_score or 0.0) * judged_count
        checked_total = sum(item.get("passed") is True for item in checked)
        score = round((judged_total + checked_total) / denominator, 4)
    else:
        score = None
    summary = f"Checked directly: {len(checked)}, judged: {judged_count}, not checked: {len(not_checked)}"
    result = dict(judge_result)
    result["score"] = score
    result["status"] = "error" if has_judge_error else "accepted" if score is not None else "not_applicable"
    old_reason = str(judge_result.get("reason") or "").strip()
    if judge_result.get("status") == "not_applicable" and checked:
        old_reason = ""
    result["reason"] = f"{old_reason}; {summary}" if old_reason else summary
    judged_results = judge_result.get("results")
    judged_items = (
        [item for item in judged_results if isinstance(item, dict)] if isinstance(judged_results, list) else []
    )
    result["results"] = [*judged_items, *checked]
    result["declarative_checks"] = {
        "results": declarative_results,
        "checked_count": len(checked),
        "judged_count": judged_count,
        "not_checked_count": len(not_checked),
        "summary": summary,
    }
    return result
