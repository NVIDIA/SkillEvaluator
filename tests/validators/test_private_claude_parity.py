# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in `claude plugin validate` parity check (the subprocess is always mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.models.result import Severity
from skillevaluator.utils.tool_runner import ToolResult
from skillevaluator.validators.claude_plugin_validate import ClaudePluginValidateParity


class FakeClaude:
    def __init__(self, responses: list[ToolResult], *, available: bool = True) -> None:
        self.responses = list(responses)
        self.available = available
        self.calls: list[dict[str, Any]] = []

    @property
    def is_available(self) -> bool:
        return self.available

    def get_install_hint(self) -> str:
        return "Install Claude Code"

    def run(self, args: list[str], **kwargs: Any) -> ToolResult:
        self.calls.append({"args": list(args), **kwargs})
        return self.responses.pop(0)


def _json(payload: dict, exit_code: int) -> ToolResult:
    return ToolResult(True, json.dumps(payload), "", exit_code)


def _plugin(root: Path, manifest: dict | None = None) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest or {"name": "demo"}))
    return root


PASSED = {
    "success": True,
    "strict": True,
    "target": "/p",
    "manifest": {"file": "plugin.json", "errors": [], "warnings": []},
    "contents": [],
}
FAILED = {
    "success": False,
    "strict": True,
    "target": "/p",
    "manifest": {
        "file": ".claude-plugin/plugin.json",
        "errors": [{"path": "name", "message": "Invalid input"}],
        "warnings": [],
    },
    "contents": [{"file": "agents/a.md", "errors": [], "warnings": ["Missing description"], "notes": []}],
}


def test_absent_cli_is_skipped_with_reason(tmp_path: Path) -> None:
    result = ClaudePluginValidateParity(FakeClaude([], available=False)).validate(
        _plugin(tmp_path), skillevaluator_verdict="passed"
    )
    assert result.passed
    assert not result.is_incomplete
    parity = result.metadata["plugin"]["validator_parity"]
    assert parity["status"] == "skipped"
    assert "not found" in parity["reason"]


def test_exact_command_env_and_agreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_not_leak")
    fake = FakeClaude([_json(PASSED, 0)])
    root = _plugin(tmp_path)
    result = ClaudePluginValidateParity(fake).validate(root, skillevaluator_verdict="passed")
    [call] = fake.calls
    # Not --strict: parity is about what Claude Code refuses to load (proof M38).
    assert call["args"] == ["plugin", "validate", str(root.absolute()), "--json"]
    assert call["replace_env"] is True
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "GITHUB_TOKEN" not in call["env"]
    assert call["env"]["DISABLE_AUTOUPDATER"] == "1"
    assert call["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert Path(call["cwd"]) != root
    parity = result.metadata["plugin"]["validator_parity"]
    assert parity["claude_verdict"] == "passed"
    assert parity["agree"] is True
    assert result.findings == []


def test_disagreement_is_info_and_claude_errors_are_advisory(tmp_path: Path) -> None:
    result = ClaudePluginValidateParity(FakeClaude([_json(FAILED, 1)])).validate(
        _plugin(tmp_path), skillevaluator_verdict="passed"
    )
    by_check = {}
    for finding in result.findings:
        by_check.setdefault(finding.check_name, []).append(finding)
    assert [f.severity for f in by_check["claude_validate_disagreement"]] == [Severity.INFO]
    assert [f.severity for f in by_check["claude_validate_error"]] == [Severity.MEDIUM]
    assert "name: Invalid input" in by_check["claude_validate_error"][0].message
    assert [f.severity for f in by_check["claude_validate_warning"]] == [Severity.LOW]
    assert result.passed
    parity = result.metadata["plugin"]["validator_parity"]
    assert parity["agree"] is False
    assert parity["error_count"] == 1
    assert parity["warning_count"] == 1


def test_old_cli_without_json_falls_back_to_text(tmp_path: Path) -> None:
    fake = FakeClaude(
        [
            ToolResult(True, "", "error: unknown option '--json'", 1),
            ToolResult(True, "Validating plugin\n✘ Error: agents: Path not found\nValidation failed\n", "", 1),
        ]
    )
    result = ClaudePluginValidateParity(fake).validate(_plugin(tmp_path), skillevaluator_verdict="failed")
    assert [call["args"][3:] for call in fake.calls] == [["--json"], []]
    parity = result.metadata["plugin"]["validator_parity"]
    assert parity["format"] == "text"
    assert parity["claude_verdict"] == "failed"
    assert parity["agree"] is True
    assert parity["errors"] == ["Error: agents: Path not found"]


@pytest.mark.parametrize(
    "response",
    [
        ToolResult(True, "", "Unexpected error during validation: EACCES", 2),
        ToolResult(False, "", "", -1, error_message="Claude Code timed out after 120 seconds"),
    ],
)
def test_crash_or_timeout_is_incomplete(tmp_path: Path, response: ToolResult) -> None:
    result = ClaudePluginValidateParity(FakeClaude([response])).validate(
        _plugin(tmp_path), skillevaluator_verdict="passed"
    )
    assert result.is_incomplete
    assert result.metadata["plugin"]["validator_parity"]["status"] == "error"
