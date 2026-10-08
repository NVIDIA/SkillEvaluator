# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``claude plugin validate`` parity: not always ``--strict``, and findings compared, not only verdicts (proof M38).

The fake CLI replays what Claude Code 2.1.284 printed for the proof's
examples (check-01 n01 and k03, check-07 edge-05): with ``--strict`` the
three metadata warnings of a minimal manifest fail it, without ``--strict``
it passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.claude_plugin_validate import ClaudePluginValidateParity

if TYPE_CHECKING:
    import pytest

N01_WARNINGS = [
    {"path": "version", "message": 'No version specified. Consider adding a version following semver (e.g., "1.0.0")'},
    {"path": "description", "message": "No description provided."},
    {"path": "author", "message": "No author information provided."},
]
# k03: Claude Code's report on the g01 manifest.
K03_ERRORS = [
    {"path": "name", "message": 'Plugin name cannot contain spaces. Use kebab-case (e.g., "my-plugin")'},
    {"path": "version", "message": "Invalid input"},
    {"path": "description", "message": "Invalid input"},
    {"path": "author", "message": "Invalid input"},
    {"path": "homepage", "message": "Invalid input"},
    {"path": "keywords", "message": "Invalid input"},
]
K03_WARNINGS = [{"path": "bogusKey", "message": "Unknown field 'bogusKey'. Claude Code ignores it at load time."}]
G01 = {
    "name": "My Plugin!",
    "version": 123,
    "description": ["not", "a", "string"],
    "author": "me",
    "keywords": "not-a-list",
    "homepage": 42,
    "bogusKey": True,
}


class OracleClaude:
    """Answers like Claude Code: warnings fail the run only under --strict."""

    def __init__(self, errors: list[dict], warnings: list[dict]) -> None:
        self.errors = errors
        self.warnings = warnings
        self.calls: list[list[str]] = []

    @property
    def is_available(self) -> bool:
        return True

    def get_install_hint(self) -> str:
        return "Install Claude Code"

    def run(self, args: list[str], **_kwargs: Any) -> ToolResult:
        self.calls.append(list(args))
        strict = "--strict" in args
        success = not self.errors and not (strict and self.warnings)
        payload = {
            "success": success,
            "strict": strict,
            "manifest": {"file": ".claude-plugin/plugin.json", "errors": self.errors, "warnings": self.warnings},
            "contents": [],
        }
        return ToolResult(True, json.dumps(payload), "", 0 if success else 1)


def _plugin(root: Path, manifest: dict) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _parity(monkeypatch: pytest.MonkeyPatch, root: Path, fake: OracleClaude):
    monkeypatch.setattr(Tools, "claude", fake)
    results = run_validation(root, checks="schema,claude-validate", content_type=CONTENT_TYPE_PLUGIN)
    parity_result = results[-1]
    return parity_result, parity_result.metadata["plugin"]["validator_parity"]


def test_clean_minimal_manifest_agrees_n01(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = OracleClaude([], N01_WARNINGS)
    result, parity = _parity(monkeypatch, _plugin(tmp_path, {"name": "demo"}), fake)

    assert "--strict" not in fake.calls[0]
    assert parity["claude_verdict"] == "passed"
    assert parity["strict_verdict"] == "failed"
    assert parity["agree"] is True
    assert not [f for f in result.findings if f.check_name == "claude_validate_disagreement"]
    assert [f.check_name for f in result.findings] == ["claude_validate_warning"] * 3


def test_same_failures_on_the_same_fields_agree_k03(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, parity = _parity(monkeypatch, _plugin(tmp_path, G01), OracleClaude(K03_ERRORS, K03_WARNINGS))

    assert parity["claude_verdict"] == "failed"
    assert parity["skillevaluator_verdict"] == "failed"
    assert parity["fields"]["claude_only"] == []
    assert parity["fields"]["skillevaluator_only"] == []
    assert parity["agree"] is True
    assert not [f for f in result.findings if f.check_name == "claude_validate_disagreement"]


def test_lsp_schema_errors_agree_edge05(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "name": "c7-lsp-gaps",
        "version": "1.0.0",
        "description": "Check 7 example plugin c7-lsp-gaps.",
        "author": {"name": "Check Seven"},
        "lspServers": {"nocommand": {"args": ["--stdio"], "extensionToLanguage": {".py": "python"}}},
    }
    oracle = OracleClaude([{"path": "lspServers", "message": "Invalid input"}], [])
    _result, parity = _parity(monkeypatch, _plugin(tmp_path, manifest), oracle)

    assert parity["fields"]["claude_error_fields"] == ["lspServers"]
    assert parity["agree"] is True


def test_different_failures_are_not_an_agreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SkillEvaluator fails 'version'; the (synthetic) Claude Code report fails only 'homepage'.
    oracle = OracleClaude([{"path": "homepage", "message": "Invalid input"}], [])
    result, parity = _parity(monkeypatch, _plugin(tmp_path, {"name": "demo", "version": 5}), oracle)

    assert parity["claude_verdict"] == parity["skillevaluator_verdict"] == "failed"
    assert parity["fields"]["claude_only"] == ["homepage"]
    assert parity["fields"]["skillevaluator_only"] == ["version"]
    assert parity["agree"] is False
    [info] = [f for f in result.findings if f.check_name == "claude_validate_disagreement"]
    assert "for different reasons" in info.message
    assert "homepage" in info.message
    assert "version" in info.message


def test_without_field_data_only_verdicts_are_compared(tmp_path: Path) -> None:
    fake = OracleClaude([], N01_WARNINGS)
    result = ClaudePluginValidateParity(fake).validate(
        _plugin(tmp_path, {"name": "demo"}), skillevaluator_verdict="passed"
    )

    parity = result.metadata["plugin"]["validator_parity"]
    assert parity["agree"] is True
    assert parity["strict"] is False
    assert parity["command"] == "claude plugin validate <plugin-root> --json"
    assert "fields" not in parity


# Verifier round 2 (M38): both sides fail the same field, but parity said "for different reasons". The
# Claude Code reports are what `claude plugin validate --json` 2.1.284 printed for these manifests.
SAME_FIELD_PROBES = {
    "commands-map-missing-source": (
        {"commands": {"x": {"source": "./a.md"}}},
        [{"path": "commands.x.source", "message": "Path not found: ./a.md."}],
    ),
    "mcp-entry-not-object": ({"mcpServers": {"x": 5}}, [{"path": "mcpServers", "message": "Invalid input"}]),
    "mcp-not-object": ({"mcpServers": 5}, [{"path": "mcpServers", "message": "Invalid input"}]),
    "experimental-themes-number": (
        {"experimental": {"themes": 5}},
        [{"path": "experimental.themes", "message": "Invalid input"}],
    ),
    "experimental-monitors-number": (
        {"experimental": {"monitors": 5}},
        [{"path": "experimental.monitors", "message": "Invalid input"}],
    ),
    "monitors-number": ({"monitors": 5}, [{"path": "monitors", "message": "Invalid input"}]),
    "hooks-number": ({"hooks": 5}, [{"path": "hooks", "message": "Invalid input"}]),
    "homepage-space": ({"homepage": "https://exa mple.com"}, [{"path": "homepage", "message": "Invalid input"}]),
}


def test_same_field_failures_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, (fields, errors) in SAME_FIELD_PROBES.items():
        manifest = {"name": "demo", "version": "1.0.0", "description": "d", **fields}
        result, parity = _parity(monkeypatch, _plugin(tmp_path / name, manifest), OracleClaude(errors, []))

        assert parity["skillevaluator_verdict"] == "failed", name
        assert parity["fields"]["claude_only"] == [], (name, parity["fields"])
        assert parity["fields"]["skillevaluator_only"] == [], (name, parity["fields"])
        assert parity["agree"] is True, name
        assert not [f for f in result.findings if f.check_name == "claude_validate_disagreement"], name


def test_mcp_findings_outside_the_manifest_are_not_manifest_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bad server in .mcp.json is not a plugin.json field, so it does not cover a Claude Code mcpServers error.
    root = _plugin(tmp_path, {"name": "demo", "version": "1.0.0", "description": "d"})
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"x": 5}}), encoding="utf-8")
    oracle = OracleClaude([{"path": "mcpServers", "message": "Invalid input"}], [])
    _result, parity = _parity(monkeypatch, root, oracle)

    assert parity["fields"]["skillevaluator_error_fields"] == []
    assert parity["fields"]["claude_only"] == ["mcpServers"]
