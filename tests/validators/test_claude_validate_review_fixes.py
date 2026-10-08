# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review fixes for the opt-in `claude plugin validate` parity check.

* The child gets an allowlisted environment and a throwaway HOME.
* Only Claude Code plugins are compared; other formats are not applicable.
* The text report's messages are parsed, not just its section headers.
* Output with no JSON and no verdict leaves the check INCOMPLETE.
* Messages name files relative to the plugin root.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ExternalTool, ToolResult, Tools
from skillevaluator.validators.claude_plugin_validate import ClaudePluginValidateParity

_AP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"


class FakeClaude:
    def __init__(self, responses: list[ToolResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def is_available(self) -> bool:
        return True

    def get_install_hint(self) -> str:
        return "Install Claude Code"

    def run(self, args: list[str], **kwargs: Any) -> ToolResult:
        env = kwargs.get("env") or {}
        home = Path(env["HOME"]) if "HOME" in env else None
        self.calls.append({"args": list(args), "home_existed": bool(home and home.is_dir()), **kwargs})
        if not self.responses:
            raise AssertionError("claude was run more times than expected")
        return self.responses.pop(0)


def _write(root: Path, files: dict[str, Any]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _claude_plugin(root: Path) -> Path:
    return _write(root, {".claude-plugin/plugin.json": {"name": "demo", "version": "1.0.0", "description": "Demo"}})


def _json(payload: dict, exit_code: int) -> ToolResult:
    return ToolResult(True, json.dumps(payload), "", exit_code)


def _parity(result) -> dict[str, Any]:
    return result.metadata["plugin"]["validator_parity"]


# --- Child environment ------------------------------------------------------------

_FAKE_SECRETS = {
    "ACME_INFERENCE_KEY": "fake-inference-key",
    "AWS_ACCESS_KEY_ID": "AKIAFAKEFAKEFAKE",
    "AWS_SECRET_ACCESS_KEY": "fake-secret",
    "GITLAB_PAT_RO": "fake-pat",
    "GH_PAT": "fake-pat",
    "DATADOG_APP_KEY": "fake-app-key",
    "DATABASE_URL": "postgres://user:fake-pw@db/app",
    "MYSQL_PWD": "fake-pwd",
    "SSH_PRIVATE_KEY": "fake-private-key",
    "SIGNING_KEY": "fake-signing-key",
    "SESSION_ID": "fake-session",
    "MY_SERVICE_PASS": "fake-pass",
    "SENTRY_DSN": "https://fake@sentry.invalid/1",
    "OPENAI_KEY": "fake-openai",
    "ANTHROPIC_API_KEY": "fake-anthropic",
    "XDG_CONFIG_HOME": "/real/config",
    "CLAUDE_CONFIG_DIR": "/real/claude",
}
_KEPT = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TERM": "xterm",
    "NO_PROXY": "localhost,127.0.0.1",
    "SSL_CERT_FILE": "/etc/ssl/cert.pem",
}


def test_child_env_is_an_allowlist_with_a_throwaway_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {**_FAKE_SECRETS, **_KEPT}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxyuser:proxy-pw@proxy.invalid:3128")
    real_home = os.environ.get("HOME", "")
    fake = FakeClaude([_json({"success": True, "manifest": {"file": "plugin.json"}, "contents": []}, 0)])

    result = ClaudePluginValidateParity(fake).validate(_claude_plugin(tmp_path / "p"), skillevaluator_verdict="passed")

    [call] = fake.calls
    env = call["env"]
    assert call["replace_env"] is True
    leaked = sorted(set(env) & set(_FAKE_SECRETS) - {"CLAUDE_CONFIG_DIR"})
    assert leaked == []
    assert not any(value in env.values() for value in _FAKE_SECRETS.values())
    for name, value in _KEPT.items():
        assert env[name] == value
    assert env["HTTPS_PROXY"] == "http://proxy.invalid:3128"
    assert env["DISABLE_AUTOUPDATER"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    # A temporary HOME and Claude config directory that exist during the run and are gone after it.
    assert env["HOME"] != real_home
    assert Path(env["CLAUDE_CONFIG_DIR"]).parent == Path(env["HOME"])
    assert call["home_existed"] is True
    assert not Path(env["HOME"]).exists()
    assert Path(call["cwd"]) != Path(env["HOME"])
    assert _parity(result)["status"] == "compared"


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script as the fake claude binary")
def test_real_child_process_sees_only_the_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through ExternalTool: a fake `claude` records its environment and writes to HOME."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "child-env.txt"
    script = bin_dir / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'env > "{record}"\n'
        'echo state > "$HOME/.claude.json"\n'
        'echo \'{"success": true, "manifest": {"file": "plugin.json", "errors": [], "warnings": []}, '
        '"contents": []}\'\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    home = tmp_path / "real-home"
    home.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(home))
    for name, value in _FAKE_SECRETS.items():
        monkeypatch.setenv(name, value)

    tool = ExternalTool("Claude Code", "claude")
    assert Path(tool.path).resolve() == script.resolve()
    result = ClaudePluginValidateParity(tool).validate(_claude_plugin(tmp_path / "p"), skillevaluator_verdict="passed")

    child = dict(line.split("=", 1) for line in record.read_text(encoding="utf-8").splitlines() if "=" in line)
    assert sorted(set(child) & set(_FAKE_SECRETS) - {"CLAUDE_CONFIG_DIR"}) == []
    assert child["HOME"] != str(home)
    assert child["CLAUDE_CONFIG_DIR"].startswith(child["HOME"])
    assert list(home.iterdir()) == []  # nothing written into the caller's HOME
    assert _parity(result)["agree"] is True


_REAL_CLAUDE = (
    "#!/bin/sh\n"
    'echo state > "$HOME/.claude.json"\n'
    'echo \'{"success": true, "manifest": {"file": "plugin.json", "errors": [], "warnings": []}, "contents": []}\'\n'
)
# Shims shaped like Volta's and asdf's: they find the real claude through a
# location variable, or under HOME when it is unset.
_VOLTA_SHIM = '#!/bin/sh\nexec "${VOLTA_HOME:-$HOME/.volta}/tools/claude" "$@"\n'
_ASDF_SHIM = (
    "#!/bin/sh\n"
    'version=$(sed -n "s/^nodejs //p" "$HOME/.tool-versions" 2>/dev/null)\n'
    '[ -n "$version" ] || { echo "No version is set for command claude" >&2; exit 126; }\n'
    'exec "${ASDF_DATA_DIR:-$HOME/.asdf}/installs/nodejs/$version/bin/claude" "$@"\n'
)


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX shell scripts as the version-manager shim")
@pytest.mark.parametrize(
    ("shim", "real", "location_var", "pins"),
    [
        pytest.param(_VOLTA_SHIM, ".volta/tools/claude", None, None, id="volta-under-home"),
        pytest.param(_VOLTA_SHIM, "custom-volta/tools/claude", ("VOLTA_HOME", "custom-volta"), None, id="volta-home"),
        pytest.param(_ASDF_SHIM, ".asdf/installs/nodejs/22.1.0/bin/claude", None, "nodejs 22.1.0\n", id="asdf"),
    ],
)
def test_version_manager_shim_still_finds_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shim: str,
    real: str,
    location_var: tuple[str, str] | None,
    pins: str | None,
) -> None:
    """The throwaway HOME must not hide a claude that a version manager keeps under the caller's HOME."""
    home = tmp_path / "real-home"
    _executable(home / real, _REAL_CLAUDE)
    if pins is not None:
        (home / ".tool-versions").write_text(pins, encoding="utf-8")
    bin_dir = tmp_path / "shims"
    _executable(bin_dir / "claude", shim)
    for name in ("VOLTA_HOME", "ASDF_DIR", "ASDF_DATA_DIR", "MISE_DATA_DIR", "NVM_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    if location_var is not None:
        monkeypatch.setenv(location_var[0], str(home / location_var[1]))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("HOME", str(home))
    before = sorted(path.relative_to(home) for path in home.rglob("*"))

    result = ClaudePluginValidateParity(ExternalTool("Claude Code", "claude")).validate(
        _claude_plugin(tmp_path / "p"), skillevaluator_verdict="passed"
    )

    parity = _parity(result)
    assert parity["status"] == "compared", parity.get("reason")
    assert parity["agree"] is True
    assert not result.is_incomplete
    # The real claude wrote its state into the throwaway HOME, not the caller's.
    assert sorted(path.relative_to(home) for path in home.rglob("*")) == before


# --- Only Claude Code plugins are compared ---------------------------------------

_NO_MANIFEST = {
    "success": False,
    "manifest": {
        "file": "/abs/plugin",
        "errors": [{"path": "directory", "message": "No manifest found in directory. Expected .claude-plugin/..."}],
        "warnings": [],
    },
    "contents": [],
}


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({".codex-plugin/plugin.json": {"name": "demo", "version": "1.0.0"}}, id="codex"),
        pytest.param({".cursor-plugin/plugin.json": {"name": "demo"}}, id="cursor"),
        pytest.param({"plugin.json": {"$schema": _AP_SCHEMA, "name": "demo", "version": "1.0.0"}}, id="agent-plugins"),
        pytest.param(
            {
                "agent_plugin.yaml": "name: demo\nversion: 1.0.0\nskills: []\n",
                ".claude-plugin/plugin.json": {"name": "demo"},
            },
            id="bundle-reference-wins-precedence",
        ),
    ],
)
def test_other_plugin_formats_are_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, Any]
) -> None:
    fake = FakeClaude([_json(_NO_MANIFEST, 1)])  # what Claude Code says about a non-Claude root
    monkeypatch.setattr(Tools, "claude", fake)
    root = _write(tmp_path / "p", files)

    results = run_validation(root, checks="claude-validate", content_type=CONTENT_TYPE_PLUGIN)

    [parity_result] = [r for r in results if r.validator_name == "Claude Plugin Validate Parity"]
    parity = _parity(parity_result)
    assert fake.calls == []
    assert parity["status"] == "not_applicable"
    assert ".claude-plugin/plugin.json" in parity["reason"]
    assert parity_result.findings == []
    assert parity_result.metadata["skipped"] is True
    assert not parity_result.is_incomplete


def test_claude_finding_no_manifest_is_not_applicable(tmp_path: Path) -> None:
    """A vacuous pass (`manifest: null`, nothing validated) is not a parity agreement."""
    fake = FakeClaude([_json({"success": True, "strict": True, "manifest": None, "contents": []}, 0)])

    result = ClaudePluginValidateParity(fake).validate(_claude_plugin(tmp_path / "p"), skillevaluator_verdict="passed")

    parity = _parity(result)
    assert parity["status"] == "not_applicable"
    assert parity["agree"] is None
    assert "claude_verdict" not in parity
    assert result.findings == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"success": True}, id="verdict-only"),
        pytest.param({"success": True, "strict": True, "contents": []}, id="empty-contents"),
    ],
)
def test_json_report_without_a_manifest_key_is_incomplete(tmp_path: Path, payload: dict) -> None:
    """Only an explicit ``"manifest": null`` means "nothing to validate"; a missing key says nothing."""
    fake = FakeClaude([_json(payload, 0)])

    result = ClaudePluginValidateParity(fake).validate(_claude_plugin(tmp_path / "p"), skillevaluator_verdict="passed")

    parity = _parity(result)
    assert parity["status"] == "error"
    assert "no manifest" in parity["reason"]
    assert result.is_incomplete
    assert result.metadata.get("skipped") is not True


def test_json_report_without_a_manifest_key_still_compares_its_files(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "p")
    payload = {
        "success": False,
        "contents": [{"file": f"{root}/agents/a.md", "errors": [{"path": "name", "message": "Required"}]}],
    }
    fake = FakeClaude([_json(payload, 1)])

    parity = _parity(ClaudePluginValidateParity(fake).validate(root, skillevaluator_verdict="failed"))

    assert parity["status"] == "compared"
    assert parity["errors"] == ["agents/a.md: name: Required"]
    assert parity["agree"] is True


# --- Text report parsing ------------------------------------------------------------


def _text_report(root: Path) -> str:
    """The shape of Claude Code 2.1.284's `plugin validate --strict` text report."""
    base = os.path.realpath(root)
    return (
        f"Validating plugin manifest: {base}/.claude-plugin/plugin.json\n\n"
        "\u2718 Found 3 errors:\n\n"
        "  \u276f version: Invalid input\n"
        "  \u276f description: Invalid input\n"
        "  \u276f keywords: Invalid input\n\n"
        "\u26a0 Found 1 warning:\n\n"
        "  \u276f bogus: Unknown field 'bogus'. Claude Code ignores it at load time.\n\n"
        f"Validating agent: {base}/agents/bad.md\n\n"
        "\u26a0 Found 1 warning:\n\n"
        "  \u276f description: No description in frontmatter.\n\n"
        f"Validating command: {base}/commands/c.md\n\n"
        "\u2718 Found 1 error:\n\n"
        "  \u276f allowed-tools: allowed-tools must be a string or array of strings, got number.\n\n"
        "\u2718 Validation failed\n"
    )


def test_text_report_messages_are_parsed_under_their_sections(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "p")
    fake = FakeClaude(
        [
            ToolResult(True, "", "error: unknown option '--json'", 1),
            ToolResult(True, _text_report(root), "", 1),
        ]
    )

    result = ClaudePluginValidateParity(fake).validate(root, skillevaluator_verdict="passed")

    parity = _parity(result)
    assert parity["format"] == "text"
    assert parity["claude_verdict"] == "failed"
    assert parity["errors"] == [
        ".claude-plugin/plugin.json: version: Invalid input",
        ".claude-plugin/plugin.json: description: Invalid input",
        ".claude-plugin/plugin.json: keywords: Invalid input",
        "commands/c.md: allowed-tools: allowed-tools must be a string or array of strings, got number.",
    ]
    assert parity["warnings"] == [
        ".claude-plugin/plugin.json: bogus: Unknown field 'bogus'. Claude Code ignores it at load time.",
        "agents/bad.md: description: No description in frontmatter.",
    ]
    assert (parity["error_count"], parity["warning_count"]) == (4, 2)
    severities = [f.severity for f in result.findings if f.check_name != "claude_validate_disagreement"]
    assert severities.count(Severity.MEDIUM) == 4
    assert severities.count(Severity.LOW) == 2


def test_text_report_that_passes_with_warnings(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "p")
    report = (
        f"Validating plugin manifest: {os.path.realpath(root)}/.claude-plugin/plugin.json\n\n"
        "\u26a0 Found 1 warning:\n\n"
        "  \u276f author: No author information provided.\n\n"
        "\u2714 Validation passed with warnings\n"
    )
    fake = FakeClaude([ToolResult(True, "", "error: unknown option '--json'", 1), ToolResult(True, report, "", 0)])

    parity = _parity(ClaudePluginValidateParity(fake).validate(root, skillevaluator_verdict="passed"))

    assert parity["claude_verdict"] == "passed"
    assert parity["errors"] == []
    assert parity["warnings"] == [".claude-plugin/plugin.json: author: No author information provided."]


_NOTE = "types ./types.d.ts declares on $: nothing (no EngineInterface member)"


def _report_with_notes(root: Path) -> tuple[str, dict]:
    """Claude Code 2.1.284 prints a file's notes after a blank line, with no header of their own."""
    base = os.path.realpath(root)
    text = (
        f"Validating plugin manifest: {base}/.claude-plugin/plugin.json\n\n"
        "\u26a0 Found 1 warning:\n\n"
        "  \u276f bogus: Unknown field 'bogus'. Claude Code ignores it at load time.\n\n"
        f"  \u276f {_NOTE}\n\n"
        "  \u276f error-handling: a note whose text starts with error\n\n"
        f"Validating agent: {base}/agents/bad.md\n\n"
        "\u2718 Found 1 error:\n\n"
        "  \u276f name: Required\n\n"
        "\u2718 Validation failed (--strict treats warnings as errors)\n"
    )
    payload = {
        "success": False,
        "manifest": {
            "file": f"{base}/.claude-plugin/plugin.json",
            "errors": [],
            "warnings": [{"path": "bogus", "message": "Unknown field 'bogus'. Claude Code ignores it at load time."}],
            "notes": [_NOTE, "error-handling: a note whose text starts with error"],
        },
        "contents": [{"file": f"{base}/agents/bad.md", "errors": [{"path": "name", "message": "Required"}]}],
    }
    return text, payload


def test_text_report_notes_are_not_findings(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "p")
    text, payload = _report_with_notes(root)
    from_text = _parity(
        ClaudePluginValidateParity(
            FakeClaude([ToolResult(True, "", "error: unknown option '--json'", 1), ToolResult(True, text, "", 1)])
        ).validate(root, skillevaluator_verdict="failed")
    )
    from_json = _parity(ClaudePluginValidateParity(FakeClaude([_json(payload, 1)])).validate(root))

    assert from_text["format"] == "text"
    assert from_text["warnings"] == [
        ".claude-plugin/plugin.json: bogus: Unknown field 'bogus'. Claude Code ignores it at load time."
    ]
    assert from_text["errors"] == ["agents/bad.md: name: Required"]
    # Both report formats give the same findings.
    assert (from_text["errors"], from_text["warnings"]) == (from_json["errors"], from_json["warnings"])


# --- Unusable output is INCOMPLETE --------------------------------------------------


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize(
    "responses",
    [
        pytest.param(lambda code: [ToolResult(True, "Loading...\nsome banner\n", "", code)], id="json-accepted"),
        pytest.param(
            lambda code: [
                ToolResult(True, "", "error: unknown option '--json'", 1),
                ToolResult(True, "Loading...\nsome banner\n", "", code),
            ],
            id="text-fallback",
        ),
        pytest.param(
            lambda code: [_json({"contents": [], "manifest": {"file": "plugin.json"}}, code)], id="no-success"
        ),
    ],
)
def test_output_without_a_verdict_is_incomplete(tmp_path: Path, responses, exit_code: int) -> None:
    fake = FakeClaude(responses(exit_code))

    result = ClaudePluginValidateParity(fake).validate(_claude_plugin(tmp_path / "p"), skillevaluator_verdict="passed")

    parity = _parity(result)
    assert result.is_incomplete
    assert parity["status"] == "error"
    assert "claude_verdict" not in parity
    assert "agree" not in parity
    assert result.findings == []


# --- Messages use plugin-relative paths ---------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="needs a directory symlink")
def test_json_messages_use_plugin_relative_paths(tmp_path: Path) -> None:
    # The root is reached through a linked parent, so its lexical and resolved
    # spellings differ; Claude Code prints the resolved one.
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    root = _claude_plugin(tmp_path / "link" / "p")
    lexical = os.path.abspath(root)  # noqa: PTH100 - the lexical spelling, not resolved
    resolved = os.path.realpath(root)
    assert lexical != resolved
    long_message = "allowed-tools must be a string or array of strings, got number. " * 3
    payload = {
        "success": False,
        "manifest": {
            "file": f"{resolved}/.claude-plugin/plugin.json",
            "errors": [{"path": "name", "message": f"Invalid input in {resolved}/.claude-plugin/plugin.json"}],
            "warnings": [],
        },
        "contents": [
            {"file": f"{lexical}/commands/c.md", "errors": [{"path": "allowed-tools", "message": long_message}]},
            {"file": "/somewhere/else/outside.md", "warnings": ["Missing description"]},
        ],
    }
    fake = FakeClaude([_json(payload, 1)])

    parity = _parity(ClaudePluginValidateParity(fake).validate(root, skillevaluator_verdict="failed"))

    assert parity["errors"] == [
        ".claude-plugin/plugin.json: name: Invalid input in .claude-plugin/plugin.json",
        f"commands/c.md: allowed-tools: {' '.join(long_message.split())}",
    ]
    assert parity["warnings"] == ["outside.md: Missing description"]
    text = json.dumps(parity)
    assert lexical not in text
    assert resolved not in text
    assert "truncated" not in text
