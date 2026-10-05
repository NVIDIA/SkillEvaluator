# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 4 at Tier 3: MCP staging gives the same answer as Tier 1, under the same policy.

Shapes follow the proof's check-04 cases: the skeptic's policy probes x1
(override down) and x7 (override up), a private host allowlist, edge-09 (a
runnable ``agent_plugin.yaml`` entry), x3 (a Codex plugin whose root
``.mcp.json`` only Claude Code loads), and edge-10 (a secret in the refusal
text). ``tier3 evaluate-plugin`` takes ``--policy`` like ``validate``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.validators.policy import resolve_policy

FAKE_TOKEN = "ghp_" + "FAKE" + "0" * 36


def _claude_plugin(root: Path, servers: dict[str, Any]) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    manifest = {"name": "demo", "version": "1.0.0", "description": "demo", "mcpServers": servers}
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "evals").mkdir()
    (root / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p", "expected_output": "o"}]))
    return root


def _policy(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _prepare(root: Path, tmp_path: Path, policy_path: Path | None = None) -> Any:
    extra = {"policy": resolve_policy(policy_path=policy_path)} if policy_path else {}
    return prepare_plugin_eval_package(root, stage_root=tmp_path / "stage", **extra)


def test_policy_override_down_lets_tier3_stage_what_tier1_accepts(tmp_path: Path) -> None:
    # Skeptic x1: a pipe inside one argv argument (a regex) is plain text, so Tier 3 stages it by default.
    plain = _claude_plugin(
        tmp_path / "plain",
        {"grep": {"command": "npx", "args": ["-y", "@scope/grep-server@1.2.3", "--pattern", "foo|bar"]}},
    )
    assert _prepare(plain, tmp_path / "plain-stage").runnable_mcp_servers == ("grep",)
    # Command substitution in an argument is CRITICAL by default; the policy lowers it.
    root = _claude_plugin(
        tmp_path / "p",
        {"grep": {"command": "npx", "args": ["-y", "@scope/grep-server@1.2.3", "--pattern", "$(cat pattern.txt)"]}},
    )
    with pytest.raises(ValueError, match="mcp_command_shell_metacharacters"):
        _prepare(root, tmp_path)
    policy = _policy(tmp_path, "severity_overrides:\n  MCP_DECLARATION.mcp_command_shell_metacharacters: low\n")
    assert _prepare(root, tmp_path, policy).runnable_mcp_servers == ("grep",)


def test_policy_override_up_blocks_tier3_like_tier1(tmp_path: Path) -> None:
    # Skeptic x7: an unpinned package is MEDIUM by default; the policy raises it to HIGH.
    root = _claude_plugin(tmp_path / "p", {"loose": {"command": "npx", "args": ["-y", "@scope/server"]}})
    assert _prepare(root, tmp_path).runnable_mcp_servers == ("loose",)
    policy = _policy(tmp_path, "severity_overrides:\n  MCP_DECLARATION.mcp_unpinned_package: high\n")
    with pytest.raises(ValueError, match="mcp_unpinned_package"):
        _prepare(root, tmp_path, policy)


def test_policy_private_hosts_reach_tier3_staging(tmp_path: Path) -> None:
    root = _claude_plugin(tmp_path / "p", {"intra": {"url": "https://10.0.0.5/mcp", "type": "http"}})
    raised = _policy(tmp_path, "severity_overrides:\n  MCP_DECLARATION.mcp_endpoint_private: high\n")
    with pytest.raises(ValueError, match="mcp_endpoint_private"):
        _prepare(root, tmp_path, raised)
    allowed = _policy(
        tmp_path,
        "severity_overrides:\n  MCP_DECLARATION.mcp_endpoint_private: high\nmcp:\n  allowed_private_hosts:\n"
        "    - 10.0.0.0/8\n",
    )
    assert _prepare(root, tmp_path, allowed).runnable_mcp_servers == ("intra",)


def test_agent_plugin_yaml_mcp_entries_take_only_name_and_provider(tmp_path: Path) -> None:
    # check-04 edge-09: Tier 1 rejects the runnable fields as schema errors; Tier 3 staged them.
    root = tmp_path / "b"
    (root / "evals").mkdir(parents=True)
    (root / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p"}]))
    (root / "agent_plugin.yaml").write_text(
        "name: bundle\nauthor:\n  email: dev@example.com\nmcp:\n"
        "  - name: fs\n    command: npx\n    args: ['-y', '@scope/fs@1.2.3']\n    transport: stdio\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"mcp\[0\].*name and provider.*args, command, transport"):
        _prepare(root, tmp_path)


P12_HEAD = (
    "name: c21-yaml\ndescription: Check 21 agent_plugin.yaml form.\n"
    "author: {name: SkillEvaluator verification, email: verification@example.com}\nmcp:\n"
    "  - name: tracker\n    provider: example-provider\n"
)


def _bundle(root: Path, manifest: str) -> Path:
    (root / "evals").mkdir(parents=True)
    (root / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p"}]))
    (root / "agent_plugin.yaml").write_text(manifest, encoding="utf-8")
    return root


def test_check21_p12_url_entry_in_agent_plugin_yaml_is_refused_before_any_probe(tmp_path: Path) -> None:
    # check-21 p12 (expected behavior change): a provider entry plus a URL entry
    # used to stage and probe the URL server; Tier 1's manifest schema rejects
    # the URL entry, and Tier 3 now refuses it the same way.
    manifest = P12_HEAD + "  - name: deepwiki\n    url: https://mcp.example.com/mcp\n    transport: http\n"
    with pytest.raises(
        ValueError, match=r"mcp\[1\] is not a name and provider entry.*unsupported keys: transport, url"
    ):
        _prepare(_bundle(tmp_path / "p12", manifest), tmp_path)


def test_check21_p12_provider_only_manifest_stays_provider_only(tmp_path: Path) -> None:
    prepared = _prepare(_bundle(tmp_path / "p12", P12_HEAD), tmp_path)
    assert prepared.runnable_mcp_servers == ()
    assert prepared.provenance()["provider_only_mcp_servers"] == ["tracker"]


def test_codex_root_mcp_json_that_claude_code_loads_blocks_tier3_too(tmp_path: Path) -> None:
    # Skeptic x3: the Codex manifest points at mcp/servers.json, but Claude Code loads the
    # folder's root .mcp.json (--plugin-dir), and Tier 1 blocks on its shell server.
    root = tmp_path / "x3"
    (root / ".codex-plugin").mkdir(parents=True)
    (root / "mcp").mkdir()
    (root / "evals").mkdir()
    (root / ".codex-plugin" / "plugin.json").write_text(
        json.dumps({"name": "x3", "version": "1.0.0", "description": "x3", "mcpServers": "./mcp/servers.json"}),
        encoding="utf-8",
    )
    (root / "mcp" / "servers.json").write_text(
        json.dumps({"mcpServers": {"ok": {"command": "npx", "args": ["-y", "@scope/server@1.2.3"]}}}),
        encoding="utf-8",
    )
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"root-view": {"command": "sh", "args": ["-c", "id"]}}}), encoding="utf-8"
    )
    (root / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "prompt": "p"}]))
    with pytest.raises(ValueError, match=r"root-view.*mcp_command_dangerous_form"):
        _prepare(root, tmp_path)


def test_tier3_refusal_never_echoes_an_inline_secret(tmp_path: Path) -> None:
    # check-04 edge-10 (H9, Tier 3 half): the refusal printed the token itself.
    root = _claude_plugin(
        tmp_path / "p", {"leaky-arg": {"command": "npx", "args": ["-y", "@scope/s@1.0.0", FAKE_TOKEN]}}
    )
    with pytest.raises(ValueError) as refused:
        _prepare(root, tmp_path)
    message = str(refused.value)
    assert "mcp_command_inline_secret" in message
    assert FAKE_TOKEN not in message and "FAKE0000" not in message


def test_tier3_refusal_withholds_a_password_in_a_default_url(tmp_path: Path) -> None:
    # Skeptic x01: a password inside ``${VAR:-https://u:pw@host}``.
    root = _claude_plugin(
        tmp_path / "p", {"docs": {"url": "${DOCS_URL:-https://deploy:hunter2pw@docs.example.com/mcp}", "type": "http"}}
    )
    with pytest.raises(ValueError) as refused:
        _prepare(root, tmp_path)
    assert "hunter2pw" not in str(refused.value)


def test_evaluate_plugin_takes_a_policy_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # L25: --policy reaches staging (severity overrides) and the probe (private hosts) before any agent runs.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _claude_plugin(tmp_path / "p", {"loose": {"command": "npx", "args": ["-y", "@scope/server"]}})
    policy = _policy(tmp_path, "severity_overrides:\n  MCP_DECLARATION.mcp_unpinned_package: high\n")

    result = CliRunner().invoke(
        cli, ["tier3", "evaluate-plugin", str(root), "--policy", str(policy), "-a", "claude-code"]
    )

    assert result.exit_code == 1, result.output
    assert "mcp_unpinned_package" in result.output


def test_evaluate_plugin_policy_allowlist_reaches_the_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import mcp_proof

    seen: dict[str, Any] = {}

    def fake_probe(targets: Any, *, allowed_private_hosts: Any = (), expand_env: Any = ()) -> dict[str, Any]:
        seen["hosts"] = tuple(allowed_private_hosts)
        raise RuntimeError("stop before the run")

    monkeypatch.setattr(mcp_proof, "probe_mcp_servers", fake_probe)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = _claude_plugin(tmp_path / "p", {"intra": {"url": "https://10.0.0.5/mcp", "type": "http"}})
    policy = _policy(tmp_path, "mcp:\n  allowed_private_hosts:\n    - 10.0.0.0/8\n")

    result = CliRunner().invoke(
        cli, ["tier3", "evaluate-plugin", str(root), "--probe-mcp", "--policy", str(policy), "-a", "claude-code"]
    )

    assert "stop before the run" in result.output
    assert seen["hosts"] == ("10.0.0.0/8",)
