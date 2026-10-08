# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for native launch rewriting and the Codex, OpenCode, and Hermes adapters.

Covers review findings for the launch regexes (CodeQL py/redos and
setup commands that looked like the launch), Codex and wrapper MCP TOML,
OpenCode built-in agent names and tool limits, Hermes labeling and provider
routing, and Cursor rule scope. Harbor agents run against a recording fake
environment; nothing starts Docker, a model, or a harness CLI.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

import skillevaluator
from skillevaluator.plugin_components import parse_markdown
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.plugin_native import (
    HARNESS_ADAPTERS,
    SETUP_SCRIPT,
    NativePluginSource,
    NativeTextComponent,
    normalize_census,
    resolve_plugin_load,
)

_SRC = str(Path(skillevaluator.__file__).resolve().parents[1])


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
class _Result:
    return_code = 0
    stdout = ""
    stderr = ""


class _RecordingEnv:
    """A Harbor environment that records every command and runs nothing."""

    default_user = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def exec(self, command: str, **_kwargs: Any) -> _Result:
        self.calls.append(command)
        return _Result()

    async def upload_file(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _harbor_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    for name in (
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "CODEX_AUTH_JSON_PATH",
        "CODEX_FORCE_AUTH_JSON",
        "CLAUDE_FORCE_OAUTH",
        "CLAUDE_CODE_USE_BEDROCK",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _source(tmp_path: Path, **fields: Any) -> NativePluginSource:
    return NativePluginSource(
        plugin_name="demo",
        description="Demo plugin",
        contained=True,
        plugin_root=tmp_path,
        manifest={"name": "demo"},
        manifest_rel=".claude-plugin/plugin.json",
        **fields,
    )


def _agent_text(name: str, frontmatter: str) -> NativeTextComponent:
    return NativeTextComponent("agent", name, f"agents/{name}.md", f"---\n{frontmatter}\n---\nYou are {name}.\n")


def _launch_env(root: Path, **env: str) -> dict[str, str]:
    """The env the harness launch sets, pointed at the bundle re-rooted under ``root``.

    The OpenCode census only lists a config-backed component when the launch
    env names the staged bundle, the way Harbor's OpenCode launch does.
    """
    container = root / "skilleval"
    return {
        "OPENCODE_CONFIG": f"{container}/native/opencode/opencode.json",
        "OPENCODE_CONFIG_DIR": f"{container}/native/opencode/config",
        **env,
    }


def _run_census(bundle: Any, root: Path, env: dict[str, str]) -> dict[str, Any]:
    """Write the bundle into a fake container layout, run its setup.sh, and return the census."""
    container = root / "skilleval"
    for rel, text in bundle.generated.items():
        target = container / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    logs = root / "logs" / "agent"
    logs.mkdir(parents=True)
    script = (container / "native" / "setup.sh").read_text(encoding="utf-8")
    script = script.replace("/skilleval/", f"{container}/").replace("/logs/agent", str(logs))
    (root / "setup.sh").write_text(script, encoding="utf-8")
    completed = subprocess.run(
        ["/bin/sh", str(root / "setup.sh")],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    census = normalize_census(json.loads((logs / "skilleval-load-census.json").read_text(encoding="utf-8")))
    assert census is not None
    return census


# --------------------------------------------------------------------------- #
# Launch rewriting: CodeQL py/redos and setup commands that look like launches #
# --------------------------------------------------------------------------- #
_LINEAR_CHILD = r"""
import asyncio, json, os, sys, time
from pathlib import Path
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from skillevaluator.tier3.harbor.native_agents import NativeHermes, _NativeHermesMixin

class R:
    return_code = 0; stdout = ""; stderr = ""

class Env:
    default_user = None
    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        return R()

payloads = [
    "hermes\t-" + "--\t-" * 40,  # the CodeQL alert's input shape
    ";hermes" + "  --a" * 40,
    "x;hermes" + " --a" * 40 + " x",
]
start = time.perf_counter()
for payload in payloads:
    _NativeHermesMixin._SKILLEVAL_LAUNCH_RE.search(payload)
    NativeHermes.__new__(NativeHermes).skilleval_native_command(payload, None)
# A remote MCP URL a plugin controls reaches Harbor's MCP heredoc before the launch.
url = "https://mcp.example.com/mcp?x=1&hermes" + "  --a" * 40
agent = NativeHermes(
    logs_dir=Path(sys.argv[1]),
    model_name="anthropic/claude-test",
    mcp_servers=[MCPServerConfig(name="docs", transport="streamable-http", url=url)],
)
try:
    asyncio.run(agent.run("Do it.", Env(), AgentContext()))
except Exception:
    pass
print(json.dumps({"elapsed": time.perf_counter() - start}))
"""


def test_hermes_launch_pattern_is_linear_on_the_codeql_input_and_a_plugin_url(tmp_path: Path) -> None:
    # The old pattern backtracked 2**n steps (n=40 never finished), inside
    # Harbor's event loop and with no outer deadline. Run it in a child process
    # so a regression fails on the timeout instead of hanging the suite.
    env = {**os.environ, "PYTHONPATH": _SRC, "ANTHROPIC_API_KEY": "sk-test-not-a-real-key", "HOME": str(tmp_path)}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _LINEAR_CHILD, str(tmp_path / "logs")],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the Hermes launch pattern backtracked exponentially (no result in 60 s)")
    assert completed.returncode == 0, completed.stderr[-2000:]
    elapsed = json.loads(completed.stdout.strip().splitlines()[-1])["elapsed"]
    assert elapsed < 2.0


def _hostile_servers(agent: str) -> list[Any]:
    from harbor.models.task.config import MCPServerConfig

    text = {
        "NativeClaudeCode": "claude --verbose mcp serve",
        "NativeCodex": "codex exec --json",
        "NativeOpenCode": "opencode --model=openai/x run",
    }
    if agent == "NativeHermes":
        return [
            MCPServerConfig(name="docs", transport="streamable-http", url="https://mcp.example.com/m?x=1&hermes chat")
        ]
    return [MCPServerConfig(name="helper", transport="stdio", command="sh", args=["-c", f"echo {text[agent]}"])]


@pytest.mark.parametrize(
    ("agent", "model", "launch_marker"),
    [
        # Each marker is text only Harbor's launch command has, whatever the rewrite did.
        ("NativeClaudeCode", "anthropic/claude-test", "--output-format=stream-json"),
        ("NativeCodex", "openai/gpt-test", "--dangerously-bypass-approvals-and-sandbox"),
        ("NativeOpenCode", "openai/gpt-test", "run --format=json"),
        ("NativeHermes", "anthropic/claude-test", 'chat -q "$HARBOR_INSTRUCTION"'),
    ],
)
def test_native_setup_runs_on_harbors_real_launch_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent: str, model: str, launch_marker: str
) -> None:
    """Harbor's own run() with task MCP text that contains the launch words: only the launch is rewritten."""
    from harbor.models.agent.context import AgentContext

    from skillevaluator.tier3.harbor import native_agents

    _harbor_env(monkeypatch, tmp_path)
    cls = getattr(native_agents, agent)
    instance = cls(logs_dir=tmp_path / "logs", model_name=model, mcp_servers=_hostile_servers(agent))
    environment = _RecordingEnv()
    asyncio.run(instance.run("Do the task.", environment, AgentContext()))
    with_setup = [index for index, call in enumerate(environment.calls) if SETUP_SCRIPT in call]
    launches = [index for index, call in enumerate(environment.calls) if launch_marker in call]
    assert len(launches) == 1, environment.calls
    assert with_setup == launches
    with_plugin_dir = [index for index, call in enumerate(environment.calls) if "--plugin-dir" in call]
    assert with_plugin_dir == (launches if agent == "NativeClaudeCode" else [])


def test_a_run_whose_launch_never_appears_fails_instead_of_running_without_the_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import re

    from harbor.models.agent.context import AgentContext

    from skillevaluator.tier3.harbor import native_agents

    _harbor_env(monkeypatch, tmp_path)
    instance = native_agents.NativeCodex(logs_dir=tmp_path / "logs", model_name="openai/gpt-test")
    # Simulate a Harbor upgrade that changes the launch text.
    instance._SKILLEVAL_LAUNCH_RE = re.compile(r"codex[ \t]+run-something-else\b")
    environment = _RecordingEnv()
    with pytest.raises(native_agents.NativeLaunchError, match="did not find the Harbor launch"):
        asyncio.run(instance.run("Do the task.", environment, AgentContext()))
    assert not any(SETUP_SCRIPT in call for call in environment.calls)


def test_a_second_launch_in_one_run_fails_loudly() -> None:
    from skillevaluator.tier3.harbor import native_agents

    agent = native_agents.NativeOpenCode.__new__(native_agents.NativeOpenCode)
    launch = ". ~/.nvm/nvm.sh; opencode --model=openai/m run --format=json -- 'x'"
    first, env = agent.skilleval_native_command(launch, None)
    assert first.startswith(f"/bin/sh {SETUP_SCRIPT}") and env and env["OPENCODE_CONFIG_DIR"]
    with pytest.raises(native_agents.NativeLaunchError, match="second launch"):
        agent.skilleval_native_command(launch, None)


# --------------------------------------------------------------------------- #
# TOML: Codex native config and the wrapper plugin MCP file                    #
# --------------------------------------------------------------------------- #
_ROCKET_ARGS = ["-y", "@scope/rocket-mcp@1.2.3", "--banner=\U0001f680 ready", "del\x7fchar", 'quote"and\\slash']


def test_codex_native_mcp_toml_keeps_emoji_and_del_valid(tmp_path: Path) -> None:
    # json.dumps wrote the emoji as a surrogate pair and left DEL raw; Codex then
    # refused config.toml and every with-plugin trial died at startup.
    source = _source(
        tmp_path,
        mcp_servers=(
            {"name": "rocket", "command": "npx", "args": list(_ROCKET_ARGS), "transport": "stdio"},
            {"name": "docs", "url": "https://mcp.example.com/mcp", "transport": "http"},
        ),
    )
    bundle = HARNESS_ADAPTERS["codex"].build(source)
    parsed = tomllib.loads(bundle.generated["native/codex/mcp_servers.toml"])
    assert parsed["mcp_servers"]["rocket"] == {"command": "npx", "args": _ROCKET_ARGS}
    assert parsed["mcp_servers"]["docs"] == {"url": "https://mcp.example.com/mcp"}


def test_opencode_native_remote_mcp_disables_oauth(tmp_path: Path) -> None:
    # Like Harbor and the wrapper arm: without oauth false, a 401 starts OpenCode's
    # OAuth discovery and client registration, which a headless trial cannot finish.
    source = _source(
        tmp_path,
        mcp_servers=(
            {"name": "docs", "url": "https://mcp.example.com/mcp", "transport": "http"},
            {"name": "legacy", "url": "https://mcp.example.com/sse", "transport": "sse"},
            {"name": "local", "command": "npx", "args": ["-y", "@scope/local-mcp@1.0.0"], "transport": "stdio"},
        ),
    )
    bundle = HARNESS_ADAPTERS["opencode"].build(source)
    mcp = json.loads(bundle.generated["native/opencode/opencode.json"])["mcp"]
    assert mcp["docs"] == {"type": "remote", "url": "https://mcp.example.com/mcp", "enabled": True, "oauth": False}
    assert mcp["legacy"]["oauth"] is False
    assert "oauth" not in mcp["local"]


def test_wrapper_plugin_mcp_toml_round_trips_through_the_task_loader(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import _load_mcp_servers
    from skillevaluator.tier3.plugin_eval import PLUGIN_MCP_SERVERS_FILENAME, _write_plugin_mcp_servers_toml

    skill = tmp_path / "demo-plugin-eval"
    _write_plugin_mcp_servers_toml(
        skill / "evals", [{"name": "rocket", "command": "npx", "args": list(_ROCKET_ARGS), "transport": "stdio"}]
    )
    servers = _load_mcp_servers(skill, PLUGIN_MCP_SERVERS_FILENAME)
    assert servers == [{"name": "rocket", "command": "npx", "transport": "stdio", "args": _ROCKET_ARGS}]


def test_a_broken_plugin_mcp_file_fails_instead_of_dropping_every_plugin_server(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import _load_mcp_servers

    skill = tmp_path / "demo-plugin-eval"
    environment = skill / "evals" / "environment"
    environment.mkdir(parents=True)
    broken = '[[mcp_servers]]\nname = "rocket"\nargs = ["\\ud83d\\ude80"]\ncommand = "npx"\n'
    (environment / "plugin_mcp_servers.toml").write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match=r"plugin_mcp_servers\.toml"):
        _load_mcp_servers(skill, "plugin_mcp_servers.toml")
    # The shared task-environment file keeps its lenient behavior.
    (environment / "mcp_servers.toml").write_text(broken, encoding="utf-8")
    assert _load_mcp_servers(skill) == []


# --------------------------------------------------------------------------- #
# OpenCode: built-in agent names and subagent tool limits                      #
# --------------------------------------------------------------------------- #
def _opencode_agent(bundle: Any, stem: str) -> dict[str, Any]:
    return parse_markdown(bundle.generated[f"native/opencode/config/agents/{stem}.md"]).frontmatter


def test_opencode_plugin_agents_named_like_built_ins_do_not_replace_them(tmp_path: Path) -> None:
    # A plugin agent named build was forced to mode: subagent, so `opencode run`
    # fell back to the read-only plan agent and the with-plugin arm could not edit.
    texts = tuple(_agent_text(name, f"description: {name} helper") for name in ("build", "General", "reviewer"))
    bundle = HARNESS_ADAPTERS["opencode"].build(_source(tmp_path, texts=texts))
    agents = sorted(rel.rsplit("/", 1)[-1] for rel in bundle.generated if "/config/agents/" in rel)
    assert agents == ["demo-General.md", "demo-build.md", "reviewer.md"]
    assert _opencode_agent(bundle, "demo-build")["mode"] == "subagent"
    census = _run_census(
        bundle, tmp_path / "container", _launch_env(tmp_path / "container", HOME=str(tmp_path / "home"))
    )
    evidence = {row["name"]: row["evidence"] for row in census["listed"] if row["type"] == "agent"}
    assert set(evidence) == {"build", "General", "reviewer"}
    assert "staged as demo-build: build is an OpenCode built-in agent" in evidence["build"]
    assert "built-in" not in evidence["reviewer"]


def test_opencode_subagent_tool_lists_become_permission_rules(tmp_path: Path) -> None:
    texts = (
        _agent_text("reader", "description: Reads\ntools: Read, Grep, Glob"),
        _agent_text("git-reader", "description: Git\ntools:\n  - Read\n  - Bash(git status:*)"),
        _agent_text("writer", "description: Writes\ntools: Read, Write, NotebookEdit, mcp__tracker__search"),
        _agent_text("no-shell", "description: No shell\ndisallowedTools: Bash, WebFetch"),
        _agent_text("free", "description: Everything"),
    )
    bundle = HARNESS_ADAPTERS["opencode"].build(_source(tmp_path, texts=texts))
    assert _opencode_agent(bundle, "reader")["permission"] == {
        "*": "deny",
        "read": "allow",
        "grep": "allow",
        "glob": "allow",
    }
    assert _opencode_agent(bundle, "git-reader")["permission"] == {
        "*": "deny",
        "read": "allow",
        "bash": {"*": "deny", "git status": "allow", "git status *": "allow"},
    }
    # Unmapped tools stay denied (access only shrinks), and the census says so.
    assert _opencode_agent(bundle, "writer")["permission"] == {"*": "deny", "read": "allow", "edit": "allow"}
    assert _opencode_agent(bundle, "no-shell")["permission"] == {"bash": "deny", "webfetch": "deny"}
    assert "permission" not in _opencode_agent(bundle, "free")
    census = _run_census(
        bundle, tmp_path / "container", _launch_env(tmp_path / "container", HOME=str(tmp_path / "home"))
    )
    evidence = {row["name"]: row["evidence"] for row in census["listed"] if row["type"] == "agent"}
    assert "no OpenCode permission for NotebookEdit, mcp__tracker__search" in evidence["writer"]


def test_opencode_does_not_stage_a_subagent_whose_denies_it_cannot_enforce(tmp_path: Path) -> None:
    texts = (_agent_text("careful", "description: Careful\ndisallowedTools: mcp__tracker__delete"),)
    bundle = HARNESS_ADAPTERS["opencode"].build(_source(tmp_path, texts=texts))
    assert not any("/config/agents/" in rel for rel in bundle.generated)
    census = _run_census(bundle, tmp_path / "container", {"HOME": str(tmp_path / "home")})
    [row] = [row for row in census["not_loaded"] if row["type"] == "agent"]
    assert row["name"] == "careful" and "cannot deny mcp__tracker__delete" in row["reason"]


# --------------------------------------------------------------------------- #
# Hermes: labeled as the wrapper it is                                         #
# --------------------------------------------------------------------------- #
def test_hermes_is_labeled_wrapper_and_auto_picks_the_wrapper() -> None:
    # The hermes-home task is the wrapper task plus a census file, so auto must
    # not report a native lift for it.
    auto = resolve_plugin_load("auto", ["hermes", "codex"], env_mode="docker")
    assert auto["hermes"].mode == "wrapper"
    assert auto["hermes"].reason.startswith("auto: hermes has no native plugin path yet")
    assert auto["codex"].mode == "native"
    explicit = resolve_plugin_load("native", ["hermes"], env_mode="docker")["hermes"]
    assert explicit.adapter == "hermes-home"
    assert {explicit.components[kind] for kind in ("skill", "rule", "mcp")} == {"wrapper"}
    assert "native" not in HARNESS_ADAPTERS["hermes"].component_modes().values()


# --------------------------------------------------------------------------- #
# Rules: only always-on rules go to the always-on channels                     #
# --------------------------------------------------------------------------- #
_RULES = (
    ("ts-only.mdc", '---\ndescription: TypeScript style\nglobs: "**/*.ts"\nalwaysApply: false\n---\nUse strict TS.'),
    ("ask-me.mdc", "---\ndescription: Ask for review\nalwaysApply: false\n---\nAsk before merging."),
    ("always.mdc", "---\ndescription: Always\nalwaysApply: true\n---\nCite ticket IDs."),
    ("plain.md", "Prefer small commits."),
)


@pytest.mark.parametrize(
    ("agent", "rules_file"), [("codex", "native/codex/AGENTS.md"), ("opencode", "native/opencode/AGENTS.md")]
)
def test_scoped_cursor_rules_are_not_staged_as_always_on_rules(tmp_path: Path, agent: str, rules_file: str) -> None:
    bundle = HARNESS_ADAPTERS[agent].build(_source(tmp_path, rules=_RULES))
    text = bundle.generated[rules_file]
    assert "Cite ticket IDs." in text and "Prefer small commits." in text
    assert "Use strict TS." not in text and "Ask before merging." not in text
    assert "alwaysApply" not in text and "globs" not in text
    census = _run_census(
        bundle,
        tmp_path / "container",
        _launch_env(tmp_path / "container", HOME=str(tmp_path / "home"), CODEX_HOME=str(tmp_path / "codex-home")),
    )
    reasons = {row["name"]: row["reason"] for row in census["not_loaded"] if row["type"] == "rule"}
    assert set(reasons) == {"ts-only.mdc", "ask-me.mdc"}
    assert reasons["ts-only.mdc"].startswith("scoped rule (applies only to files matching **/*.ts)")
    assert reasons["ask-me.mdc"].startswith("agent-requested rule")


def test_claude_code_scopes_cursor_rules_by_paths_and_skips_agent_requested_ones(tmp_path: Path) -> None:
    # Claude Code loads a user rule on every task unless its frontmatter has paths.
    rules = (*_RULES, ("multi.mdc", '---\nglobs: "src/**/*.ts, test/**"\n---\nScoped twice.'))
    bundle = HARNESS_ADAPTERS["claude-code"].build(_source(tmp_path, rules=rules))
    staged = {rel.rsplit("/", 1)[-1]: text for rel, text in bundle.generated.items() if "/claude-code/rules/" in rel}

    assert set(staged) == {"ts-only.mdc.md", "always.mdc.md", "plain.md", "multi.mdc.md"}
    scoped = parse_markdown(staged["ts-only.mdc.md"])
    assert scoped.frontmatter == {"paths": ["**/*.ts"]} and scoped.body.strip() == "Use strict TS."
    assert parse_markdown(staged["multi.mdc.md"]).frontmatter == {"paths": ["src/**/*.ts", "test/**"]}
    assert staged["always.mdc.md"] == "Cite ticket IDs.\n"
    assert staged["plain.md"] == "Prefer small commits.\n"
    assert not any("alwaysApply" in text or "globs" in text for text in staged.values())
    census = _run_census(
        bundle, tmp_path / "container", {"HOME": str(tmp_path / "home"), "CLAUDE_CONFIG_DIR": str(tmp_path / "cfg")}
    )
    reasons = {row["name"]: row["reason"] for row in census["not_loaded"] if row["type"] == "rule"}
    assert set(reasons) == {"ask-me.mdc"}
    assert reasons["ask-me.mdc"].startswith("agent-requested rule")


# --------------------------------------------------------------------------- #
# Runner: Hermes never gets an OpenAI credential                               #
# --------------------------------------------------------------------------- #
def _provider(name: str, base_url: str | None) -> ProviderConfig:
    return ProviderConfig(
        provider=name, model="vendor/model-x", api_key="provider-key", base_url=base_url, litellm_model="openai/m"
    )


@pytest.mark.parametrize(
    "provider",
    [
        _provider("openai-compatible", "https://gateway.example/v1"),
        _provider("openai", "https://api.openai.com/v1"),
        _provider("openai", None),
    ],
)
def test_hermes_with_an_openai_provider_is_refused_before_the_key_reaches_openrouter(provider: ProviderConfig) -> None:
    from skillevaluator.tier3.harbor import runner

    errors = runner._validate_agent_provider_credentials(
        provider, ["hermes"], {}, {"hermes": "CLI"}, env_mode="docker", agent_models={"hermes": "openai/vendor/x"}
    )
    assert errors and "openrouter.ai" in errors[0]
    assert runner._agent_credentials(provider=provider, agent="hermes", env_mode="docker") == {}
    # Codex and OpenCode keep their OpenAI-compatible routes.
    assert runner._agent_credentials(provider=provider, agent="codex", env_mode="docker")["OPENAI_API_KEY"]


# --------------------------------------------------------------------------- #
# Docs                                                                         #
# --------------------------------------------------------------------------- #
def test_docs_do_not_claim_codex_plugins_have_no_non_interactive_install() -> None:
    docs = (Path(_SRC).parent / "docs" / "plugin-evaluation.mdx").read_text(encoding="utf-8")
    assert "no documented non-interactive local install path" not in docs
    assert "codex plugin marketplace add" in docs and "codex plugin add" in docs
