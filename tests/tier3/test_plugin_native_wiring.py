# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native plugin loading wiring: Harbor wrappers, runner arms, CLI flags, Hermes, and reports."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import (
    HARBOR_AGENTS,
    HARBOR_AGENTS_EXPERIMENTAL,
    HARBOR_AGENTS_SUPPORTED,
    LOCAL_HARBOR_AGENTS,
    runner,
)
from skillevaluator.tier3.plugin_native import (
    NATIVE_AGENT_IMPORT_PATHS,
    SETUP_SCRIPT,
    plugin_load_provenance,
    resolve_plugin_load,
)


# --------------------------------------------------------------------------- #
# Harbor agent wrappers                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("key", "path"), sorted(NATIVE_AGENT_IMPORT_PATHS.items(), key=str))
def test_every_native_import_path_resolves_to_a_wrapper_over_its_base(key: tuple[str, str | None], path: str) -> None:
    from skillevaluator.tier3.harbor.native_agents import _NativePluginLoadMixin

    agent, base = key
    module_name, class_name = path.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    assert issubclass(cls, _NativePluginLoadMixin)
    assert cls.name() == agent
    if base is not None:
        base_module, base_class = base.split(":")
        assert issubclass(cls, getattr(importlib.import_module(base_module), base_class))


def _instance(class_name: str) -> Any:
    from skillevaluator.tier3.harbor import native_agents

    cls = getattr(native_agents, class_name)
    return cls.__new__(cls)


def test_claude_code_launch_gets_setup_and_plugin_dir_once() -> None:
    agent = _instance("NativeClaudeCode")
    launch = (
        'export PATH="$HOME/.local/bin:$PATH"; claude --verbose --output-format=stream-json '
        "--permission-mode=bypassPermissions --print -- 'do claude --verbose things' 2>&1 </dev/null | tee x"
    )
    command, env = agent.skilleval_native_command(launch, {"CLAUDE_CONFIG_DIR": "/logs/agent/sessions"})
    assert command.startswith(f"/bin/sh {SETUP_SCRIPT} </dev/null >/dev/null 2>&1 || true; ")
    assert "claude --plugin-dir /skilleval/native/claude-code/plugin --verbose" in command
    # The prompt after `--` is never rewritten, and no bypass flag is added.
    assert command.endswith("--print -- 'do claude --verbose things' 2>&1 </dev/null | tee x")
    assert command.count("bypassPermissions") == launch.count("bypassPermissions")
    assert env == {"CLAUDE_CONFIG_DIR": "/logs/agent/sessions"}
    # A second launch in the same run would run without the plugin: fail loudly.
    from skillevaluator.tier3.harbor.native_agents import NativeLaunchError

    with pytest.raises(NativeLaunchError, match="second launch"):
        agent.skilleval_native_command(launch, None)


def test_setup_commands_are_left_alone_and_opencode_gets_config_env() -> None:
    codex = _instance("NativeCodex")
    setup = 'mkdir -p "$CODEX_HOME" /tmp/codex-secrets'
    assert codex.skilleval_native_command(setup, {"CODEX_HOME": "/tmp/codex-home"}) == (
        setup,
        {"CODEX_HOME": "/tmp/codex-home"},
    )
    command, _env = codex.skilleval_native_command("codex exec --skip-git-repo-check --model m -- 'x'", None)
    assert command.startswith(f"/bin/sh {SETUP_SCRIPT}")
    opencode = _instance("NativeOpenCode")
    command, env = opencode.skilleval_native_command(
        ". ~/.nvm/nvm.sh; opencode --model=openai/m run --format=json -- 'x'", {"OPENCODE_FAKE_VCS": "git"}
    )
    assert env == {
        "OPENCODE_FAKE_VCS": "git",
        "OPENCODE_CONFIG": "/skilleval/native/opencode/opencode.json",
        "OPENCODE_CONFIG_DIR": "/skilleval/native/opencode/config",
    }
    hermes = _instance("NativeHermes")
    # Harbor's own config-write and MCP-append commands come first; "chat" in the
    # model or server name must not trigger the setup before they run.
    from harbor.agents.installed.hermes import Hermes

    config_yaml = Hermes._build_config_yaml("openai/baichuan-inc/baichuan2-13b-chat")
    config_command = f"mkdir -p /tmp/hermes && cat > /tmp/hermes/config.yaml << 'EOF'\n{config_yaml}EOF"
    hermes.mcp_servers = [
        SimpleNamespace(name="team-chat", transport="streamable-http", url="https://mcp.example/chat")
    ]
    mcp_command = hermes._build_register_mcp_servers_command()
    assert "chat" in config_command and "team-chat" in mcp_command
    for setup_command in (config_command, mcp_command):
        assert hermes.skilleval_native_command(setup_command, None) == (setup_command, None)
    command, _env = hermes.skilleval_native_command(
        'export PATH="$HOME/.local/bin:$PATH" && hermes --yolo chat -q "$HARBOR_INSTRUCTION" -Q --model m', None
    )
    assert command.startswith(f"/bin/sh {SETUP_SCRIPT}")


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def test_runner_uses_the_native_wrapper_for_the_with_plugin_arm_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched: dict[str, str | None] = {}

    def fake_run(**kwargs):
        launched[str(kwargs["job_name"]).rsplit("-", 1)[-1]] = kwargs["agent_import_path"]
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run)
    errors = runner._run_agent_pair(
        skill_name="plugin",
        agent="codex",
        model="model",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "parts",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=3,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=1,
        agent_import_path="base:Agent",
        with_agent_import_path="native:Agent",
    )
    assert errors == []
    assert launched == {"with": "native:Agent", "without": "base:Agent", "sumofparts": "base:Agent"}


def test_census_plan_is_absent_in_wrapper_mode_and_declares_components_otherwise() -> None:
    decisions = resolve_plugin_load("wrapper", ["codex"], env_mode="docker")
    assert runner._plugin_load_census_plan("wrapper", decisions, {}, None) is None
    source = SimpleNamespace(
        member_skills=(Path("/x/alpha"),), rules=(("style.md", "x"),), mcp_servers=({"name": "tracker"},)
    )
    auto = resolve_plugin_load("auto", ["codex"], env_mode="local")
    plan = runner._plugin_load_census_plan("auto", auto, {}, source)
    assert plan == {
        "codex": {
            "mode": "wrapper",
            "declared": [
                {"type": "skill", "name": "alpha"},
                {"type": "rule", "name": "style.md"},
                {"type": "mcp", "name": "tracker"},
            ],
        }
    }
    from skillevaluator.tier3.plugin_native import NativeBundle

    bundle = NativeBundle(
        agent="codex",
        adapter="codex-home",
        components={"skill": "native"},
        declared=[{"type": "skill", "name": "alpha"}],
        hook_ids={"hooks/hooks.json": ["hooks/hooks.json#Stop[0].hooks[0]"]},
    )
    native = resolve_plugin_load("native", ["codex"], env_mode="docker")
    assert runner._plugin_load_census_plan("native", native, {"codex": SimpleNamespace(bundle=bundle)}, source) == {
        "codex": {
            "mode": "native",
            "declared": [{"type": "skill", "name": "alpha"}],
            "components": {"skill": "native"},
            "hook_ids": {"hooks/hooks.json": ["hooks/hooks.json#Stop[0].hooks[0]"]},
        }
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _prepared(package_path: Path, member: Path) -> SimpleNamespace:
    return SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(member,),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        native_source="snapshot",
        mcp_probe_targets=(),
        integration_evidence_error=lambda: None,
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )


@pytest.mark.parametrize("plugin_load", ["native", "auto", None])
def test_evaluate_plugin_passes_plugin_load_and_records_it_in_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plugin_load: str | None
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (tmp_path / "package").mkdir()
    (tmp_path / "member").mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured: dict[str, Any] = {}

    def fake_prepare(*_args, **kwargs):
        captured["prepare"] = kwargs
        return _prepared(tmp_path / "package", tmp_path / "member")

    requested = plugin_load or "wrapper"
    engine_result = {
        "run_dir": str(run_dir),
        "run_config": {
            "plugin_load": plugin_load_provenance(
                requested, resolve_plugin_load(requested, ["codex"], env_mode="docker")
            )
        },
    }
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", fake_prepare)
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.setdefault("options", options) and engine_result,
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.refresh_plugin_run_report", lambda *_a, **_k: None)
    written: dict[str, Any] = {}
    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.write_plugin_provenance",
        lambda _run_dir, provenance: written.update(provenance),
    )
    arguments = ["tier3", "evaluate-plugin", str(plugin), "--progress", "off"]
    if plugin_load:
        arguments += ["--plugin-load", plugin_load]
    result = CliRunner().invoke(cli_module.cli, arguments)
    assert result.exit_code == 0, result.output
    assert captured["prepare"]["plugin_load"] == requested
    assert captured["options"].plugin_load == requested
    assert captured["options"].native_plugin_source == "snapshot"
    assert written["plugin_load"]["requested"] == requested


def test_plugin_load_rejects_unknown_values() -> None:
    result = CliRunner().invoke(cli_module.cli, ["tier3", "evaluate-plugin", ".", "--plugin-load", "magic"])
    assert result.exit_code != 0
    assert "magic" in result.output


def test_validate_forwards_plugin_load_to_the_plugin_tier3_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    captured: dict[str, Any] = {}

    def fake_prepare(*_args, **kwargs):
        captured["prepare"] = kwargs
        return _prepared(tmp_path / "package", tmp_path / "member")

    (tmp_path / "package").mkdir()
    (tmp_path / "member").mkdir()
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", fake_prepare)
    monkeypatch.setattr(
        EvaluationService, "evaluate", lambda _self, options, **_kwargs: captured.setdefault("options", options) or {}
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", lambda *_a, **_k: None)
    cli_module._run_agent_eval_or_skip(
        plugin,
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
        plugin_load="auto",
    )
    assert captured["prepare"]["plugin_load"] == "auto"
    assert captured["options"].plugin_load == "auto"


def test_validate_exposes_the_plugin_load_flag() -> None:
    result = CliRunner().invoke(cli_module.cli, ["validate", "--help"])
    assert "--plugin-load" in result.output


# --------------------------------------------------------------------------- #
# Hermes as a supported agent                                                  #
# --------------------------------------------------------------------------- #
def _provider(name: str, model: str, base_url: str | None = None) -> ProviderConfig:
    prefix = "anthropic" if name == "anthropic" else "openai"
    return ProviderConfig(
        provider=name, model=model, api_key="provider-key", base_url=base_url, litellm_model=f"{prefix}/{model}"
    )


def test_hermes_is_experimental_for_containers_but_not_local_mode() -> None:
    assert "hermes" in HARBOR_AGENTS_EXPERIMENTAL
    assert "hermes" not in HARBOR_AGENTS_SUPPORTED
    assert "hermes" in HARBOR_AGENTS
    assert "hermes" not in LOCAL_HARBOR_AGENTS
    with pytest.raises(ValueError, match="does not support agent: hermes"):
        runner.build_harbor_run_command(dataset_path="d", agent="hermes", job_name="j", env_mode="local")
    command = runner.build_harbor_run_command(dataset_path="d", agent="hermes", job_name="j", env_mode="docker")
    assert command[command.index("-a") + 1] == "hermes"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (_provider("openai", "gpt-5.5"), "openai/gpt-5.5"),
        (_provider("openai-compatible", "vendor/model-x", "https://gateway.example/v1"), "openai/vendor/model-x"),
        (_provider("anthropic", "claude-opus-5"), "anthropic/claude-opus-5"),
    ],
)
def test_hermes_default_models_are_provider_qualified(provider: ProviderConfig, expected: str) -> None:
    model, source = runner._model_for_agent("hermes", cli_model=None, config_agents={}, provider=provider)
    assert (model, source) == (expected, "public provider default")
    # Defaults for the other agents are unchanged.
    assert (
        runner._model_for_agent("codex", cli_model=None, config_agents={}, provider=_provider("openai", "m"))[0] == "m"
    )


def test_hermes_credentials_and_provider_validation() -> None:
    openai = _provider("openai", "gpt-5.5", "https://api.openai.com/v1")
    # Hermes routes openai/MODEL to OpenRouter, so it never gets an OpenAI key.
    assert runner._agent_credentials(provider=openai, agent="hermes", env_mode="docker") == {}
    anthropic = _provider("anthropic", "claude-opus-5")
    assert runner._agent_credentials(provider=anthropic, agent="hermes", env_mode="docker") == {
        "ANTHROPIC_API_KEY": "provider-key"
    }
    ok = runner._validate_agent_provider_credentials(
        anthropic, ["hermes"], {}, {"hermes": "public provider default"}, agent_models={"hermes": "anthropic/m"}
    )
    assert ok == []
    refused = runner._validate_agent_provider_credentials(
        openai, ["hermes"], {}, {"hermes": "public provider default"}, agent_models={"hermes": "openai/gpt-5.5"}
    )
    assert refused and "openrouter.ai" in refused[0]
    mismatch = runner._validate_agent_provider_credentials(
        openai, ["hermes"], {}, {"hermes": "CLI"}, agent_models={"hermes": "anthropic/claude"}
    )
    assert mismatch and "provider-qualified" in mismatch[0]
    gateway = runner._validate_agent_provider_credentials(
        _provider("anthropic", "claude-opus-5", "https://proxy.example"),
        ["hermes"],
        {},
        {},
        agent_models={"hermes": "anthropic/claude-opus-5"},
    )
    assert gateway and "ANTHROPIC_BASE_URL" in gateway[0]
    nv_build = runner._validate_agent_provider_credentials(
        _provider("nv_build", "nvidia/model", "https://integrate.api.nvidia.com/v1"), ["hermes"], {}, {}
    )
    assert nv_build and "hermes" in nv_build[0]
    probe = runner._agent_provider_config(
        evaluator_provider=openai, agent="hermes", model="openai/gpt-5.5", credentials={}, env_mode="docker"
    )
    assert probe.model == "gpt-5.5"


# --------------------------------------------------------------------------- #
# Reports                                                                      #
# --------------------------------------------------------------------------- #
def _payload() -> dict[str, Any]:
    decisions = resolve_plugin_load("auto", ["claude-code"], env_mode="docker")
    return {
        "plugin_provenance": {
            "plugin_name": "release-helper",
            "plugin_load": plugin_load_provenance("auto", decisions),
            "load_census": {
                "claude-code": {
                    "agent": "claude-code",
                    "mode": "native",
                    "trials": 2,
                    "fallback_trials": 0,
                    "loaded": [{"type": "skill", "name": "alpha", "evidence": "claude-code system/init event: x"}],
                    "listed": [{"type": "hook", "name": "hooks/hooks.json", "evidence": "plugin-dir listing: /h"}],
                    "staged": [{"type": "rule", "name": "style.md", "evidence": "staged"}],
                    "not_loaded": [{"type": "lsp", "name": "go", "reason": "unsupported"}],
                }
            },
        }
    }


def test_plugin_load_view_and_renderers() -> None:
    from rich.console import Console

    from skillevaluator.reporting.cli import print_plugin_tier3
    from skillevaluator.reporting.markdown import MarkdownReporter
    from skillevaluator.reporting.plugin_sections import plugin_load_view, tier3_plugin_view

    view = plugin_load_view(_payload())
    assert view is not None and view["requested"] == "auto"
    row = view["agents"][0]
    assert (row["agent"], row["mode"], row["adapter"]) == ("claude-code", "native", "claude-code-plugin-dir")
    assert (row["confirmed"], row["listed"], row["staged_only"]) == (1, 1, 1)
    assert row["census_summary"] == (
        "1 confirmed by harness, 1 listed (files found, not confirmed), 1 staged only, 1 not loaded"
    )
    assert "hook" in row["native"] and "monitor" in row["unsupported"]
    assert plugin_load_view({"plugin_provenance": {"plugin_name": "x"}}) is None

    full = tier3_plugin_view(_payload())
    assert full is not None and full["plugin_load"] == view
    lines: list[str] = []
    MarkdownReporter._render_tier3_plugin(full, lines)
    text = "\n".join(lines)
    assert "### Plugin Loading" in text and "claude-code-plugin-dir" in text
    console = Console(record=True, width=200)
    print_plugin_tier3(full, console)
    assert "Plugin loading: requested auto" in console.export_text()


def test_plugin_load_view_reads_the_engine_result_without_provenance() -> None:
    from skillevaluator.reporting.plugin_sections import plugin_load_view

    decisions = resolve_plugin_load("native", ["codex"], env_mode="docker")
    engine = {
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions), "eval_target": {"kind": "plugin"}},
        "agents": {"codex": {"plugin_load_census": {"trials": 1, "loaded": [], "not_loaded": []}}},
    }
    view = plugin_load_view(engine)
    assert view is not None and view["agents"][0]["census"] is True


def test_html_report_template_renders_the_plugin_load_section() -> None:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from skillevaluator.reporting.plugin_sections import tier3_plugin_view

    templates = Path(cli_module.__file__).parent / "reporting" / "templates"
    env = Environment(loader=FileSystemLoader(str(templates)), autoescape=select_autoescape(["html", "j2"]))
    macro = env.get_template("plugin_sections.html.j2").module.tier3_plugin_load_section
    html = str(macro(tier3_plugin_view(_payload())))
    assert 'id="tier3-plugin-load"' in html and "claude-code-plugin-dir" in html
