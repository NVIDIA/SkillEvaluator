# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 config-key parity: stop_on_pass, base_image_mode, agent_runtime_preflight.

These keys are part of the supported ``evals/config.yml`` surface. Each test
drives the real engine entrypoint with a real config file and asserts that the
parsed values route into the corresponding engine behavior.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skillevaluator.cli import GRADING_MODE_CHOICE
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner, runtime_preflight

# The exact config reported by a Tier 3 user; it must parse and route end-to-end.
USER_CONFIG = """\
schema_version: 1
harbor:
  task_source: evals_json
  n_attempts: 1
  pass_threshold: 0.6
  stop_on_pass: false
grading:
  mode: aces_plus_custom
"""


def _provider() -> ProviderConfig:
    return ProviderConfig(
        provider="nv_build",
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )


def _run_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_yaml: str,
    **engine_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Run ``run_harbor_eval`` against a real config file with the Harbor edges mocked."""
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(config_yaml, encoding="utf-8")

    captured: dict[str, dict[str, Any]] = {"emit": {}, "pair": {}, "collect": {}, "base_image": {}}

    monkeypatch.setattr(runner, "resolve_llm_provider", _provider)
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    def emit(_skill: Path, target: Path, **kwargs: Any) -> list[Path]:
        task = target / "case-001"
        task.mkdir(parents=True)
        if kwargs.get("with_skill"):
            captured["emit"] = kwargs
        return [task]

    def pair(**kwargs: Any) -> list[str]:
        captured["pair"] = kwargs
        return []

    def collect(**kwargs: Any) -> dict[str, Any]:
        captured["collect"] = kwargs
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}}

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_run_agent_pair", pair)
    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected_provider: runtime_preflight.ModelProbeResult(
            True,
            selected_provider.provider,
            selected_provider.model,
            f"model {selected_provider.model} is available",
        ),
    )
    agents = engine_kwargs.pop("agents", ["opencode"])
    result = runner.run_harbor_eval(
        skill,
        agents,
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=engine_kwargs.pop("agent_runtime_preflight", False),
        **engine_kwargs,
    )
    return result, captured


def test_user_config_parses_and_routes_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The reported user config must run: stop_on_pass key + legacy grading mode."""
    result, captured = _run_engine(monkeypatch, tmp_path, USER_CONFIG)

    assert "error" not in result
    policy = result["attempt_policy"]
    assert policy["max_attempts"] == 1
    assert policy["pass_threshold"] == 0.6
    assert policy["stop_on_pass"] is False
    assert result["run_config"]["grading"]["mode"] == "default_plus_custom"
    assert result["run_config"]["task_source"] == "evals_json"
    assert result["run_config"]["harbor"]["stop_on_pass"] is False
    assert captured["emit"]["grading_mode"] == "default_plus_custom"
    assert captured["pair"]["stop_on_pass"] is False
    assert captured["pair"]["pass_threshold"] == 0.6
    assert captured["pair"]["n_attempts"] == 1
    assert captured["collect"]["stop_on_pass"] is False
    assert captured["collect"]["pass_threshold"] == 0.6
    assert captured["collect"]["expected_trials"] == 1


def test_stop_on_pass_config_key_routes_sequential_attempt_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = """\
schema_version: 1
harbor:
  task_source: evals_json
  n_attempts: 3
  stop_on_pass: true
"""
    result, captured = _run_engine(monkeypatch, tmp_path, config)

    assert "error" not in result
    assert result["attempt_policy"]["stop_on_pass"] is True
    assert captured["pair"]["stop_on_pass"] is True
    assert captured["pair"]["task_names"] == ["case-001"]
    assert captured["collect"]["stop_on_pass"] is True
    # Early-stopped cases legitimately use fewer trials than the maximum, so
    # the exact-count trial validation is replaced by per-case coverage.
    assert captured["collect"]["expected_trials"] is None


def test_stop_on_pass_routes_nvidia_build_codex_through_bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = """\
schema_version: 1
harbor:
  task_source: evals_json
  n_attempts: 2
  stop_on_pass: true
"""

    result, captured = _run_engine(monkeypatch, tmp_path, config, agents=["codex"])

    assert "error" not in result
    assert captured["pair"]["stop_on_pass"] is True
    assert captured["pair"]["agent_import_path"] == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"
    )


def test_cli_stop_on_pass_overrides_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = """\
schema_version: 1
harbor:
  task_source: evals_json
  n_attempts: 3
  stop_on_pass: true
"""
    result, captured = _run_engine(monkeypatch, tmp_path, config, stop_on_pass=False)

    assert "error" not in result
    assert result["attempt_policy"]["stop_on_pass"] is False
    assert captured["pair"]["stop_on_pass"] is False


def test_stop_on_pass_requires_multiple_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = """\
schema_version: 1
harbor:
  task_source: evals_json
  n_attempts: 1
  stop_on_pass: true
"""
    result, _captured = _run_engine(monkeypatch, tmp_path, config)

    assert result["error"] == ["stop_on_pass requires n_attempts > 1"]


def _observe_preflight(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def observed(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs["agent"])
        return SimpleNamespace(ok=True, detail="")

    monkeypatch.setattr(runtime_preflight, "run_agent_runtime_preflight", observed)
    return calls


def test_agent_runtime_preflight_config_key_disables_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = """\
schema_version: 1
harbor:
  task_source: evals_json
  agent_runtime_preflight: false
"""
    calls = _observe_preflight(monkeypatch)

    result, _captured = _run_engine(monkeypatch, tmp_path, config, agent_runtime_preflight=None)

    assert "error" not in result
    assert calls == []


def test_agent_runtime_preflight_defaults_to_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _observe_preflight(monkeypatch)

    result, _captured = _run_engine(monkeypatch, tmp_path, USER_CONFIG, agent_runtime_preflight=None)

    assert "error" not in result
    assert calls == ["opencode"]


def test_agent_runtime_preflight_cli_value_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = """\
schema_version: 1
harbor:
  task_source: evals_json
  agent_runtime_preflight: true
"""
    calls = _observe_preflight(monkeypatch)

    result, _captured = _run_engine(monkeypatch, tmp_path, config, agent_runtime_preflight=False)

    assert "error" not in result
    assert calls == []


@pytest.mark.parametrize(
    ("mode", "expected_force_rebuild"),
    [("reuse", False), ("rebuild", True)],
)
def test_base_image_mode_config_key_routes_to_base_image_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_force_rebuild: bool,
) -> None:
    config = f"""\
schema_version: 1
harbor:
  task_source: evals_json
  base_image_mode: {mode}
"""
    builds: list[dict[str, Any]] = []

    def build(_skill: Path, _reference: Path | None, **kwargs: Any) -> str:
        builds.append(kwargs)
        return "skillevaluator-base:test"

    monkeypatch.setattr(runner, "build_eval_base_image", build)

    result, captured = _run_engine(monkeypatch, tmp_path, config)

    assert "error" not in result
    assert len(builds) == 1
    assert builds[0]["force_rebuild"] is expected_force_rebuild
    assert captured["emit"]["base_image"] == "skillevaluator-base:test"
    assert result["run_config"]["harbor"]["base_image_mode"] == mode


def test_base_image_mode_defaults_to_self_contained_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builds: list[dict[str, Any]] = []
    monkeypatch.setattr(
        runner,
        "build_eval_base_image",
        lambda *_args, **kwargs: builds.append(kwargs) or "skillevaluator-base:test",
    )

    result, captured = _run_engine(monkeypatch, tmp_path, USER_CONFIG)

    assert "error" not in result
    assert builds == []
    assert captured["emit"]["base_image"] == ""


def test_cli_grading_mode_accepts_legacy_aliases() -> None:
    assert GRADING_MODE_CHOICE.convert("aces_plus_custom", None, None) == "default_plus_custom"
    assert GRADING_MODE_CHOICE.convert("aces_default", None, None) == "default"
    assert GRADING_MODE_CHOICE.convert("custom_only", None, None) == "custom_only"
