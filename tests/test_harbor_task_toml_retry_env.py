# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Harbor task.toml verifier environment forwarding for retry settings."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor.utils.env import resolve_env_vars

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner, runtime_preflight
from skillevaluator.tier3.harbor.adapter import (
    _VERIFIER_PROVIDER_ENV_VARS,
    _VERIFIER_RETRY_ENV_VARS,
    _verifier_env_block,
    _verifier_env_vars,
)

if TYPE_CHECKING:
    import pytest

_EVAL_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_eval_template_module(tmp_path: Path):
    """Load the bundled Harbor verifier template as an isolated module."""
    module_name = f"harbor_template_retry_env_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_provider_env_vars_includes_retry_settings() -> None:
    """Verify _VERIFIER_PROVIDER_ENV_VARS allowlist contains all retry configuration keys."""
    expected_retry_vars = {
        "SKILL_EVAL_LLM_MAX_RETRIES",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
    }
    assert expected_retry_vars == _VERIFIER_RETRY_ENV_VARS
    assert expected_retry_vars.issubset(_VERIFIER_PROVIDER_ENV_VARS)


def test_verifier_env_block_forwards_retry_settings() -> None:
    """Verify _verifier_env_block generates task.toml env lines for staged retry variables."""
    runtime_env = {
        "SKILL_EVAL_LLM_MAX_RETRIES": "5",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "2.0",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "40.0",
        "UNRELATED_HOST_VAR": "secret",
    }
    env_vars = _verifier_env_vars(runtime_env)
    assert "SKILL_EVAL_LLM_MAX_RETRIES" in env_vars
    assert "SKILL_EVAL_LLM_RETRY_BASE_DELAY" in env_vars
    assert "SKILL_EVAL_LLM_RETRY_MAX_DELAY" in env_vars
    assert "UNRELATED_HOST_VAR" not in env_vars

    env_block = _verifier_env_block(runtime_env, indent="    ")
    assert '    SKILL_EVAL_LLM_MAX_RETRIES = "${SKILL_EVAL_LLM_MAX_RETRIES}"' in env_block
    assert '    SKILL_EVAL_LLM_RETRY_BASE_DELAY = "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"' in env_block
    assert '    SKILL_EVAL_LLM_RETRY_MAX_DELAY = "${SKILL_EVAL_LLM_RETRY_MAX_DELAY}"' in env_block
    assert "UNRELATED_HOST_VAR" not in env_block


def test_runner_forwards_host_retry_settings_to_staged_task_and_verifier_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify host SKILL_EVAL_LLM_* retry settings flow through runner staging, subprocess env, and eval.py."""
    skill = tmp_path / "demo-skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo skill.\n---\n# Demo\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        '[{"id": "case-001", "question": "Verify retry env forwarding.", "expected_answer": "ok", "files": []}]\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.1")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0.5")

    provider = ProviderConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4o-mini",
    )
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "evals_json", "base_image_mode": "disabled"}}, None),
    )
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
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

    captured_launch: dict[str, Any] = {}

    def capture_launch(**kwargs: Any) -> list[str]:
        captured_launch["with_skill"] = kwargs["with_skill"]
        captured_launch["run_env"] = dict(kwargs["run_env"])
        return []

    monkeypatch.setattr(runner, "_run_agent_pair", capture_launch)
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_kwargs: {"execution_status": "complete", "execution_errors": [], "metrics": [], "agents": {}},
    )

    def render_report(_skill_path: Path, output_dir: Path, **_kwargs: Any) -> Path:
        report = output_dir / "report.html"
        report.write_text("<html></html>\n", encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "render_agent_eval_html_report", render_report)

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "gpt-4o-mini"},
        output_dir=tmp_path / "results",
        env_mode="docker",
        skip_baseline=True,
        keep_harbor_jobs=True,
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert "with_skill" in captured_launch
    task_toml_path = captured_launch["with_skill"] / "case-001" / "task.toml"
    staged_task = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    verifier_toml_env = staged_task["verifier"]["env"]

    assert verifier_toml_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "${SKILL_EVAL_LLM_MAX_RETRIES}"
    assert verifier_toml_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"
    assert verifier_toml_env["SKILL_EVAL_LLM_RETRY_MAX_DELAY"] == "${SKILL_EVAL_LLM_RETRY_MAX_DELAY}"

    run_env = captured_launch["run_env"]
    assert run_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "0"
    assert run_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "0.1"
    assert run_env["SKILL_EVAL_LLM_RETRY_MAX_DELAY"] == "0.5"

    eval_module = _load_eval_template_module(tmp_path)
    with monkeypatch.context() as harbor_proc_env:
        harbor_proc_env.setattr("os.environ", dict(run_env))
        resolved_verifier_env = resolve_env_vars(verifier_toml_env)

    with monkeypatch.context() as verifier_proc_env:
        verifier_proc_env.setattr(eval_module.os, "environ", resolved_verifier_env)
        assert eval_module._resolve_eval_retry_config() == (0, 0.1, 0.5)
