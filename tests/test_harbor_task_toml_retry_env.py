# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Harbor task.toml verifier environment forwarding for retry settings."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor.utils.env import resolve_env_vars
from tests.conftest import load_harbor_eval_template

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner, runtime_preflight
from skillevaluator.tier3.harbor.adapter import (
    _VERIFIER_PROVIDER_ENV_VARS,
    _VERIFIER_RETRY_ENV_VARS,
    _verifier_env_block,
    _verifier_env_vars,
    stage_native_harbor_tasks,
)

if TYPE_CHECKING:
    import pytest
else:
    import pytest


def test_verifier_provider_env_vars_includes_retry_settings() -> None:
    """Verify _VERIFIER_PROVIDER_ENV_VARS allowlist contains all retry configuration keys."""
    expected_retry_vars = {
        "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC",
        "SKILL_EVAL_LLM_MAX_RETRIES",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
    }
    assert expected_retry_vars == _VERIFIER_RETRY_ENV_VARS
    assert expected_retry_vars.issubset(_VERIFIER_PROVIDER_ENV_VARS)


def test_verifier_env_block_forwards_retry_settings() -> None:
    """Verify _verifier_env_block generates task.toml env lines for staged retry variables."""
    runtime_env = {
        "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC": "120.0",
        "SKILL_EVAL_LLM_MAX_RETRIES": "5",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "2.0",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "40.0",
        "UNRELATED_HOST_VAR": "secret",
    }
    env_vars = _verifier_env_vars(runtime_env)
    assert "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC" in env_vars
    assert "SKILL_EVAL_LLM_MAX_RETRIES" in env_vars
    assert "SKILL_EVAL_LLM_RETRY_BASE_DELAY" in env_vars
    assert "SKILL_EVAL_LLM_RETRY_MAX_DELAY" in env_vars
    assert "UNRELATED_HOST_VAR" not in env_vars

    env_block = _verifier_env_block(runtime_env, indent="    ")
    assert '    SKILL_EVAL_LLM_JUDGE_BUDGET_SEC = "${SKILL_EVAL_LLM_JUDGE_BUDGET_SEC}"' in env_block
    assert '    SKILL_EVAL_LLM_MAX_RETRIES = "${SKILL_EVAL_LLM_MAX_RETRIES}"' in env_block
    assert '    SKILL_EVAL_LLM_RETRY_BASE_DELAY = "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"' in env_block
    assert '    SKILL_EVAL_LLM_RETRY_MAX_DELAY = "${SKILL_EVAL_LLM_RETRY_MAX_DELAY}"' in env_block
    assert "UNRELATED_HOST_VAR" not in env_block


def _stub_harbor_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    task_source: str,
    grading_mode: str = "default",
) -> dict[str, Any]:
    """Stub Harbor runner dependencies and capture the staged launch payload."""
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-openai-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4o-mini",
    )
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {
                "harbor": {"task_source": task_source, "base_image_mode": "disabled"},
                "grading": {"mode": grading_mode},
            },
            None,
        ),
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
    return captured_launch


def _resolve_verifier_retry_tuple(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verifier_toml_env: dict[str, str],
    run_env: dict[str, str],
) -> tuple[int, float, float]:
    """Resolve Harbor task.toml [verifier.env] placeholders against run_env and evaluate eval.py retry config."""
    eval_module = load_harbor_eval_template(f"harbor_template_retry_env_{tmp_path.name}")
    with monkeypatch.context() as harbor_proc_env:
        harbor_proc_env.setattr("os.environ", dict(run_env))
        resolved_verifier_env = resolve_env_vars(verifier_toml_env)

    with monkeypatch.context() as verifier_proc_env:
        verifier_proc_env.setattr(eval_module.os, "environ", resolved_verifier_env)
        return eval_module._resolve_eval_retry_config()


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

    monkeypatch.setenv("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", "95.0")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.1")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0.5")

    captured_launch = _stub_harbor_runner(monkeypatch, tmp_path, task_source="evals_json")

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

    assert verifier_toml_env["SKILL_EVAL_LLM_JUDGE_BUDGET_SEC"] == "${SKILL_EVAL_LLM_JUDGE_BUDGET_SEC}"
    assert verifier_toml_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "${SKILL_EVAL_LLM_MAX_RETRIES}"
    assert verifier_toml_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"
    assert verifier_toml_env["SKILL_EVAL_LLM_RETRY_MAX_DELAY"] == "${SKILL_EVAL_LLM_RETRY_MAX_DELAY}"

    run_env = captured_launch["run_env"]
    assert run_env["SKILL_EVAL_LLM_JUDGE_BUDGET_SEC"] == "95.0"
    assert run_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "0"
    assert run_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "0.1"
    assert run_env["SKILL_EVAL_LLM_RETRY_MAX_DELAY"] == "0.5"

    assert _resolve_verifier_retry_tuple(monkeypatch, tmp_path, verifier_toml_env, run_env) == (0, 0.1, 0.5)


def _create_native_skill_with_authored_retry_env(skill_dir: Path, grading_mode: str) -> None:
    """Create a native Harbor skill fixture whose task.toml defines all three retry variables."""
    evals = skill_dir / "evals"
    native_task = evals / "harbor" / "case-001"
    (native_task / "environment").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill.\n---\n# Demo\n",
        encoding="utf-8",
    )
    (evals / "evals.json").write_text(
        '[{"id": "case-001", "question": "Run native case.", "expected_answer": "ok", "files": []}]\n',
        encoding="utf-8",
    )
    (evals / "config.yml").write_text(
        f"schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: {grading_mode}\n",
        encoding="utf-8",
    )
    if grading_mode == "default_plus_custom":
        (evals / "grader.py").write_text("def grade(*args, **kwargs):\n    return 1\n", encoding="utf-8")
    (native_task / "instruction.md").write_text("Run native case.\n", encoding="utf-8")
    (native_task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n'
        '[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n'
        "[verifier]\ntimeout_sec = 180.0\n\n"
        "[verifier.env]\n"
        'SKILL_EVAL_LLM_MAX_RETRIES = "2"\n'
        'SKILL_EVAL_LLM_RETRY_BASE_DELAY = "0.5"\n'
        'SKILL_EVAL_LLM_RETRY_MAX_DELAY = "10.0"\n\n'
        "[environment]\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom"])
def test_native_task_preserves_authored_retry_settings_without_host_override(
    tmp_path: Path,
    grading_mode: str,
) -> None:
    """Verify native task staging keeps authored [verifier.env] retry values when host does not override them."""
    skill = tmp_path / f"native-no-override-{grading_mode}"
    _create_native_skill_with_authored_retry_env(skill, grading_mode)

    staged = stage_native_harbor_tasks(
        skill,
        tmp_path / f"staged-no-override-{grading_mode}",
        grading_mode=grading_mode,
        verifier_env={"SKILL_EVAL_LLM_PROVIDER": "openai", "SKILL_EVAL_LLM_MODEL": "gpt-4o-mini"},
    )[0]

    staged_task = tomllib.loads((staged / "task.toml").read_text(encoding="utf-8"))
    verifier_env = staged_task["verifier"]["env"]
    assert verifier_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "2"
    assert verifier_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "0.5"
    assert verifier_env["SKILL_EVAL_LLM_RETRY_MAX_DELAY"] == "10.0"
    assert verifier_env["SKILL_EVAL_LLM_PROVIDER"] == "${SKILL_EVAL_LLM_PROVIDER}"


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom"])
@pytest.mark.parametrize(
    ("retry_var", "host_val", "expected_tuple"),
    [
        ("SKILL_EVAL_LLM_MAX_RETRIES", "0", (0, 0.5, 10.0)),
        ("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.1", (2, 0.1, 10.0)),
        ("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0.5", (2, 0.5, 0.5)),
    ],
)
def test_native_task_host_override_replaces_authored_retry_key_without_duplicate_toml_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    grading_mode: str,
    retry_var: str,
    host_val: str,
    expected_tuple: tuple[int, float, float],
) -> None:
    """Verify each host retry override replaces its colliding authored [verifier.env] key and produces valid TOML."""
    skill = tmp_path / f"native-override-{grading_mode}-{retry_var.lower()}"
    _create_native_skill_with_authored_retry_env(skill, grading_mode)

    for var in _VERIFIER_RETRY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(retry_var, host_val)

    captured_launch = _stub_harbor_runner(
        monkeypatch,
        tmp_path,
        task_source="native_harbor",
        grading_mode=grading_mode,
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "gpt-4o-mini"},
        output_dir=tmp_path / f"results-{grading_mode}-{retry_var.lower()}",
        env_mode="docker",
        grading_mode=grading_mode,
        skip_baseline=True,
        keep_harbor_jobs=True,
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    task_toml_path = captured_launch["with_skill"] / "case-001" / "task.toml"
    staged_task = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    verifier_toml_env = staged_task["verifier"]["env"]
    assert verifier_toml_env[retry_var] == f"${{{retry_var}}}"

    run_env = captured_launch["run_env"]
    assert run_env[retry_var] == host_val

    assert _resolve_verifier_retry_tuple(monkeypatch, tmp_path, verifier_toml_env, run_env) == expected_tuple


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom"])
def test_native_task_simultaneous_host_overrides_and_quoted_keys(
    tmp_path: Path,
    grading_mode: str,
) -> None:
    """Verify simultaneous host overrides replace all three authored retry keys including quoted TOML keys."""
    skill = tmp_path / f"native-all-three-{grading_mode}"
    _create_native_skill_with_authored_retry_env(skill, grading_mode)
    task_toml = skill / "evals" / "harbor" / "case-001" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            'SKILL_EVAL_LLM_MAX_RETRIES = "2"\n',
            '"SKILL_EVAL_LLM_MAX_RETRIES" = "2"\nCUSTOM_VERIFIER_FLAG = "keep-me"\n',
        ),
        encoding="utf-8",
    )

    staged = stage_native_harbor_tasks(
        skill,
        tmp_path / f"staged-all-three-{grading_mode}",
        grading_mode=grading_mode,
        verifier_env={
            "SKILL_EVAL_LLM_PROVIDER": "openai",
            "SKILL_EVAL_LLM_MODEL": "gpt-4o-mini",
            "SKILL_EVAL_LLM_MAX_RETRIES": "0",
            "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "0.1",
            "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "0.5",
        },
    )[0]

    staged_task = tomllib.loads((staged / "task.toml").read_text(encoding="utf-8"))
    verifier_env = staged_task["verifier"]["env"]
    assert verifier_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "${SKILL_EVAL_LLM_MAX_RETRIES}"
    assert verifier_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"
    assert verifier_env["SKILL_EVAL_LLM_RETRY_MAX_DELAY"] == "${SKILL_EVAL_LLM_RETRY_MAX_DELAY}"
    assert verifier_env["CUSTOM_VERIFIER_FLAG"] == "keep-me"


def test_native_task_verifier_env_with_header_comments_and_cross_section_keys(
    tmp_path: Path,
) -> None:
    """Verify [verifier.env] and next-table headers with inline comments or whitespace do not duplicate tables or keys."""
    skill = tmp_path / "native-header-comments"
    _create_native_skill_with_authored_retry_env(skill, "default")
    task_toml = skill / "evals" / "harbor" / "case-001" / "task.toml"
    task_toml.write_text(
        'schema_version = "1.3"\n\n'
        '[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n'
        "  [verifier] # verifier config\n"
        "timeout_sec = 180.0\n\n"
        "  [verifier.env] # verifier env overrides\n"
        'SKILL_EVAL_LLM_MAX_RETRIES = "2" # authored retry budget\n\n'
        '[[steps]] # step list\nname = "step-one"\n\n'
        "[steps.verifier.env]\n"
        'SKILL_EVAL_LLM_RETRY_BASE_DELAY = "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"\n\n'
        "[environment] # agent environment\n",
        encoding="utf-8",
    )

    staged = stage_native_harbor_tasks(
        skill,
        tmp_path / "staged-header-comments",
        grading_mode="default",
        verifier_env={
            "SKILL_EVAL_LLM_PROVIDER": "openai",
            "SKILL_EVAL_LLM_MAX_RETRIES": "0",
            "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "0.1",
        },
    )[0]

    staged_task = tomllib.loads((staged / "task.toml").read_text(encoding="utf-8"))
    verifier_env = staged_task["verifier"]["env"]
    assert verifier_env["SKILL_EVAL_LLM_MAX_RETRIES"] == "${SKILL_EVAL_LLM_MAX_RETRIES}"
    assert verifier_env["SKILL_EVAL_LLM_RETRY_BASE_DELAY"] == "${SKILL_EVAL_LLM_RETRY_BASE_DELAY}"
    assert verifier_env["SKILL_EVAL_LLM_PROVIDER"] == "${SKILL_EVAL_LLM_PROVIDER}"
