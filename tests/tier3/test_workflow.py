# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Tier 3 workflows preserve authored inputs and reuse evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import evaluate
from skillevaluator.evaluation import EvaluationService
from skillevaluator.tier3.workflow import build_tier3_workflow


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    path = tmp_path / "demo-skill"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Explain how to validate a configuration.\n---\n"
        "# Demo\nValidate the configuration, then show the result.\n"
    )
    return path


def _dataset(skill: Path, *, suffix: str = ".json") -> Path:
    directory = skill / "evals"
    directory.mkdir(exist_ok=True)
    path = directory / f"evals{suffix}"
    path.write_text(
        json.dumps(
            {
                "skill_name": skill.name,
                "evals": [
                    {"id": "demo-001", "prompt": "Validate this config.", "expected_output": "Validation result."}
                ],
            }
        )
    )
    return path


def _command():
    return build_tier3_workflow(evaluate)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> list:
    from skillevaluator.tier3.harbor import runner

    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-never-sent")
    for name in ("SKILL_EVAL_LLM_MODEL", "LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    captured = []
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.append(options) or {"execution_status": "succeeded"},
    )
    return captured


def test_direct_workflow_reuses_dataset_and_forwards_evaluation_options(skill, configured, monkeypatch):
    dataset = _dataset(skill)
    original = dataset.read_bytes()
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must reuse")
    )
    result = CliRunner().invoke(
        _command(),
        [
            str(skill),
            "--agents",
            "opencode",
            "--skip-baseline",
            "--n-attempts",
            "3",
            "--no-agent-runtime-preflight",
            "--progress",
            "off",
            "--agent-model",
            "opencode=nvidia/nvidia/test-model",
            "--results-dir",
            str(skill.parent / "results"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert dataset.read_bytes() == original
    assert configured[0].skip_baseline is True
    assert configured[0].n_attempts == 3
    assert configured[0].agent_runtime_preflight is False
    assert configured[0].agent_model == ("opencode=nvidia/nvidia/test-model",)
    assert configured[0].results_dir == skill.parent / "results"
    assert "reusing" in result.output.lower()


def test_direct_workflow_generates_missing_dataset_then_evaluates(skill, configured, monkeypatch):
    calls = []

    def generate(self, skill_path, *, use_llm):
        calls.append(use_llm)
        return _dataset(skill_path)

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", generate)
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert len(configured) == 1
    assert configured[0].agents == "opencode"
    assert configured[0].skip_baseline is False


@pytest.mark.parametrize(
    ("provider", "credential", "expected_agent"),
    [
        ("openai", "OPENAI_API_KEY", "codex"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-code"),
    ],
)
def test_direct_workflow_uses_the_provider_native_agent(
    skill: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    credential: str,
    expected_agent: str,
) -> None:
    from skillevaluator.tier3.harbor import runner

    _dataset(skill)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    for name in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(credential, "test-key-never-sent")
    captured = []
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.append(options) or {"execution_status": "succeeded"},
    )

    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])

    assert result.exit_code == 0, result.output
    assert captured[0].agents == expected_agent


def test_missing_provider_never_generates_dataset(skill, monkeypatch):
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preflight")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "NVIDIA_API_KEY" in result.output
    assert not (skill / "evals").exists()


@pytest.mark.parametrize("body", ['{"evals": []}', "{not valid JSON", '{"evals":[{"id":"a","prompt":""}]}'])
def test_invalid_existing_dataset_errors_without_overwrite(skill, configured, monkeypatch, body):
    dataset = _dataset(skill)
    dataset.write_text(body)
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preserve")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert dataset.read_text() == body
    assert not configured


@pytest.mark.parametrize(
    "options",
    [
        ["--agents", "unknown-agent"],
        ["--agent-model", "not-a-model-override"],
        ["--agent-model", "opencode=other-model"],
        ["--n-attempts", "0"],
        ["--stop-on-pass", "--n-attempts", "1"],
        ["--pass-threshold", "2"],
        ["--timeout-multiplier", "0"],
    ],
)
def test_invalid_options_error_before_generating(skill, configured, monkeypatch, options):
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preflight")
    )
    result = CliRunner().invoke(_command(), [str(skill), *options, "--progress", "off"])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not (skill / "evals").exists()
    assert not configured


def test_invalid_configuration_does_not_generate(skill, configured, monkeypatch):
    (skill / "evals").mkdir()
    config = skill / "evals" / "config.yml"
    config.write_text("schema_version: 1\nharbor:\n  n_attempts: -1\n")
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preflight")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "n_attempts" in result.output
    assert not (skill / "evals" / "evals.json").exists()
    assert not configured


def test_explicit_native_source_is_not_replaced_with_generated_json(skill, configured, monkeypatch):
    (skill / "evals").mkdir()
    (skill / "evals" / "config.yml").write_text("schema_version: 1\nharbor:\n  task_source: native_harbor\n")
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preserve")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "native_harbor" in result.output
    assert not (skill / "evals" / "evals.json").exists()


def test_direct_help_explains_generation_without_changing_expert_help():
    direct = CliRunner().invoke(_command(), ["--help"])
    expert = CliRunner().invoke(evaluate, ["--help"])
    assert direct.exit_code == 0
    assert "missing" in direct.output.lower()
    assert "--autopilot" not in direct.output
    assert "--autopilot" in expert.output


def test_yaml_dataset_is_reused_without_json_creation(skill, configured, monkeypatch):
    dataset = _dataset(skill, suffix=".yaml")
    original = dataset.read_bytes()
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must reuse YAML")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code == 0, result.output
    assert dataset.read_bytes() == original
    assert not (skill / "evals" / "evals.json").exists()
    assert len(configured) == 1


def test_explicit_json_source_generates_missing_dataset(skill, configured, monkeypatch):
    (skill / "evals").mkdir()
    config = skill / "evals" / "config.yml"
    original = "schema_version: 1\nharbor:\n  task_source: evals_json\n"
    config.write_text(original)
    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", lambda _self, path, **_kwargs: _dataset(path))
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code == 0, result.output
    assert config.read_text() == original
    assert len(configured) == 1


def test_valid_native_tasks_are_reused_without_generating_json(skill, configured, monkeypatch):
    harbor = skill / "evals" / "harbor"
    case = harbor / "demo-001"
    case.mkdir(parents=True)
    (harbor / "dataset.toml").write_text('[[tasks]]\nname = "demo-001"\n')
    (case / "task.toml").write_text('version = "1.0"\n')
    (case / "instruction.md").write_text("Validate the supplied configuration.")
    (skill / "evals" / "config.yml").write_text("schema_version: 1\ngrading:\n  mode: custom_only\n")
    original = {path.relative_to(skill): path.read_bytes() for path in skill.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must reuse native tasks")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code == 0, result.output
    assert {path.relative_to(skill): path.read_bytes() for path in skill.rglob("*") if path.is_file()} == original
    assert len(configured) == 1


def test_multiple_datasets_are_rejected_unchanged(skill, configured):
    first = _dataset(skill)
    second = _dataset(skill, suffix=".yaml")
    original = (first.read_bytes(), second.read_bytes())
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "multiple eval datasets" in result.output
    assert (first.read_bytes(), second.read_bytes()) == original
    assert not configured


def test_invalid_generated_dataset_is_not_evaluated(skill, configured, monkeypatch):
    def generate(_self, path, **_kwargs):
        _dataset(path).write_text('{"evals": []}')

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", generate)
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "empty" in result.output
    assert not configured


def test_generation_uses_existing_deterministic_fallback_on_model_failure(skill, configured, monkeypatch):
    from skillevaluator.tier3 import generate_dataset

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("model is unavailable")

    monkeypatch.setattr(generate_dataset, "_generate_with_llm", unavailable)
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code == 0, result.output
    dataset = json.loads((skill / "evals" / "evals.json").read_text())
    assert len(dataset["evals"]) == 1
    assert "falling back" in result.output.lower()
    assert len(configured) == 1


def test_symlinked_evals_directory_is_rejected_without_writes(skill, configured, tmp_path):
    target = tmp_path / "outside-evals"
    target.mkdir()
    (skill / "evals").symlink_to(target, target_is_directory=True)
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "real directory" in result.output
    assert list(target.iterdir()) == []
    assert not configured


def test_runtime_credential_failure_precedes_dataset_generation(skill, configured, monkeypatch):
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-sent")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preflight agent")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--agents", "codex", "--progress", "off"])
    assert result.exit_code != 0
    assert not (skill / "evals").exists()
    assert not configured


def test_invalid_resource_override_does_not_generate(skill, configured, monkeypatch):
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must preflight resources")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--override-cpus", "-1", "--progress", "off"])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not (skill / "evals").exists()
    assert not configured


def test_file_target_fails_before_dataset_preparation(skill, configured, monkeypatch):
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must reject file")
    )
    result = CliRunner().invoke(_command(), [str(skill / "SKILL.md"), "--progress", "off"])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not configured


@pytest.mark.parametrize("configured_include", [False, True])
def test_included_skills_need_group_mode_before_preparation(skill, configured, monkeypatch, configured_include):
    from skillevaluator.tier3 import workflow

    options = ["--include-skills", str(skill)]
    if configured_include:
        (skill / "evals").mkdir()
        (skill / "evals" / "config.yml").write_text(f"schema_version: 1\nskill_workspace:\n  include:\n    - {skill}\n")
        options = []
    monkeypatch.setattr(workflow, "_prepare_dataset", lambda *_args, **_kwargs: pytest.fail("must preflight include"))
    result = CliRunner().invoke(_command(), [str(skill), *options, "--progress", "off"])
    assert result.exit_code != 0
    assert "include_skills requires" in result.output
    assert not configured


def test_unavailable_environment_fails_before_generation(skill, configured, monkeypatch):
    from skillevaluator.tier3.harbor import runner

    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: ["Docker Compose v2 is required"])
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("must preflight environment"),
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "Docker Compose" in result.output
    assert not (skill / "evals").exists()
    assert not configured


@pytest.mark.parametrize("relative", ["evals.json", "config.yml", "harbor/case/task.toml"])
def test_linked_evaluator_inputs_are_rejected_without_reading(skill, configured, monkeypatch, tmp_path, relative):
    outside = tmp_path / "private-file"
    outside.write_text("private contents must never be parsed")
    linked = skill / "evals" / relative
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside)
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must reject unsafe input")
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "symlink" in result.output.lower() or "non-linked" in result.output.lower()
    assert "private contents" not in result.output
    assert not configured


def test_fifo_dataset_is_rejected_without_blocking(skill, configured):
    import os
    import subprocess
    import sys

    (skill / "evals").mkdir()
    os.mkfifo(skill / "evals" / "evals.json")
    command = (
        "from skillevaluator.cli import evaluate; "
        "from skillevaluator.tier3.workflow import build_tier3_workflow; "
        "build_tier3_workflow(evaluate).main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(skill)], capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode != 0
    assert "regular non-linked file" in result.stderr.lower()
    assert not configured


def test_hardlinked_dataset_is_rejected(skill, configured, tmp_path):
    import os

    dataset = _dataset(skill)
    os.link(dataset, tmp_path / "second-link")
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "non-linked" in result.output.lower() or "hard-linked" in result.output.lower()
    assert not configured


def test_included_skill_with_group_mode_is_preserved(skill, configured, monkeypatch):
    _dataset(skill)
    result = CliRunner().invoke(
        _command(),
        [
            str(skill),
            "--include-skills",
            str(skill),
            "--skill-workspace-mode",
            "group",
            "--progress",
            "off",
        ],
    )
    assert result.exit_code == 0, result.output
    assert configured[0].include_skills == (skill,)
    assert configured[0].skill_workspace_mode == "group"


@pytest.mark.parametrize("include_source", ["cli", "config", "both"])
def test_relative_included_skills_keep_cli_and_config_path_bases(skill, configured, monkeypatch, include_source):
    _dataset(skill)
    caller = skill.parent / "caller"
    caller.mkdir()
    cli_helper = caller / "cli-helper"
    config_helper = skill.parent / "config-helper"
    for helper in (cli_helper, config_helper):
        helper.mkdir()
        (helper / "SKILL.md").write_text("# Helper\n")
    config = skill / "evals" / "config.yml"
    config_body = "schema_version: 1\nskill_workspace:\n  mode: group\n"
    if include_source in {"config", "both"}:
        config_body += "  include:\n    - config-helper\n"
    config.write_text(config_body)
    monkeypatch.chdir(caller)
    options = ["--include-skills", "./cli-helper"] if include_source in {"cli", "both"} else []

    result = CliRunner().invoke(_command(), [str(skill), *options, "--progress", "off"])

    assert result.exit_code == 0, result.output
    assert len(configured) == 1
    assert configured[0].include_skills == ((Path("cli-helper"),) if options else ())
    assert config.read_text() == config_body


def test_remote_authentication_preflight_is_deferred(skill, configured, monkeypatch):
    from skillevaluator.tier3 import workflow
    from skillevaluator.tier3.harbor import runner

    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: pytest.fail("must not make remote RPC"))
    for mode in ("cwsandbox", "wandb", "langsmith"):
        workflow._preflight_environment({"env_mode": mode, "agents": "codex"})
    assert not configured


@pytest.mark.parametrize("with_evals", [False, True])
def test_legacy_dataset_requires_migration_before_any_read(skill, configured, monkeypatch, with_evals):
    from skillevaluator.tier3 import workflow

    legacy = skill / "eval"
    legacy.mkdir()
    (legacy / "dataset.json").write_text("legacy marker must not be parsed")
    if with_evals:
        (skill / "evals").mkdir()
    monkeypatch.setattr(workflow, "_preflight_options", lambda *_args: pytest.fail("legacy source must be rejected"))
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "eval/dataset" in result.output
    assert "evals/evals" in result.output
    assert not configured


@pytest.mark.parametrize("relative", ["SKILL.md", "evals/EVAL.md"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_generation_inputs_are_rejected(skill, configured, monkeypatch, tmp_path, relative, kind):
    import os

    outside = tmp_path / "outside-guidance"
    outside.write_text("private marker must never reach the generator")
    target = skill / relative
    target.parent.mkdir(exist_ok=True)
    target.unlink(missing_ok=True)
    if kind == "symlink":
        target.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, target)
    else:
        os.mkfifo(target)
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("must reject generation input"),
    )
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "regular non-linked file" in result.output
    assert "private marker" not in result.output
    assert not configured


def test_linked_skill_root_is_rejected(skill, configured, tmp_path):
    alias = tmp_path / "linked-skill"
    alias.symlink_to(skill, target_is_directory=True)
    _dataset(skill)
    result = CliRunner().invoke(_command(), [str(alias), "--progress", "off"])
    assert result.exit_code != 0
    assert "real directory" in result.output
    assert not configured


def test_legacy_fifo_is_rejected_without_blocking(skill, configured):
    import os
    import subprocess
    import sys

    (skill / "eval").mkdir()
    os.mkfifo(skill / "eval" / "dataset.json")
    command = (
        "from skillevaluator.cli import evaluate; "
        "from skillevaluator.tier3.workflow import build_tier3_workflow; "
        "build_tier3_workflow(evaluate).main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(skill)], capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode != 0
    assert "eval/dataset" in result.stderr
    assert "evals/evals" in result.stderr


def test_missing_optional_runtime_has_install_guidance(skill, configured, monkeypatch):
    from skillevaluator.tier3 import workflow

    def missing_runtime(*_args):
        raise ModuleNotFoundError("No module named 'harbor'")

    monkeypatch.setattr(workflow, "_preflight_options", missing_runtime)
    result = CliRunner().invoke(_command(), [str(skill), "--progress", "off"])
    assert result.exit_code != 0
    assert "Install skillevaluator[tier3]" in result.output
    assert "Traceback" not in result.output
    assert not (skill / "evals").exists()
    assert not configured


def test_duplicate_agent_model_aliases_fail_before_generation(skill, configured, monkeypatch):
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("must preflight all overrides"),
    )
    result = CliRunner().invoke(
        _command(),
        [
            str(skill),
            "--agents",
            "claude-code",
            "--agent-model",
            "claude-code=nvidia/valid-model",
            "--agent-model",
            "claude=invalid-later-model",
            "--progress",
            "off",
        ],
    )
    assert result.exit_code != 0
    assert "same agent" in result.output
    assert not (skill / "evals").exists()
    assert not configured
