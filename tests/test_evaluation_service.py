# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Tier 3 EvaluationService + CLI/API parity (Phase 4)."""

from __future__ import annotations

import asyncio
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import DatasetOptions, EvaluationOptions, EvaluationService

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


def test_dataset_result_and_error_are_public_runtime_types() -> None:
    from skillevaluator import evaluation

    assert evaluation.DatasetGenerationResult.__module__ == "skillevaluator.evaluation.results"
    assert evaluation.DatasetGenerationError.__module__ == "skillevaluator.evaluation.results"
    assert get_type_hints(EvaluationService.create_dataset)["return"] is evaluation.DatasetGenerationResult


def test_service_preserves_dataset_generation_failure_without_printing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator import evaluation

    invalid_skill = tmp_path / "not-a-skill"
    invalid_skill.mkdir()

    with pytest.raises(evaluation.DatasetGenerationError, match=rf"{invalid_skill} does not contain a SKILL\.md"):
        EvaluationService().create_dataset(DatasetOptions(skill_path=invalid_skill, no_llm=True))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("command", [("create-eval-dataset",), ("tier3", "create-eval-dataset")])
def test_cli_formats_dataset_generation_failure_without_raw_exception(
    tmp_path: Path,
    command: tuple[str, ...],
) -> None:
    invalid_skill = tmp_path / "not-a-skill"
    invalid_skill.mkdir()

    result = CliRunner().invoke(cli, [*command, str(invalid_skill), "--no-llm"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, ValueError)
    assert result.output == f"Error: {invalid_skill} does not contain a SKILL.md\n"


@pytest.mark.parametrize("dependent_option", ["--from-results", "--results-dir"])
def test_cli_formats_dataset_option_dependency_failure(
    tmp_path: Path,
    dependent_option: str,
) -> None:
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()

    result = CliRunner().invoke(
        cli,
        ["create-eval-dataset", str(skill), "--no-llm", dependent_option, str(results)],
    )

    assert result.exit_code == 1
    assert result.output == f"Error: {dependent_option} requires --refine\n"


def test_options_fields_match_engine_signature() -> None:
    """EvaluationOptions must not drift from the engine's evaluate() signature."""
    harbor = pytest.importorskip("harbor")
    assert harbor is not None
    from skillevaluator.tier3.commands import evaluate as engine_evaluate

    sig_params = set(inspect.signature(engine_evaluate).parameters) - {"skill_path", "progress_reporter"}
    option_fields = set(EvaluationOptions.__dataclass_fields__) - {"skill_path"}
    assert option_fields == sig_params


def test_dataset_options_fields_match_engine_signature() -> None:
    from skillevaluator.tier3.commands import create_dataset as engine_create_dataset

    sig_params = set(inspect.signature(engine_create_dataset).parameters) - {"skill_path"}
    option_fields = set(DatasetOptions.__dataclass_fields__) - {"skill_path"}
    assert option_fields == sig_params


def test_engine_kwargs_excludes_skill_path() -> None:
    opts = EvaluationOptions(skill_path=Path("/tmp/x"), agents="codex", env_mode="docker")
    kwargs = opts.engine_kwargs()
    assert "skill_path" not in kwargs
    assert kwargs["agents"] == "codex"
    assert kwargs["env_mode"] == "docker"


def test_service_defaults_to_null_progress_reporter(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.harbor.progress import NullProgressReporter

    captured: dict[str, object] = {}

    def _fake_engine(skill_path: Path, **kwargs: object) -> dict[str, object]:
        captured["skill_path"] = skill_path
        captured.update(kwargs)
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(commands, "evaluate", _fake_engine)
    options = EvaluationOptions(skill_path=FIXTURE, agents="codex", env_mode="docker")

    EvaluationService().evaluate(options)

    assert captured["skill_path"] == FIXTURE
    assert isinstance(captured["progress_reporter"], NullProgressReporter)


def test_cli_evaluate_uses_shared_service(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_evaluate(self, options: EvaluationOptions, *, progress_reporter=None) -> dict:
        captured["options"] = options
        captured["progress_reporter"] = progress_reporter
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(
        cli, ["evaluate", str(FIXTURE), "-a", "codex", "--env-mode", "docker", "--skip-baseline"]
    )
    assert result.exit_code == 0, result.output
    opts = captured["options"]
    assert isinstance(opts, EvaluationOptions)
    assert opts.agents == "codex"
    assert opts.env_mode == "docker"
    assert opts.skip_baseline is True


def _autopilot_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    return skill


def _write_one_case(skill: Path, marker: str = "generated") -> None:
    evals = skill / "evals"
    evals.mkdir(exist_ok=True)
    (evals / "evals.json").write_text(
        json.dumps({"skill_name": skill.name, "evals": [{"id": f"{skill.name}-001", "prompt": marker}]}),
        encoding="utf-8",
    )


def test_cli_autopilot_reuses_existing_dataset_unchanged_and_preserves_skip_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = _autopilot_skill(tmp_path)
    _write_one_case(skill, "existing-sentinel")
    dataset = skill / "evals" / "evals.json"
    original = dataset.read_bytes()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must reuse")
    )
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: (
            captured.setdefault("options", options) and {"execution_status": "succeeded", "execution_errors": []}
        ),
    )

    result = CliRunner().invoke(cli, ["evaluate", str(skill), "--autopilot", "--skip-baseline", "--progress", "off"])

    assert result.exit_code == 0, result.output
    assert dataset.read_bytes() == original
    assert captured["options"].skip_baseline is True
    assert "reusing" in result.output.lower()


def test_cli_autopilot_uses_keyed_llm_one_case_generation_and_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator import provider_config

    skill = _autopilot_skill(tmp_path)
    calls = []
    captured: dict[str, object] = {}
    monkeypatch.setattr(provider_config, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))

    def create(_self, _skill_path, *, use_llm):
        calls.append(use_llm)
        _write_one_case(skill)

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", create)
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: (
            captured.setdefault("options", options) and {"execution_status": "succeeded", "execution_errors": []}
        ),
    )
    results_dir = tmp_path / "results"

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(skill),
            "--autopilot",
            "--agents",
            "opencode",
            "--env-mode",
            "e2b",
            "--skip-baseline",
            "--n-attempts",
            "3",
            "--results-dir",
            str(results_dir),
            "--progress",
            "off",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [True]
    options = captured["options"]
    assert options.agents == "opencode" and options.env_mode == "e2b"
    assert options.skip_baseline is True and options.n_attempts == 3 and options.results_dir == results_dir


def test_cli_autopilot_uses_deterministic_one_case_generation_without_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator import provider_config

    skill = _autopilot_skill(tmp_path)
    monkeypatch.setattr(
        provider_config,
        "resolve_llm_provider",
        lambda: (_ for _ in ()).throw(provider_config.ProviderConfigurationError("key missing")),
    )
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda *_args, **_kwargs: {"execution_status": "succeeded", "execution_errors": []},
    )

    result = CliRunner().invoke(cli, ["evaluate", str(skill), "--autopilot", "--progress", "off"])

    assert result.exit_code == 0, result.output
    payload = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert len(payload["evals"]) == 1
    assert "deterministic" in result.output.lower()


def test_cli_autopilot_warns_and_falls_back_when_llm_generation_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator import provider_config
    from skillevaluator.inference.client import LLMClient

    skill = _autopilot_skill(tmp_path)
    monkeypatch.setattr(provider_config, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))
    monkeypatch.setattr(
        LLMClient,
        "completions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda *_args, **_kwargs: {"execution_status": "succeeded", "execution_errors": []},
    )

    result = CliRunner().invoke(cli, ["evaluate", str(skill), "--autopilot", "--progress", "off"])

    assert result.exit_code == 0, result.output
    payload = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert len(payload["evals"]) == 1
    assert "provider unavailable" not in result.output and "falling back" in result.output.lower()


def test_autopilot_llm_generator_surfaces_model_error_without_printing_raw_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator import provider_config
    from skillevaluator.inference.client import LLMClient
    from skillevaluator.tier3 import generate_dataset

    monkeypatch.setattr(provider_config, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai", model="m"))
    monkeypatch.setattr(
        LLMClient,
        "completions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic outage")),
    )

    with pytest.raises(RuntimeError, match="LLM dataset generation failed"):
        asyncio.run(
            generate_dataset._generate_with_llm(
                {"name": "simple", "description": "Summarize input", "scripts": [], "content": "", "eval_prompt": ""},
                fallback_to_template=False,
            )
        )

    assert "synthetic outage" not in capsys.readouterr().out


def test_llm_generator_rejects_malformed_case_and_uses_safe_one_case_template(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator import provider_config
    from skillevaluator.inference.client import LLMClient
    from skillevaluator.tier3 import generate_dataset

    monkeypatch.setattr(provider_config, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai", model="m"))
    monkeypatch.setattr(
        LLMClient,
        "completions",
        lambda *_args, **_kwargs: json.dumps(
            [{"id": "../../escape", "question": 5, "ground_truth": None, "expected_behavior": []}]
        ),
    )

    cases = asyncio.run(
        generate_dataset._generate_with_llm(
            {"name": "simple", "description": "Summarize input", "scripts": [], "content": "", "eval_prompt": ""}
        )
    )

    assert [case["id"] for case in cases] == ["simple-001"]
    assert "Warning: LLM generation failed" in capsys.readouterr().out


def test_cli_autopilot_rejects_symlinked_evals_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator import provider_config

    skill = _autopilot_skill(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (skill / "evals").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        provider_config,
        "resolve_llm_provider",
        lambda: (_ for _ in ()).throw(provider_config.ProviderConfigurationError("key missing")),
    )

    result = CliRunner().invoke(cli, ["evaluate", str(skill), "--autopilot", "--progress", "off"])

    assert result.exit_code == 1
    assert "evals directory must be a real directory" in result.output
    assert not (outside / "evals.json").exists()


def test_local_mode_notice_precedes_progress_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor.progress import Tier3RunPlan

    def _fake_evaluate(self, options: EvaluationOptions, *, progress_reporter=None) -> dict:
        del self, options
        progress_reporter.start(Tier3RunPlan(skill_name="simple", environment="local", agents=("opencode",)))
        progress_reporter.close()
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(
        cli,
        ["evaluate", str(FIXTURE), "--agents", "opencode", "--env-mode", "local", "--progress", "plain"],
    )

    assert result.exit_code == 0, result.output
    assert "Local mode · Experimental" in result.output
    assert "Intended for trusted skills and workspaces" in result.output
    assert result.output.index("Local mode · Experimental") < result.output.index("Tier 3 live evaluation")


@pytest.mark.parametrize("env_mode", ["docker", "e2b"])
def test_non_local_modes_do_not_show_experimental_local_notice(
    monkeypatch: pytest.MonkeyPatch,
    env_mode: str,
) -> None:
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, _options, **_kwargs: {"execution_status": "succeeded", "execution_errors": []},
        raising=True,
    )
    result = CliRunner().invoke(
        cli,
        ["evaluate", str(FIXTURE), "--agents", "opencode", "--env-mode", env_mode, "--progress", "off"],
    )

    assert result.exit_code == 0, result.output
    assert "Local mode · Experimental" not in result.output


def test_cli_evaluate_accepts_harbor_native_cloud_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_evaluate(self, options: EvaluationOptions, *, progress_reporter=None) -> dict:
        captured["options"] = options
        captured["progress_reporter"] = progress_reporter
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(
        cli,
        ["evaluate", str(FIXTURE), "-a", "codex", "--env-mode", "e2b"],
    )

    assert result.exit_code == 0, result.output
    opts = captured["options"]
    assert isinstance(opts, EvaluationOptions)
    assert opts.env_mode == "e2b"


def test_cli_evaluate_renders_persisted_score_and_report_truth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result_payload = {
        "execution_status": "succeeded",
        "execution_errors": [],
        "duration_seconds": 3.5,
        "report_path": str(tmp_path / "report.html"),
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "lift": {"overall": {"with_skill": 0.8, "without_skill": 0.4, "delta": 0.4}},
                "pass_at_k": {"with_skill": {"rate": 1.0}, "without_skill": {"rate": 0.5}},
            }
        },
    }
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, _options, **_kwargs: result_payload,
        raising=True,
    )

    result = CliRunner().invoke(cli, ["evaluate", str(FIXTURE), "--skip-baseline", "--progress", "off"])

    assert result.exit_code == 0, result.output
    assert "With Skill" in result.output
    assert "+0.40" in result.output
    assert "report.html" in result.output
    assert "Time: 3.5s" in result.output


def test_cli_evaluate_renders_real_command_failure_before_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from skillevaluator.tier3 import commands

    jobs = tmp_path / "jobs"
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))
    monkeypatch.setattr(commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(
        commands,
        "run_harbor_eval",
        lambda **_kwargs: {
            "execution_status": "failed",
            "execution_errors": ["opencode with-skill failed: 401 Unauthorized"],
            "error": ["opencode with-skill failed: 401 Unauthorized"],
            "duration_seconds": 2.0,
            "harbor_jobs_dir": str(jobs),
            "harbor_jobs_retained": True,
            "agents": {},
        },
    )

    result = CliRunner().invoke(cli, ["evaluate", str(FIXTURE), "--skip-baseline", "--progress", "off"])

    assert result.exit_code != 0
    assert "FAILED" in result.output
    assert "401 Unauthorized" in result.output
    assert "🔍 Inspect jobs" not in result.output


def test_cli_failure_keeps_compact_artifacts_as_absolute_footer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.html"
    report.touch()
    payload = {
        "execution_status": "failed",
        "execution_errors": ["agent failed"],
        "report_path": str(report),
        "run_dir": str(tmp_path),
        "agents": {},
    }
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, _options, **_kwargs: payload,
        raising=True,
    )

    result = CliRunner().invoke(cli, ["evaluate", str(FIXTURE), "--progress", "off"])

    assert result.exit_code == 1
    assert "agent failed" in result.output
    assert "Error:" not in result.output
    assert "Artifacts" in result.output
    artifacts = result.output[result.output.rindex("Artifacts") :]
    normalized_artifacts = "".join(line.strip(" │") for line in artifacts.splitlines())
    assert str(tmp_path) in normalized_artifacts
    assert "…" not in artifacts
    assert artifacts.rstrip().splitlines()[-1].lstrip().startswith("╰")


@pytest.mark.parametrize(
    ("engine_result", "expected"),
    [
        ({}, "empty result"),
        ({"execution_status": "failed", "execution_errors": ["job incomplete"]}, "job incomplete"),
        ({"execution_status": "unknown"}, "status is unknown"),
        ({"execution_status": "succeeded", "error": ["late failure"]}, "late failure"),
        (None, "non-mapping result"),
    ],
)
def test_cli_evaluate_fails_closed_on_unusable_engine_result(
    monkeypatch: pytest.MonkeyPatch,
    engine_result: object,
    expected: str,
) -> None:
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, _options, **_kwargs: engine_result,
        raising=True,
    )

    result = CliRunner().invoke(cli, ["evaluate", str(FIXTURE), "--skip-baseline"])

    assert result.exit_code != 0
    assert expected in result.output


def test_combined_evaluate_keeps_exit_advisory_but_result_false_for_empty_engine_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator import cli as cli_module

    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options: {}, raising=True)

    result = cli_module._run_agent_eval_or_skip(
        FIXTURE,
        agents="codex",
        env_mode="docker",
        skip_baseline=True,
        n_concurrent=1,
        max_agents=1,
        validate_source=False,
    )

    assert result.passed is False
    assert result.metadata["agent_eval"]["execution_status"] == "skipped"
    assert "empty result" in result.metadata["agent_eval"]["execution_errors"][0]


def test_combined_evaluate_converts_normalization_error_to_advisory_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import tier3_report

    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, _options: {"execution_status": "succeeded", "execution_errors": []},
        raising=True,
    )

    def _raise_normalizer(*_args, **_kwargs):
        raise ValueError("corrupt Harbor artifact")

    monkeypatch.setattr(tier3_report, "agent_eval_result_from_run", _raise_normalizer, raising=True)

    result = cli_module._run_agent_eval_or_skip(
        FIXTURE,
        agents="codex",
        env_mode="docker",
        skip_baseline=True,
        n_concurrent=1,
        max_agents=1,
        validate_source=False,
    )

    assert result.passed is False
    payload = result.metadata["agent_eval"]
    assert payload["execution_status"] == "skipped"
    assert "Tier 3 result normalization failed: corrupt Harbor artifact" in payload["execution_errors"]


def test_combined_evaluate_forwards_results_dir_to_normalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import tier3_report
    from skillevaluator.models.result import ValidationResult

    engine_result = {"execution_status": "succeeded", "execution_errors": []}
    expected = ValidationResult(validator_name="AGENT_EVAL")
    expected.add_success("agent_eval", "ok")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, _options: engine_result,
        raising=True,
    )

    def _normalize(_skill_path: Path, **kwargs) -> ValidationResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(tier3_report, "agent_eval_result_from_run", _normalize, raising=True)
    results_root = tmp_path / "external-results"

    actual = cli_module._run_agent_eval_or_skip(
        FIXTURE,
        agents="codex",
        env_mode="docker",
        skip_baseline=True,
        n_concurrent=1,
        max_agents=1,
        results_dir=results_root,
        validate_source=False,
    )

    assert actual is expected
    assert captured["results_dir"] == results_root
    assert captured["engine_result"] is engine_result
    assert captured["env_mode"] == "docker"


def test_report_discovery_handles_timestamped_results(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    skill = tmp_path / "myskill"
    skill.mkdir()
    run_dir = results_root / "myskill" / "2026-06-18_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name}), encoding="utf-8")
    (results_root / "myskill" / "latest").symlink_to(run_dir.name)

    service = EvaluationService()
    latest = service.discover_latest_results(skill, results_dir=results_root)
    assert latest is not None and latest.name == "2026-06-18_120000"
    report = service.discover_latest_report(skill, results_dir=results_root)
    assert report is not None and report.name == "report.html"


def test_report_discovery_returns_none_when_absent(tmp_path: Path) -> None:
    skill = tmp_path / "empty"
    skill.mkdir()
    service = EvaluationService()
    assert service.discover_latest_results(skill, results_dir=tmp_path / "nope") is None
    assert service.discover_latest_report(skill, results_dir=tmp_path / "nope") is None
