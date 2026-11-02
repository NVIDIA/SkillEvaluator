# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ``max_cases`` limit for Tier 3 runs.

Unset, every staged case runs. When set, only the first N staged cases run in
every agent's with-skill and baseline datasets, and the persisted dataset
snapshot, collector expectations and run config all describe that subset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor import runner, runtime_preflight

CASES = ("case-001", "case-002", "case-003", "case-004", "case-005")
FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


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
    config_yaml: str = "schema_version: 1\nharbor:\n  task_source: evals_json\n",
    **engine_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``run_harbor_eval`` over five staged cases with the Harbor edges mocked."""
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(config_yaml, encoding="utf-8")

    captured: dict[str, Any] = {"pairs": [], "collect": {}, "progress": []}

    monkeypatch.setattr(runner, "resolve_llm_provider", _provider)
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    def emit(_skill: Path, target: Path, **_kwargs: Any) -> list[Path]:
        tasks = []
        for case_id in CASES:
            task = target / case_id
            (task / "tests").mkdir(parents=True)
            (task / "tests" / "entry.json").write_text(json.dumps({"id": case_id}), encoding="utf-8")
            tasks.append(task)
        return tasks

    def pair(**kwargs: Any) -> list[str]:
        datasets = [kwargs["with_skill"], kwargs["baseline"]]
        captured["pairs"].append(
            {
                "task_names": kwargs["task_names"],
                "expected_trials": kwargs["expected_trials"],
                "staged": [sorted(path.name for path in dataset.iterdir() if path.is_dir()) for dataset in datasets],
            }
        )
        return []

    def collect(**kwargs: Any) -> dict[str, Any]:
        captured["collect"] = kwargs
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}}

    class _Reporter:
        is_active = False

        def start(self, _plan: Any) -> None:
            pass

        def set_secret_values(self, _values: Any) -> None:
            pass

        def emit(self, event: Any) -> None:
            captured["progress"].append((event.stage, event.state, event.detail))

        def heartbeat(self) -> None:
            pass

        def close(self) -> None:
            pass

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
    result = runner.run_harbor_eval(
        skill,
        engine_kwargs.pop("agents", ["opencode", "codex"]),
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=False,
        keep_harbor_jobs=True,
        progress_reporter=_Reporter(),
        **engine_kwargs,
    )
    return result, captured


def _snapshot_ids(result: dict[str, Any]) -> list[str]:
    snapshot = json.loads(Path(result["dataset_snapshot_path"]).read_text(encoding="utf-8"))
    return [entry["id"] for entry in snapshot["dataset"]]


def test_unset_max_cases_runs_every_staged_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _run_engine(monkeypatch, tmp_path)

    assert "error" not in result
    assert "case_selection" not in result["run_config"]
    assert "max_cases" not in result["run_config"]["harbor"]
    assert captured["collect"]["expected_case_ids"] == list(CASES)
    assert all(pair["task_names"] == list(CASES) for pair in captured["pairs"])
    assert not any(stage == "case-selection" for stage, _state, _detail in captured["progress"])


def test_max_cases_runs_only_the_first_cases_in_every_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, captured = _run_engine(monkeypatch, tmp_path, max_cases=2)

    assert "error" not in result
    selected = ["case-001", "case-002"]
    assert len(captured["pairs"]) == 2
    for pair in captured["pairs"]:
        assert pair["task_names"] == selected
        assert pair["expected_trials"] == 2
        # Harbor discovers local tasks from the dataset directory, so the
        # unselected cases must be gone from both arms of every agent.
        assert pair["staged"] == [selected, selected]
    assert captured["collect"]["expected_cases"] == 2
    assert captured["collect"]["expected_case_ids"] == selected
    assert captured["collect"]["case_id_by_task_selector"] == {case: case for case in selected}
    assert _snapshot_ids(result) == selected
    assert result["dataset_summary"]["total_tasks"] == 2
    assert result["run_config"]["harbor"]["max_cases"] == 2
    assert result["run_config"]["case_selection"] == {
        "policy": "first_n",
        "limit": 2,
        "total": 5,
        "selected": 2,
        "skipped": 3,
        "selected_ids": selected,
    }
    persisted = json.loads((Path(result["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
    assert persisted["case_selection"] == result["run_config"]["case_selection"]
    assert ("case-selection", "complete", "running the first 2 of 5 cases (max_cases=2)") in captured["progress"]


def test_max_cases_from_config_applies_and_cli_value_overrides_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = "schema_version: 1\nharbor:\n  task_source: evals_json\n  max_cases: 3\n"

    configured, configured_capture = _run_engine(monkeypatch, tmp_path / "config", config)
    overridden, overridden_capture = _run_engine(monkeypatch, tmp_path / "cli", config, max_cases=1)

    assert configured_capture["collect"]["expected_case_ids"] == ["case-001", "case-002", "case-003"]
    assert configured["run_config"]["case_selection"]["skipped"] == 2
    assert overridden_capture["collect"]["expected_case_ids"] == ["case-001"]
    assert overridden["run_config"]["case_selection"]["limit"] == 1


def test_max_cases_larger_than_the_suite_keeps_every_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _run_engine(monkeypatch, tmp_path, max_cases=10)

    assert captured["collect"]["expected_case_ids"] == list(CASES)
    assert _snapshot_ids(result) == list(CASES)
    assert result["run_config"]["case_selection"]["selected"] == 5
    assert result["run_config"]["case_selection"]["skipped"] == 0


@pytest.mark.parametrize("value", [0, -3, True])
def test_invalid_max_cases_is_rejected_before_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: object,
) -> None:
    result, captured = _run_engine(monkeypatch, tmp_path, max_cases=value)

    assert result["error"] == ["max_cases must be >= 1"]
    assert captured["pairs"] == []


def test_max_cases_is_configurable_per_skill(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    config_path = skill / "evals" / "config.yml"

    config_path.write_text("schema_version: 1\nharbor:\n  max_cases: 6\n", encoding="utf-8")
    config, _path = load_evals_config(skill)
    assert config["harbor"]["max_cases"] == 6

    for invalid in ("0", "true", "two"):
        config_path.write_text(f"schema_version: 1\nharbor:\n  max_cases: {invalid}\n", encoding="utf-8")
        with pytest.raises(EvalsConfigError, match=r"harbor\.max_cases"):
            load_evals_config(skill)


def test_payload_carries_the_recorded_case_selection() -> None:
    agents = {
        "codex": {
            "model": "test-model",
            "execution_status": "succeeded",
            "with_skill": {"overall": 0.8},
            "without_skill": {"overall": 0.4},
        }
    }
    selection = {"policy": "first_n", "limit": 2, "total": 5, "selected": 2, "skipped": 3, "selected_ids": ["a", "b"]}

    limited = build_agent_eval_payload("demo", agents, run_config={"case_selection": selection})
    unlimited = build_agent_eval_payload("demo", agents, run_config={})

    expected = {"policy": "first_n", "limit": 2, "total": 5, "selected": 2, "skipped": 3}
    assert limited is not None
    assert limited["case_selection"] == expected
    assert limited["summary"]["case_selection"] == expected
    assert unlimited is not None
    assert "case_selection" not in unlimited
    assert "case_selection" not in unlimited["summary"]


@pytest.mark.parametrize("command", [["evaluate"], ["tier3", "evaluate"]])
def test_evaluate_forwards_max_cases(monkeypatch: pytest.MonkeyPatch, command: list[str]) -> None:
    from skillevaluator.evaluation import EvaluationService

    captured: dict[str, Any] = {}

    def _fake_evaluate(self, options, *, progress_reporter=None):
        captured["options"] = options
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(cli, [*command, str(FIXTURE), "--progress", "off", "--max-cases", "3"])

    assert result.exit_code == 0, result.output
    assert captured["options"].engine_kwargs()["max_cases"] == 3


def test_evaluate_rejects_a_non_positive_max_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.evaluation import EvaluationService

    def _unreachable(self, options, *, progress_reporter=None):
        raise AssertionError("the engine must not run with an invalid case limit")

    monkeypatch.setattr(EvaluationService, "evaluate", _unreachable, raising=True)
    result = CliRunner().invoke(cli, ["tier3", "evaluate", str(FIXTURE), "--max-cases", "0"])

    assert result.exit_code == 2
    assert "Invalid value for '--max-cases'" in result.output


def test_validate_forwards_max_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    captured: dict[str, Any] = {}

    def _tier3(*_args: Any, **kwargs: Any) -> ValidationResult:
        captured.update(kwargs)
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "--no-autopilot",
            str(FIXTURE),
            "--no-llm",
            "--no-tier2",
            "--tier3",
            "--max-cases",
            "4",
            "--checks",
            "schema",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["max_cases"] == 4
