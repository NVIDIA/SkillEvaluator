# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""New Harbor features must not silently weaken Gym execution controls."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.test_nemo_gym_backend import _evaluate, _fake_gym_skill, _run_fake_gym


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("reuse_baseline", "latest"),
        ("compare_to", "latest"),
        ("max_agent_tokens", 100),
        ("max_judge_calls", 2),
        ("max_attempts_total", 2),
        ("max_cost_usd", 0.25),
        ("max_agent_tokens", 0),
        ("max_judge_calls", 0),
        ("max_attempts_total", 0),
        ("max_cost_usd", 0.0),
        ("price_file", Path("prices.json")),
    ],
)
def test_gym_dispatch_rejects_unsupported_baseline_and_budget_options(monkeypatch, tmp_path, option, value):
    with pytest.raises(ValueError, match=option.replace("_", "-")):
        _evaluate(monkeypatch, tmp_path, backend="nemo-gym", **{option: value})


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("reuse-baseline", "latest"),
        ("compare-to", "latest"),
        ("max-agent-tokens", "100"),
        ("max-judge-calls", "2"),
        ("max-attempts-total", "2"),
        ("max-cost-usd", "0.25"),
        ("max-agent-tokens", "0"),
        ("max-judge-calls", "0"),
        ("max-attempts-total", "0"),
        ("max-cost-usd", "0"),
        ("price-file", "prices.json"),
    ],
)
def test_gym_direct_workflow_rejects_unsupported_options_before_generation(monkeypatch, tmp_path, option, value):
    from skillevaluator.cli import evaluate
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.tier3.workflow import build_tier3_workflow

    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    dataset = skill / "evals" / "evals.json"
    dataset.unlink()
    if option == "price-file":
        price_file = tmp_path / value
        price_file.write_text("{}")
        value = str(price_file)
    extra_args = []
    if option == "max-cost-usd":
        price_file = tmp_path / "prices.json"
        price_file.write_text("{}")
        extra_args = ["--price-file", str(price_file)]
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("unsupported Gym options must fail before generation"),
    )
    result = CliRunner().invoke(
        build_tier3_workflow(evaluate),
        [
            str(skill),
            "--backend",
            "nemo-gym",
            "--agents",
            "codex",
            "--progress",
            "off",
            f"--{option}",
            value,
            *extra_args,
        ],
    )
    assert result.exit_code != 0, result.output
    assert f"--{option}" in result.output, result.output
    assert "not supported" in result.output, result.output
    assert not dataset.exists()


@pytest.mark.parametrize(
    ("option", "config"),
    [
        ("max_agent_tokens", "max_agent_tokens: 100"),
        ("max_judge_calls", "max_judge_calls: 2"),
        ("max_attempts_total", "max_attempts_total: 2"),
        ("max_cost_usd", "max_cost_usd: 0.25\n  price_file: prices.json"),
        ("max_agent_tokens", "max_agent_tokens: 0"),
        ("max_judge_calls", "max_judge_calls: 0"),
        ("max_attempts_total", "max_attempts_total: 0"),
        ("max_cost_usd", "max_cost_usd: 0\n  price_file: prices.json"),
        ("price_file", "price_file: prices.json"),
        ("tolerate_baseline_timeouts", "tolerate_baseline_timeouts: true"),
    ],
)
@pytest.mark.parametrize("entrypoint", ["runner", "workflow"])
def test_gym_rejects_config_budget_before_generation_or_execution(monkeypatch, tmp_path, option, config, entrypoint):
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, f"schema_version: 1\nharbor:\n  {config}\n")
    if entrypoint == "runner":
        result = _run_fake_gym(skill, tmp_path)
        assert result.get("error"), result
        assert f"harbor.{option}" in "; ".join(result["error"]), result
    else:
        from skillevaluator.cli import evaluate
        from skillevaluator.evaluation import EvaluationService
        from skillevaluator.tier3.workflow import build_tier3_workflow

        dataset = skill / "evals" / "evals.json"
        dataset.unlink()
        monkeypatch.setattr(
            EvaluationService,
            "create_autopilot_dataset",
            lambda *_args, **_kwargs: pytest.fail("unsupported Gym config must fail before generation"),
        )
        result = CliRunner().invoke(
            build_tier3_workflow(evaluate),
            [str(skill), "--backend", "nemo-gym", "--agents", "codex", "--progress", "off"],
        )
        assert result.exit_code != 0, result.output
        assert f"harbor.{option}" in result.output, result.output
        assert "not supported" in result.output, result.output
        assert not dataset.exists()
    assert not (tmp_path / "results").exists()
    assert not env_log.exists()


def test_gym_dispatch_rejects_enabled_baseline_timeout_tolerance(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="--tolerate-baseline-timeouts is not supported with --backend nemo-gym"):
        _evaluate(monkeypatch, tmp_path, backend="nemo-gym", tolerate_baseline_timeouts=True)
    # Off is Gym's own behavior, so an explicit --no-tolerate-baseline-timeouts reaches the runner.
    calls = _evaluate(monkeypatch, tmp_path, backend="nemo-gym", tolerate_baseline_timeouts=False)
    assert calls["nemo-gym"]["tolerate_baseline_timeouts"] is False


@pytest.mark.parametrize(
    ("flag", "config", "expected"),
    [
        (True, "", "--tolerate-baseline-timeouts is not supported"),
        (None, "tolerate_baseline_timeouts: true", "harbor.tolerate_baseline_timeouts is not supported"),
        (False, "tolerate_baseline_timeouts: true", None),
        (None, "tolerate_baseline_timeouts: false", None),
    ],
)
def test_gym_preflight_baseline_timeout_tolerance_flag_overrides_config(tmp_path, flag, config, expected):
    from skillevaluator.tier3.workflow import nemo_gym_preflight_errors

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "config.yml").write_text(f"schema_version: 1\nharbor:\n  {config}\n" if config else "")
    params = {"agents": "codex", "env_mode": "docker", "tolerate_baseline_timeouts": flag}
    tolerance_errors = [error for error in nemo_gym_preflight_errors(tmp_path, params) if "tolerate" in error]
    assert tolerance_errors == ([f"{expected} with --backend nemo-gym yet"] if expected else [])


def test_gym_runner_rejects_enabled_tolerance_but_lets_the_flag_override_config(monkeypatch, tmp_path):
    skill, env_log = _fake_gym_skill(
        tmp_path, monkeypatch, "schema_version: 1\nharbor:\n  tolerate_baseline_timeouts: true\n"
    )
    rejected = _run_fake_gym(skill, tmp_path, tolerate_baseline_timeouts=True)
    assert "--tolerate-baseline-timeouts is not supported with --backend nemo-gym yet" in rejected["error"], rejected
    assert not any("harbor.tolerate_baseline_timeouts" in error for error in rejected["error"]), rejected
    assert not (tmp_path / "results").exists()
    assert not env_log.exists()

    allowed = _run_fake_gym(skill, tmp_path, tolerate_baseline_timeouts=False)
    assert allowed.get("execution_status") == "succeeded", allowed


@pytest.mark.parametrize(
    "runtime_env",
    ["FOO: ${SKILL_EVAL_TEST_UNSET_RUNTIME_VAR}", "MY_API_KEY: ${SKILL_EVAL_TEST_UNSET_RUNTIME_VAR}"],
)
def test_gym_rejects_bad_runtime_env_before_reserving_a_run_folder(monkeypatch, tmp_path, runtime_env):
    monkeypatch.delenv("SKILL_EVAL_TEST_UNSET_RUNTIME_VAR", raising=False)
    skill, env_log = _fake_gym_skill(
        tmp_path, monkeypatch, f"schema_version: 1\nharbor:\n  runtime_env:\n    {runtime_env}\n"
    )
    result = _run_fake_gym(skill, tmp_path)
    assert result.get("error"), result
    assert not (tmp_path / "results").exists()
    assert not env_log.exists()


@pytest.mark.parametrize(("cli_value", "config"), [(0, ""), (None, "schema_version: 1\nharbor:\n  max_agents: 0\n")])
def test_gym_rejects_max_agents_below_one_like_harbor(monkeypatch, tmp_path, cli_value, config):
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, config)
    result = _run_fake_gym(skill, tmp_path, max_agents=cli_value)
    assert "max_agents must be >= 1" in "; ".join(result.get("error") or []), result
    assert not (tmp_path / "results").exists()
    assert not env_log.exists()


def test_gym_dispatch_passes_max_agents(monkeypatch, tmp_path):
    calls = _evaluate(monkeypatch, tmp_path, backend="nemo-gym", max_agents=3)
    assert calls["nemo-gym"]["max_agents"] == 3
