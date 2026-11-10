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
