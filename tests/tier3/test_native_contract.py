# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.coverage import ContractError
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.native_contract import (
    finalize_contract,
    seal_plan,
    validate_contract_requests,
    validate_evidence_bindings,
)


def _staged_task(root: Path, *, with_skill: bool) -> Path:
    task = root / "case-1"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[metadata]\nentry_id = "case-1"\n', encoding="utf-8")
    (task / "instruction.md").write_text("Evaluate the skill.\n", encoding="utf-8")
    if with_skill:
        payload = task / "environment" / "skills" / "demo"
        payload.mkdir(parents=True)
        (payload / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    return task


def _sealed(
    tmp_path: Path,
    *,
    agent_entries: list[dict] | None = None,
    policy: str = "all-selected",
    required_agents: tuple[str, ...] = (),
    grading_mode: str = "default",
    evals_config: dict | None = None,
):
    skill = tmp_path / "demo"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    evals_file = evals / "evals.json"
    evals_file.write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Use the demo skill",
                        "expected_output": "done",
                        "assertions": ["done"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if grading_mode != "default":
        (evals / "grader.py").write_text("# sealed custom grader\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    with_root = run_dir / "staged" / "with_skill"
    baseline_root = run_dir / "staged" / "baseline"
    with_task = _staged_task(with_root, with_skill=True)
    _staged_task(baseline_root, with_skill=False)
    sealed = seal_plan(
        run_dir=run_dir,
        run_id="run-1",
        skill_path=skill,
        task_source="evals_json",
        evals_file=evals_file,
        native_harbor_dir=evals / "harbor",
        evals_config=evals_config or {"schema_version": 1},
        grading_mode=grading_mode,
        agent_entries=agent_entries
        or [
            {
                "result_agent": "codex",
                "agent": "codex",
                "occurrence": 1,
                "model": "gpt-5",
                "model_source": "default",
            }
        ],
        task_paths=[with_task],
        with_skill_root=with_root,
        baseline_root=baseline_root,
        skip_baseline=False,
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        agent_validity_policy=policy,
        min_valid_agents=None,
        required_agents=required_agents,
        contract_requests=("agent-coverage/1", "tier3-result/3"),
    )
    return run_dir, sealed


def _agent_result() -> dict:
    scores = {
        "security": 0.8,
        "skill_execution": 0.8,
        "skill_efficiency": 0.8,
        "accuracy": 0.8,
        "goal_accuracy": 0.8,
        "behavior_check": 0.8,
    }
    condition = {
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 1,
        "scored_attempts": 1,
    }
    attempt = {"attempt": 1, "trial": "trial-1", "score": 0.8, "passed": True}
    return {
        "execution_status": "succeeded",
        "with_skill": scores,
        "without_skill": dict.fromkeys(scores, 0.6),
        "conditions": {"with_skill": condition, "without_skill": condition},
        "pass_at_k": {
            "with_skill": {"rate": 1.0, "cases": {"case-1": {"attempts": [attempt]}}},
            "without_skill": {"rate": 1.0, "cases": {"case-1": {"attempts": [attempt]}}},
        },
    }


def test_any_valid_runner_excludes_optional_provider_before_staging_and_emits_v3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = tmp_path / "demo"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Use the demo skill",
                        "expected_output": "done",
                        "assertions": ["done"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provider = ProviderConfig(
        provider="openai",
        model="gpt-5.4-mini",
        api_key="provider-test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-5.4-mini",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    staged_agents: list[str] = []

    def fake_generate(_skill_path: Path, output_dir: Path, *, with_skill: bool, runtime_env: dict, **_kwargs):
        staged_agents.append("claude-code" if "ANTHROPIC_API_KEY" in runtime_env else "codex")
        task = output_dir / "case-1"
        task.mkdir(parents=True)
        (task / "task.toml").write_text('[metadata]\nentry_id = "case-1"\n', encoding="utf-8")
        (task / "instruction.md").write_text("Use the demo skill.\n", encoding="utf-8")
        if with_skill:
            payload = task / "environment" / "skills" / "demo"
            payload.mkdir(parents=True)
            (payload / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
        return [task]

    monkeypatch.setattr(runner, "generate_harbor_tasks", fake_generate)
    launched_agents: list[str] = []
    monkeypatch.setattr(
        runner,
        "_run_agent_pair",
        lambda **kwargs: launched_agents.append(str(kwargs["agent"])) or [],
    )

    def fake_collect(**kwargs):
        assert kwargs["agents"] == ["codex"]
        result = _agent_result()
        result["without_skill"] = {}
        result["conditions"]["without_skill"] = {
            "execution_status": "skipped",
            "execution_errors": [],
            "expected_attempts": 0,
            "scored_attempts": 0,
        }
        return {
            "execution_status": "succeeded",
            "execution_errors": [],
            "metrics": list(DEFAULT_METRICS),
            "agents": {"codex": result},
        }

    monkeypatch.setattr(runner, "collect_harbor_results", fake_collect)

    def fake_report(_skill_path: Path, run_dir: Path, **_kwargs) -> Path:
        report = run_dir / "report.html"
        report.write_text("<html></html>\n", encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "render_agent_eval_html_report", fake_report)
    monkeypatch.setattr(runner, "record_agent_eval_summary", lambda **_kwargs: None)

    result = runner.run_harbor_eval(
        skill,
        ["codex", "claude-code"],
        skip_baseline=True,
        agent_runtime_preflight=False,
        env_mode="docker",
        output_dir=tmp_path / "results",
        agent_models={"claude-code": "claude-sonnet-4-5"},
        agent_validity_policy="any-valid",
        min_valid_agents=1,
        required_agents=("codex",),
        contract_requests=("agent-coverage/1", "tier3-result/3"),
        tier3_evidence_mode=True,
    )

    assert staged_agents == ["codex"]
    assert launched_agents == ["codex"]
    assert result["coverage_status"] == "valid_degraded"
    assert result["eligible_agents"] == ["codex"]
    assert result["excluded_agents"] == ["claude-code"]
    assert result["tier3_result"]["agents"]["claude-code"]["with_skill"] is None
    assert result["evidence_job_status"] == "succeeded"
    run_dir = Path(result["run_dir"])
    for name in ("expected_attempt_plan.json", "execution_ledger.json", "agent_coverage.json", "tier3-result.json"):
        assert (run_dir / name).is_file()


def test_all_agent_task_roots_must_match_the_sealed_plan(tmp_path: Path) -> None:
    run_dir, sealed = _sealed(tmp_path)
    second_with = run_dir / "_harbor-tasks" / "opencode" / "with"
    second_baseline = run_dir / "_harbor-tasks" / "opencode" / "without"
    shutil.copytree(run_dir / "staged" / "with_skill", second_with)
    shutil.copytree(run_dir / "staged" / "baseline", second_baseline)
    task_roots = {
        "codex": (run_dir / "staged" / "with_skill", run_dir / "staged" / "baseline"),
        "opencode": (second_with, second_baseline),
    }

    runner._verify_agent_task_roots(
        run_dir=run_dir,
        plan=sealed.plan,
        agent_task_dirs=task_roots,
        skip_baseline=False,
    )

    (second_with / "case-1" / "instruction.md").write_text("Different agent bytes.\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"opencode.*digest disagrees"):
        runner._verify_agent_task_roots(
            run_dir=run_dir,
            plan=sealed.plan,
            agent_task_dirs=task_roots,
            skip_baseline=False,
        )


def test_dataset_controls_are_removed_from_every_agent_root(tmp_path: Path) -> None:
    roots: dict[str, tuple[Path, Path | None]] = {}
    for agent in ("codex", "opencode"):
        with_root = tmp_path / agent / "with"
        baseline_root = tmp_path / agent / "without"
        for root in (with_root, baseline_root):
            root.mkdir(parents=True)
            (root / "dataset.toml").write_text("[datasets]\n", encoding="utf-8")
            (root / "metric.py").write_text("# generated\n", encoding="utf-8")
        roots[agent] = (with_root, baseline_root)

    runner._remove_staged_dataset_controls(roots)

    for with_root, baseline_root in roots.values():
        assert baseline_root is not None
        for root in (with_root, baseline_root):
            assert not (root / "dataset.toml").exists()
            assert not (root / "metric.py").exists()


def test_evidence_runner_rejects_agent_specific_task_bytes_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = tmp_path / "demo"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Use the demo skill",
                        "expected_output": "done",
                        "assertions": ["done"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provider = ProviderConfig(
        provider="openai",
        model="gpt-5.4-mini",
        api_key="provider-test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-5.4-mini",
    )
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    def fake_generate(_skill_path: Path, output_dir: Path, *, with_skill: bool, **_kwargs):
        task = output_dir / "case-1"
        task.mkdir(parents=True)
        (task / "task.toml").write_text('[metadata]\nentry_id = "case-1"\n', encoding="utf-8")
        agent_marker = "opencode" if "opencode" in output_dir.parts else "codex"
        (task / "instruction.md").write_text(f"Use the demo skill with {agent_marker}.\n", encoding="utf-8")
        if with_skill:
            payload = task / "environment" / "skills" / "demo"
            payload.mkdir(parents=True)
            (payload / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
        return [task]

    monkeypatch.setattr(runner, "generate_harbor_tasks", fake_generate)
    launched_agents: list[str] = []
    monkeypatch.setattr(
        runner,
        "_run_agent_pair",
        lambda **kwargs: launched_agents.append(str(kwargs["agent"])) or [],
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        skip_baseline=True,
        agent_runtime_preflight=False,
        env_mode="docker",
        output_dir=tmp_path / "results",
        contract_requests=("agent-coverage/1", "tier3-result/3"),
        tier3_evidence_mode=True,
    )

    assert launched_agents == []
    assert result["error"]
    assert "staged task integrity check failed for agent opencode" in result["error"][0]
    assert "digest disagrees with the immutable plan" in result["error"][0]


def test_contract_request_pair_is_all_or_nothing() -> None:
    assert validate_contract_requests(()) == ()
    assert validate_contract_requests(("agent-coverage/1", "tier3-result/3"))
    with pytest.raises(ContractError, match="requires both"):
        validate_contract_requests(("agent-coverage/1",))


def test_pipeline_binding_formats_fail_closed() -> None:
    validate_evidence_bindings(
        occurrence_id="skill-eval/occ-1",
        expected_content_digest="sha256:" + "a" * 64,
        validated_sha="b" * 40,
        gate_policy_digest="sha256:" + "c" * 64,
    )
    with pytest.raises(ContractError, match="validated_sha"):
        validate_evidence_bindings(
            occurrence_id=None,
            expected_content_digest=None,
            validated_sha="not-a-sha",
            gate_policy_digest=None,
        )


def test_custom_grader_and_declared_metric_are_bound_into_plan(tmp_path: Path) -> None:
    _run_dir, sealed = _sealed(
        tmp_path,
        grading_mode="default_plus_custom",
        evals_config={
            "schema_version": 1,
            "grading": {"mode": "default_plus_custom", "custom_metrics": ["domain_quality"]},
        },
    )

    reward = sealed.plan["reward_contract"]
    assert reward["custom_metrics"] == [{"name": "domain_quality", "range": "unit_interval"}]
    assert reward["custom_grader_schema_digest"].startswith("sha256:")


def test_native_contract_publishes_schema_valid_full_result(tmp_path: Path) -> None:
    run_dir, sealed = _sealed(tmp_path)
    result = finalize_contract(
        run_dir=run_dir,
        sealed=sealed,
        results={"agents": {"codex": _agent_result()}},
        skill_name="demo",
        environment="local",
        duration_seconds=1.5,
    )

    assert result["coverage_status"] == "valid_full"
    assert result["quality"] == "pass"
    assert result["tier3_result"]["schema_version"] == "3.0"
    assert result["tier3_result"]["overall_score"] == pytest.approx(0.8)
    assert (run_dir / "expected_attempt_plan.json").is_file()
    assert (run_dir / "execution_ledger.json").is_file()
    assert (run_dir / "agent_coverage.json").is_file()
    assert (run_dir / "tier3-result.json").is_file()


@pytest.mark.parametrize("missing_field", ["score", "passed"])
def test_missing_collector_attempt_field_raises_contract_error(tmp_path: Path, missing_field: str) -> None:
    run_dir, sealed = _sealed(tmp_path)
    agent = _agent_result()
    del agent["pass_at_k"]["with_skill"]["cases"]["case-1"]["attempts"][0][missing_field]

    with pytest.raises(
        ContractError,
        match=rf"collector attempt codex/with_skill/case-1/1.*{missing_field}",
    ):
        finalize_contract(
            run_dir=run_dir,
            sealed=sealed,
            results={"agents": {"codex": agent}},
            skill_name="demo",
            environment="local",
            duration_seconds=1.5,
        )

    assert not (run_dir / "execution_ledger.json").exists()
    assert not (run_dir / "tier3-result.json").exists()


def test_invalid_coverage_never_synthesizes_zero_score(tmp_path: Path) -> None:
    run_dir, sealed = _sealed(tmp_path)
    failed = _agent_result()
    failed["execution_status"] = "failed"
    failed["conditions"]["with_skill"] = {
        "execution_status": "failed",
        "execution_errors": ["trial failed"],
        "expected_attempts": 1,
        "scored_attempts": 0,
    }
    failed["pass_at_k"]["with_skill"]["cases"]["case-1"]["attempts"] = []

    result = finalize_contract(
        run_dir=run_dir,
        sealed=sealed,
        results={"agents": {"codex": failed}},
        skill_name="demo",
        environment="local",
        duration_seconds=1.0,
    )

    assert result["coverage_status"] == "invalid"
    assert result["quality"] == "not_evaluated"
    assert result["tier3_result"]["overall_score"] is None
    assert result["tier3_result"]["agents"]["codex"]["with_skill"] is None


def test_optional_presemantic_agent_failure_is_valid_degraded(tmp_path: Path) -> None:
    run_dir, sealed = _sealed(
        tmp_path,
        policy="any-valid",
        agent_entries=[
            {
                "result_agent": "codex",
                "agent": "codex",
                "occurrence": 1,
                "model": "gpt-5",
                "model_source": "default",
            },
            {
                "result_agent": "opencode",
                "agent": "opencode",
                "occurrence": 1,
                "model": "openai/gpt-5",
                "model_source": "default",
            },
        ],
    )
    failed = _agent_result()
    failed["execution_status"] = "failed"
    failed["agent_runtime_failures"] = {"with_skill": [{"reason": "adapter failed"}]}
    failed["conditions"]["with_skill"] = {
        "execution_status": "failed",
        "execution_errors": ["adapter failed"],
        "expected_attempts": 1,
        "scored_attempts": 0,
    }
    failed["pass_at_k"]["with_skill"]["cases"]["case-1"]["attempts"] = []

    result = finalize_contract(
        run_dir=run_dir,
        sealed=sealed,
        results={"agents": {"codex": _agent_result(), "opencode": failed}},
        skill_name="demo",
        environment="local",
        duration_seconds=2.0,
    )

    assert result["coverage_status"] == "valid_degraded"
    assert result["eligible_agents"] == ["codex"]
    assert result["tier3_result"]["agents"]["opencode"]["with_skill"] is None
    assert result["agent_coverage"]["warnings"][0]["code"] == "optional_agent_excluded"


def test_required_agent_failure_invalidates_any_valid_policy(tmp_path: Path) -> None:
    run_dir, sealed = _sealed(
        tmp_path,
        policy="any-valid",
        required_agents=("opencode",),
        agent_entries=[
            {
                "result_agent": "codex",
                "agent": "codex",
                "occurrence": 1,
                "model": "gpt-5",
                "model_source": "default",
            },
            {
                "result_agent": "opencode",
                "agent": "opencode",
                "occurrence": 1,
                "model": "openai/gpt-5",
                "model_source": "default",
            },
        ],
    )
    failed = _agent_result()
    failed["execution_status"] = "failed"
    failed["agent_runtime_failures"] = {"with_skill": [{"reason": "adapter failed"}]}

    result = finalize_contract(
        run_dir=run_dir,
        sealed=sealed,
        results={"agents": {"codex": _agent_result(), "opencode": failed}},
        skill_name="demo",
        environment="local",
        duration_seconds=2.0,
    )

    assert result["coverage_status"] == "invalid"
    assert result["tier3_result"]["quality"]["status"] == "not_evaluated"


def test_result_path_cannot_escape_run_directory(tmp_path: Path) -> None:
    run_dir, sealed = _sealed(tmp_path)

    with pytest.raises(ContractError, match="confined"):
        finalize_contract(
            run_dir=run_dir,
            sealed=sealed,
            results={"agents": {"codex": _agent_result()}},
            skill_name="demo",
            environment="local",
            duration_seconds=1.0,
            result_file=Path("../escaped.json"),
        )

    assert not (tmp_path / "escaped.json").exists()
