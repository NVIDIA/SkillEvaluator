# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs with a baseline arm must pass their own run-scoped baseline alias check.

The runner checks baseline skill sources once per run, then hands that proof to
every baseline-style emitter call. The emitter only accepts the proof when it
is asked to stage the exact same source set: target skill, reference skills,
workspace skills and excluded roots. These tests use the real emitter and the
real check, so any drift between the two calls fails the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner


def _provider() -> ProviderConfig:
    return ProviderConfig(
        provider="openai",
        model="test-model",
        api_key="placeholder-value",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
        region=None,
    )


def _write_skill(root: Path, name: str, *, with_evals: bool = True) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Public test skill {name}\n---\n# {name}\n\nAnswer with the word done.\n",
        encoding="utf-8",
    )
    if with_evals:
        (skill / "evals").mkdir()
        (skill / "evals" / "evals.json").write_text(
            json.dumps(
                {"skill_name": name, "evals": [{"id": "case-1", "prompt": "Say done.", "expected_output": "done"}]}
            ),
            encoding="utf-8",
        )
    return skill


@pytest.fixture
def harbor_jobs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace Harbor execution with a recorder and return the job names it saw."""
    jobs: list[str] = []

    def record(**kwargs: Any) -> tuple[bool, str]:
        jobs.append(str(kwargs["job_name"]))
        return True, ""

    def render(_skill_path: Path, output_dir: Path, **_kwargs: Any) -> Path:
        report = output_dir / "report.html"
        report.write_text("<html></html>\n", encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "_run_harbor", record)
    monkeypatch.setattr(runner, "resolve_llm_provider", _provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "_harbor_supports_yes", lambda: True)
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_kwargs: {"execution_status": "complete", "execution_errors": [], "metrics": [], "agents": {}},
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", render)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "NVIDIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return jobs


def _arms(jobs: list[str]) -> list[str]:
    return sorted(job.rsplit("-", 1)[-1] for job in jobs)


def _run(skill: Path, **kwargs: Any) -> dict[str, Any]:
    return runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "openai/gpt-test"},
        env_mode="docker",
        keep_harbor_jobs=True,
        agent_runtime_preflight=False,
        n_attempts=1,
        n_concurrent=1,
        **kwargs,
    )


@pytest.mark.parametrize("custom_output_root", [False, True], ids=["default-root", "custom-root"])
def test_skill_run_stages_the_baseline_arm(
    harbor_jobs: list[str],
    tmp_path: Path,
    custom_output_root: bool,
) -> None:
    skill = _write_skill(tmp_path, "baseline-skill")
    kwargs: dict[str, Any] = {"output_dir": tmp_path / "custom-results"} if custom_output_root else {}

    result = _run(skill, **kwargs)

    assert result.get("error") is None, result.get("error")
    assert _arms(harbor_jobs) == ["with", "without"]
