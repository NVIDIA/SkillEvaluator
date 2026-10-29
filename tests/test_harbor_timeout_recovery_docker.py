# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Docker/Harbor recovery check; run explicitly with pytest -m integration.

Uses a deterministic agent, not an LLM provider. Requires Docker and access to
python:3.12-slim. No service credentials, private images, or datasets are used.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("harbor")
from harbor.agents.base import BaseAgent
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.harbor.runner import _merge_attempt_jobs


class RecoveryExampleAgent(BaseAgent):
    @staticmethod
    def name():
        return "recovery-example"

    def version(self):
        return "1"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        del instruction, context
        if "-with-task-attempt001" in str(self.logs_dir):
            await environment.exec("sleep 30", timeout_sec=40)
        elif "-with-" in str(self.logs_dir):
            await environment.exec("printf recovered > /app/answer.txt")
        else:
            await environment.exec("printf wrong > /app/answer.txt")


@pytest.mark.integration
def test_docker_timeout_retry_and_scored_wrong_baseline(tmp_path: Path):
    asyncio.run(_docker_timeout_retry_and_scored_wrong_baseline(tmp_path))


async def _docker_timeout_retry_and_scored_wrong_baseline(tmp_path: Path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment/Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    (task / "instruction.md").write_text("Write recovered to /app/answer.txt.\n")
    (task / "task.toml").write_text(
        "[agent]\ntimeout_sec = 3\n[verifier]\ntimeout_sec = 30\n[environment]\ncpus = 1\nmemory_mb = 512\n"
    )
    (task / "tests/test.sh").write_text("""#!/bin/bash
mkdir -p /logs/verifier
python - <<'CHECK'
import json
from pathlib import Path
answer = Path('/app/answer.txt')
score = float(answer.exists() and answer.read_text() == 'recovered')
metrics = ('security', 'skill_execution', 'skill_efficiency', 'accuracy', 'goal_accuracy', 'behavior_check')
Path('/logs/verifier/reward.json').write_text(json.dumps(dict.fromkeys(metrics, score)))
CHECK
""")
    failed_snapshot = None
    failed_file = None
    for arm, attempts in (("with", (1, 2)), ("without", (1, 2, 3))):
        jobs = []
        for attempt in attempts:
            job = tmp_path / "jobs" / f"demo-opencode-{arm}-task-attempt{attempt:03d}"
            config = TrialConfig(
                task={"path": task},
                trial_name=f"task__attempt{attempt:03d}",
                trials_dir=job,
                agent={"import_path": f"{__name__}:RecoveryExampleAgent"},
                environment={"type": "docker"},
            )
            trial = await Trial.create(config)
            result = await trial.run()
            native = JobResult(
                id=uuid4(),
                started_at=result.started_at,
                updated_at=result.finished_at,
                finished_at=result.finished_at,
                n_total_trials=1,
                stats=JobStats.from_trial_results([result], n_total_trials=1),
                trial_results=[result],
            )
            (job / "result.json").write_text(native.model_dump_json(indent=2))
            if arm == "with" and attempt == 1:
                assert result.exception_info.exception_type == "AgentTimeoutError"
                failed_file = job / config.trial_name / "result.json"
                failed_snapshot = failed_file.read_bytes()
            else:
                assert result.exception_info is None
            jobs.append(job)
        _merge_attempt_jobs(jobs, tmp_path / "jobs" / f"demo-opencode-{arm}")
    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=3,
        stop_on_pass=True,
        expected_cases=1,
        expected_case_ids=["task"],
        case_id_by_task_selector={"task": "task"},
    )
    assert results["execution_status"] == "succeeded", results.get("execution_errors")
    agent = results["agents"]["opencode"]
    assert agent["overall_with_skill"] == 1.0
    assert agent["overall_without_skill"] == 0.0
    assert agent["conditions"]["with_skill"]["recovered_attempts_total"] == 1
    assert results["scored_attempts"] == 4
    assert failed_file.read_bytes() == failed_snapshot
    assert load_agent_data(tmp_path / "results")["opencode"]["execution_errors"] == []
    skill = tmp_path / "demo"
    skill.mkdir()
    report = render_agent_eval_html_report(skill, tmp_path / "results", use_llm_judge=False)
    assert report.is_file()
    assert json.loads((tmp_path / "results/opencode/with-skill/summary.json").read_text())["overall_score"] == 1.0
