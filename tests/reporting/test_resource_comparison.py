# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def write_arm(run, arm, tokens, seconds):
    directory = run / "opencode" / arm
    directory.mkdir(parents=True)
    scores = dict.fromkeys(DEFAULT_METRICS, 1.0)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "scores": scores,
                "metrics": list(DEFAULT_METRICS),
                "num_trials": 1,
                "expected_attempts": 1,
                "scored_attempts": 1,
                "execution_status": "succeeded",
            }
        )
    )
    trial = directory / "trials" / "case__1"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps({"entry_id": "case", **scores}))
    (trial / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [{}],
                "final_metrics": {
                    "total_prompt_tokens": tokens,
                    "total_completion_tokens": 2,
                    "total_cached_tokens": 3,
                },
            }
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {"agent_execution": {"started_at": "2026-10-11T00:00:00Z", "finished_at": f"2026-10-11T00:00:0{seconds}Z"}}
        )
    )
    return trial


@pytest.mark.parametrize("missing_tokens", [True, False])
def test_resources_flow_from_harbor_artifacts_to_json_and_html(tmp_path, missing_tokens):
    run = tmp_path / "run"
    write_arm(run, "with-skill", 10, 3)
    baseline = write_arm(run, "without-skill", 20, 5)
    if missing_tokens:
        (baseline / "trajectory.json").unlink()
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    resources = result.metadata["agent_eval"]["agents"]["opencode"]["resources"]
    assert resources["agent_seconds"]["delta"] == -2
    assert resources["tokens"]["delta"] == (None if missing_tokens else -10)
    assert resources["tokens"]["with_skill"]["total"] == 12
    html = HTMLReporter(target_path=str(tmp_path)).render_all([result])
    assert "Measured Agent Resources" in html
    assert "Agent execution seconds" in html
    assert "-2.00" in html
    assert "N/A" in html if missing_tokens else "-10.00" in html
    output = json.loads(JSONReporter().render_all([result]))
    assert "resources" in json.dumps(output)
