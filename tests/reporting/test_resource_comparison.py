# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def write_arm(run, arm, tokens, seconds, steps=None, score=1.0):
    directory = run / "opencode" / arm
    directory.mkdir(parents=True)
    scores = dict.fromkeys(DEFAULT_METRICS, score)
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
                "steps": steps if steps is not None else [{}],
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
    assert "-2.0 s (-" in html
    assert "N/A" in html if missing_tokens else "-10 (-" in html
    output = json.loads(JSONReporter().render_all([result]))
    assert "resources" in json.dumps(output)


def codex_steps(tool, llm_after, name="exec_command"):
    """A user prompt, a model call that runs one tool 2 s later, and a final answer."""
    final = 2 + tool + llm_after
    return [
        {"step_id": 1, "source": "user", "message": "go", "timestamp": "2026-10-11T00:00:00.000Z"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "timestamp": "2026-10-11T00:00:02.000Z",
            "tool_calls": [{"tool_call_id": "call_1", "function_name": name, "arguments": {"cmd": "ls"}}],
            "observation": {
                "results": [
                    {
                        "source_call_id": "call_1",
                        "content": f"Chunk ID: 0a1b2c\nWall time: {tool:.4f} seconds\n"
                        "Process exited with code 0\nOutput:\nok\n",
                    }
                ]
            },
        },
        {"step_id": 3, "source": "agent", "message": "done", "timestamp": f"2026-10-11T00:00:{final:06.3f}Z"},
    ]


def native_steps(llm_ms, tool_ms):
    return [
        {"step_id": 1, "source": "user", "message": "go"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "done",
            "extra": {"llm_duration_ms": llm_ms},
            "tool_calls": [{"tool_call_id": "call_1", "function_name": "Bash", "extra": {"duration_ms": tool_ms}}],
        },
    ]


def render(tmp_path, run):
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    resources = result.metadata["agent_eval"]["agents"]["opencode"]["resources"]
    html = HTMLReporter(target_path=str(tmp_path)).render_all([result])
    output = json.loads(JSONReporter().render_all([result]))
    return result, resources, html, output


@pytest.mark.parametrize("reused_baseline", [False, True])
def test_failed_task_outcomes_keep_observed_deltas_and_disclosures(tmp_path, reused_baseline):
    run = tmp_path / "run"
    write_arm(run, "with-skill", 10, 3, score=0.0)
    write_arm(run, "without-skill", 20, 5)
    if reused_baseline:
        (run / "run_config.json").write_text(
            json.dumps(
                {
                    "baseline": {
                        "mode": "reused",
                        "source_run_id": "earlier-run",
                        "source_evaluated_at": "2026-10-01T00:00:00Z",
                    }
                }
            )
        )
    result, resources, html, output = render(tmp_path, run)
    agent = output["tier3"]["agents"]["opencode"]
    assert agent["execution_status"] == "succeeded"
    assert agent["trials"][0]["scores"]["goal_accuracy"] == 0.0
    assert agent["trials"][0]["attempt_verdict"]["passed"] is False
    assert agent["trials_baseline"][0]["scores"]["goal_accuracy"] == 1.0
    assert agent["trials_baseline"][0]["attempt_verdict"]["passed"] is True
    assert agent["resources"]["tokens"]["delta"] == -10
    assert agent["resources"]["agent_seconds"]["delta"] == -2
    assert "-10 (-45%)" in html
    assert "-2.0 s (-40%)" in html
    assert "including failed task outcomes" in resources["scope"]
    assert "including failed task outcomes" in html
    for disclosure in (
        "Lower use alone does not establish improved task quality or Efficiency",
        "Pairing does not certify matched concurrency or budget, or judge-model agreement",
        "Reused baseline measurements are historical",
    ):
        assert disclosure in agent["resources"]["interpretation"]
        assert disclosure in html
    if reused_baseline:
        assert output["tier3"]["token_usage"]["baseline_reused"] is True
    # Reports retained before interpretation metadata existed still disclose its limits.
    del resources["interpretation"]
    legacy_html = HTMLReporter(target_path=str(tmp_path)).render_all([result])
    assert "Lower use alone does not establish improved task quality or Efficiency" in legacy_html


def test_llm_and_tool_seconds_flow_from_trajectories_to_json_and_html(tmp_path):
    run = tmp_path / "run"
    write_arm(run, "with-skill", 10, 3, steps=codex_steps(tool=1.5, llm_after=4.5))
    write_arm(run, "without-skill", 20, 5, steps=codex_steps(tool=2.5, llm_after=6.5))
    result, resources, html, output = render(tmp_path, run)
    trials = result.metadata["agent_eval"]["agents"]["opencode"]["trials"]
    assert trials[0]["timing"] == {"llm_seconds": 6.5, "tool_seconds": 1.5, "source": "step_timestamps"}
    assert resources["llm_seconds"]["with_skill"]["total"] == 6.5
    assert resources["llm_seconds"]["baseline"]["total"] == 8.5
    assert resources["llm_seconds"]["delta"] == -2.0
    assert resources["tool_seconds"]["delta"] == -1.0
    # write_arm records prompt + 2 completion tokens: 12 with the skill, 22 without.
    assert resources["llm_ms_per_token"]["with_skill"]["value"] == round(6500 / 12, 3)
    assert resources["llm_ms_per_token"]["baseline"]["value"] == round(8500 / 22, 3)
    assert resources["timing_sources"] == {"with_skill": ["step_timestamps"], "baseline": ["step_timestamps"]}
    assert resources["caveats"] == []
    for label in ("LLM seconds", "Tool seconds", "LLM ms per token"):
        assert label in html
    assert "-2.0 s (-24%)" in html
    assert "-1.0 s (-40%)" in html
    assert "541.67 ms" in html
    assert (
        "opencode timing source: with skill inferred from step timestamps; baseline inferred from step timestamps."
        in html
    )
    assert "not like-for-like" not in html
    assert '"llm_seconds"' in json.dumps(output)


def test_unmeasured_arm_timing_is_na_without_a_delta(tmp_path):
    run = tmp_path / "run"
    write_arm(run, "with-skill", 10, 3, steps=codex_steps(tool=1.5, llm_after=4.5))
    write_arm(run, "without-skill", 20, 5, steps=codex_steps(tool=2.5, llm_after=6.5, name="update_plan"))
    _result, resources, html, _output = render(tmp_path, run)
    assert resources["llm_seconds"]["baseline"] == {
        "status": "unavailable",
        "measured_trials": 0,
        "trials": 1,
        "total": None,
    }
    assert resources["llm_seconds"]["delta"] is None
    assert resources["agent_seconds"]["delta"] == -2
    assert "<td>opencode</td><td>LLM seconds</td>" in html
    assert "1/1 / 0/1" in html
    assert "baseline none." in html


def test_mixed_timing_sources_are_flagged_in_html(tmp_path):
    run = tmp_path / "run"
    write_arm(run, "with-skill", 10, 3, steps=codex_steps(tool=1.5, llm_after=4.5))
    write_arm(run, "without-skill", 20, 5, steps=native_steps(llm_ms=7000, tool_ms=2000))
    _result, resources, html, _output = render(tmp_path, run)
    assert resources["llm_seconds"]["delta"] == -0.5
    assert resources["caveats"] == ["mixed_timing_sources"]
    assert "with skill inferred from step timestamps; baseline native ATIF timing." in html
    assert "so LLM and tool seconds are not like-for-like" in html
