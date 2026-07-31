# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-facing Tier 3 result summary regressions."""

import io
import re
from pathlib import Path

import pytest
from rich.console import Console
from rich.panel import Panel

from skillevaluator.tier3.harbor.metrics import CUSTOM_ONLY_METRIC_SET, DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.result_display import _with_skill_overall, render_evaluation_result, render_result


def _skip_baseline_agent(with_skill: dict[str, float], *, pass_rate: float = 0.5) -> dict[str, object]:
    return {
        "execution_status": "succeeded",
        "with_skill": with_skill,
        "without_skill": {},
        "lift": {},
        "pass_at_k": {
            "with_skill": {"rate": pass_rate},
            "without_skill": {},
            "lift": {},
        },
        "conditions": {
            "with_skill": {"execution_status": "succeeded"},
            "without_skill": {"execution_status": "skipped"},
        },
    }


def test_success_output_has_scores_report_and_elapsed(tmp_path: Path) -> None:
    result = {
        "execution_status": "succeeded",
        "execution_errors": [],
        "duration_seconds": 12.4,
        "report_path": str(tmp_path / "report.html"),
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "lift": {"overall": {"with_skill": 0.85, "without_skill": 0.55, "delta": 0.3}},
                "pass_at_k": {
                    "with_skill": {"rate": 0.75},
                    "without_skill": {"rate": 0.5},
                },
            }
        },
    }

    output = render_result(result)

    assert "with skill" in output.lower()
    assert "no skill" in output.lower()
    assert "lift" in output.lower()
    assert "0.85" in output
    assert "+0.30" in output
    assert "report.html" in output
    assert "Time: 12.4s" in output


def test_skip_baseline_success_uses_canonical_default_metric_aggregate() -> None:
    result = {
        "execution_status": "succeeded",
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {
            "opencode": _skip_baseline_agent(
                {
                    "security": 1.0,
                    "skill_execution": 0.8,
                    "skill_efficiency": 0.6,
                    "accuracy": 0.7,
                    "goal_accuracy": 0.6666,
                    "behavior_check": 0.5,
                }
            )
        },
    }

    output = render_result(result)

    assert "Security" in output
    assert "1.00" in output
    assert "Behavior Check" in output
    assert "0.50" in output
    assert output.count("No Skill") == 1  # Dimension context remains explicit when baseline is skipped.
    assert "skipped" in output


def test_skip_baseline_custom_only_uses_persisted_attempt_scores() -> None:
    agent = _skip_baseline_agent({}, pass_rate=0.5)
    agent["pass_at_k"] = {
        "with_skill": {
            "rate": 0.5,
            "attempts_used": 4,
            "cases": {
                "case-001": {"attempts": [{"score": 0.4}, {"score": 0.8}]},
                "case-002": {"attempts": [{"score": 0.9}, {"score": 0.7}]},
            },
        },
        "without_skill": {},
        "lift": {},
    }
    result = {
        "execution_status": "succeeded",
        "metric_set": CUSTOM_ONLY_METRIC_SET,
        "metrics": [],
        "agents": {"custom-agent": agent},
    }

    output = render_result(result)

    assert "custom-agent" in output
    assert "Overall" in output
    assert "0.70" in output
    assert output.count("No Skill") == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_skip_baseline_aggregates_reject_nonfinite_scores(invalid: float) -> None:
    default_agent = _skip_baseline_agent(dict.fromkeys(DEFAULT_METRICS, 0.8))
    default_agent["with_skill"]["security"] = invalid  # type: ignore[index]
    custom_agent = _skip_baseline_agent({})
    custom_agent["pass_at_k"] = {
        "with_skill": {
            "attempts_used": 1,
            "cases": {"case-001": {"attempts": [{"score": invalid}]}},
        },
        "without_skill": {},
        "lift": {},
    }

    assert _with_skill_overall(default_agent, DEFAULT_METRIC_SET) is None
    assert _with_skill_overall(custom_agent, CUSTOM_ONLY_METRIC_SET) is None


def test_skip_baseline_preserves_real_zero_and_multiple_agents() -> None:
    result = {
        "execution_status": "succeeded",
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {
            "zero-agent": _skip_baseline_agent(dict.fromkeys(DEFAULT_METRICS, 0.0), pass_rate=0.0),
            "perfect-agent": _skip_baseline_agent(dict.fromkeys(DEFAULT_METRICS, 1.0), pass_rate=1.0),
        },
    }

    output = render_result(result)

    assert "zero-agent" in output
    assert "perfect-agent" in output
    assert "0.00" in output
    assert "1.00" in output
    assert "██████████" in output


def test_skip_baseline_does_not_coerce_missing_default_metric_to_zero() -> None:
    incomplete_scores = dict.fromkeys(DEFAULT_METRICS, 0.8)
    incomplete_scores.pop("behavior_check")
    result = {
        "execution_status": "succeeded",
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {"opencode": _skip_baseline_agent(incomplete_scores)},
    }

    output = render_result(result)

    assert re.search(r"Behavior Check\s+NO SCORE", output)
    assert output.count("0.80") == len(DEFAULT_METRICS) - 1
    assert "0.667" not in output


def test_skip_baseline_condition_overrides_stale_comparison_values() -> None:
    agent = _skip_baseline_agent(dict.fromkeys(DEFAULT_METRICS, 0.6))
    agent["lift"] = {"overall": {"with_skill": 0.9, "without_skill": 0.2, "delta": 0.7}}
    agent["pass_at_k"] = {
        "with_skill": {"rate": 0.5},
        "without_skill": {"rate": 0.8},
        "lift": {"delta": -0.3},
    }
    result = {
        "execution_status": "succeeded",
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {"opencode": agent},
    }

    output = render_result(result)

    assert output.count("0.60") == len(DEFAULT_METRICS)
    assert "0.900" not in output
    assert "0.200" not in output
    assert "+0.700" not in output
    assert "0.800" not in output


@pytest.mark.parametrize("agent_status", ["failed", "unknown"])
def test_skip_baseline_unusable_agent_does_not_synthesize_with_skill_score(agent_status: str) -> None:
    agent = _skip_baseline_agent(
        {
            "security": 1.0,
            "skill_execution": 0.8,
            "skill_efficiency": 0.6,
            "accuracy": 0.7,
            "goal_accuracy": 0.6666,
            "behavior_check": 0.5,
        }
    )
    agent["execution_status"] = agent_status
    agent["lift"] = {"overall": {"with_skill": 0.9, "without_skill": 0.2, "delta": 0.7}}
    agent["conditions"] = {
        "with_skill": {"execution_status": agent_status, "execution_errors": ["job incomplete"]},
        "without_skill": {"execution_status": "skipped"},
    }
    result = {
        "execution_status": agent_status,
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {"opencode": agent},
    }

    output = render_result(result)

    assert "opencode" in output
    assert re.search(r"Security\s+NO SCORE", output)
    assert "0.71" not in output
    assert "0.50" not in output
    assert "job incomplete" in output


@pytest.mark.parametrize("agent_status", ["failed", "unknown"])
def test_unusable_ab_agent_hides_all_stale_numeric_scores(agent_status: str) -> None:
    result = {
        "execution_status": agent_status,
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {
            "opencode": {
                "execution_status": agent_status,
                "lift": {"overall": {"with_skill": 0.9, "without_skill": 0.2, "delta": 0.7}},
                "pass_at_k": {
                    "with_skill": {"rate": 0.5},
                    "without_skill": {"rate": 0.8},
                },
                "conditions": {
                    "with_skill": {"execution_status": agent_status, "execution_errors": ["job incomplete"]},
                    "without_skill": {"execution_status": "succeeded"},
                },
            }
        },
    }

    output = render_result(result)

    assert "opencode" in output
    assert "0.20" in output
    for stale in ("0.90", "+0.70", "0.50", "0.80"):
        assert stale not in output


def test_failure_output_names_agent_condition_log_and_retained_command(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    result = {
        "execution_status": "failed",
        "execution_errors": ["opencode with-skill Harbor run failed: 401 Unauthorized"],
        "duration_seconds": 4.2,
        "harbor_jobs_dir": str(jobs),
        "harbor_jobs_retained": True,
        "agents": {
            "opencode": {
                "execution_status": "failed",
                "conditions": {
                    "with_skill": {"status": "failed", "detail": "agent exited"},
                    "without_skill": {"status": "not_run"},
                },
            }
        },
    }

    output = render_result(result)

    assert "opencode" in output
    assert "with-skill" in output
    assert "401 Unauthorized" in output
    assert f"skillevaluator tier3 harbor-view {jobs}" in output
    assert "Time: 4.2s" in output


def test_failure_output_surfaces_the_concrete_timed_out_trial() -> None:
    result = {
        "execution_status": "failed",
        "execution_errors": ["Harbor job did not complete successfully: 1 errored"],
        "agents": {
            "opencode": {
                "execution_status": "failed",
                "conditions": {
                    "with_skill": {
                        "execution_status": "failed",
                        "execution_errors": ["Scored attempt coverage is 3/4"],
                    },
                    "without_skill": {"execution_status": "skipped"},
                },
                "trial_failures": {
                    "with_skill": [
                        {
                            "trial": "case-002__attempt",
                            "reason": "AgentTimeoutError: Agent execution timed out after 300.0 seconds",
                        }
                    ],
                    "without_skill": [],
                },
            }
        },
    }

    output = render_result(result)

    assert "case-002__attempt" in output
    assert "AgentTimeoutError: Agent execution timed out after 300.0 seconds" in output


def test_result_display_never_recomputes_missing_scores() -> None:
    result = {
        "execution_status": "failed",
        "execution_errors": ["no scored trials"],
        "agents": {"opencode": {"execution_status": "failed", "with_skill": {"accuracy": 1.0}}},
    }

    output = render_result(result)

    assert "1.000" not in output
    assert "no scored trials" in output


def test_result_display_redacts_host_credentials_and_terminal_controls(monkeypatch) -> None:
    secret = "sk-AbCdEf1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = {
        "execution_status": "failed",
        "execution_errors": [f"credential={secret}\x1b[2JINJECT"],
        "agents": {},
    }

    output = render_result(result)

    assert secret not in output
    assert "\x1b" not in output
    assert "<redacted>" in output


def test_degraded_report_output_shows_warning_result_and_output_paths(tmp_path: Path) -> None:
    result = {
        "execution_status": "succeeded",
        "execution_errors": [],
        "warnings": ["HTML report was not generated: template failed"],
        "result_path": str(tmp_path / "result.json"),
        "run_dir": str(tmp_path),
        "duration_seconds": 1.0,
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": list(DEFAULT_METRICS),
        "agents": {"opencode": _skip_baseline_agent(dict.fromkeys(DEFAULT_METRICS, 0.6))},
    }

    output = render_result(result)

    assert "DEGRADED" in output
    assert output.count("0.60") == len(DEFAULT_METRICS)
    assert "HTML report was not generated" in output
    assert "Result JSON:" not in output
    assert "📁 Output" in output
    assert str(tmp_path) in output


def test_result_uses_horizontal_bars_and_includes_dimension_summary() -> None:
    result = {
        "execution_status": "succeeded",
        "duration_seconds": 8.5,
        "skill_name": "sample-skill",
        "metrics": ["security", "accuracy"],
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "with_skill": {"security": 1.0, "accuracy": 0.8},
                "without_skill": {"security": 0.5, "accuracy": 0.4},
                "lift": {
                    "security": {"with_skill": 1.0, "without_skill": 0.5, "delta": 0.5},
                    "accuracy": {"with_skill": 0.8, "without_skill": 0.4, "delta": 0.4},
                    "overall": {"with_skill": 0.9, "without_skill": 0.45, "delta": 0.45},
                },
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "succeeded"},
                },
                "dimensions_with_skill": {"security": {"score": 1.0}, "correctness": {"score": 0.8}},
                "dimensions_without_skill": {"security": {"score": 0.5}, "correctness": {"score": 0.4}},
            }
        },
    }

    output = render_result(result)

    assert output.index("Time: 8.5s") < output.index("Evaluator")
    assert "██████████" in output
    assert "Skill Lift" in output
    assert "Results by Evaluator" in output
    assert "Results by Dimension" in output
    assert "Results by Dimension" in output
    assert "Correctness" in output
    assert output.count("Results by Dimension") == 1


def test_single_agent_dimensions_use_a_separate_summary_panel() -> None:
    printed: list[object] = []

    class RecordingConsole:
        def print(self, value: object) -> None:
            printed.append(value)

    result = {
        "execution_status": "succeeded",
        "skill_name": "sample-skill",
        "metrics": ["security"],
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "with_skill": {"security": 1.0},
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "skipped"},
                },
                "dimensions_with_skill": {"security": {"score": 1.0}},
            }
        },
    }

    render_evaluation_result(result, console=RecordingConsole())  # type: ignore[arg-type]

    panels = [value for value in printed if isinstance(value, Panel)]
    assert len(panels) == 2
    rendered = Console(width=160, force_terminal=False).render_str(str(panels[0].title)).plain
    assert rendered == "Results by Evaluator"
    panel_output = render_result(result)
    assert "sample-skill / opencode" in panel_output
    assert panel_output.count("Results by Dimension") == 1


def test_result_title_infers_skill_from_real_run_directory_shape(tmp_path: Path) -> None:
    run_dir = tmp_path / "managing-teams" / "20260709_120000"
    result = {
        "execution_status": "succeeded",
        "run_dir": str(run_dir),
        "metrics": ["security"],
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "with_skill": {"security": 0.8},
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "skipped"},
                },
            }
        },
    }

    output = render_result(result)

    assert "managing-teams / opencode" in output


def test_result_bars_do_not_coerce_invalid_or_missing_scores_to_zero() -> None:
    result = {
        "execution_status": "succeeded",
        "metrics": ["security", "accuracy", "goal_accuracy"],
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "with_skill": {"security": True, "accuracy": float("nan")},
                "without_skill": {"security": 0.5, "accuracy": float("inf"), "goal_accuracy": 0.25},
                "lift": {
                    "security": {"delta": 0.5},
                    "accuracy": {"delta": 0.4},
                    "goal_accuracy": {"delta": 0.25},
                },
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "succeeded"},
                },
            }
        },
    }

    output = render_result(result)

    assert "nan" not in output.lower()
    assert "inf" not in output.lower()
    assert re.search(r"Security\s+NO SCORE\s+\s*0\.50\s+", output)
    assert re.search(r"Goal Accuracy\s+NO SCORE\s+\s*0\.25\s+", output)
    assert "+0.25" not in output


def test_failed_condition_hides_stale_metric_values() -> None:
    result = {
        "execution_status": "failed",
        "metrics": ["security"],
        "agents": {
            "opencode": {
                "execution_status": "failed",
                "with_skill": {"security": 0.91},
                "without_skill": {"security": 0.81},
                "lift": {"security": {"delta": 0.10}},
                "conditions": {
                    "with_skill": {"execution_status": "failed", "detail": "agent exited"},
                    "without_skill": {"execution_status": "succeeded"},
                },
            }
        },
    }

    output = render_result(result)

    assert "0.91" not in output
    # Conditions are gated independently: the completed baseline remains useful,
    # while the failed with-skill condition and its stale lift are hidden.
    assert "0.81" in output
    assert "+0.10" not in output
    assert "agent exited" in output


def test_failed_agent_without_diagnostic_says_no_score_and_explains_failure() -> None:
    output = render_result(
        {
            "execution_status": "failed",
            "metrics": ["security"],
            "agents": {"opencode": {"execution_status": "failed", "with_skill": {}}},
        }
    )

    assert "NO SCORE" in output
    assert "Findings" in output
    assert "opencode evaluation failed without diagnostic details" in output


def test_multi_agent_result_has_one_shared_dimension_panel() -> None:
    agents = {
        name: {
            "execution_status": "succeeded",
            "with_skill": {"security": score},
            "conditions": {
                "with_skill": {"execution_status": "succeeded"},
                "without_skill": {"execution_status": "skipped"},
            },
            "dimensions_with_skill": {"security": {"score": score}},
        }
        for name, score in (("opencode", 0.8), ("codex", 0.6))
    }

    output = render_result({"execution_status": "succeeded", "metrics": ["security"], "agents": agents})

    assert output.count("Results by Dimension") == 1
    assert "opencode" in output
    assert "codex" in output


def test_multi_agent_dimension_panel_preserves_agents_and_lift_at_80_columns() -> None:
    agents = {
        name: {
            "execution_status": "succeeded",
            "conditions": {
                "with_skill": {"execution_status": "succeeded"},
                "without_skill": {"execution_status": "succeeded"},
            },
            "dimensions_with_skill": {"security": {"score": with_score}},
            "dimensions_without_skill": {"security": {"score": baseline_score}},
        }
        for name, with_score, baseline_score in (
            ("opencode", 0.8, 0.4),
            ("codex", 0.7, 0.5),
        )
    }
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, width=80)

    render_evaluation_result(
        {"execution_status": "succeeded", "metrics": [], "agents": agents},
        console=console,
    )
    output = stream.getvalue()

    assert "Agent: opencode" in output
    assert "Agent: codex" in output
    assert "+0.40" in output
    assert "+0.20" in output


def test_compact_footer_is_last_and_omits_result_json(tmp_path: Path) -> None:
    report = tmp_path / "report.html"
    report.touch()
    output = render_result(
        {
            "execution_status": "succeeded",
            "duration_seconds": 1.0,
            "report_path": str(report),
            "result_path": str(tmp_path / "result.json"),
            "run_dir": str(tmp_path),
            "agents": {},
        }
    )

    assert "Result JSON:" not in output
    artifacts = output[output.rindex("Artifacts") :]
    normalized_artifacts = "".join(line.strip(" │") for line in artifacts.splitlines())
    assert str(tmp_path) in normalized_artifacts
    assert "…" not in artifacts
    assert artifacts.rstrip().splitlines()[-1].lstrip().startswith("╰")
    assert output.index("Time: 1.0s") < output.index("📊 HTML report")


@pytest.mark.parametrize("report_path", ["report.html", "./report.html", r".\report.html"])
def test_artifact_footer_does_not_duplicate_basename_only_report_path(report_path: str) -> None:
    output = render_result(
        {
            "execution_status": "succeeded",
            "report_path": report_path,
            "agents": {},
        }
    )

    assert output.count("report.html") == 1
    assert "report.html · report.html" not in output


def test_artifact_footer_uses_windows_report_basename() -> None:
    output = render_result(
        {
            "execution_status": "succeeded",
            "report_path": r"C:\reports\report.html",
            "agents": {},
        }
    )

    normalized = " ".join(output.split())
    assert r"report.html · C:\reports\report.html" in normalized
    assert r"C:\reports\report.html · C:\reports\report.html" not in normalized


def test_feedback_and_suggestions_render_before_artifacts(tmp_path: Path) -> None:
    report = tmp_path / "report.html"
    report.touch()
    result = {
        "execution_status": "succeeded",
        "report_path": str(report),
        "run_dir": str(tmp_path),
        "agents": {},
        "tier3_feedback": {
            "conclusions": [
                {"severity": "warn", "title": "Execution gap", "message": "The skill did not run."},
            ],
            "recommendations": [
                {"category": "Fix", "message": "Add the required authentication flow."},
                {"category": "Test", "message": "Add a successful summarization case."},
            ],
            "suggestions": ["Add the required authentication flow."],
        },
    }

    output = render_result(result)

    assert "Feedback & Suggestions" in output
    assert "Execution gap" in output
    assert "The skill did not run." in output
    assert "Add the required authentication flow." in output
    assert "Add a successful summarization case." in output
    assert output.count("Add the required authentication flow.") == 1
    assert output.index("Feedback & Suggestions") < output.index("Artifacts")


def test_detailed_findings_only_suppress_matching_payload_feedback(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import report as harbor_report

    monkeypatch.setattr(
        harbor_report,
        "display_findings_report",
        lambda *_args, **_kwargs: {
            "SHARED_CONCLUSION",
            "SHARED_RECOMMENDATION",
        },
    )
    result = {
        "execution_status": "succeeded",
        "skill_name": "simple",
        "run_dir": "/tmp/run",
        "agents": {
            "codex": {
                "execution_status": "succeeded",
                "with_skill": {"security": 1.0},
                "without_skill": {"security": 1.0},
                "lift": {"overall": {"with_skill": 1.0, "without_skill": 1.0, "delta": 0.0}},
            }
        },
        "tier3_feedback": {
            "conclusions": [
                {"severity": "warn", "message": "SHARED_CONCLUSION"},
                {"severity": "warn", "message": "UNIQUE_CONCLUSION_ONLY"},
            ],
            "recommendations": [
                {"message": "SHARED_RECOMMENDATION"},
                {"message": "UNIQUE_RECOMMENDATION_ONLY"},
            ],
        },
    }
    stream = io.StringIO()

    render_evaluation_result(result, console=Console(file=stream, force_terminal=False, width=180))
    output = stream.getvalue()

    assert output.count("UNIQUE_CONCLUSION_ONLY") == 1
    assert output.count("UNIQUE_RECOMMENDATION_ONLY") == 1
    assert "SHARED_CONCLUSION" not in output
    assert "SHARED_RECOMMENDATION" not in output
    assert output.count("Feedback & Suggestions") == 1


def test_feedback_display_redacts_secrets_controls_and_rich_markup(monkeypatch) -> None:
    secret = "sk-AbCdEf1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = {
        "execution_status": "succeeded",
        "agents": {},
        "tier3_feedback": {
            "conclusions": [],
            "recommendations": [
                {
                    "category": "Fix",
                    "message": f"[link=https://evil.example]credential={secret}\x1b]52;c;INJECT\x07[/link]",
                }
            ],
            "suggestions": [],
        },
    }
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system="standard", width=180)

    render_evaluation_result(result, console=console)
    output = stream.getvalue()

    assert secret not in output
    assert "<redacted>" in output
    assert "\x1b]52;" not in output
    assert "evil.example" in output
    assert "\x1b]8;" not in output


def test_feedback_overflow_does_not_claim_a_missing_html_report() -> None:
    output = render_result(
        {
            "execution_status": "succeeded",
            "warnings": ["HTML report was not generated"],
            "agents": {},
            "tier3_feedback": {
                "conclusions": [],
                "recommendations": [{"message": f"Suggestion {index}"} for index in range(6)],
            },
        }
    )

    assert "1 more suggestion(s) not shown" in output
    assert "1 more suggestion(s) in the HTML report" not in output


def test_feedback_display_supports_legacy_agent_eval_results() -> None:
    output = render_result(
        {
            "execution_status": "succeeded",
            "agents": {},
            "agent_eval": {
                "conclusions": [],
                "recommendations": [{"message": "Legacy recommendation"}],
            },
        }
    )

    assert "Feedback & Suggestions" in output
    assert "Legacy recommendation" in output


def test_inspect_jobs_requires_existing_retained_directory_and_quotes_path(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs with spaces"
    jobs.mkdir()
    retained = render_result(
        {
            "execution_status": "failed",
            "harbor_jobs_dir": str(jobs),
            "harbor_jobs_retained": True,
            "agents": {},
        }
    )
    deleted = render_result(
        {
            "execution_status": "failed",
            "harbor_jobs_dir": str(tmp_path / "missing"),
            "harbor_jobs_retained": True,
            "agents": {},
        }
    )

    assert "🔍 Inspect jobs" in retained
    assert f"skillevaluator tier3 harbor-view '{jobs}'" in retained
    assert "🔍 Inspect jobs" not in deleted


def test_bar_geometry_clamps_without_changing_numeric_truth() -> None:
    result = {
        "execution_status": "succeeded",
        "metrics": ["security", "accuracy"],
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "with_skill": {"security": 1.5, "accuracy": -0.25},
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "skipped"},
                },
            }
        },
    }

    output = render_result(result)

    assert re.search(r"Security\s+1\.50\s+██████████", output)
    assert re.search(r"Accuracy\s+-0\.25\s+░░░░░░░░░░", output)


def test_result_panel_title_cannot_inject_rich_markup_or_links() -> None:
    result = {
        "execution_status": "succeeded",
        "skill_name": "[link=https://evil.example]skill[/link]",
        "metrics": ["security"],
        "run_config": {
            "agents": {
                "opencode": {
                    "model": "[link=https://evil.example/model]model[/link]",
                    "source": "test",
                }
            }
        },
        "agents": {
            "opencode": {
                "execution_status": "succeeded",
                "with_skill": {"security": 1.0},
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "skipped"},
                },
            }
        },
    }

    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system="standard", width=180)
    render_evaluation_result(result, console=console)

    assert "evil.example" in stream.getvalue()
    assert "\x1b]8;" not in stream.getvalue()
