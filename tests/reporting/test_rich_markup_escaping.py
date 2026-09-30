# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Untrusted text must render literally in Rich console output.

Skill names, descriptions, paths, finding messages, LLM rubric notes, judge
reasoning, and external tool output are interpolated into Rich markup strings.
Unescaped, ``[/x]`` raises ``rich.errors.MarkupError`` and aborts the report,
while ``[bold]...[/bold]`` or ``[link=...]...[/link]`` restyles or spoofs it.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import CLIReporter

PAYLOADS = ("[/x]", "[bold]evil[/bold]", "[link=http://x]y[/link]")
# pathlib collapses "//", so path payloads avoid it; a payload with "/" spans
# directory components ("[" and "x]" for "[/x]").
PATH_PAYLOADS = ("[/x]", "[bold]evil[/bold]")
# A single path component (a skill or agent directory name) cannot contain "/".
NAME_PAYLOADS = ("[bold]evil", "[green]PASS")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _recording_console(width: int = 200) -> Console:
    # Tests that print absolute tmp paths pass a wider width so long paths do not wrap.
    return Console(file=io.StringIO(), record=True, width=width)


def _quality_result(payload: str) -> ValidationResult:
    result = ValidationResult(validator_name="QUALITY", validator_description=f"Quality of {payload}")
    dimensions = {name: {"score": 90.0} for name in ("correctness", "discoverability", "reliability", "efficiency")}
    result.metadata["quality_scores"] = {
        "overall_score": 88.0,
        "grade": "B",
        "skill_count": 2,
        "dimensions": dimensions,
    }
    result.metadata["quality_scores_all"] = [
        {"skill_name": f"alpha {payload}", "grade": "A", "overall_score": 91.0, "dimensions": dimensions},
        {"skill_name": f"beta {payload}", "grade": "B", "overall_score": 85.0, "dimensions": dimensions},
    ]
    return result


def _rubric_result(payload: str) -> ValidationResult:
    result = ValidationResult(validator_name="RUBRIC_EVAL", validator_description="LLM rubric evaluation")
    result.metadata["rubric_eval"] = {
        "overall_score": 64.0,
        "summary": f"Judge summary {payload}",
        "checks": [{"id": "scope_definition", "score": 6, "pass": False, "notes": f"Judge notes {payload}"}],
    }
    result.add_error(f"Rubric error {payload}")
    return result


def _finding_result(payload: str) -> ValidationResult:
    result = ValidationResult(validator_name="SECURITY", validator_description="Security checks")
    result.add_finding(
        Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="mcp-least-privilege",
            message=f"Finding message {payload}",
            file_path=f"skills/{payload}/SKILL.md",
            line_number=3,
            line_content=f"content {payload}",
            suggestion=f"Suggested fix {payload}",
        )
    )
    return result


def _agent_eval_result(payload: str) -> ValidationResult:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live agent evaluation")
    result.metadata["agent_eval"] = {
        "verdict": "pass",
        "composite_lift": 0.2,
        "runtime_seconds": 3.0,
        "harbor_viewer": {
            "job_url": f"https://harbor.example.test/jobs/{payload}",
            "analysis_url": f"https://harbor.example.test/analysis/{payload}",
        },
        "evaluators": {f"custom {payload}": {"with_skill": 0.9, "baseline": 0.5, "lift": 0.4}},
        "recommendations": [
            {
                "message": f"Recommendation {payload}",
                "evidence": {"url": f"https://harbor.example.test/trials/{payload}", "label": f"trial {payload}"},
            }
        ],
        "insights": {f"dim {payload}": {"score": f"PASS {payload}", "explanation": f"Judge reasoning {payload}"}},
    }
    return result


def _advisory_skip_result(payload: str) -> ValidationResult:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live agent evaluation")
    result.metadata["agent_eval"] = {
        "provenance": {"advisory": True, "reason": "skipped", "message": f"Skipped because {payload}"}
    }
    return result


def _incomplete_result(payload: str) -> ValidationResult:
    result = ValidationResult(validator_name=f"Scanner {payload}", validator_description="External scanner")
    result.mark_scan_incomplete(f"tool {payload}")
    return result


@pytest.mark.parametrize("payload", PAYLOADS)
def test_cli_report_renders_untrusted_text_literally(payload: str) -> None:
    console = _recording_console()
    reporter = CLIReporter(console=console)
    results = [
        _quality_result(payload),
        _rubric_result(payload),
        _finding_result(payload),
        _agent_eval_result(payload),
        _advisory_skip_result(payload),
        _incomplete_result(payload),
    ]

    for result in results:
        reporter.print(result)  # every per-result section
    reporter.print_all(results)  # summary table, failure details, overall verdict

    html = console.export_html(clear=False)
    text = console.export_text()
    expected = [
        f"alpha {payload}",  # quality table skill names
        f"beta {payload}",
        f"Quality of {payload}",  # validator description
        f"Judge summary {payload}",  # rubric summary (LLM)
        f"Judge notes {payload}",  # rubric notes (LLM)
        f"Rubric error {payload}",
        f"Finding message {payload}",
        f"skills/{payload}/SKILL.md",
        f"content {payload}",
        f"Suggested fix {payload}",
        f"https://harbor.example.test/jobs/{payload}",
        f"https://harbor.example.test/analysis/{payload}",
        f"custom {payload}".title(),  # evaluator (custom grader metric) name
        f"Recommendation {payload}",
        f"View trial {payload}",  # evidence link text
        f"https://harbor.example.test/trials/{payload}",
        f"dim {payload}".title(),  # judge insight dimension
        f"PASS {payload}",  # judge insight verdict string
        f"Judge reasoning {payload}",
        f"Skipped because {payload}",  # advisory skip provenance
        f"[Scanner {payload}]",  # result header
        f"tool {payload} did not complete",  # incomplete scanners
    ]
    assert [snippet for snippet in expected if snippet not in text] == []
    assert 'href="http://x"' not in html


def test_escaping_does_not_disturb_trusted_status_markup() -> None:
    console = _recording_console()
    reporter = CLIReporter(console=console)
    passing = ValidationResult(validator_name="SCHEMA", validator_description="Schema checks")
    failing = _finding_result("[/x]")

    reporter.print_all([passing, failing])

    text = console.export_text()
    assert "[SCHEMA]" not in text  # passing results only appear in the summary table
    assert "[SECURITY]" in text
    assert "[FAIL] Validation failed" in text
    assert "\\[" not in text  # no escape backslashes leak into the output


def test_log_handlers_render_untrusted_text_literally(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator import logging_config

    console = _recording_console()
    monkeypatch.setattr(logging_config, "console", console)
    root = logging.getLogger()
    # setup_logging() rewires the root logger; keep pytest's handlers and levels intact.
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "level", root.level)
    for name in ("httpx", "urllib3"):
        noisy = logging.getLogger(name)
        monkeypatch.setattr(noisy, "level", noisy.level)
    named = logging.getLogger("skillevaluator.tests.rich_markup_escaping")
    monkeypatch.setattr(named, "handlers", [])
    monkeypatch.setattr(named, "propagate", named.propagate)
    monkeypatch.setattr(named, "level", named.level)

    logging_config.get_logger(named.name)
    logging_config.setup_logging()
    handlers = [*named.handlers, *(h for h in root.handlers if isinstance(h, RichHandler))]
    assert len(handlers) == 2

    for handler in handlers:
        for payload in PAYLOADS:
            record = logging.LogRecord(
                named.name, logging.WARNING, __file__, 1, "Could not read %s", (f"skills/{payload}/SKILL.md",), None
            )
            handler.handle(record)

    text = console.export_text()
    for payload in PAYLOADS:
        assert text.count(f"Could not read skills/{payload}/SKILL.md") == 2


@pytest.mark.parametrize("payload", PATH_PAYLOADS)
def test_validate_run_banner_renders_target_and_profile_literally(
    payload: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator import cli as cli_module

    console = _recording_console()
    monkeypatch.setattr(cli_module, "console", console)
    target = Path("/work/skills") / payload

    cli_module._print_run_banner(target, "skill", f"custom {payload}")

    text = console.export_text()
    assert f"Target: {target}" in text
    assert f"Profile: custom {payload}" in text


@pytest.mark.parametrize("payload", PATH_PAYLOADS)
def test_report_path_announcement_renders_literally(
    payload: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier1 import commands as tier1_commands

    console = _recording_console(width=1000)
    monkeypatch.setattr(tier1_commands, "console", console)
    output_dir = tmp_path / payload
    output_dir.mkdir(parents=True)
    result = ValidationResult(validator_name="SCHEMA", validator_description="Schema checks")

    tier1_commands.emit_reports([result], report_formats=("json",), output_dir=output_dir)

    assert f"json report: {output_dir / 'skillevaluator-output.json'}" in console.export_text()


@pytest.mark.parametrize("name", NAME_PAYLOADS)
def test_tier3_validate_panel_renders_skill_name_literally(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    console = _recording_console()
    monkeypatch.setattr(tier3_commands, "console", console)

    tier3_commands._print_validate_results(
        tmp_path / name, [SimpleNamespace(status="ok", path="evals/evals.json", message="found")]
    )

    assert f"Validate: {name}/evals/" in console.export_text()


@pytest.mark.parametrize("payload", PATH_PAYLOADS)
def test_tier3_grader_starter_renders_created_path_literally(
    payload: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    console = _recording_console(width=1000)
    monkeypatch.setattr(tier3_commands, "console", console)
    skill_path = tmp_path / payload
    skill_path.mkdir(parents=True)

    assert (
        tier3_commands.init_custom_grader(
            skill_path, language="python", mode="custom_only", force=False, no_config=True
        )
        == 0
    )
    assert (
        tier3_commands.init_custom_grader(
            skill_path, language="python", mode="custom_only", force=False, no_config=True
        )
        == 1
    )

    text = console.export_text()
    grader_path = skill_path.resolve() / "evals" / "grader.py"
    assert f"custom grader starter at {grader_path}" in text
    assert f"custom grader already exists: {grader_path}." in text


def test_tier3_compare_renders_agent_and_summary_values_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    console = _recording_console()
    monkeypatch.setattr(tier3_commands, "console", console)
    skill_name, agent = "[green]PASS", "[bold]evil"
    skill_path = tmp_path / skill_name
    skill_path.mkdir()
    results_root = tmp_path / "results"
    run_dir = results_root / skill_name / "20260709_010000"
    summary_dir = run_dir / agent / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.9}, "num_trials": "[/x]"}),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name, "agents": {}}), encoding="utf-8")

    assert tier3_commands.compare_results(skill_path, results_dir=results_root) == 0

    text = console.export_text()
    assert f"Skill Evaluation - {skill_name}" in text
    assert f"{agent:<16s} 20260709_010000 (Harbor, [/x] trials)" in text


def test_tier3_doctor_renders_provider_and_prerequisite_errors_literally(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.provider_config import ProviderConfigurationError
    from skillevaluator.tier3 import commands as tier3_commands

    console = _recording_console()
    monkeypatch.setattr(tier3_commands, "console", console)

    def _unconfigured_provider():
        raise ProviderConfigurationError("provider [/x] is not configured")

    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", _unconfigured_provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: ["docker: [bold]evil[/bold]"])

    assert tier3_commands.doctor(agents=None, env_mode="local") == 1

    text = console.export_text()
    assert "provider [/x] is not configured" in text
    assert "docker: [bold]evil[/bold]" in text


def test_harbor_findings_report_renders_skill_and_model_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from skillevaluator.tier3.harbor import report

    monkeypatch.setenv("COLUMNS", "200")
    scores = {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "goal_accuracy": 0.1,
        "behavior_check": 1.0,
    }
    condition_dir = tmp_path / "codex" / "with-skill"
    trial_dir = condition_dir / "trials" / "case-001"
    trial_dir.mkdir(parents=True)
    (condition_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "codex",
                "scores": scores,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    reward = {"entry_id": "case-001", **scores, "details": {"goal_accuracy": {"reason": "no result [/x]"}}}
    (trial_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    monkeypatch.setattr(
        report,
        "_generate_suggestions_structured",
        lambda _skill, _findings, _rewards: [
            {"suggestion": "Write the result file [/x]", "dimension": "goal_accuracy", "evidence_refs": []}
        ],
    )
    skill_name = "skill [bold]evil[/bold]"

    report.display_findings_report(
        {
            "agents": {
                "codex": {"execution_status": "succeeded", "model": "gpt-[/x]", "with_skill": scores},
                "opencode": {"execution_status": "succeeded", "model": "m", "with_skill": {"security": 0.1}},
            }
        },
        skill_name,
        ["codex", "opencode"],
        tmp_path,
    )

    output = _ANSI.sub("", capsys.readouterr().out)
    assert "combination for your skill: codex / gpt-[/x]" in output
    assert f"{skill_name} / codex / gpt-[/x] — Findings" in output
    assert "no result [/x]" in output


@pytest.mark.parametrize("name", NAME_PAYLOADS)
def test_rubric_progress_status_renders_skill_name_literally(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.validators.rubric_eval import RubricEvalValidator, RubricJudge

    statuses: list[str] = []

    class _RecordingStatus:
        def __init__(self, status: str, **_kwargs: object) -> None:
            statuses.append(status)

        def __enter__(self) -> _RecordingStatus:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr("rich.status.Status", _RecordingStatus)
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: demo\ndescription: test\n---\n# Demo\n", encoding="utf-8")

    with patch.object(RubricJudge, "process", return_value={}):
        RubricEvalValidator().validate(skill_dir)

    assert [Text.from_markup(status).plain for status in statuses] == [f"Evaluating {name} with LLM judge..."]
