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
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
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
# A single path component (a skill or agent directory name) cannot contain "/",
# and on Windows it cannot contain ":" either, so the emoji name is POSIX-only there.
_WINDOWS = sys.platform == "win32"
_EMOJI_NAME_SUFFIX = "" if _WINDOWS else ":x:"
NAME_PAYLOADS = (
    "[bold]evil",
    "[green]PASS",
    pytest.param("demo:x:", marks=pytest.mark.skipif(_WINDOWS, reason="':' is not valid in Windows file names")),
)
# Rich turns ":x:", ":key:" and ":white_check_mark:" into emoji unless the console,
# Panel title or Status text disables it.
EMOJI_TEXT = "grep root:x:0:0:root:/root:/bin/bash :key: :white_check_mark:"
EMOJI_GLYPHS = ("\u274c", "\U0001f511", "\u2705")

# Terminal control sequences untrusted text could carry, and what stays visible.
CONTROL_PAYLOADS = {
    "\x1b[1A\x1b[2K": "",  # cursor up one line, erase it
    "\x1b[2J": "",  # clear the screen
    "\x1b]8;;https://evil.test/\x1b\\see docs\x1b]8;;\x1b\\": "see docs",  # OSC 8 hyperlink
    "\x1bc": "",  # RIS: full terminal reset
    "\x9b2J": "2J",  # 8-bit CSI
}
_SGR = re.compile(r"\x1b\[[0-9;]*m")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _recording_console(width: int = 200) -> Console:
    # Tests that print absolute tmp paths pass a wider width so long paths do not wrap.
    return Console(file=io.StringIO(), record=True, width=width)


def _terminal_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, color_system="truecolor", width=200)


def _terminal_text(output: str) -> str:
    """Drop Rich's own SGR styling and fail if any other control character is left."""
    text = _SGR.sub("", output)
    assert _CONTROL.search(text) is None, repr(text)
    return text


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


@pytest.mark.parametrize(("payload", "visible"), CONTROL_PAYLOADS.items())
def test_cli_report_strips_terminal_control_sequences(payload: str, visible: str) -> None:
    console = _terminal_console()
    reporter = CLIReporter(console=console)
    results = [
        _rubric_result(payload),
        _finding_result(payload),
        _agent_eval_result(payload),
        _advisory_skip_result(payload),
    ]

    for result in results:
        reporter.print(result)
    reporter.print_all(results)

    text = _terminal_text(console.file.getvalue())
    for label in ("Judge notes", "Finding message", "Judge reasoning", "Skipped because"):
        assert f"{label} {visible}".rstrip() in text


def test_trailing_backslashes_render_literally(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator import cli as cli_module

    console = _recording_console(width=250)
    monkeypatch.setattr(cli_module, "console", console)
    reporter = CLIReporter(console=console)
    errored = ValidationResult(validator_name="SCHEMA", validator_description="Schema checks")
    errored.add_error("Output directory is not writable: D:\\reports\\")
    # The judge explanation is cut at 80 characters, here right after a backslash.
    prefix = "Agent wrote its report to C:\\Users\\runner\\AppData\\Local\\Temp\\"
    explanation = prefix + "r" * (79 - len(prefix)) + "\\ and then exited without writing result.json"
    judged = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live agent evaluation")
    judged.metadata["agent_eval"] = {
        "verdict": "pass",
        "composite_lift": 0.1,
        "insights": {"goal": {"score": "PASS", "explanation": explanation}},
    }

    cli_module._print_run_banner(PureWindowsPath("\\\\fileserver\\skills\\"), "skill", None)
    reporter.print(errored)
    reporter.print(judged)

    lines = [line.rstrip(" │") for line in console.export_text().splitlines()]
    assert "Target: \\\\fileserver\\skills\\" in lines
    assert "  • Output directory is not writable: D:\\reports\\" in lines
    assert any(line.endswith(f" {explanation[:80]}") for line in lines)


def test_cli_reporter_consoles_print_emoji_codes_literally(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "250")
    results = [
        _rubric_result(EMOJI_TEXT),
        _finding_result(EMOJI_TEXT),
        _agent_eval_result(EMOJI_TEXT),
        _advisory_skip_result(EMOJI_TEXT),
    ]
    reporter = CLIReporter()  # the reporter's own default console

    for result in results:
        reporter.print(result)
    reporter.print_all(results)
    outputs = [capsys.readouterr().out, reporter.render(results[1]), reporter.render_all(results)]

    for output in outputs:
        text = _terminal_text(output)
        assert [glyph for glyph in EMOJI_GLYPHS if glyph in text] == []
        assert f"Finding message {EMOJI_TEXT}" in text
        assert f"Suggested fix {EMOJI_TEXT}" in text
    assert f"Judge notes {EMOJI_TEXT}" in outputs[0]
    assert f"Skipped because {EMOJI_TEXT}" in outputs[0]


def test_command_consoles_print_emoji_codes_literally(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.tier3 import commands as tier3_commands

    monkeypatch.setenv("COLUMNS", "200")

    # The run banner prints through the tier1 command console.
    cli_module._print_run_banner(PurePosixPath("/work/root:x:0:0:/skill"), "skill", "custom :key:")
    tier3_commands._print_validate_results(
        PurePosixPath("/work/demo:x:"),
        [SimpleNamespace(status="ok", path="evals/evals.json", message="found :white_check_mark:")],
    )

    text = _terminal_text(capsys.readouterr().out)
    assert [glyph for glyph in EMOJI_GLYPHS if glyph in text] == []
    assert "Target: /work/root:x:0:0:/skill" in text
    assert "Profile: custom :key:" in text
    assert "Validate: demo:x:/evals/" in text
    assert "found :white_check_mark:" in text


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

        for index, payload in enumerate(CONTROL_PAYLOADS):
            record = logging.LogRecord(
                named.name, logging.WARNING, __file__, 1, "Tool %d said: %s", (index, f"done{payload}"), None
            )
            handler.handle(record)

    text = console.export_text()
    for payload in PAYLOADS:
        assert text.count(f"Could not read skills/{payload}/SKILL.md") == 2
    lines = [line.rstrip() for line in text.splitlines()]
    for index, visible in enumerate(CONTROL_PAYLOADS.values()):
        assert sum(line.endswith(f"Tool {index} said: done{visible}") for line in lines) == 2
    assert _CONTROL.search(text) is None


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


def test_tier3_validate_panel_strips_terminal_control_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    console = _terminal_console()
    monkeypatch.setattr(tier3_commands, "console", console)

    tier3_commands._print_validate_results(
        tmp_path / "demo",
        [SimpleNamespace(status="error", path="evals/\x1b[2Jevals.json", message="bad\x1bc value")],
    )

    text = _terminal_text(console.file.getvalue())
    assert "evals/evals.json" in text
    assert "bad value" in text


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
    skill_name, agent = f"[green]PASS{_EMOJI_NAME_SUFFIX}", "[bold]evil"
    skill_path = tmp_path / skill_name
    skill_path.mkdir()
    results_root = tmp_path / "results"
    run_dir = results_root / skill_name / "20260709_010000"
    summary_dir = run_dir / agent / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "execution_status": "succeeded",
                "scores": {"security": 0.9, "custom: grader\x1b[2J": 0.5},
                "num_trials": "[/x]",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name, "agents": {}}), encoding="utf-8")

    assert tier3_commands.compare_results(skill_path, results_dir=results_root) == 0

    text = console.export_text()
    assert f"Skill Evaluation - {skill_name}" in text
    assert f"{agent:<16s} 20260709_010000 (Harbor, [/x] trials)" in text
    assert "custom: grader" in text
    assert "\x1b" not in text


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


def test_tier3_doctor_strips_terminal_control_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.provider_config import ProviderConfigurationError
    from skillevaluator.tier3 import commands as tier3_commands

    console = _terminal_console()
    monkeypatch.setattr(tier3_commands, "console", console)

    def _unconfigured_provider():
        raise ProviderConfigurationError("provider\x1b[1A\x1b[2K is not configured\x1bc")

    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", _unconfigured_provider)
    monkeypatch.setattr(
        tier3_commands,
        "_check_prerequisites",
        lambda **_kwargs: ["docker: \x1b]8;;https://evil.test/\x1b\\see docs\x1b]8;;\x1b\\"],
    )

    assert tier3_commands.doctor(agents=None, env_mode="local") == 1

    text = _terminal_text(console.file.getvalue())
    assert "provider is not configured" in text
    assert "docker: see docs" in text


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
    reward = {
        "entry_id": "case-001",
        **scores,
        "details": {"goal_accuracy": {"reason": "no result [/x]\x1b]8;;https://evil.test/\x1b\\"}},
    }
    (trial_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    monkeypatch.setattr(
        report,
        "_generate_suggestions_structured",
        lambda _skill, _findings, _rewards: [
            {"suggestion": "Write the result file [/x]\x1bc", "dimension": "goal_accuracy", "evidence_refs": []}
        ],
    )
    skill_name = "skill [bold]evil[/bold] :x:"

    report.display_findings_report(
        {
            "agents": {
                "codex": {"execution_status": "succeeded", "model": "gpt-[/x]:key:", "with_skill": scores},
                "opencode": {"execution_status": "succeeded", "model": "m", "with_skill": {"security": 0.1}},
            }
        },
        skill_name,
        ["codex", "opencode"],
        tmp_path,
    )

    output = _terminal_text(capsys.readouterr().out)
    assert "combination for your skill: codex / gpt-[/x]:key:" in output
    assert f"{skill_name} / codex / gpt-[/x]:key: — Findings" in output
    assert "no result [/x]" in output
    assert "Write the result file [/x]" in output


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

    assert all(isinstance(status, Text) for status in statuses)
    assert [status.plain for status in statuses] == [f"Evaluating {name} with LLM judge..."]
