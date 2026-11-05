# SPDX-License-Identifier: Apache-2.0
"""Operator-visible terminal evidence, wrapping and progress contracts."""

import io

import pytest
from rich.console import Console
from rich.text import Text

from skillevaluator.models import ValidationResult
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.console_ui import ValidateView, Verdict, detail_row, lift_row, summarize_tier1


@pytest.mark.parametrize(
    ("grade", "style"), [("A", "#76B900"), ("B", "#76B900"), ("C", "#E5A33B"), ("D", "#F5524C"), ("F", "#F5524C")]
)
def test_compact_quality_grade_has_semantic_color(grade, style):
    result = ValidationResult(
        validator_name="Quality", passed=True, metadata={"quality_scores": {"grade": grade, "overall_score": 65}}
    )
    rows = summarize_tier1([result])[1]
    assert style in next(row for row in rows if row.label == "quality").segments[0][1]


@pytest.mark.parametrize("lift", [0.0, 0.004, -0.004])
def test_displayed_zero_lift_is_neutral(lift):
    assert "#76B900" not in lift_row(lift, 0.8, 0.8).segments[0][1]
    assert "#F5524C" not in lift_row(lift, 0.8, 0.8).segments[0][1]


@pytest.mark.parametrize("width", [98, 80, 60])
def test_each_lift_condition_keeps_label_number_and_complete_bar(width):
    out = io.StringIO()
    view = ValidateView(
        "試験🧪", [(3, "Live Agent Eval", "live")], console=Console(file=out, width=width, color_system=None)
    )
    view.tier_done(0, rows=[detail_row("長い条件名🧪非常に長いラベル", "測定結果 0.85"), lift_row(0.25, 0.85, 0.60)])
    lines = out.getvalue().splitlines()
    for label, score in [("with-skill", "0.85"), ("baseline", "0.60")]:
        condition = next(line for line in lines if label in line)
        assert score in condition
        assert sum(condition.count(c) for c in "█░") >= 10
    assert all(Text(line).cell_len <= width for line in lines)


def test_full_summary_keeps_successful_quality_and_lift_evidence():
    quality = ValidationResult(
        validator_name="Quality", passed=True, metadata={"quality_scores": {"grade": "B", "overall_score": 87.5}}
    )
    live = ValidationResult(
        validator_name="AGENT_EVAL", passed=True, metadata={"agent_eval": {"verdict": "pass", "composite_lift": 0.25}}
    )
    output = io.StringIO()
    CLIReporter(console=Console(file=output, width=120)).print_all([quality, live])
    text = output.getvalue()
    assert "B" in text and "87.5/100" in text and "+0.25" in text
    assert "Quality Scores by Skill" not in text


def test_noninteractive_progress_is_durable_deduplicated_and_separate(monkeypatch):
    progress = io.StringIO()
    output = io.StringIO()
    monkeypatch.setattr("sys.stderr", progress)
    view = ValidateView("demo", [(1, "Static", "static")], console=Console(file=output, width=80))
    try:
        view.start()
        view.tier_start(0)
        view.tier_progress(0, [detail_row("checks", "schema running")])
        view.tier_progress(0, [detail_row("checks", "schema running")])
        assert "Tier 1" in progress.getvalue() and "schema running" in progress.getvalue()
        assert progress.getvalue().count("schema running") == 1
        view.tier_done(0)
        view.finish(Verdict(True, "Passed required checks"), [("report", "/tmp/actual-name.html")])
    finally:
        view.stop()
    assert "complete" in progress.getvalue()
    assert "Passed required checks" in output.getvalue()
    assert "still running" not in output.getvalue()
    assert "\x1b]8" not in output.getvalue()


def test_plain_progress_heartbeat_waits_for_idle_and_deduplicates(monkeypatch):
    from skillevaluator.tier3.harbor import progress

    clock = [100.0]
    monkeypatch.setattr(progress, "monotonic", lambda: clock[0])
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output)
    try:
        reporter.start(progress.Tier3RunPlan("demo", "docker", ()))
        worker = reporter._thread
        event = progress.ProgressEvent("agent", "running", "one measured trial")
        reporter.emit(event)
        reporter.emit(event)
        clock[0] = 159
        reporter.heartbeat()
        assert "still running" not in output.getvalue()
        clock[0] = 160
        reporter.heartbeat()
        assert output.getvalue().count("still running") == 1
        clock[0] = 180
        reporter.emit(progress.ProgressEvent("agent", "running", "two measured trials"))
        clock[0] = 220
        reporter.heartbeat()
        assert output.getvalue().count("still running") == 1
        assert output.getvalue().count("one measured trial") == 1
    finally:
        reporter.close()
    assert not reporter.is_active
    assert not worker.is_alive()
    assert reporter._thread is None


def test_plain_progress_closed_stream_stops_worker():
    from skillevaluator.tier3.harbor.progress import PlainProgressReporter, ProgressEvent, Tier3RunPlan

    output = io.StringIO()
    reporter = PlainProgressReporter(stream=output)
    reporter.start(Tier3RunPlan("demo", "docker", ()))
    output.close()
    reporter.emit(ProgressEvent("agent", "running"))
    reporter.close()
    assert reporter._thread is None
    assert not reporter.is_active


@pytest.mark.parametrize("width", [98, 80, 60])
def test_direct_tier3_preserves_conditions_and_bars_at_narrow_width(width):
    from skillevaluator.tier3.result_display import render_evaluation_result

    output = io.StringIO()
    result = {
        "execution_status": "succeeded",
        "agents": {
            "試験🧪" * 10: {
                "execution_status": "succeeded",
                "with_skill": {"accuracy": 0.85},
                "without_skill": {"accuracy": 0.60},
                "lift": {
                    "accuracy": {"delta": 0.25},
                    "overall": {"with_skill": 0.85, "without_skill": 0.60, "delta": 0.25},
                },
            }
        },
    }
    render_evaluation_result(result, console=Console(file=output, width=width, color_system=None))
    lines = output.getvalue().splitlines()
    for number in ("0.85", "0.60"):
        condition = next(line for line in lines if number in line)
        assert sum(condition.count(c) for c in "█░") >= 10
    assert "+0.25" in output.getvalue()
    assert all(Text(line).cell_len <= width for line in lines)


def test_interactive_footer_links_require_capability_and_preserve_real_path(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    for stream, no_color, expect_link in [(TTY(), False, True), (TTY(), True, False), (io.StringIO(), False, False)]:
        console = Console(file=stream, force_terminal=True, no_color=no_color, color_system="truecolor", width=98)
        view = ValidateView("demo", [], console=console)
        view.links = [("report", "/tmp/actual [name].html")]
        console.print(next(view._footer_lines()))
        text = stream.getvalue()
        assert "actual [name].html" in text
        assert ("\x1b]8" in text) is expect_link


def test_noninteractive_check_ticker_reports_only_active_check(monkeypatch):
    from skillevaluator.reporting.console_ui import check_ticker_row

    output = io.StringIO()
    monkeypatch.setattr("sys.stderr", output)
    view = ValidateView("demo", [(1, "Static", "static")], console=Console(file=io.StringIO()))
    try:
        view.start()
        view.tier_start(0)
        view.tier_progress(0, [check_ticker_row(["schema", "quality", "license"], [], "schema")])
        assert "schema" in output.getvalue()
        assert "license" not in output.getvalue()
    finally:
        view.stop()


@pytest.mark.parametrize(
    "target", ["https://[invalid", "javascript:alert(1)", "/tmp/name\x1b]8;;https://evil\x1b\\.html"]
)
def test_unsafe_footer_target_is_readable_without_hyperlink(monkeypatch, target):
    from skillevaluator.reporting.terminal_styles import artifact_text

    monkeypatch.setenv("TERM", "xterm-256color")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    stream = TTY()
    console = Console(file=stream, force_terminal=True, no_color=False)
    console.print(artifact_text(target, console))
    assert "\x1b]8" not in stream.getvalue()


@pytest.mark.parametrize(
    ("passed", "extra", "expected"),
    [
        (False, {}, "#F5524C"),
        (False, {"incomplete_scans": ["gitleaks"]}, "#E5A33B"),
        (True, {"skipped": True}, "#8A8F98"),
    ],
)
def test_quality_grade_cannot_override_failed_incomplete_or_skipped_status(passed, extra, expected):
    result = ValidationResult(
        validator_name="Quality",
        passed=passed,
        metadata={
            **extra,
            "quality_scores": {"grade": "A", "overall_score": 95},
        },
    )
    row = next(row for row in summarize_tier1([result])[1] if row.label == "quality")
    assert expected in row.segments[0][1]


def test_full_agent_verdict_keeps_displayed_zero_lift_neutral():
    console = Console(file=io.StringIO(), width=100)
    rendered = []
    original_print = console.print

    def capture(value="", *args, **kwargs):
        if isinstance(value, str) and "composite lift" in value:
            rendered.append(Text.from_markup(value))
        original_print(value, *args, **kwargs)

    console.print = capture
    CLIReporter._print_agent_eval_tables({"verdict": "pass", "composite_lift": 0.004}, console)
    text = rendered[0]
    style = text.get_style_at_offset(console, text.plain.index("+0.00"))
    assert style.color.name == "#8a8f98"


def test_verdict_badge_background_does_not_fill_headline():
    from rich.console import Group

    view = ValidateView("demo", [])
    for verdict in (Verdict(True, "Passed required checks"), Verdict(False, "Changes required")):
        panel = view._verdict_panel(verdict)
        assert isinstance(panel.renderable, Group)
        body = panel.renderable.renderables[0]
        style = body.get_style_at_offset(view.console, body.plain.index(verdict.headline))
        assert style.bgcolor is None
