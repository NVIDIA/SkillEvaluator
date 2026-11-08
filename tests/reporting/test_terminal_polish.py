# SPDX-License-Identifier: Apache-2.0
"""Operator-visible terminal evidence, wrapping and progress contracts."""

import io
import math

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
    # Manual fake-clock steps must not race the autonomous heartbeat worker.
    monkeypatch.setattr(reporter, "_heartbeat_loop", reporter._stop.wait)
    try:
        reporter.start(progress.Tier3RunPlan("demo", "docker", ()))
        worker = reporter._thread
        event = progress.ProgressEvent("agent", "running", "one measured trial")
        reporter.emit(event)
        clock[0] = 130
        reporter.emit(event)
        clock[0] = math.nextafter(160.0, 100.0)
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
        clock[0] = math.nextafter(240.0, 180.0)
        reporter.heartbeat()
        assert output.getvalue().count("still running") == 1
        clock[0] = 240.0
        reporter.heartbeat()
        assert output.getvalue().count("still running") == 2
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


@pytest.mark.parametrize(
    ("is_tty", "no_color", "legacy_windows", "expect_link"),
    [(True, False, False, True), (True, True, False, False), (False, False, False, False), (True, False, True, False)],
    ids=["modern-tty", "no-color", "redirected", "legacy-windows-tty"],
)
def test_interactive_footer_links_require_capability_and_preserve_real_path(
    monkeypatch, tmp_path, is_tty, no_color, legacy_windows, expect_link
):
    monkeypatch.setenv("TERM", "xterm-256color")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    report_path = tmp_path / "actual [name].html"
    expected_uri = report_path.as_uri()
    assert expected_uri.endswith("/actual%20%5Bname%5D.html")
    stream = TTY() if is_tty else io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        no_color=no_color,
        color_system="truecolor",
        legacy_windows=legacy_windows,
        width=max(120, len(str(report_path)) + 30),
    )
    view = ValidateView("demo", [], console=console)
    view.links = [("report", str(report_path))]
    console.print(next(view._footer_lines()))
    text = stream.getvalue()
    assert str(report_path) in text
    assert ("\x1b]8" in text) is expect_link
    assert (expected_uri in text) is expect_link


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


def test_agent_eval_summary_shows_required_counts_and_failed_case_reason():
    output = io.StringIO()
    CLIReporter._print_agent_eval_tables(
        {
            "verdict": "fail",
            "verdict_reason": "1 of 2 cases failed: 1 goal not met",
            "composite_lift": 0.2,
            "required_checks": {
                "with_skill": {"passed": 1, "scored": 2},
                "without_skill": {"passed": 2, "scored": 2},
                "difference": -1,
            },
            "correct_answers": {
                "with_skill": {"correct": 1, "scored": 2},
                "without_skill": {"correct": 2, "scored": 2},
                "difference": -1,
            },
            "security_clean": {
                "with_skill": {"clean": 2, "scored": 2},
                "without_skill": {"clean": 2, "scored": 2},
                "difference": 0,
            },
            "pass_at_k": {
                "with_skill": {
                    "cases": {
                        "wrong-ground-truth": {
                            "passed": False,
                            "attempts_used": 1,
                            "best_score": 0.82,
                            "reason": "goal not met (goal accuracy 0.00)",
                            "warnings": ["missed expected step 1 of 1 (behavior 0.00)"],
                        }
                    }
                }
            },
        },
        Console(file=output, width=140, color_system=None),
    )

    rendered = output.getvalue()
    assert "Verdict: FAIL — 1 of 2 cases failed: 1 goal not met" in rendered
    assert "composite lift (for reference)" in rendered
    assert "Required checks passed: 1/2 | 2/2 | -1" in rendered
    assert "Goals met: 1/2 | 2/2 | -1" in rendered
    assert "Security clean: 2/2 | 2/2 | +0" in rendered
    assert "wrong-ground-truth" in rendered
    assert "0.82" in rendered
    assert "goal not met (goal accuracy 0.00)" in rendered
    assert "WARNING" in rendered


def test_failed_agent_eval_summary_row_shows_verdict_reason_and_lift():
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        passed=True,
        metadata={
            "agent_eval": {
                "verdict": "fail",
                "verdict_reason": "2 of 4 cases failed: 1 goal not met, 1 security issue",
                "composite_lift": 0.0,
            }
        },
    )
    result.summary.checks_performed = 1
    output = io.StringIO()

    CLIReporter(console=Console(file=output, width=180, color_system=None)).print_summary([result])

    rendered = output.getvalue()
    assert "AGENT_EVAL" in rendered
    assert "FAIL" in rendered
    assert "2 of 4 cases failed: 1 goal not met, 1 security issue; lift +0.00" in rendered
    assert "1 checks passed" not in rendered


def test_full_summary_shows_completed_agent_eval_counts_without_findings():
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        passed=True,
        metadata={
            "agent_eval": {
                "verdict": "fail",
                "verdict_reason": "1 of 2 cases failed: 1 goal not met",
                "composite_lift": 0.2,
                "required_checks": {
                    "with_skill": {"passed": 1, "scored": 2},
                    "without_skill": {"passed": 2, "scored": 2},
                    "difference": -1,
                },
                "correct_answers": {
                    "with_skill": {"correct": 1, "scored": 2},
                    "without_skill": {"correct": 2, "scored": 2},
                    "difference": -1,
                },
                "security_clean": {
                    "with_skill": {"clean": 2, "scored": 2},
                    "without_skill": {"clean": 2, "scored": 2},
                    "difference": 0,
                },
                "pass_at_k": {
                    "with_skill": {
                        "cases": {
                            "wrong-ground-truth": {
                                "passed": False,
                                "attempts_used": 1,
                                "best_score": 0.82,
                                "reason": "goal not met (goal accuracy 0.00)",
                                "warnings": ["missed expected step 1 of 1 (behavior 0.00)"],
                            }
                        }
                    }
                },
            }
        },
    )
    output = io.StringIO()

    CLIReporter(console=Console(file=output, width=140, color_system=None)).print_all([result])

    rendered = output.getvalue()
    assert "Verdict: FAIL — 1 of 2 cases failed: 1 goal not met" in rendered
    assert "Required checks passed: 1/2 | 2/2 | -1" in rendered
    assert "Goals met: 1/2 | 2/2 | -1" in rendered
    assert "Security clean: 2/2 | 2/2 | +0" in rendered
    assert "wrong-ground-truth" in rendered
    assert "0.82" in rendered
    assert "goal not met (goal accuracy 0.00)" in rendered
    assert "missed expected step 1 of 1 (behavior 0.00)" in rendered


def test_verdict_badge_background_does_not_fill_headline():
    from rich.console import Group

    view = ValidateView("demo", [])
    for verdict in (Verdict(True, "Passed required checks"), Verdict(False, "Changes required")):
        panel = view._verdict_panel(verdict)
        assert isinstance(panel.renderable, Group)
        body = panel.renderable.renderables[0]
        style = body.get_style_at_offset(view.console, body.plain.index(verdict.headline))
        assert style.bgcolor is None


def test_real_tty_capture_excludes_links_but_ordinary_print_keeps_them(monkeypatch):
    import os

    from skillevaluator.reporting.terminal_styles import artifact_text

    if not hasattr(os, "openpty"):
        pytest.skip("PTY capture regression runs on POSIX; Windows URI policy is tested separately")
    monkeypatch.setenv("TERM", "xterm-256color")
    master, slave = os.openpty()
    try:
        with os.fdopen(slave, "w") as stream:
            console = Console(file=stream, color_system="truecolor", no_color=False, width=120)
            line = artifact_text("/tmp/actual [name].html", console)
            with console.capture() as capture:
                console.print(line)
            assert "actual [name].html" in capture.get()
            assert "\x1b]8" not in capture.get()
            console.print(line)
            assert "\x1b]8" in os.read(master, 65536).decode()
    finally:
        os.close(master)


@pytest.mark.parametrize("legacy_windows", [False, True], ids=["modern-tty", "legacy-windows-tty"])
def test_footer_links_follow_capture_at_render_time(monkeypatch, tmp_path, legacy_windows):
    monkeypatch.setenv("TERM", "xterm-256color")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    report_path = tmp_path / "actual [name].html"
    expected_uri = report_path.as_uri()
    assert expected_uri.endswith("/actual%20%5Bname%5D.html")
    stream = TTY()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        legacy_windows=legacy_windows,
        width=max(120, len(str(report_path)) + 30),
    )
    view = ValidateView("demo", [], console=console)
    view.links = [("report", str(report_path))]
    view.verdict = Verdict(True, "Passed required checks")
    with console.capture() as capture:
        console.print(view)
    assert str(report_path) in capture.get()
    assert "\x1b]8" not in capture.get()
    assert expected_uri not in capture.get()
    console.print(view)
    assert str(report_path) in stream.getvalue()
    assert ("\x1b]8" in stream.getvalue()) is (not legacy_windows)
    assert (expected_uri in stream.getvalue()) is (not legacy_windows)


@pytest.mark.parametrize(
    "target",
    [
        "https://user:password@example.com/report.html",
        "https://user@example.com/report.html",
        "https://example.com/line\nbreak.html",
        "https://example.com/tab\tname.html",
        "https://example.com/%0areport.html",
    ],
)
def test_remote_credential_and_control_targets_are_not_linked(monkeypatch, target):
    from skillevaluator.reporting.terminal_styles import artifact_text

    monkeypatch.setenv("TERM", "xterm-256color")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    output = TTY()
    console = Console(file=output, color_system="truecolor", no_color=False, width=120)
    console.print(artifact_text(target, console, label="Readable report path"))
    assert "Readable report path" in output.getvalue()
    assert "\x1b]8" not in output.getvalue()


@pytest.mark.parametrize(
    ("target", "windows", "expected"),
    [
        (r"C:\reports\actual [name].html", True, "file:///C:/reports/actual%20%5Bname%5D.html"),
        ("C:/reports/actual.html", True, "file:///C:/reports/actual.html"),
        (r"C:\reports\actual.html", False, None),
        (r"C:report.html", True, None),
        ("/tmp/report.html", True, None),
        ("javascript:alert(1)", True, None),
        ("https://example.com/report.html", True, "https://example.com/report.html"),
    ],
)
def test_artifact_uri_uses_only_native_absolute_windows_paths(target, windows, expected):
    from skillevaluator.reporting.terminal_styles import _artifact_uri

    assert _artifact_uri(target, windows=windows) == expected


@pytest.mark.parametrize("captured", [False, True])
def test_real_tty_live_lifecycle_prints_footer_once_with_capture_safe_links(monkeypatch, captured):
    import os
    import threading

    from skillevaluator.utils.rich_markup import strip_terminal_controls

    if not hasattr(os, "openpty"):
        pytest.skip("Live PTY regression runs on POSIX")
    monkeypatch.setenv("TERM", "xterm-256color")
    master, slave = os.openpty()
    chunks = []

    def drain():
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                return
            if not chunk:
                return
            chunks.append(chunk)

    reader = threading.Thread(target=drain, daemon=True)
    try:
        with os.fdopen(slave, "w") as stream:
            # Drain concurrently: a real Live frame can fill a PTY's small buffer.
            reader.start()
            console = Console(file=stream, color_system="truecolor", no_color=False, width=98)
            view = ValidateView("demo", [], console=console)

            def lifecycle():
                view.start()
                worker = view._live._refresh_thread
                try:
                    view.finish(Verdict(True, "Passed required checks"), [("report", "/tmp/actual.html")])
                finally:
                    view.stop()
                assert view._live is None
                worker.join(timeout=1)
                assert not worker.is_alive()

            if captured:
                with console.capture() as capture:
                    lifecycle()
                output = capture.get()
            else:
                lifecycle()
        reader.join(timeout=1)
        assert not reader.is_alive()
        if not captured:
            output = b"".join(chunks).decode()
        assert strip_terminal_controls(output).count("/tmp/actual.html") == 1
        assert ("\x1b]8" in output) is (not captured)
    finally:
        os.close(master)


@pytest.mark.parametrize(
    "value", [10**400, -(10**400), True, False, float("nan"), float("inf"), float("-inf"), None, "95"]
)
def test_success_summary_omits_unusable_quality_evidence_without_changing_payload(value):
    import json

    result = ValidationResult(
        validator_name="CHECK",
        passed=True,
        metadata={
            "quality_scores": {"grade": "A", "overall_score": value},
        },
    )
    before = json.dumps(result.metadata, sort_keys=True)
    output = io.StringIO()
    CLIReporter(console=Console(file=output, width=120, color_system=None)).print_summary([result])
    text = output.getvalue()
    assert "PASS" in text
    assert "quality" not in text.casefold()
    assert "/100" not in text
    assert json.dumps(result.metadata, sort_keys=True) == before


@pytest.mark.parametrize(
    "value", [10**400, -(10**400), True, False, float("nan"), float("inf"), float("-inf"), None, "0.25"]
)
def test_success_summary_omits_unusable_lift_evidence_without_changing_payload(value):
    import json

    result = ValidationResult(
        validator_name="CHECK",
        passed=True,
        metadata={
            "agent_eval": {"verdict": "pass", "summary": {"overall_lift": value}},
        },
    )
    before = json.dumps(result.metadata, sort_keys=True)
    output = io.StringIO()
    CLIReporter(console=Console(file=output, width=120, color_system=None)).print_summary([result])
    assert "PASS" in output.getvalue()
    assert "lift" not in output.getvalue().casefold()
    assert json.dumps(result.metadata, sort_keys=True) == before


def test_success_summary_keeps_valid_numeric_zero_quality_and_lift():
    result = ValidationResult(
        validator_name="CHECK",
        passed=True,
        metadata={
            "quality_scores": {"grade": "F", "overall_score": 0},
            "agent_eval": {"verdict": "pass", "summary": {"overall_lift": 0}},
        },
    )
    output = io.StringIO()
    CLIReporter(console=Console(file=output, width=120, color_system=None)).print_summary([result])
    assert "quality F 0.0/100" in output.getvalue()
    assert "lift +0.00" in output.getvalue()


@pytest.mark.parametrize("env", ["FORCE_COLOR", "TTY_COMPATIBLE"])
def test_forced_terminal_pipe_keeps_durable_progress_instead_of_live(monkeypatch, env):
    # CI often forces colour, so Rich treats the pipe as a terminal; it is not a TTY.
    monkeypatch.setenv(env, "1")
    progress = io.StringIO()
    output = io.StringIO()
    monkeypatch.setattr("sys.stderr", progress)
    view = ValidateView("demo", [(1, "Static", "static")], console=Console(file=output, width=80))
    assert view.console.is_terminal
    try:
        view.start()
        assert view._live is None
        view.tier_start(0)
        view.tier_progress(0, [detail_row("checks", "schema running")])
        view.tier_done(0)
    finally:
        view.stop()
    assert "Tier 1" in progress.getvalue() and "schema running" in progress.getvalue()
    assert "\x1b[?25l" not in output.getvalue()


def test_durable_tier3_progress_logs_the_count_without_bar_cells(monkeypatch):
    from skillevaluator.reporting.console_ui import tier3_case_rows

    progress = io.StringIO()
    monkeypatch.setattr("sys.stderr", progress)
    view = ValidateView("demo", [(3, "Live Agent Eval", "live")], console=Console(file=io.StringIO()))
    marks = {"codex": {"with_skill": {"001": ["pass"]}, "without_skill": {"001": []}}}
    try:
        view.start()
        view.tier_start(0)
        view.tier_progress(0, tier3_case_rows(marks, agents=["codex"], k=4, progress=(3, 8), show_agent=False))
    finally:
        view.stop()
    assert "Tier 3/progress: running - 3/8 attempts" in progress.getvalue()
    assert "█" not in progress.getvalue() and "░" not in progress.getvalue()


def test_windows_artifact_uri_avoids_deprecated_pure_path_as_uri():
    import warnings

    from skillevaluator.reporting.terminal_styles import _artifact_uri

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _artifact_uri(r"D:\a\b #%?;é.html", windows=True) == "file:///D:/a/b%20%23%25%3F%3B%C3%A9.html"


@pytest.mark.parametrize("surrogate", ["\udcff", "\ud800"])
def test_windows_artifact_paths_with_invalid_unicode_remain_unlinked(surrogate):
    from skillevaluator.reporting.terminal_styles import _artifact_uri

    assert _artifact_uri("C:\\reports\\surrogate" + surrogate + ".html", windows=True) is None


def test_compact_dimension_table_keeps_agent_headers_free_of_condition_labels():
    from skillevaluator.tier3 import result_display

    dimension = next(iter(result_display.DIMENSION_DISPLAY))
    agent = {
        "execution_status": "succeeded",
        "dimensions_with_skill": {dimension: {"score": 0.9}},
        "dimensions_without_skill": {dimension: {"score": 0.7}},
    }
    output = io.StringIO()
    result_display._render_dimensions(
        console=Console(file=output, width=80, color_system=None), agents={"codex": agent, "claude": agent}, safe=str
    )
    header = next(line for line in output.getvalue().splitlines() if "Agent: codex" in line)
    assert "With Skill" not in header
