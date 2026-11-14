# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An agent arm that ran but failed is reported as incomplete, never as not run.

When one attempt of an agent's no-skill baseline times out, the baseline arm
fails while the with-skill arm keeps its scores. Every report must say that the
baseline ran but did not finish, how many of its attempts scored and what
failed first, instead of printing ``baseline not run``.
"""

from __future__ import annotations

import copy
import html as html_lib
import io
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from rich.console import Console
from tests.test_harbor_collector_runtime_failures import _write_actual_harbor_022_result

from skillevaluator.cli import _effective_report_formats
from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    agent_eval_result_from_directory,
    build_agent_eval_payload,
)
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter
from skillevaluator.reporting.console_ui import summarize_tier3
from skillevaluator.reporting.html import _TIER3_HTML_PREVIEW_TRIGGER_BYTES
from skillevaluator.reporting.tier3_coverage import IncompleteArm, coverage_notices, incomplete_arms
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult

TIMEOUT_ERROR = (
    "Without skill trial case-1__attempt2: AgentTimeoutError: Agent execution timed out after 1200.0 seconds"
)
SETUP_ERROR = (
    "Agent runtime failed in case-1__attempt1: AgentSetupTimeoutError: Agent setup timed out after 360 seconds"
)
# A job whose statistics miss a trial names the first trial or step exception
# Harbor recorded under it.
JOB_STATISTICS_ERROR = "Harbor evaluation statistics account for 1/2 completed trials"
RECORDED_STEP_ERROR = (
    f"{JOB_STATISTICS_ERROR}; trial case-1__attempt2 recorded RewardFileNotFoundError: reward.json missing"
)
FAILED_AGENT = "The agent and the Tier 3 run are marked failed."
REPO_ROOT = Path(__file__).resolve().parents[2]


def _condition(status: str, scored: int, expected: int = 2, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "execution_status": status,
        "execution_errors": list(errors or []),
        "expected_attempts": expected,
        "scored_attempts": scored,
    }


def _agent(
    *,
    baseline: dict[str, Any] | None,
    with_skill: dict[str, Any] | None = None,
    activated: bool = True,
) -> dict[str, Any]:
    """Return loaded agent data in the shape ``report_data.load_agent_data`` produces."""
    with_skill = with_skill or _condition("succeeded", 2)
    scores = dict.fromkeys(DEFAULT_METRICS, 0.9)
    scores["skill_execution"] = 0.9 if activated else 0.0
    reward = {
        **scores,
        "entry_id": "case-1",
        "trial_id": "case-1__attempt1",
        "details": {"skill_execution": {"activation": {"passed": activated, "score": float(activated)}}},
    }
    conditions = {"with_skill": with_skill}
    agent: dict[str, Any] = {
        "with_skill": scores,
        "rewards": [reward],
        "num_trials": 1,
        "conditions": conditions,
    }
    if baseline is not None:
        conditions["without_skill"] = baseline
        baseline_scores = dict.fromkeys(DEFAULT_METRICS, 0.5)
        agent["without_skill"] = baseline_scores
        agent["rewards_baseline"] = [{**baseline_scores, "entry_id": "case-1", "trial_id": "case-1__attempt1"}]
        agent["num_trials_baseline"] = 1
    active = [condition for condition in conditions.values() if condition["execution_status"] != "skipped"]
    agent["execution_status"] = (
        "succeeded" if all(condition["execution_status"] == "succeeded" for condition in active) else "failed"
    )
    agent["execution_errors"] = [error for condition in active for error in condition["execution_errors"]]
    agent["expected_attempts"] = sum(condition["expected_attempts"] for condition in active)
    agent["scored_attempts"] = sum(condition["scored_attempts"] for condition in active)
    return agent


def _render(agents: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ValidationResult, str, str]:
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    assert payload is not None
    result = _validation_result_from_payload(payload)
    assert result is not None
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])
    report = HTMLReporter(include_timestamp=False).render_all([result])
    return payload, result, benchmark, report


def _benchmark_rows(benchmark: str) -> dict[str, list[str]]:
    """Return the score rows (Overall and each dimension) of the Results at a Glance table.

    The count rows (required checks, goals met, security clean) compare counts,
    not a baseline score, so they are left out.
    """
    score_rows = {"Overall", *(dim_id.title() for dim_id in DIMENSION_MAPPING)}
    section = benchmark.split("## Results at a Glance\n", 1)[1].split("\n## ", 1)[0].split("\n### ", 1)[0]
    return {
        cells[0]: cells[1:]
        for line in section.splitlines()
        if line.startswith("| ")
        for cells in [[cell.strip() for cell in line.strip("|").split("|")]]
        if cells[0] in score_rows
    }


def _coverage_section(benchmark: str) -> list[str]:
    if "\n### Agent coverage\n" not in benchmark:
        return []
    section = benchmark.split("\n### Agent coverage\n", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if line.startswith("- ")]


def _tier3_evidence(benchmark: str) -> str:
    row = next(line for line in benchmark.splitlines() if line.startswith("| Tier 3 |"))
    return [cell.strip() for cell in row.strip("|").split("|")][3]


def _html_coverage_notes(report: str) -> list[str]:
    match = re.search(r'<div[^>]*id="tier3-coverage-notices"[^>]*>(.*?)</div>', report, flags=re.DOTALL)
    if match is None:
        return []
    return [html_lib.unescape(item.strip()) for item in re.findall(r"<li>(.*?)</li>", match.group(1), re.DOTALL)]


def _cli_output(result: ValidationResult) -> str:
    output = io.StringIO()
    CLIReporter(console=Console(file=output, width=400, color_system=None)).print_all([result])
    return output.getvalue()


def _doc_block(name: str, lead: str) -> str:
    """Return the one blank-line separated block of ``name`` that starts with ``lead``, whitespace-normalized."""
    blocks = [" ".join(block.split()) for block in (REPO_ROOT / name).read_text(encoding="utf-8").split("\n\n")]
    [block] = [block for block in blocks if block.startswith(lead)]
    return block


# -- arm classification -------------------------------------------------------


@pytest.mark.parametrize(
    ("scored", "expected", "state", "text"),
    [
        (1, 2, "incomplete", "baseline incomplete (1 of 2 attempts scored)"),
        (0, 2, "unavailable", "baseline unavailable (0 of 2 attempts scored)"),
        # The arm failed for another reason although no attempt is missing.
        (2, 2, "failed", "baseline failed (2 of 2 attempts scored)"),
        # Unusable counters are not published.
        (3, 2, "failed", "baseline failed"),
        (0, 0, "failed", "baseline failed"),
        (None, 2, "failed", "baseline failed"),
    ],
)
def test_incomplete_arm_states_and_counts(scored: int | None, expected: int, state: str, text: str) -> None:
    arm = IncompleteArm("without_skill", scored, expected)
    assert arm.state == state
    assert str(arm) == text


@pytest.mark.parametrize(
    ("errors", "failure"),
    [
        ([TIMEOUT_ERROR], "agent timeout error"),
        ([SETUP_ERROR], "agent setup timeout error"),
        (["Scored attempt coverage is 1/2", TIMEOUT_ERROR], "agent timeout error"),
        (["Without skill trial case-1__attempt2: harbor.trial.errors.VerifierTimeoutError"], "verifier timeout error"),
        (["Without skill trial case-1__attempt2: HTTPError: 503"], "http error"),
        # An acronym keeps its trailing digits.
        (["Without skill trial case-1__attempt2: S3UploadFailedError: denied"], "s3 upload failed error"),
        (["Without skill aggregate job: HTTP2Error: stream reset"], "http2 error"),
        (["Without skill trial case-1__attempt2: Http2ProtocolError"], "http2 protocol error"),
        # A case ID shaped like an exception type is not a failure type.
        (["Without skill trial FakeError: AgentTimeoutError: timed out"], "agent timeout error"),
        (["Missing scored attempts for cases: ParseError", "Scored attempt coverage is 1/2"], None),
        (["Excess scored attempts for cases: ParseError", TIMEOUT_ERROR], "agent timeout error"),
        (["Unexpected scored cases: ValueError"], None),
        # The exception Harbor recorded under a failed job, as the collector and
        # the saved-run loader word it, is the first failure, also before a later
        # typed error.
        (
            [RECORDED_STEP_ERROR, "Missing scored attempts for cases: case-1", "Scored attempt coverage is 1/2"],
            "reward file not found error",
        ),
        ([f"Without skill aggregate job: {RECORDED_STEP_ERROR}"], "reward file not found error"),
        ([RECORDED_STEP_ERROR, TIMEOUT_ERROR], "reward file not found error"),
        (
            [f"{JOB_STATISTICS_ERROR}; trial case-1__attempt2 recorded harbor.trial.errors.VerifierTimeoutError"],
            "verifier timeout error",
        ),
        # A recorded message without a type, or a trial label shaped like a type, names nothing.
        ([f"{JOB_STATISTICS_ERROR}; trial case-1__attempt2 recorded reward.json missing"], None),
        ([f"{JOB_STATISTICS_ERROR}; trial FakeError recorded reward.json missing"], None),
        (["Scored attempt coverage is 1/2", "Unscoreable reward in case-1__attempt2: judge failed"], None),
        ("AgentTimeoutError: not a list", None),
    ],
)
def test_first_failure_names_the_first_harbor_exception_type(errors: object, failure: str | None) -> None:
    agent = {"conditions": {"without_skill": {"execution_status": "failed", "execution_errors": errors}}}
    (arm,) = incomplete_arms(agent)
    assert arm.failure == failure


def test_notice_says_the_agent_failed_only_when_the_agent_is_marked_failed() -> None:
    baseline = _condition("failed", 1, errors=[TIMEOUT_ERROR])
    failed = {"execution_status": "failed", "conditions": {"without_skill": baseline}}
    # An agent kept scored although its baseline arm failed.
    scored = {"execution_status": "succeeded", "conditions": {"without_skill": baseline}}

    notices = coverage_notices({"failed-agent": failed, "scored-agent": scored})

    assert [notice["code"] for notice in notices] == ["baseline_failed", "baseline_failed"]
    assert [notice["message"] for notice in notices] == [
        "failed-agent baseline incomplete (1 of 2 attempts scored; first failure: agent timeout error). "
        f"{FAILED_AGENT} Baseline and uplift are not reported for this agent.",
        "scored-agent baseline incomplete (1 of 2 attempts scored; first failure: agent timeout error). "
        "Baseline and uplift are not reported for this agent.",
    ]


@pytest.mark.parametrize(
    "conditions",
    [
        None,
        "failed",
        {"with_skill": _condition("succeeded", 2), "without_skill": _condition("succeeded", 2)},
        {"with_skill": _condition("succeeded", 2), "without_skill": _condition("skipped", 0, 0)},
        {"with_skill": _condition("succeeded", 2), "without_skill": _condition("unknown", 1)},
        {"with_skill": _condition("succeeded", 2), "without_skill": ["failed"]},
    ],
)
def test_completed_skipped_or_malformed_arms_are_not_incomplete(conditions: object) -> None:
    assert incomplete_arms({"conditions": conditions}) == []
    assert coverage_notices({"agent": {"conditions": conditions}}) == []


# -- canonical payload ----------------------------------------------------------


def test_payload_records_a_coverage_notice_for_a_failed_baseline_arm() -> None:
    payload, _, _, _ = _render(
        {
            "claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR])),
            "codex": _agent(baseline=_condition("succeeded", 2)),
        }
    )

    assert payload["coverage_notices"] == [
        {
            "code": "baseline_failed",
            "agent": "claude-code",
            "condition": "without_skill",
            "state": "incomplete",
            "scored_attempts": 1,
            "expected_attempts": 2,
            "failure": "agent timeout error",
            "message": (
                "claude-code baseline incomplete (1 of 2 attempts scored; first failure: agent timeout error). "
                f"{FAILED_AGENT} Baseline and uplift are not reported for this agent."
            ),
        }
    ]
    # The with-skill arm keeps its scores; the failed baseline has none.
    agent = payload["agents"]["claude-code"]
    assert agent["with_skill"] == pytest.approx(0.9)
    assert agent["baseline"] is None
    assert agent["lift"] is None


def test_payload_has_no_coverage_notices_when_every_arm_completed_or_was_skipped() -> None:
    payload, _, benchmark, report = _render(
        {
            "codex": _agent(baseline=_condition("succeeded", 2)),
            "opencode": _agent(baseline=_condition("skipped", 0, 0)),
        }
    )

    assert "coverage_notices" not in payload
    assert _coverage_section(benchmark) == []
    assert _html_coverage_notes(report) == []
    assert "tier3-coverage-notices" not in report


def test_run_directory_failed_baseline_trial_reaches_the_coverage_notice(tmp_path: Path) -> None:
    """The loader's condition errors carry the Harbor exception type the notice names."""
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results"
    for variant, status, scored, trial_failures in (
        ("with-skill", "succeeded", 2, []),
        (
            "without-skill",
            "failed",
            1,
            [{"trial": "case-1__attempt2", "reason": "AgentTimeoutError: Agent execution timed out after 1200.0 s"}],
        ),
    ):
        summary = run_dir / "claude-code" / variant / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "scores": dict.fromkeys(DEFAULT_METRICS, 0.8 if variant == "with-skill" else 0.4),
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": scored,
                    "execution_status": status,
                    "execution_errors": [] if status == "succeeded" else ["Scored attempt coverage is 1/2"],
                    "trial_failures": trial_failures,
                    "expected_attempts": 2,
                    "scored_attempts": scored,
                }
            ),
            encoding="utf-8",
        )

    result = agent_eval_result_from_directory(skill, run_dir, use_llm_judge=False)

    assert result is not None
    payload = result.metadata["agent_eval"]
    assert [(notice["agent"], notice["state"], notice["failure"]) for notice in payload["coverage_notices"]] == [
        ("claude-code", "incomplete", "agent timeout error")
    ]
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])
    assert _benchmark_rows(benchmark)["Overall"] == [
        "80.0% — baseline incomplete (1 of 2 attempts scored); uplift unavailable"
    ]


def _collect_failed_baseline_step(tmp_path: Path) -> dict[str, Any]:
    """Collect a run whose baseline job recorded a ``RewardFileNotFoundError`` step exception."""
    jobs_dir = tmp_path / "jobs"
    _write_actual_harbor_022_result(jobs_dir / "demo-opencode-with", reward=1.0)
    _write_actual_harbor_022_result(
        jobs_dir / "demo-opencode-without",
        verifier_mode="missing",
        step_rewards=(1.0,),
        step_exception_type="RewardFileNotFoundError",
    )
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )


def test_harbor_step_exception_reaches_the_coverage_notice(tmp_path: Path) -> None:
    """A step exception Harbor recorded under a failed baseline job is the first failure, live and reloaded."""
    results = _collect_failed_baseline_step(tmp_path)
    skill = tmp_path / "demo"
    skill.mkdir()
    saved = agent_eval_result_from_directory(skill, tmp_path / "results", use_llm_judge=False)

    assert saved is not None
    expected = [("opencode", "baseline_failed", "unavailable", "reward file not found error")]
    for notices in (coverage_notices(results["agents"]), saved.metadata["agent_eval"]["coverage_notices"]):
        assert [(notice["agent"], notice["code"], notice["state"], notice["failure"]) for notice in notices] == expected


def test_changelog_quotes_the_recorded_clause_as_the_end_of_the_job_failure(tmp_path: Path) -> None:
    """The job failure opens with the trial count, so the entry quotes only how it ends."""
    results = _collect_failed_baseline_step(tmp_path)
    job_failure = results["agents"]["opencode"]["conditions"]["without_skill"]["execution_errors"][0]
    assert job_failure.startswith("Harbor evaluation statistics account for 0/1 completed trials; trial ")
    assert re.search(r"; trial \S+ recorded RewardFileNotFoundError: [^;]+\Z", job_failure)

    entry = _doc_block("CHANGELOG.md", "- Tier 3 coverage notices now name the exception type Harbor recorded")
    assert "job failure then ends with `trial <label> recorded VerifierOutputParseError: <message>`" in entry
    assert "reports that job failure as" not in entry


# -- BENCHMARK.md ---------------------------------------------------------------


def test_benchmark_says_a_failed_baseline_is_incomplete_not_not_run() -> None:
    _, _, benchmark, _ = _render(
        {
            "claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR])),
            "codex": _agent(baseline=_condition("succeeded", 2)),
            "opencode": _agent(baseline=_condition("skipped", 0, 0)),
        }
    )

    rows = _benchmark_rows(benchmark)
    assert len(rows) == len(DIMENSION_MAPPING) + 1
    for cells in rows.values():
        incomplete, completed, skipped = cells
        assert incomplete == "90.0% — baseline incomplete (1 of 2 attempts scored); uplift unavailable"
        assert completed == "50.0% → 90.0% (+40.0 points)"
        # A skipped baseline still reads as not run.
        assert skipped == "90.0% — baseline not run; uplift unavailable"
    assert _coverage_section(benchmark) == [
        "- Claude Code: baseline incomplete (1 of 2 attempts scored; first failure: agent timeout error). "
        f"{FAILED_AGENT} Baseline and uplift are not reported for this agent."
    ]
    assert _tier3_evidence(benchmark).endswith("; Claude Code baseline incomplete (1 of 2 attempts scored)")


def test_benchmark_says_baseline_unavailable_when_no_baseline_attempt_scored() -> None:
    _, _, benchmark, _ = _render({"claude-code": _agent(baseline=_condition("failed", 0, errors=[TIMEOUT_ERROR]))})

    for (cell,) in _benchmark_rows(benchmark).values():
        assert cell == "90.0% — baseline unavailable (0 of 2 attempts scored); uplift unavailable"
    assert _tier3_evidence(benchmark).endswith("; Claude Code baseline unavailable (0 of 2 attempts scored)")


def test_benchmark_explains_a_failed_with_skill_arm_below_the_table() -> None:
    _, _, benchmark, _ = _render(
        {
            "codex": _agent(
                with_skill=_condition("failed", 0, errors=[SETUP_ERROR]),
                baseline=_condition("succeeded", 2),
            ),
            "claude-code": _agent(baseline=_condition("succeeded", 2)),
        }
    )

    # Agents are ordered by name: claude-code, then codex. Codex's with-skill
    # scores (90.0%) and uplift are not published.
    for _completed, failed in _benchmark_rows(benchmark).values():
        assert "90.0%" not in failed
        assert "points" not in failed
    assert _coverage_section(benchmark) == [
        "- Codex: with-skill run unavailable (0 of 2 attempts scored; first failure: agent setup timeout error). "
        f"{FAILED_AGENT} With-skill scores and uplift are not reported for this agent."
    ]
    assert _tier3_evidence(benchmark).endswith("; Codex with-skill run unavailable (0 of 2 attempts scored)")


def test_benchmark_keeps_never_activated_wording_over_an_incomplete_baseline() -> None:
    _, _, benchmark, _ = _render(
        {"claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR]), activated=False)}
    )

    for (cell,) in _benchmark_rows(benchmark).values():
        assert "INCONCLUSIVE: skill never activated; uplift unavailable" in cell
        assert "baseline" not in cell
    assert len(_coverage_section(benchmark)) == 1


# -- HTML and terminal ---------------------------------------------------------------


def test_html_and_terminal_show_agent_coverage_notes() -> None:
    payload, result, _, report = _render(
        {
            "claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR])),
            "codex": _agent(
                with_skill=_condition("failed", 0, errors=[SETUP_ERROR]),
                baseline=_condition("succeeded", 2),
            ),
        }
    )
    messages = [notice["message"] for notice in payload["coverage_notices"]]
    assert len(messages) == 2

    assert _html_coverage_notes(report) == messages

    output = _cli_output(result)
    assert "Agent coverage notes" in output
    for message in messages:
        assert f"• {message}" in output


def test_html_and_terminal_escape_agent_names_in_coverage_notes() -> None:
    name = "<b>[red]agent"
    payload, result, _, report = _render({name: _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR]))})
    (message,) = [notice["message"] for notice in payload["coverage_notices"]]
    assert message.startswith(f"{name} baseline incomplete")

    assert "<li><b>" not in report
    assert _html_coverage_notes(report) == [message]
    assert f"• {message}" in _cli_output(result)


@pytest.mark.parametrize(
    "notices",
    [5, True, {"message": "claude-code baseline incomplete"}, ["claude-code", None, {"message": 5}, {"message": ""}]],
)
def test_html_drops_malformed_coverage_notices(notices: object) -> None:
    _, result, _, _ = _render({"claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR]))})
    result.metadata["agent_eval"]["coverage_notices"] = notices

    report = HTMLReporter(include_timestamp=False).render_all([result])

    assert "tier3-coverage-notices" not in report
    assert _html_coverage_notes(report) == []


def test_html_keeps_coverage_notes_whole_in_the_bounded_preview() -> None:
    """A payload above the preview trigger still shows every coverage message in full."""
    payload, result, _, _ = _render(
        {
            "claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR])),
            "codex": _agent(baseline=_condition("succeeded", 2)),
        }
    )
    messages = [notice["message"] for notice in payload["coverage_notices"]]
    large = copy.deepcopy(result)
    prompt = "Summarize the attached quarterly report and list each follow-up action. " * 40
    large.metadata["agent_eval"]["dataset"] = [{"id": f"case-{index}", "prompt": prompt} for index in range(100)]
    assert len(json.dumps(large.metadata["agent_eval"]).encode("utf-8")) > _TIER3_HTML_PREVIEW_TRIGGER_BYTES

    report = HTMLReporter(include_timestamp=False).render_all([large])

    assert "HTML preview omitted" in report
    assert _html_coverage_notes(report) == messages


def test_terminal_prints_coverage_notes_for_a_non_blocking_tier3_result() -> None:
    """``validate`` runs Tier 3 non-blocking by default; the failed run has errors but no findings.

    The full terminal report (``validate --verbose`` or ``-r cli``) must still print the notes.
    """
    payload, result, _, _ = _render({"claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR]))})
    result.metadata["gating"] = {"tier": 3, "blocking": False}
    assert not result.findings

    output = _cli_output(result)

    # Printed once, with the result's errors, under Non-blocking Findings
    # rather than as a bare agent table above it.
    assert output.count("Agent coverage notes") == 1
    assert "Non-blocking Findings" in output
    assert output.index("Non-blocking Findings") < output.index("Agent coverage notes")
    (notice,) = payload["coverage_notices"]
    assert f"• {notice['message']}" in output


def test_compact_validate_view_leaves_coverage_notes_to_the_report_files_and_the_docs_say_so() -> None:
    """Only the full terminal report prints the notes; a plain ``validate`` shows the compact view."""
    payload, result, _, _ = _render({"claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR]))})
    result.metadata["gating"] = {"tier": 3, "blocking": False}
    (notice,) = payload["coverage_notices"]

    # A plain ``validate`` swaps the ``cli`` reporter for HTML and JSON; ``--verbose`` keeps it.
    assert "cli" not in _effective_report_formats(("cli",), quiet=True)
    assert _effective_report_formats(("cli",), quiet=False) == ("cli",)
    # The compact Tier 3 rows show at most the first raw execution error, never the notes.
    _, _, rows, _ = summarize_tier3(result, report_formats=("html", "json"))
    compact = " ".join(text for row in rows for text, _ in row.segments)
    assert "AgentTimeoutError" in compact
    assert notice["message"] not in compact
    assert "baseline incomplete" not in compact

    docs = _doc_block("docs/reports.mdx", "The HTML report lists each message under **Agent coverage notes**")
    changelog = _doc_block("CHANGELOG.md", "- Tier 3 reports no longer say `baseline not run`")
    for name, text in (("docs/reports.mdx", docs), ("CHANGELOG.md", changelog)):
        assert "`validate --verbose`" in text, name
        assert "`-r cli`" in text, name
        assert "compact default" in text, name
        assert "`validate` prints" not in text, name


def test_html_agent_details_label_a_failed_baseline_like_a_skipped_one_and_the_docs_say_so() -> None:
    """Only the callout tells a baseline that ran and failed from one that was skipped."""
    _, _, _, report = _render(
        {
            "claude-code": _agent(baseline=_condition("failed", 1, errors=[TIMEOUT_ERROR])),
            "opencode": _agent(baseline=_condition("skipped", 0, 0)),
        }
    )

    kpi_subs = re.findall(r'<div class="t3-kpi-sub">(.*?)</div>', report)
    assert kpi_subs.count("baseline N/A") == 2
    assert kpi_subs.count("no baseline") == 2
    (note,) = _html_coverage_notes(report)
    assert note.startswith("claude-code baseline incomplete")

    definition = _doc_block("docs/reports.mdx", "When the HTML report has no lift for an agent or dimension")
    assert "**no baseline** means" in definition
    assert "a baseline arm that ran but failed" in definition
    assert "**Agent coverage notes**" in definition
    details = _doc_block("docs/reports.mdx", "The HTML report lists each message under **Agent coverage notes**")
    assert "**no baseline** and `baseline N/A`" in details
