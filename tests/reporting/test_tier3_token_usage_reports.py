# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 token usage reaches the payload and every report, and is never scored.

A run directory with a timed-out baseline attempt goes through the loader, the
canonical payload, the terminal, the Markdown report, ``BENCHMARK.md`` and the
HTML report. The failed attempt's tokens count as usage; scores and the
complete-only ``resources`` comparison stay as they were.
"""

from __future__ import annotations

import copy
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from scripts.ci import check_public_benchmarks as benchmark_gate

from skillevaluator.evaluation import tier3_report
from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    agent_eval_result_from_directory,
    build_agent_eval_payload,
    render_agent_eval_html_report,
)
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter, MarkdownReporter
from skillevaluator.reporting import base as reporting_base
from skillevaluator.reporting import html as html_reporting
from skillevaluator.reporting.base import (
    agent_eval_publication_evidence_complete,
    agent_eval_report_serialization_issue,
)
from skillevaluator.reporting.dashboard_data import comparison_rows, parse_dashboard_report
from skillevaluator.reporting.tier3_token_usage import EVALUATE_CASE_POINTER, VALIDATE_CASE_POINTER
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.result_display import render_result
from skillevaluator.tier3.token_usage import drop_token_usage_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests" / "golden" / "benchmark_token_usage" / "TOKEN_USAGE.md"


def _arm(
    run: Path,
    agent: str,
    variant: str,
    trials: list[tuple[str, tuple[int, int, int] | None, bool]],
    *,
    status: str = "succeeded",
) -> None:
    """Write one arm: ``trials`` are (trial directory, (prompt, completion, cached), has a reward)."""
    directory = run / agent / variant
    directory.mkdir(parents=True)
    scored = sum(rewarded for _name, _tokens, rewarded in trials)
    scores = dict.fromkeys(DEFAULT_METRICS, 0.9 if variant == "with-skill" else 0.5)
    summary: dict[str, Any] = {
        "scores": scores,
        "metrics": list(DEFAULT_METRICS),
        "num_trials": scored,
        "expected_attempts": len(trials),
        "scored_attempts": scored,
        "execution_status": status,
    }
    if status == "failed":
        summary["execution_errors"] = [f"Scored attempt coverage is {scored}/{len(trials)}"]
        summary["trial_failures"] = [
            {"trial": name, "reason": "AgentTimeoutError: Agent execution timed out after 1200.0 seconds"}
            for name, _tokens, rewarded in trials
            if not rewarded
        ]
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name, tokens, rewarded in trials:
        trial = directory / "trials" / name
        trial.mkdir(parents=True)
        if rewarded:
            # Harbor's ``__<id>`` suffix never contains ``__``; a case ID may.
            reward = {"entry_id": name.rsplit("__", 1)[0], "trial_id": name, **scores}
            (trial / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
        else:
            failure = {"status": "unscored", "trial": name, "error": {"type": "AgentTimeoutError"}}
            (trial / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
        if tokens is not None:
            prompt, completion, cached = tokens
            metrics = {
                "total_prompt_tokens": prompt,
                "total_completion_tokens": completion,
                "total_cached_tokens": cached,
            }
            (trial / "trajectory.json").write_text(
                json.dumps({"steps": [{}], "final_metrics": metrics}), encoding="utf-8"
            )


def _run(tmp_path: Path) -> Path:
    """Claude Code's baseline times out on case-2; Codex completes both arms."""
    run = tmp_path / "run"
    _arm(run, "claude-code", "with-skill", [("case-1__a", (1000, 100, 400), True), ("case-2__a", (2000, 200, 0), True)])
    _arm(
        run,
        "claude-code",
        "without-skill",
        [("case-1__a", (1500, 150, 0), True), ("case-2__b", (900, 47522, 0), False)],
        status="failed",
    )
    _arm(run, "codex", "with-skill", [("case-1__a", (500, 50, 0), True), ("case-2__a", (600, 60, 0), True)])
    _arm(run, "codex", "without-skill", [("case-1__a", (700, 70, 0), True), ("case-2__a", (800, 80, 0), True)])
    return run


@pytest.fixture
def run_result(tmp_path: Path) -> tuple[Path, Any]:
    skill = tmp_path / "demo"
    skill.mkdir()
    run = _run(tmp_path)
    result = agent_eval_result_from_directory(skill, run, use_llm_judge=False)
    assert result is not None
    return run, result


def _set_reward_case(run: Path, agent: str, variant: str, trial: str, case: str) -> None:
    reward_file = run / agent / variant / "trials" / trial / "reward.json"
    reward = json.loads(reward_file.read_text(encoding="utf-8"))
    reward_file.write_text(json.dumps({**reward, "entry_id": case}), encoding="utf-8")


def _nodes(value: Any) -> int:
    """Count JSON values the way the report serialization limit does: every value is one node."""
    if isinstance(value, dict):
        return 1 + sum(_nodes(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_nodes(item) for item in value)
    return 1


def _without_token_usage(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"token_usage", "report_truncation"}}


def _cli_output(result: Any) -> str:
    output = io.StringIO()
    CLIReporter(console=Console(file=output, width=220, color_system=None)).print_all([result])
    return output.getvalue()


def _section(text: str, heading: str) -> str:
    return text.split(f"{heading}\n", 1)[1].split("\n## ", 1)[0]


def _table_rows(section: str) -> list[list[str]]:
    """Split Markdown table rows on unescaped pipes; ``\\|`` is a literal pipe in a cell."""
    return [
        [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| Agent |")
    ]


# -- loader and payload -----------------------------------------------------------


def test_loader_keeps_usage_of_attempts_that_scores_never_read(tmp_path: Path) -> None:
    agents = load_agent_data(_run(tmp_path))

    baseline_usage = agents["claude-code"]["usage_only_attempts_baseline"]
    assert baseline_usage == [
        # The failed arm's scored attempt: its scores are withheld, its usage is not.
        {
            "entry_id": "case-1",
            "trial_id": "case-1__a",
            "scored": True,
            "tokens": {"prompt": 1500, "completion": 150, "cached": 0},
        },
        {
            "entry_id": "case-2",
            "trial_id": "case-2__b",
            "scored": False,
            "tokens": {"prompt": 900, "completion": 47522, "cached": 0},
        },
    ]
    # Scoring is unchanged: the failed arm keeps no rewards and stays incomplete.
    assert agents["claude-code"]["rewards_baseline"] == []
    assert agents["claude-code"]["rewards_baseline_complete"] is False
    assert agents["claude-code"]["usage_only_attempts"] == []
    assert agents["codex"]["usage_only_attempts_baseline"] == []


def test_an_unscored_attempt_of_a_case_id_with_double_underscores_keeps_its_case(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _arm(run, "codex", "with-skill", [("auth__login__a1b2c3d", (100, 10, 0), True)])
    _arm(
        run,
        "codex",
        "without-skill",
        [("auth__login__a1b2c3d", (90, 10, 0), True), ("auth__login__e4f5g6h", (400, 40, 0), False)],
        status="failed",
    )

    agents = load_agent_data(run)
    usage = build_agent_eval_payload("demo", agents, use_llm_judge=False)["token_usage"]["agents"]["codex"]

    [case] = usage["cases"]
    assert case["entry_id"] == "auth__login"
    assert case["baseline"]["coverage"]["observed_attempts"] == 2
    assert case["baseline"]["coverage"]["unscored_attempts"] == 1
    assert case["baseline"]["totals"]["total_tokens"] == 540


def test_a_stop_on_pass_unscored_attempt_takes_its_case_from_its_harbor_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    task_files = {
        "with-skill": ("result.json", {"task_id": {"path": "/runs/x/tasks/codex/with/case-1"}}),
        "without-skill": ("config.json", {"task": {"path": "C:\\runs\\x\\tasks\\codex\\without\\case-1"}}),
    }
    for variant, arm in (("with-skill", "with"), ("without-skill", "without")):
        timed_out = f"demo-codex-{arm}-case-1-attempt001__case-1__a1b2c3d"
        scored = f"demo-codex-{arm}-case-1-attempt002__case-1__e4f5g6h"
        _arm(run, "codex", variant, [(timed_out, (400, 40, 0), False), (scored, (100, 10, 0), True)])
        _set_reward_case(run, "codex", variant, scored, "case-1")
        name, content = task_files[variant]
        (run / "codex" / variant / "trials" / timed_out / name).write_text(json.dumps(content), encoding="utf-8")

    usage = build_agent_eval_payload("demo", load_agent_data(run), use_llm_judge=False)["token_usage"]

    codex = usage["agents"]["codex"]
    assert [case["entry_id"] for case in codex["cases"]] == ["case-1"]
    assert codex["with_skill"]["coverage"]["unscored_attempts"] == 1
    assert codex["delta"]["status"] == "available"
    assert codex["delta"]["total_tokens"] == 0


def test_payload_counts_the_failed_attempt_and_keeps_resources_complete_only(run_result: tuple[Path, Any]) -> None:
    _run_dir, result = run_result
    payload = result.metadata["agent_eval"]
    usage = payload["token_usage"]

    claude = usage["agents"]["claude-code"]
    assert claude["baseline"]["totals"]["total_tokens"] == 1650 + 48422
    assert claude["baseline"]["coverage"]["status"] == "complete"
    assert claude["baseline"]["coverage"]["unscored_attempts"] == 1
    assert claude["delta"]["reason_codes"] == ["baseline_failed"]
    assert claude["delta"]["reason"] == "No change: baseline incomplete (1 of 2 attempts scored)."
    codex = usage["agents"]["codex"]
    assert (codex["delta"]["total_tokens"], codex["delta"]["total_tokens_change_percent"]) == (-440, -26.67)
    assert usage["dataset"]["delta"]["reason_codes"] == ["agent_scope_mismatch"]

    # ``resources`` still needs every attempt of an arm scored and measured.
    claude_resources = payload["agents"]["claude-code"]["resources"]["tokens"]
    assert claude_resources["baseline"] == {"status": "unavailable", "measured_trials": 0, "trials": 0, "total": None}
    assert claude_resources["delta"] is None
    codex_resources = payload["agents"]["codex"]["resources"]["tokens"]
    assert codex_resources["with_skill"]["total"] == codex["with_skill"]["totals"]["total_tokens"] == 1210
    assert codex_resources["delta"] == codex["delta"]["total_tokens"]
    # Token usage changes no score.
    assert payload["agents"]["claude-code"]["baseline"] is None
    assert payload["agents"]["codex"]["with_skill"] == pytest.approx(0.9)


def test_payload_has_no_token_usage_without_token_counts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _arm(run, "codex", "with-skill", [("case-1__a", None, True)])
    _arm(run, "codex", "without-skill", [("case-1__a", None, True)])

    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)

    assert result is not None
    assert "token_usage" not in result.metadata["agent_eval"]


def _cases_run(tmp_path: Path, agents: int, cases: int) -> Path:
    run = tmp_path / "run"
    rows = [(f"case-{index:03d}__a", (100 + index, 10, 0), True) for index in range(cases)]
    for agent in range(agents):
        _arm(run, f"agent-{agent}", "with-skill", rows)
        _arm(run, f"agent-{agent}", "without-skill", rows)
    return run


def test_token_case_rows_go_before_any_score_evidence_under_the_size_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agents = load_agent_data(_cases_run(tmp_path, agents=3, cases=10))
    full = build_agent_eval_payload("demo", copy.deepcopy(agents), use_llm_judge=False, enforce_embedded_size=False)
    assert full is not None
    sums = copy.deepcopy(full["token_usage"])
    assert drop_token_usage_cases(sums) == 30
    # Room for every trial and the token sums, but not the token case rows.
    budget = tier3_report._serialized_payload_size({**_without_token_usage(full), "token_usage": sums}) + 2048
    assert tier3_report._serialized_payload_size(full) > budget
    monkeypatch.setattr(tier3_report, "_MAX_EMBEDDED_REPORT_BYTES", budget)

    bounded = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert bounded is not None
    assert _without_token_usage(bounded) == _without_token_usage(full)
    assert bounded["report_truncation"]["omitted"] == {"token_usage_case_rows": 30}
    assert bounded["token_usage"] == sums
    assert bounded["token_usage"]["case_rows_omitted"] == 30


@pytest.mark.parametrize("room", ["sums", "truncation_record", "none"])
def test_token_usage_never_pushes_a_payload_over_the_report_node_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, room: str
) -> None:
    agents = load_agent_data(_cases_run(tmp_path, agents=2, cases=6))
    full = build_agent_eval_payload("demo", copy.deepcopy(agents), use_llm_judge=False)
    assert full is not None and "report_truncation" not in full
    base = _without_token_usage(full)
    sums = copy.deepcopy(full["token_usage"])
    drop_token_usage_cases(sums)
    # A truncation record is about 20 nodes; the sums are more and the case rows far more.
    extra = {"sums": _nodes(sums) + 40, "truncation_record": 30, "none": 0}[room]
    assert extra < _nodes(full["token_usage"])
    monkeypatch.setattr(reporting_base, "AGENT_EVAL_REPORT_MAX_NODES", _nodes(base) + extra)
    assert agent_eval_report_serialization_issue(full) is not None
    assert agent_eval_report_serialization_issue(base) is None

    bounded = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert bounded is not None
    assert agent_eval_report_serialization_issue(bounded) is None
    assert _without_token_usage(bounded) == base
    assert agent_eval_publication_evidence_complete(bounded) == agent_eval_publication_evidence_complete(base)
    if room == "sums":
        assert bounded["report_truncation"]["omitted"] == {"token_usage_case_rows": 12}
        assert bounded["token_usage"] == sums
        benchmark = BenchmarkReporter(include_timestamp=False).render_all([_validation_result_from_payload(bounded)])
        assert "12 case row(s) were left out of this report" in _section(benchmark, "## Token Usage")
    elif room == "truncation_record":
        assert bounded["report_truncation"]["omitted"] == {"token_usage": 1, "token_usage_case_rows": 12}
        assert "token_usage" not in bounded
    else:
        assert bounded == base


def test_dashboards_do_not_read_dropped_token_rows_as_missing_trials(run_result: tuple[Path, Any]) -> None:
    _run_dir, result = run_result
    report = copy.deepcopy(result.metadata["agent_eval"])
    report["report_truncation"] = {"truncated": True, "omitted": {"token_usage_case_rows": 4, "token_usage": 1}}

    data = parse_dashboard_report(report, "report.json")

    codex = [row for row in comparison_rows(data.rows) if row["_agent_key"] == "codex"]
    assert [row["comparison_status"] for row in codex] == ["comparable"]


# -- terminal ----------------------------------------------------------------------


def test_validate_terminal_prints_agent_rows_and_notes(run_result: tuple[Path, Any]) -> None:
    _run_dir, result = run_result
    output = _cli_output(result)

    assert "Token Usage (measured, not scored)" in output
    rows = {
        line.split("│")[1].strip(): [cell.strip() for cell in line.split("│")[2:-1]]
        for line in output.splitlines()
        if line.startswith("│")
    }
    assert rows["claude-code"] == ["3,300", "50,072", "N/A", "skill 2/2; baseline 2/2 (1 unscored)"]
    assert rows["codex"] == ["1,210", "1,650", "-440 (-26.7%)", "skill 2/2; baseline 2/2"]
    assert rows["All agents"] == ["4,510", "51,722", "N/A", "skill 4/4; baseline 4/4 (1 unscored)"]
    assert "claude-code: No change: baseline incomplete (1 of 2 attempts scored)." in output
    assert VALIDATE_CASE_POINTER in output
    assert "case-1" not in output.split("Token Usage (measured, not scored)", 1)[1].split("Token usage per case", 1)[0]


def test_tier3_evaluate_terminal_prints_the_agent_sums_the_report_step_recorded(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run = _run(tmp_path)
    engine_result: dict[str, Any] = {"execution_status": "failed", "run_dir": str(run)}

    render_agent_eval_html_report(skill, run, engine_result=engine_result, use_llm_judge=False)

    recorded = engine_result["token_usage"]
    assert set(recorded["agents"]) == {"claude-code", "codex"}
    assert all("cases" not in agent for agent in recorded["agents"].values())
    assert recorded["dataset"]["baseline"]["totals"]["total_tokens"] == 51722
    display = {
        "execution_status": "succeeded",
        "agents": {"codex": {"execution_status": "succeeded"}},
        "token_usage": recorded,
    }
    output = render_result(display)
    assert "Token Usage (measured, not scored)" in output
    assert re.search(r"claude-code\s+│\s+3,300\s+│\s+50,072\s+│\s+N/A", output)
    assert EVALUATE_CASE_POINTER in " ".join(output.split())


def test_terminal_tables_print_agent_names_as_text_not_markup() -> None:
    usage = {
        "agents": {
            "[bold red]agent[/bold red]": {
                "with_skill": {
                    "totals": {"total_tokens": 10},
                    "coverage": {"status": "complete", "measured_attempts": 1, "expected_attempts": 1},
                },
                "baseline": {"coverage": {"status": "not_run"}},
                "delta": {"status": "unavailable", "reason": "The baseline did not run."},
            }
        }
    }
    output = render_result({"execution_status": "succeeded", "agents": {"codex": {}}, "token_usage": usage})
    assert "[bold red]agent[/bold red]" in output
    assert "not run" in output


# -- Markdown, BENCHMARK.md and HTML ---------------------------------------------------------


def test_benchmark_token_usage_section_matches_the_golden_section(run_result: tuple[Path, Any]) -> None:
    _run_dir, result = run_result
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    section = "## Token Usage\n" + _section(benchmark, "## Token Usage")
    assert section.rstrip("\n") + "\n" == GOLDEN.read_text(encoding="utf-8")
    # It sits after the score table and before the tier status.
    assert (
        benchmark.index("## Results at a Glance")
        < benchmark.index("## Token Usage")
        < benchmark.index("## Tier Status")
    )


def test_benchmark_with_token_usage_passes_the_public_gate(run_result: tuple[Path, Any], tmp_path: Path) -> None:
    _run_dir, result = run_result
    card = tmp_path / "BENCHMARK.md"
    card.write_text(BenchmarkReporter(include_timestamp=False).render_all([result]), encoding="utf-8")

    assert [offender.reason for offender in benchmark_gate.scan_file(card)] == []


def test_markdown_report_lists_token_usage_under_tier3(run_result: tuple[Path, Any]) -> None:
    _run_dir, result = run_result
    markdown = MarkdownReporter().render_all([result])

    section = markdown.split("### Token Usage\n", 1)[1].split("\n## ", 1)[0]
    assert "It is not scored" in section
    rows = _table_rows(section)
    assert ["codex", "case-2", "660", "880", "-220 (-25.0%)", "skill 1/1; baseline 1/1"] in rows
    assert ["claude-code", "case-2", "2,200", "48,422", "N/A", "skill 1/1; baseline 1/1 (1 unscored)"] in rows


def test_html_report_lists_case_rows_under_measured_agent_resources(run_result: tuple[Path, Any]) -> None:
    _run_dir, result = run_result
    report = HTMLReporter(include_timestamp=False).render_all([result])

    resources = report.split("<summary>Measured Agent Resources</summary>", 1)[1].split("</details>", 1)[0]
    block = resources.split('<div id="t3-token-usage">', 1)[1]
    rows = [
        [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<td>(.*?)</td>", row)]
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", block, flags=re.DOTALL)
    ]
    assert ["claude-code", "All cases", "3,300", "50,072", "N/A", "skill 2/2; baseline 2/2 (1 unscored)"] in rows
    assert ["codex", "case-1", "550", "770", "-220 (-28.6%)", "skill 1/1; baseline 1/1"] in rows
    assert ["All agents", "All cases", "4,510", "51,722", "N/A", "skill 4/4; baseline 4/4 (1 unscored)"] in rows
    assert "claude-code: No change: baseline incomplete (1 of 2 attempts scored)." in block
    # Extends the resources section; no second token chart.
    assert report.count('id="tier3-token-chart"') <= 1


def test_html_rows_stay_whole_when_the_tier3_payload_is_above_the_preview_trigger(tmp_path: Path) -> None:
    result = agent_eval_result_from_directory(tmp_path, _cases_run(tmp_path, agents=2, cases=70), use_llm_judge=False)
    assert result is not None
    payload = result.metadata["agent_eval"]
    assert tier3_report._serialized_payload_size(payload) > html_reporting._TIER3_HTML_PREVIEW_TRIGGER_BYTES

    report = HTMLReporter(include_timestamp=False).render_all([result])

    assert "HTML preview omitted" in report
    block = report.split('<div id="t3-token-usage">', 1)[1].split("</details>", 1)[0]
    rows = [
        [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<td>(.*?)</td>", row)]
        for row in re.findall(
            r"<tr[^>]*>(.*?)</tr>", block.split("<tbody>", 1)[1].split("</table>", 1)[0], flags=re.DOTALL
        )
    ]
    agent_row = next(row for row in rows if row[:2] == ["agent-0", "All cases"])
    assert agent_row[2:] == ["10,115", "10,115", "+0 (+0.0%)", "skill 70/70; baseline 70/70"]
    case_rows = [row for row in rows if row[0] == "agent-1" and row[1] != "All cases"]
    assert [row[1] for row in case_rows] == [f"case-{index:03d}" for index in range(64)]
    assert all(row[2] == row[3] and "(partial)" not in row[2] for row in case_rows)
    assert "12 more case row(s) are in the JSON payload under token_usage." in block


def test_untrusted_case_names_are_escaped_in_every_report(tmp_path: Path) -> None:
    hostile = "<img src=x onerror=alert(1)>|`x`"
    run = tmp_path / "run"
    for variant in ("with-skill", "without-skill"):
        _arm(run, "codex", variant, [("case__a", (100, 10, 0), True)])
        reward_file = run / "codex" / variant / "trials" / "case__a" / "reward.json"
        reward = json.loads(reward_file.read_text(encoding="utf-8"))
        reward_file.write_text(json.dumps({**reward, "entry_id": hostile}), encoding="utf-8")
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    assert result is not None
    assert result.metadata["agent_eval"]["token_usage"]["agents"]["codex"]["cases"][0]["entry_id"] == hostile

    report = HTMLReporter(include_timestamp=False).render_all([result])
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])
    markdown = MarkdownReporter().render_all([result])

    block = report.split('<div id="t3-token-usage">', 1)[1].split("</table>", 1)[0]
    assert "<img" not in block
    assert "&lt;img src=x onerror=alert(1)&gt;" in block
    for text, heading in ((benchmark, "## Token Usage"), (markdown, "### Token Usage")):
        section = _section(text, heading)
        assert "<img" not in section
        [row] = [line for line in section.splitlines() if "onerror" in line]
        # The pipe in the case name cannot add a column.
        assert len(_table_rows(row)[0]) == 6


def test_markdown_tables_cap_case_rows_and_say_where_the_rest_are(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cases = [(f"case-{index:02d}__a", (100, 10, 0), True) for index in range(60)]
    _arm(run, "codex", "with-skill", cases)
    _arm(run, "codex", "without-skill", cases)
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    assert result is not None

    for text, heading in (
        (BenchmarkReporter(include_timestamp=False).render_all([result]), "## Token Usage"),
        (MarkdownReporter().render_all([result]), "### Token Usage"),
    ):
        section = _section(text, heading)
        assert len([row for row in _table_rows(section) if row[1].startswith("case-")]) == 50
        assert "10 more case row(s) are in the JSON payload under `token_usage`." in section


def test_docs_and_changelog_describe_the_token_usage_contract() -> None:
    docs = (REPO_ROOT / "docs" / "reports.mdx").read_text(encoding="utf-8")
    section = docs.split("### Token usage\n", 1)[1].split("\n### ", 1)[0]
    for field in (
        "token_usage",
        "total_tokens",
        "uncached_prompt_tokens",
        "expected_attempts",
        "observed_attempts",
        "measured_attempts",
        "unscored_attempts",
        "missing_token_counters",
        "invalid_token_counters",
        "missing_attempts",
        "total_tokens_change_percent",
        "baseline_failed",
        "case_scope_mismatch",
        "agent_scope_mismatch",
        "validate --verbose",
        "tier3 evaluate --verbose",
        "first 50 case rows",
    ):
        assert field in section, field
    unreleased = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split("\n## ", 2)[1]
    added = unreleased.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    assert "token usage per case" in added
