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
    agent_eval_result_from_directory,
    build_agent_eval_payload,
    render_agent_eval_html_report,
)
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter, MarkdownReporter
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
            reward = {"entry_id": name.split("__", 1)[0], "trial_id": name, **scores}
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


def test_payload_budget_drops_case_rows_with_the_trials_and_keeps_the_sums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    cases = [(f"case-{index}__a", (100 + index, 10, 0), True) for index in range(40)]
    _arm(run, "codex", "with-skill", cases)
    _arm(run, "codex", "without-skill", cases)
    agents = load_agent_data(run)
    full = build_agent_eval_payload("demo", copy.deepcopy(agents), use_llm_judge=False, enforce_embedded_size=False)
    assert full is not None
    slim = copy.deepcopy(full)
    slim["dataset"], slim["trials"] = [], []
    for agent in slim["agents"].values():
        agent["trials"], agent["trials_baseline"], agent["cases"] = [], [], []
    assert drop_token_usage_cases(slim["token_usage"]) == 40
    monkeypatch.setattr(tier3_report, "_MAX_EMBEDDED_REPORT_BYTES", tier3_report._serialized_payload_size(slim) + 4096)

    bounded = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert bounded is not None
    assert bounded["token_usage"]["agents"]["codex"]["cases"] == []
    assert (
        bounded["token_usage"]["agents"]["codex"]["with_skill"] == full["token_usage"]["agents"]["codex"]["with_skill"]
    )
    assert bounded["report_truncation"]["omitted"]["dataset_and_trial_items"] >= 40


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
    assert "Token usage per case is in the HTML report and BENCHMARK.md." in output
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
