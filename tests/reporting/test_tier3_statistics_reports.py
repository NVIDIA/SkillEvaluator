# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only run statistics reach the loader, the payload and every report, and never change a score.

A run directory collected like a real run goes through the loader, the canonical
payload, the terminal, the Markdown report and the HTML report. Runs saved before
statistics existed render exactly as they did, and the publication card never
carries them.
"""

from __future__ import annotations

import copy
import io
import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from tests.statistics_run_fixtures import AGENT, CASES, SKILL, Trial, collect, graded_baseline, uniform, write_job

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter, MarkdownReporter
from skillevaluator.reporting.tier3_statistics import (
    LIFT_TITLE,
    RELIABILITY_TITLE,
    STATISTICS_NOTE,
    statistics_markdown_lines,
    statistics_section,
    statistics_tables,
)
from skillevaluator.tier3.harbor import report_data, stats
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.result_display import render_result

FIELDS = stats.RUN_STATISTICS_FIELDS
ZERO_CROSSING = "Skill Lift not distinguishable from zero"


def _noisy_with_skill() -> dict[str, list[Trial]]:
    """Cases that score above and below an even baseline, so the lift interval spans zero."""
    qualities = [0.9, 0.3, 0.9, 0.3, 0.9, 0.6]
    return {case: [Trial(quality)] for case, quality in zip(CASES, qualities, strict=True)}


def _good_run(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    return collect(tmp_path, uniform(0.9), graded_baseline())


def _repeated_run(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Two attempts per case: the skill passes both every time, the baseline never passes both."""
    with_skill = {case: [Trial(0.9), Trial(0.9)] for case in CASES}
    baseline = {
        case: [Trial(0.6), Trial(0.2, passed=False)] if index % 2 else [Trial(0.2, passed=False)] * 2
        for index, case in enumerate(CASES)
    }
    return collect(tmp_path, with_skill, baseline)


def _result(run: Path):
    return agent_eval_result_from_directory(run.parent / SKILL, run, use_llm_judge=False)


def _payload(run: Path) -> dict[str, Any]:
    return _result(run).metadata["agent_eval"]


def _terminal(payload: dict[str, Any]) -> str:
    buffer = io.StringIO()
    CLIReporter._print_agent_eval_tables(payload, Console(file=buffer, width=200, force_terminal=False))
    return buffer.getvalue()


def _statistics_block(html: str) -> str:
    match = re.search(r'<details class="t3-details" id="tier3-run-statistics">(.*?)</details>', html, re.DOTALL)
    assert match, "the HTML report has no run statistics block"
    return match.group(1)


def test_load_agent_data_reads_statistics_and_ignores_oversize_linked_or_malformed_files(tmp_path: Path) -> None:
    _collected, run = _good_run(tmp_path)
    file = run / AGENT / "statistics.json"
    published = report_data.load_agent_data(run)[AGENT]
    assert {field for field in FIELDS if field in published} == set(FIELDS)
    file.unlink()
    plain = report_data.load_agent_data(run)[AGENT]
    assert not [field for field in FIELDS if field in plain]
    assert set(published) - set(FIELDS) == set(plain)

    # Only the two report blocks are read, and only when they are mappings.
    file.write_text(
        json.dumps({"lift_uncertainty": [1], "reliability": {"with_skill": {"pass_at_k": 0.5}}, "cost": {"x": 1}}),
        encoding="utf-8",
    )
    assert report_data.load_agent_data(run)[AGENT] == {**plain, "reliability": {"with_skill": {"pass_at_k": 0.5}}}

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"reliability": {"with_skill": {"pass_at_k": 0.5}}}), encoding="utf-8")
    variants = {
        "oversize": lambda: file.write_text(" " * (2 * 1024 * 1024 + 1), encoding="utf-8"),
        "malformed": lambda: file.write_text("{not json", encoding="utf-8"),
        "list": lambda: file.write_text("[1, 2, 3]", encoding="utf-8"),
        "deep": lambda: file.write_text("[" * 5_000 + "]" * 5_000, encoding="utf-8"),
        "directory": file.mkdir,
        "linked": lambda: file.symlink_to(outside),
    }
    for name, make in variants.items():
        if file.is_symlink() or file.is_file():
            file.unlink()
        elif file.is_dir():
            file.rmdir()
        try:
            make()
        except (OSError, NotImplementedError):
            continue  # symlinks are not available here
        loaded = report_data.load_agent_data(run)[AGENT]
        # No block and no diagnostic about the file: the report reads as a run without statistics.
        assert loaded == plain, name


def test_payload_carries_statistics_for_each_agent(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    write_job(jobs / f"{SKILL}-codex-with", uniform(0.9))
    write_job(jobs / f"{SKILL}-codex-without", graded_baseline())
    write_job(jobs / f"{SKILL}-claude-code-with", uniform(0.7))
    write_job(jobs / f"{SKILL}-claude-code-without", graded_baseline())
    run = tmp_path / "run"
    collect_harbor_results(
        SKILL, ["codex", "claude-code"], run, jobs, expected_case_ids=CASES, n_attempts=1, expected_trials=6
    )

    payload = _payload(run)

    assert set(payload["agents"]) == {"codex", "claude-code"}
    for agent in payload["agents"].values():
        assert all(isinstance(agent[field], dict) for field in FIELDS)
    lifts = {name: agent["lift_uncertainty"]["effectiveness"]["estimate"] for name, agent in payload["agents"].items()}
    assert lifts["codex"] != lifts["claude-code"]


def test_old_runs_without_statistics_render_unchanged(tmp_path: Path) -> None:
    _collected, run = _good_run(tmp_path)
    with_statistics = _result(run)
    stripped = copy.deepcopy(with_statistics)
    for agent in stripped.metadata["agent_eval"]["agents"].values():
        for field in FIELDS:
            agent.pop(field, None)
    (run / AGENT / "statistics.json").unlink()
    old = _result(run)
    markdown, html = MarkdownReporter(), HTMLReporter(include_timestamp=False)

    assert old.metadata["agent_eval"] == stripped.metadata["agent_eval"]
    old_markdown, old_html = markdown.render_all([old]), html.render_all([old])
    old_terminal = _terminal(old.metadata["agent_eval"])
    assert old_markdown == markdown.render_all([stripped])
    assert old_html == html.render_all([stripped])
    assert old_terminal == _terminal(stripped.metadata["agent_eval"])
    assert LIFT_TITLE not in old_terminal
    assert "Run Statistics" not in old_markdown
    assert "tier3-run-statistics" not in old_html
    assert LIFT_TITLE in _terminal(with_statistics.metadata["agent_eval"])


def test_zero_crossing_interval_adds_warning_and_keeps_verdict_and_lift(tmp_path: Path) -> None:
    _collected, run = collect(tmp_path, _noisy_with_skill(), uniform(0.6))
    with_statistics = _payload(run)
    (run / AGENT / "statistics.json").unlink()
    without = _payload(run)

    interval = with_statistics["agents"][AGENT]["lift_uncertainty"]["effectiveness"]
    assert interval["ci_includes_zero"] is True
    added = [c for c in with_statistics["conclusions"] if c not in without["conclusions"]]
    assert [(c["severity"], c["title"]) for c in added] == [("warn", ZERO_CROSSING)]
    assert "paired cases" in added[0]["message"]
    assert "verdict are unchanged" in added[0]["message"]
    assert len(with_statistics["conclusions"]) == len(without["conclusions"]) + 1
    for key in ("verdict", "overall_lift", "composite_lift", "overall_score", "dataset_sufficiency", "summary"):
        assert with_statistics[key] == without[key], key
    assert with_statistics["agents"][AGENT]["lift"] == without["agents"][AGENT]["lift"]
    assert "includes zero" in " ".join(_terminal(with_statistics).split())


def test_an_interval_that_excludes_zero_adds_no_warning(tmp_path: Path) -> None:
    _collected, run = _good_run(tmp_path)

    payload = _payload(run)

    assert payload["agents"][AGENT]["lift_uncertainty"]["effectiveness"]["ci_includes_zero"] is False
    assert ZERO_CROSSING not in [conclusion["title"] for conclusion in payload["conclusions"]]


def test_terminal_lift_table(tmp_path: Path) -> None:
    collected, run = _good_run(tmp_path)

    saved = _terminal(_payload(run))
    live = " ".join(render_result(collected).split())

    for output in (" ".join(saved.split()), live):
        assert LIFT_TITLE in output
        assert "[+0.21, +0.59]" in output
        assert "includes zero" not in output
        # One attempt per case: pass^k would repeat pass@k, so there is no reliability table.
        assert RELIABILITY_TITLE not in output


def test_terminal_reliability_table_for_repeated_attempts(tmp_path: Path) -> None:
    collected, run = _repeated_run(tmp_path)

    saved = " ".join(_terminal(_payload(run)).split())
    live = " ".join(render_result(collected).split())

    for output in (saved, live):
        assert RELIABILITY_TITLE in output
        assert "│ codex │ With skill │ 100% │ 100% │ 2 │" in output
        assert "│ codex │ Baseline │ 50% │ 0% │ 2 │" in output


def test_terminal_shows_no_interval_for_one_case(tmp_path: Path) -> None:
    _collected, run = collect(tmp_path, uniform(0.9, count=1), uniform(0.5, count=1))

    output = _terminal(_payload(run))

    assert LIFT_TITLE in output
    row = next(line for line in output.splitlines() if line.startswith("│ codex") and "N/A" in line)
    assert "[" not in row
    assert "includes zero" not in output
    assert re.search(r"codex\s+│\s+\+0\.\d\d\s+│\s+N/A\s+│\s+insufficient\s+│\s+1\s", output)


def test_markdown_report_tables(tmp_path: Path) -> None:
    _collected, run = _repeated_run(tmp_path)

    markdown = MarkdownReporter().render_all([_result(run)])

    section = markdown[markdown.index("### Run Statistics (advisory)") :].split("\n### ", 1)[0]
    assert STATISTICS_NOTE in section
    assert f"**{LIFT_TITLE}**" in section
    assert f"**{RELIABILITY_TITLE}**" in section
    assert "| codex | With skill | 100% | 100% | 2 |" in section
    assert "| codex | Baseline | 50% | 0% | 2 |" in section


def test_markdown_report_has_no_reliability_table_for_one_attempt(tmp_path: Path) -> None:
    _collected, run = _good_run(tmp_path)

    markdown = MarkdownReporter().render_all([_result(run)])

    section = markdown[markdown.index("### Run Statistics (advisory)") :].split("\n### ", 1)[0]
    assert "| codex | +0.40 | [+0.21, +0.59] | low | 6 |" in section
    assert RELIABILITY_TITLE not in section


def _hostile(payload: dict[str, Any]) -> dict[str, Any]:
    name = "<img src=x onerror=alert(1)>|agent"
    agent = payload["agents"][AGENT]
    payload["agents"] = {name: agent}
    payload["best_agent"] = name
    payload["agents_run"] = [name]
    return payload


def test_markdown_tables_escape_hostile_text(tmp_path: Path) -> None:
    _collected, run = _repeated_run(tmp_path)
    payload = _hostile(_payload(run))

    lines = statistics_markdown_lines(
        payload["agents"],
        heading="### Run Statistics (advisory)",
        agent_label=lambda value: value.replace("|", "&#124;").replace("<", "&lt;"),
    )

    text = "\n".join(lines)
    assert "<img" not in text
    assert "&lt;img" in text
    rows = [line for line in lines if line.startswith("| &lt;img")]
    assert len(rows) == 3
    for line in rows:
        assert len(line.split(" | ")) == 5


def test_html_report_tables_escape_hostile_agent_names(tmp_path: Path) -> None:
    _collected, run = _repeated_run(tmp_path)
    result = _result(run)
    result.metadata["agent_eval"] = _hostile(result.metadata["agent_eval"])

    html = HTMLReporter(include_timestamp=False).render_all([result])

    block = _statistics_block(html)
    assert "<img" not in block
    assert "&lt;img src=x onerror=alert(1)&gt;|agent" in block


def test_html_report_tables(tmp_path: Path) -> None:
    _collected, run = _repeated_run(tmp_path)

    html = HTMLReporter(include_timestamp=False).render_all([_result(run)])

    block = _statistics_block(html)
    assert html.index('id="tier3-run-statistics"') < html.index('id="tier3-agent-resources"')
    assert STATISTICS_NOTE in block.replace("&#39;", "'")
    for text in (LIFT_TITLE, RELIABILITY_TITLE, "pass^k", "Baseline"):
        assert text in block
    assert block.count("<table") == 2


def test_html_report_has_only_the_lift_table_for_one_attempt(tmp_path: Path) -> None:
    _collected, run = _good_run(tmp_path)

    block = _statistics_block(HTMLReporter(include_timestamp=False).render_all([_result(run)]))

    assert "[+0.21, +0.59]" in block
    assert RELIABILITY_TITLE not in block
    assert block.count("<table") == 1


def test_statistics_never_reach_the_publication_card(tmp_path: Path) -> None:
    _collected, run = _repeated_run(tmp_path)
    with_statistics = _result(run)
    (run / AGENT / "statistics.json").unlink()
    without = _result(run)

    cards = [
        BenchmarkReporter(skill_name=SKILL, report_scope="tier3").render(result)
        for result in (with_statistics, without)
    ]

    assert cards[0] == cards[1]
    assert LIFT_TITLE not in cards[0]
    assert RELIABILITY_TITLE not in cards[0]
    assert "pass^k" not in cards[0]


def test_row_builders_read_nothing_they_cannot_trust() -> None:
    agents = {
        "ok": {"lift_uncertainty": {"effectiveness": {"estimate": 0.2, "ci_low": -0.1, "ci_high": 0.5}}},
        "weird": {
            "lift_uncertainty": {"effectiveness": {"estimate": True}},
            "reliability": {"with_skill": {"pass_at_k": float("nan"), "pass_hat_k": 7, "k": True}},
        },
        "repeated": {"reliability": {"without_skill": {"pass_at_k": 0.5, "pass_hat_k": 0.25, "k": 2}}},
        7: {"lift_uncertainty": {}},
        "text": "not a mapping",
    }

    tables = {table.title: [(row.agent, row.cells) for row in table.rows] for table in statistics_tables(agents)}

    assert tables == {
        LIFT_TITLE: [("ok", ("+0.20", "[-0.10, +0.50]", "N/A", "N/A"))],
        RELIABILITY_TITLE: [("repeated", ("Baseline", "50%", "25%", "2"))],
    }
    assert statistics_tables(None) == []
    assert statistics_section({"nothing": {}}) is None
    assert statistics_section(agents) is not None


def test_reliability_rows_are_left_out_for_a_single_attempt_or_an_early_stop() -> None:
    single = {"with_skill": {"pass_at_k": 1.0, "pass_hat_k": 1.0, "k": 1}}
    stopped = {"with_skill": {"pass_at_k": 1.0, "pass_hat_k": None, "k": 3}}
    repeated = {"with_skill": {"pass_at_k": 1.0, "pass_hat_k": 0.5, "k": 3}}

    rows = {
        name: [row.cells for table in statistics_tables({"codex": {"reliability": block}}) for row in table.rows]
        for name, block in (("single", single), ("stopped", stopped), ("repeated", repeated))
    }

    assert rows["single"] == rows["stopped"] == []
    assert rows["repeated"] == [("With skill", "100%", "50%", "3")]
