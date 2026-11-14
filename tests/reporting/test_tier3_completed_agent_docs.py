# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the completed-agent docs aligned with what Tier 3 reports render."""

from __future__ import annotations

import base64
import json
import re
from html import unescape
from pathlib import Path

import pytest
from tests.reporting.test_tier3_scoped_reports import result_with_completed_peer

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter
from skillevaluator.reporting.html import VERDICT_WORDS
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

REPO = Path(__file__).resolve().parents[2]
HEADING = "### Completed agents in an incomplete run"
ANCHOR = "completed-agents-in-an-incomplete-run"
NEEDS_ATTENTION = "Needs attention: other execution failures remain"


def _plain(text: str) -> str:
    """Drop HTML tags, entities and Markdown emphasis, and collapse whitespace."""
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).replace("*", "").split())


def _reports_page() -> str:
    return (REPO / "docs" / "reports.mdx").read_text(encoding="utf-8")


def _section() -> str:
    page = _reports_page()
    start = page.index(HEADING)
    return _plain(page[start : page.index("\n## ", start)])


def _run_details(raw_html: str) -> str:
    """Return the text of the collapsed Run details & provenance panel."""
    start = raw_html.index('<details class="run-metadata"')
    return _plain(raw_html[start : raw_html.index("</details>", start)])


def _embedded_tier3(raw_html: str) -> dict:
    """Return the full Tier 3 payload the HTML report embeds for download."""
    match = re.search(
        r'<script type="application/json" id="tier3-full"( data-encoding="base64")?>(.*?)</script>', raw_html, re.DOTALL
    )
    assert match is not None
    body = match.group(2)
    return json.loads(base64.b64decode(body).decode("utf-8") if match.group(1) else body)


def _hero_command_status(raw_html: str) -> str:
    match = re.search(r'<p class="hero-command-status[^"]*">([^<]*)</p>', raw_html)
    assert match is not None
    return match.group(1).strip()


def _agent(score: float, *, baseline_scored: int = 1, recovered: int = 0) -> dict:
    conditions = {
        "with_skill": {
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 1,
            "scored_attempts": 1,
            "recovered_attempts_total": recovered,
        },
        "without_skill": {
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 1,
            "scored_attempts": baseline_scored,
        },
    }
    return {
        "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 2,
        "scored_attempts": 1 + baseline_scored,
        "conditions": conditions,
        "num_trials": 1,
        "num_trials_baseline": 1,
    }


def _write_scored_run(run: Path) -> None:
    """Write one complete opencode agent's arm summaries into a run directory."""
    for arm in ("with-skill", "without-skill"):
        directory = run / "opencode" / arm
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "scores": dict.fromkeys(DEFAULT_METRICS, 0.8),
            "metrics": list(DEFAULT_METRICS),
            "num_trials": 1,
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 1,
            "scored_attempts": 1,
        }
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _three_agent_run(*, best_baseline_scored: int, peer_recovered: int) -> ValidationResult:
    """opencode scores best, claude-code finishes lower, and codex fails."""
    agents = {
        "opencode": _agent(0.8, baseline_scored=best_baseline_scored),
        "claude-code": _agent(0.6, recovered=peer_recovered),
        "codex": {
            "execution_status": "failed",
            "execution_errors": ["Agent setup failed"],
            "with_skill": {},
            "without_skill": {},
        },
    }
    payload = build_agent_eval_payload(
        "demo",
        agents,
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload["execution_status"] == "failed"
    assert payload["best_agent"] == "opencode"
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = payload
    result.add_error("Agent setup failed")
    return result


def test_documented_labels_match_a_run_with_a_failed_peer() -> None:
    result = result_with_completed_peer()
    payload = result.metadata["agent_eval"]
    payload["agents"]["opencode"]["conditions"]["with_skill"]["recovered_attempts_total"] = 1
    section = _section()
    raw = HTMLReporter(report_scope="tier3").render(result)
    html = _plain(raw)
    details = _run_details(raw)
    scoped = _plain(BenchmarkReporter(report_scope="tier3").render(result))
    card = _plain(BenchmarkReporter().render(result))

    assert payload["execution_status"] == "failed"
    assert payload["summary"]["overall_score"] is None
    assert payload["summary"]["overall_lift"] == pytest.approx(0.4)
    assert payload["summary"]["verdict"] == payload["verdict"] == "neutral"
    for text in ("summary.overall_score", "summary.overall_lift", "`neutral`"):
        assert text in section
    for label in (
        "Completed-agent result. Other execution failures need attention",
        "Completed Agent Score",
        "Quality Verdict",
        "Tier 3 quality",
        "Run details & provenance",
        "Best performing agent",
        "Skill Lift",
        "Best-agent lift",
    ):
        assert label in section
        assert label in html
    # The standalone quality labels follow the run; completed-agent details keep its score.
    assert "card follow the run's verdict" in section
    # The JSON report's Tier 3 payload keeps neutral, but the HTML report, also its
    # embedded payload, shows fail. Both payloads keep the best agent's lift.
    report_tier3 = json.loads(JSONReporter().render_all([result]))["tier3"]
    embedded = _embedded_tier3(raw)
    assert "The payload keeps `neutral`" not in section
    assert (
        "The Tier 3 payload of a `validate --tier3` JSON report, its `tier3` object, keeps `neutral` for a failed run"
    ) in section
    assert report_tier3["verdict"] == report_tier3["summary"]["verdict"] == "neutral"
    assert f"shows a failed run's verdict as `fail`, labelled {VERDICT_WORDS['fail']}" in section
    assert "the full Tier 3 payload embedded in the HTML report records `fail` too" in section
    assert embedded["verdict"] == "fail"
    assert 'data-verdict="fail"' in raw
    assert (
        "The JSON report's `tier3` object and the payload embedded in the HTML report both keep this lift in "
        "`summary.overall_lift`."
    ) in section
    assert "records the failed run's `verdict` as `neutral`" not in section
    assert report_tier3["summary"]["overall_lift"] == embedded["summary"]["overall_lift"] == pytest.approx(0.4)
    assert "Tier 3 quality: Changes required" in html
    assert "Changes required" in section
    assert _hero_command_status(raw) == "Changes required"
    assert "Skill Lift +0.40" in html
    assert "1 timeout attempt(s) recovered" in details
    assert "Best-agent lift: +0.40" in details
    assert NEEDS_ATTENTION in section
    assert NEEDS_ATTENTION in scoped
    assert "Recovered timeout attempts: 1" in scoped
    # A validate card shows the same lines under its unchanged verdict.
    assert NEEDS_ATTENTION in card
    assert "Recovered timeout attempts: 1" in card
    assert "Overall verdict: FAIL" in card


def test_documented_label_matches_a_successful_neutral_run() -> None:
    # Dimensions at 0.45 over a 0.40 baseline: the run succeeded with a neutral verdict.
    payload = build_agent_eval_payload(
        "demo",
        {"opencode": _agent(0.45)},
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert (payload["execution_status"], payload["verdict"]) == ("succeeded", "neutral")
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = payload
    raw = HTMLReporter(report_scope="tier3").render(result)

    assert f"A `neutral` verdict of a successful run reads {VERDICT_WORDS['neutral']}." in _section()
    assert f"Tier 3 quality: {VERDICT_WORDS['neutral']}" in _plain(raw)
    assert _embedded_tier3(raw)["verdict"] == "neutral"


def test_recovered_count_is_the_shown_agents_own() -> None:
    # claude-code recovered two timeouts, but the reports show opencode.
    result = _three_agent_run(best_baseline_scored=1, peer_recovered=2)
    raw = HTMLReporter(report_scope="tier3").render(result)
    scoped = _plain(BenchmarkReporter(report_scope="tier3").render(result))

    assert "Timeouts that other agents recovered are not counted" in _section()
    assert "Completed agent: opencode" in scoped
    assert "Recovered timeout attempts" not in scoped
    assert "timeout attempt(s) recovered" not in _plain(raw)


def test_recovered_count_when_every_agent_finished() -> None:
    result = result_with_completed_peer(peer_failed=False)
    payload = result.metadata["agent_eval"]
    payload["agents"]["opencode"]["conditions"]["with_skill"]["recovered_attempts_total"] = 1
    raw = HTMLReporter(report_scope="tier3").render(result)

    assert payload["execution_status"] == "succeeded"
    assert "also give this count when every agent finished" in _section()
    assert "1 timeout attempt(s) recovered" in _run_details(raw)
    assert "Recovered timeout attempts: 1" in _plain(BenchmarkReporter(report_scope="tier3").render(result))
    # A validate card shows the completed-agent lines only when the run failed.
    assert "Recovered timeout attempts" not in _plain(BenchmarkReporter().render(result))


def test_no_fallback_to_a_lower_complete_agent() -> None:
    # The best agent's baseline is one attempt short. claude-code is complete
    # but scored lower, so it never stands in for opencode.
    result = _three_agent_run(best_baseline_scored=0, peer_recovered=2)
    payload = result.metadata["agent_eval"]
    section = _section()
    raw = HTMLReporter(report_scope="tier3").render(result)
    html = _plain(raw)
    scoped_md = BenchmarkReporter(report_scope="tier3").render(result)
    line = "Tier 3 completed-agent result: NEUTRAL"

    assert "the reports do not fall back to another agent" in section
    assert line in section
    assert line in _plain(scoped_md)
    assert "Completed agent:" not in scoped_md
    assert "Recovered timeout attempts" not in scoped_md
    assert "Completed Agent Score" not in html
    assert "Completed-agent result" not in html
    # The HTML still names the best agent and shows its lift.
    assert "even when its evidence is incomplete" in section
    assert payload["summary"]["overall_lift"] == pytest.approx(0.4)
    assert "Best performing agent: opencode" in html
    assert "Skill Lift +0.40" in html
    assert "Best-agent lift: +0.40" in _run_details(raw)


def test_view_regenerates_only_the_html_report(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.tier3 import commands

    skill = tmp_path / "demo"
    skill.mkdir()
    run = tmp_path / "results" / "demo" / "20260709_120002"
    _write_scored_run(run)
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"run_id": run.name}), encoding="utf-8")
    render = commands.render_agent_eval_html_report
    calls: list[dict] = []

    def offline_render(skill_path: Path, target: Path, **kwargs) -> Path:
        calls.append(kwargs)
        return render(skill_path, target, use_llm_judge=False, **kwargs)

    monkeypatch.setattr(commands, "render_agent_eval_html_report", offline_render)
    monkeypatch.setattr(commands.webbrowser, "open", lambda _uri: True)

    report = commands.view_results(skill, results_dir=tmp_path / "results")

    assert calls == [{}]
    assert report.resolve() == (run / "report.html").resolve()
    assert report.is_file()
    assert not (run / "tier3-benchmark.md").exists()
    assert "[`view`](#view) regenerates only a missing `report.html`" in _plain(_reports_page())


def test_validate_tier3_writes_the_same_report_pair(monkeypatch, tmp_path: Path) -> None:
    # validate runs Tier 3 on the same engine as tier3 evaluate, so it writes the
    # report pair into the results tree and degrades the same way.
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import tier3_report
    from skillevaluator.provider_config import ProviderConfig
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.harbor import runner, runtime_preflight

    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  task_source: evals_json\n", encoding="utf-8"
    )
    provider = ProviderConfig(
        provider="nv_build",
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )

    def stage(_skill: Path, target: Path, **_kwargs) -> list[Path]:
        task = target / "case-a"
        (task / "tests").mkdir(parents=True)
        (task / "tests" / "entry.json").write_text(json.dumps({"id": "case-a"}), encoding="utf-8")
        return [task]

    def collect(**_kwargs) -> dict:
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}}

    render = runner.render_agent_eval_html_report
    calls: list[tuple[Path, dict]] = []

    def offline_render(skill_path: Path, run_dir: Path, **kwargs) -> Path:
        calls.append((run_dir, kwargs))
        _write_scored_run(run_dir)
        return render(skill_path, run_dir, use_llm_judge=False, **kwargs)

    def unwritable(*_args, **_kwargs) -> None:
        raise OSError("tier3-benchmark.md is not writable")

    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "generate_harbor_tasks", stage)
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", offline_render)
    monkeypatch.setattr(BenchmarkReporter, "save", unwritable)
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "available"),
    )
    # Normalizing the run for the combined report is not under test here.
    monkeypatch.setattr(
        tier3_report, "agent_eval_result_from_run", lambda *_a, **_kw: ValidationResult(validator_name="AGENT_EVAL")
    )

    cli_module._run_agent_eval_or_skip(
        skill,
        agents="opencode",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        results_dir=tmp_path / "results",
        agent_runtime_preflight=False,
        validate_source=False,
    )

    assert calls
    run_dir, kwargs = calls[-1]
    assert kwargs["include_benchmark"] is True
    assert run_dir.resolve().parent == (tmp_path / "results" / "demo").resolve()
    assert not (run_dir / "report.html").exists()
    assert not (run_dir / "tier3-benchmark.md").exists()
    saved = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert saved["report_status"] == "degraded"
    assert any("tier3-benchmark.md is not writable" in warning for warning in saved["warnings"])


def test_tier3_benchmark_gives_its_own_reason_for_not_assessed() -> None:
    scoped = _plain(BenchmarkReporter(report_scope="tier3").render(result_with_completed_peer()))
    page = _plain(_reports_page())

    assert "Publication: NOT ASSESSED. This report covers Tier 3 only" in scoped
    reason = "publication as NOT ASSESSED, because it covers Tier 3 only, even when it comes from a `validate` run"
    assert reason in page
    assert "because Tiers 1 and 2 did not run" not in page


def test_results_tree_and_tier3_guide_point_to_the_section() -> None:
    page = _reports_page()
    guide = (REPO / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")

    assert "      - tier3-benchmark.md\n" in page
    every_run = "At the end of every Tier 3 run, from `tier3 evaluate` or a `validate` run that includes Tier 3"
    assert every_run in _plain(page)
    assert f"(#{ANCHOR})" in page
    assert f"(/skills/skillevaluator/reports#{ANCHOR})" in guide
    assert "The verdict table above applies only when the run succeeds." in guide


def test_changelog_entry_matches_the_docs() -> None:
    # The entry is found by its opening words so that a release heading above it
    # can change. test_no_fallback_to_a_lower_complete_agent shows that no score
    # is shown when the best agent's evidence is incomplete.
    lead = "- Reports & Results now explains the completed-agent result in Tier 3 reports."
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entries = [_plain(block) for block in changelog.split("\n\n") if _plain(block).startswith(lead)]

    assert len(entries) == 1
    assert "show the best agent's score when that agent's evidence is complete" in entries[0]
    assert "every Tier 3 run, from `tier3 evaluate` or a `validate` run that includes Tier 3" in entries[0]


def _changelog_entry(lead: str) -> tuple[str, str, str]:
    """Return the release heading, subsection heading and plain text of the one entry starting with ``lead``."""
    release = subsection = ""
    found: list[tuple[str, str, str]] = []
    for block in (REPO / "CHANGELOG.md").read_text(encoding="utf-8").split("\n\n"):
        first = block.strip().splitlines()[0] if block.strip() else ""
        if first.startswith("## "):
            release, subsection = first, ""
        elif first.startswith("### "):
            subsection = first
        elif _plain(block).startswith(lead):
            found.append((release, subsection, _plain(block)))
    assert len(found) == 1
    return found[0]


def _newer_than_0_9_0(release: str) -> bool:
    """Whether ``release`` is ``## Unreleased`` or the heading of a release after 0.9.0."""
    if release == "## Unreleased":
        return True
    match = re.match(r"## (\d+)\.(\d+)\.(\d+)\b", release)
    return match is not None and tuple(int(part) for part in match.groups()) > (0, 9, 0)


def test_changelog_documents_the_failed_run_verdict_label() -> None:
    # The entry is found by its opening words in any section, so a release that
    # moves Unreleased into a dated section keeps it valid. It must stay out of
    # 0.9.0 and older released sections.
    release, subsection, entry = _changelog_entry(
        "- Reports & Results said that the HTML report labels a failed Tier 3 run's `neutral` verdict"
    )
    assert subsection == "### Documentation"
    assert _newer_than_0_9_0(release), release
    result = result_with_completed_peer()
    raw = HTMLReporter(report_scope="tier3").render(result)
    report_tier3 = json.loads(JSONReporter().render_all([result]))["tier3"]
    assert report_tier3["execution_status"] == "failed"
    assert report_tier3["verdict"] == "neutral"
    assert _embedded_tier3(raw)["verdict"] == "fail"
    assert (
        "The `tier3` object of a `validate --tier3` JSON report keeps `neutral` for a failed run, but the HTML "
        f"report and the full Tier 3 payload it embeds show the verdict as `fail`, labelled {VERDICT_WORDS['fail']}."
    ) in entry
    assert f"A `neutral` verdict of a successful run reads {VERDICT_WORDS['neutral']}." in entry
