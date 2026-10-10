# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One ``validate`` run, one headline: every surface shows the status ``resolve_run_status`` decides.

Each scenario runs the real ``validate`` command (only the Tier 3 engine and, where named, the SkillSpector scan are
replaced) and reads the headline back from every report it writes: the terminal summary, the quiet footer, JSON,
Markdown, HTML, SARIF and BENCHMARK.md. The rule is the documented one (plugin-evaluation.mdx, "Reports and
verdicts"): a real failure outranks missing evidence, missing evidence outranks a pass, and a tier outside the exit
gate never changes the headline. BENCHMARK.md is a publication card, so it may only be stricter than the gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _f90_gate_cli import plugin_dir, scores, tier3_result
from click.testing import CliRunner

import skillevaluator.cli as cli_module
from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    advisory_skip_result,
    build_agent_eval_payload,
    incomplete_reason,
)
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.utils.tool_runner import Tools
from skillevaluator.validators.claude_plugin_validate import VALIDATOR_NAME as PARITY
from skillevaluator.validators.security import SecurityValidator

if TYPE_CHECKING:
    import pytest

PARTIAL = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}

_MARKDOWN = {"✅ PASSED": "passed", "❌ FAILED": "failed", "⚠️ INCOMPLETE": "incomplete"}
_HTML_HEADER = {"All Passed": "passed", "Issues Found": "failed", "Incomplete": "incomplete"}
_HTML_DASHBOARD = {"Passed": "passed", "Failed": "failed", "Incomplete": "incomplete"}
_FOOTER = {"✓ PASS": "passed", "✗ FAIL": "failed", "! INCOMPLETE": "incomplete"}
# The terminal summary's last line; a per-result block above it can also say "[FAIL] Validation failed".
_SUMMARY = re.compile(
    r"\[PASS\] (?:All|Required) validations passed|\[FAIL\] Validation failed|"
    r"\[INCOMPLETE\] Validation evidence is incomplete"
)


def _summary_status(terminal: str) -> str | None:
    lines = _SUMMARY.findall(terminal)
    if not lines:
        return None
    return {"[PASS]": "passed", "[FAIL]": "failed", "[INCOMPLETE]": "incomplete"}[lines[-1].split(" ", 1)[0]]


def _hookify_plugin(tmp_path: Path) -> Path:
    """The saved hookify run's Tier 1 failure: the skill's name differs from its folder and it names no author."""
    plugin = plugin_dir(tmp_path)
    (plugin / "skills" / "tidy" / "SKILL.md").write_text(
        "---\nname: tidy-csv\ndescription: Tidy CSV files. Use when asked to tidy a CSV.\nversion: 1.0.0\n---\n\n"
        "# Tidy\n\n## Instructions\nTidy it.\n",
        encoding="utf-8",
    )
    return plugin


def _partial_skillspector(*, blocking: bool):
    """SkillSpector returns a partial analysis; with ``blocking`` it also reports a HIGH credential read."""

    def validate_security_only(_self: SecurityValidator, _target: Path) -> ValidationResult:
        result = ValidationResult()
        if blocking:
            result.add_finding(
                Finding(
                    category="SECURITY",
                    severity=Severity.HIGH,
                    check_name="Credential Access (PE3)",
                    message="Privilege Escalation: ~/.ssh/id_rsa",
                    file_path="agents/exfil.md",
                    line_number=9,
                )
            )
        result.add_finding(
            Finding(
                category="SECURITY",
                severity=Severity.LOW,
                check_name="Bundled hooks can execute when matching lifecycle events occur. (BH1)",
                message="Bundled Execution Surface: document:hooks/hooks.json",
                file_path="hooks/hooks.json",
                line_number=1,
            )
        )
        result.add_error("skillspector reported partial analysis; scanner diagnostics were redacted")
        result.mark_scan_incomplete("skillspector")
        return result

    return validate_security_only


def _tier3_that_did_not_complete() -> ValidationResult:
    """The saved hookify run's Tier 3: two trials errored, so the partial plugin run is INCOMPLETE."""
    info: dict[str, Any] = {
        "with_skill": scores(),
        "without_skill": {},
        "execution_status": "failed",
        "rewards": [],
        "num_trials": 2,
        "model": "test-model",
    }
    provenance = {
        "partial": True,
        "execution_incomplete": "Tier 3 plugin evaluation did not complete: Harbor job did not complete "
        "successfully: 2 errored",
    }
    payload = build_agent_eval_payload(
        "gate-demo",
        {"claude-code": info},
        use_llm_judge=False,
        run_config={"eval_target": {"kind": "plugin"}},
        plugin_provenance=provenance,
    )
    result = _validation_result_from_payload(payload)
    assert result is not None
    # The stamp validate puts on a plugin run that did not complete (_incomplete_plugin_agent_eval_result).
    result.passed = False
    result.metadata["execution_status"] = "skipped"
    result.metadata["skip_reason"] = incomplete_reason(provenance)
    return result


class Surfaces:
    """Everything one ``validate`` run wrote, and the headline each surface shows."""

    def __init__(self, exit_code: int, output: str, out: Path) -> None:
        self.exit_code = exit_code
        self.output = output
        self.report = json.loads(_one(out, "skillevaluator-output-*[0-9].json").read_text(encoding="utf-8"))
        self.markdown = _one(out, "skillevaluator-output-*.md").read_text(encoding="utf-8")
        self.html = _one(out, "skillevaluator-output-*.html").read_text(encoding="utf-8")
        self.sarif = json.loads(_one(out, "skillevaluator-output-*.sarif.json").read_text(encoding="utf-8"))
        self.benchmark = (out / "BENCHMARK.md").read_text(encoding="utf-8")

    def headlines(self) -> dict[str, str | None]:
        return {
            "json": self.report.get("overall_status"),
            "markdown": _lookup(_MARKDOWN, r"\*\*Status:\*\* (✅ PASSED|❌ FAILED|⚠️ INCOMPLETE)", self.markdown),
            "html": _lookup(_HTML_HEADER, r'class="header-status-badge [a-z]+"[^>]*>([^<]+)<', self.html),
            "html-dashboard": _lookup(
                _HTML_DASHBOARD,
                r'Overall Status</h3>\s*<p class="dashboard-card-value"><span class="status-badge [a-z]+">([^<]+)<',
                self.html,
            ),
            "sarif": self.sarif["runs"][0].get("properties", {}).get("overallStatus"),
            "cli-summary": _summary_status(self.output),
            "cli-footer": next((status for text, status in _FOOTER.items() if text in self.output), None),
        }

    def benchmark_verdict(self) -> str:
        match = re.search(r"Overall verdict: (PASS|FAIL|INCOMPLETE|NEUTRAL)", self.benchmark)
        assert match is not None
        return match.group(1)

    def html_tier_cards(self) -> dict[str, str]:
        cards = re.findall(
            r'data-target-tab="(tier\d)".*?<span class="tier-card-verdict">([^<]+)</span>', self.html, flags=re.DOTALL
        )
        return dict(cards)


def _one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    assert len(matches) == 1, (pattern, matches)
    return matches[0]


def _lookup(table: dict[str, str], pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return table.get(match.group(1)) if match else None


def validate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tier3: ValidationResult | None,
    *flags: str,
    tiers: str = "1,3",
    checks: str = "schema",
    plugin: Path | None = None,
) -> Surfaces:
    if tier3 is not None:
        monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", lambda *_a, **_k: tier3)
    out = tmp_path / "reports"
    target = plugin or plugin_dir(tmp_path)
    args = ["validate", str(target), "--type", "plugin", "--tiers", tiers, "--no-llm", "--checks", checks]
    run = CliRunner().invoke(cli_module.cli, [*args, "-r", "cli,json,html,markdown,sarif", "-o", str(out), *flags])
    # Rich wraps the footer panel; join it into one line so phrases survive the wrap.
    return Surfaces(run.exit_code, " ".join(run.output.split()), out)


def _assert_one_headline(surfaces: Surfaces, expected: str) -> None:
    assert surfaces.headlines() == dict.fromkeys(surfaces.headlines(), expected)
    # The headline is the exit gate.
    assert surfaces.exit_code == (0 if expected == "passed" else 1)


# ---------------------------------------------------------------------------
# The saved hookify run: Tier 1 failed AND SkillSpector was partial AND Tier 3 did not complete (advisory)
# ---------------------------------------------------------------------------


def test_hookify_shaped_run_says_fail_on_every_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SecurityValidator, "validate_security_only", _partial_skillspector(blocking=True))

    surfaces = validate(
        monkeypatch,
        tmp_path,
        _tier3_that_did_not_complete(),
        checks="schema,security",
        plugin=_hookify_plugin(tmp_path),
    )

    _assert_one_headline(surfaces, "failed")
    assert surfaces.benchmark_verdict() == "FAIL"
    # Every surface still names what did not complete.
    assert "evidence is also incomplete: skillspector did not complete" in surfaces.output
    assert "Evidence is also incomplete: skillspector did not complete." in surfaces.benchmark
    assert surfaces.report["incomplete_scans"] == ["skillspector"]
    # The Tier 1 card reflects the failed validators; the Tier 3 card the partial run.
    assert surfaces.html_tier_cards() == {"tier1": "FAIL", "tier3": "INCOMPLETE"}
    assert surfaces.report["tier_status"] == {"1": "failed", "3": "incomplete"}
    assert surfaces.sarif["runs"][0]["properties"]["tierStatus"] == {"1": "failed", "3": "incomplete"}
    # The security scan found a HIGH issue, so its row is FAIL everywhere, not INCOMPLETE.
    assert re.search(r"Security Scan\s*│\s*FAIL", surfaces.output)
    assert "### ❌ Security Scan" in surfaces.markdown
    security = next(item for item in surfaces.report["results"] if item["validator"] == "Security Scan")
    assert security["status"] == "failed"


def test_missing_evidence_alone_says_incomplete_on_every_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SecurityValidator, "validate_security_only", _partial_skillspector(blocking=False))

    surfaces = validate(monkeypatch, tmp_path, None, tiers="1", checks="schema,security")

    _assert_one_headline(surfaces, "incomplete")
    assert surfaces.benchmark_verdict() == "INCOMPLETE"
    assert surfaces.html_tier_cards() == {"tier1": "INCOMPLETE"}
    assert "### ⚠️ INCOMPLETE Security Scan" in surfaces.markdown


def test_a_clean_run_passes_on_every_gate_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    surfaces = validate(monkeypatch, tmp_path, None, tiers="1")

    _assert_one_headline(surfaces, "passed")
    assert surfaces.html_tier_cards() == {"tier1": "PASS"}
    # BENCHMARK.md requires Tier 3 evidence for publication by default (reports.mdx), so it stays stricter.
    assert surfaces.benchmark_verdict() == "INCOMPLETE"


# ---------------------------------------------------------------------------
# Tier 3 inside and outside the exit gate
# ---------------------------------------------------------------------------


def test_a_gated_partial_run_says_incomplete_on_every_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    surfaces = validate(
        monkeypatch, tmp_path, tier3_result(scores(), plugin_provenance=PARTIAL), "--block-on-agent-eval"
    )

    _assert_one_headline(surfaces, "incomplete")
    assert surfaces.benchmark_verdict() == "INCOMPLETE"
    assert surfaces.html_tier_cards() == {"tier1": "PASS", "tier3": "INCOMPLETE"}


def test_a_gated_run_that_never_ran_says_fail_on_every_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    skipped = advisory_skip_result("Tier 3 live evaluation skipped: docker missing", skill_name="gate-demo")

    surfaces = validate(monkeypatch, tmp_path, skipped, "--block-on-agent-eval")

    _assert_one_headline(surfaces, "failed")
    assert surfaces.benchmark_verdict() == "FAIL"
    assert surfaces.html_tier_cards() == {"tier1": "PASS", "tier3": "FAIL"}


def test_an_advisory_tier3_fail_never_changes_the_headline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    surfaces = validate(monkeypatch, tmp_path, tier3_result(scores(security=0.0)))

    _assert_one_headline(surfaces, "passed")
    assert "Tier 3 verdict FAIL is advisory" in surfaces.output
    assert surfaces.html_tier_cards() == {"tier1": "PASS", "tier3": "FAIL"}
    # The publication card still says FAIL, and says the exit code was not changed (plugin-evaluation.mdx).
    assert "FAIL — Not recommended for publication (Tier 3 was advisory in this run)" in surfaces.benchmark


def test_an_advisory_partial_run_is_named_but_never_changes_the_headline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    surfaces = validate(monkeypatch, tmp_path, tier3_result(scores(), plugin_provenance=PARTIAL))

    _assert_one_headline(surfaces, "passed")
    assert "Tier 3 INCOMPLETE is advisory" in surfaces.output
    assert surfaces.html_tier_cards() == {"tier1": "PASS", "tier3": "INCOMPLETE"}
    assert surfaces.benchmark_verdict() == "INCOMPLETE"
    # The Tier 1 plugin row stays fail-closed on the missing Tier 3 evidence.
    assert "| ⚠️ INCOMPLETE |" in surfaces.markdown


# ---------------------------------------------------------------------------
# A skipped check is not a pass
# ---------------------------------------------------------------------------


class _MissingClaude:
    is_available = False

    @staticmethod
    def get_install_hint() -> str:
        return "Install Claude Code"


def test_a_skipped_check_is_never_shown_as_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Tools, "claude", _MissingClaude())

    surfaces = validate(monkeypatch, tmp_path, None, tiers="1", checks="schema,claude-validate")

    parity = next(item for item in surfaces.report["results"] if item["validator"] == PARITY)
    assert parity["status"] == "skipped"
    assert re.search(rf"{re.escape(PARITY)}\s*│\s*SKIP\s*│", surfaces.output)
    assert f"### ⏭️ SKIPPED {PARITY}" in surfaces.markdown
    badge = rf'class="name-col">{re.escape(PARITY)}</td>\s*<td><span class="status-badge skipped">Skipped<'
    assert re.search(badge, surfaces.html)
    # Skipping a check does not fail the exit gate.
    _assert_one_headline(surfaces, "passed")
