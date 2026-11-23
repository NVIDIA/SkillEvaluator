# SPDX-License-Identifier: Apache-2.0
"""Incomplete scanner evidence must not be diagnosed as a missing installation.

The Tier 1 panel also lists checks with not-applicable evidence apart from the checks that ran.
"""

import io

import pytest
from rich.console import Console

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting.console_ui import ValidateView, summarize_tier1


def _text(rows):
    return "\n".join("".join(chunk for chunk, _ in row.segments) for row in rows)


def _headline(tmp_path, results):
    """The quiet-mode verdict text for blocking *results*."""
    from skillevaluator.cli import _finish_pipeline_view

    stream = io.StringIO()
    view = ValidateView("demo", [(1, "Static", "static")], console=Console(file=stream, width=180, color_system=None))
    _finish_pipeline_view(
        view,
        tier_gate_results=results,
        tier3_result=None,
        gate_failed=True,
        output_dir=tmp_path,
        basename="partial",
        report_formats=(),
        target_path=tmp_path,
        agent_eval=False,
    )
    return stream.getvalue()


@pytest.mark.parametrize(
    ("diagnostic", "gate", "runtime_version"),
    [
        (
            "skillspector JSON field 'analysis_completeness.analyzer_statuses' has a status that contradicts "
            "its work accounting; security scan did not complete",
            False,
            None,
        ),
        (
            "skillspector reports incomplete analysis (status 'partial'): SKILL.md:4 "
            "[reference_resolution; reference_unresolved]; and 11 more. Missing references require "
            "a standalone target with complete, matching reference and scan evidence.",
            True,
            "2.11.2",
        ),
    ],
)
def test_partial_analysis_keeps_its_diagnostic_and_version(diagnostic, gate, runtime_version):
    # An incomplete marker is not evidence that the installed scanner never ran.
    result = ValidationResult(validator_name="Security Scan")
    result.record_scanner("skillspector", None if runtime_version else "2.11.2")
    if runtime_version:
        result.metadata["skillspector_version"] = runtime_version
    result.mark_scan_incomplete("skillspector")
    if gate:
        result.fail_gate(diagnostic)
    else:
        result.add_error(diagnostic)
    result.add_finding(Finding("SECURITY", Severity.MEDIUM, "rug-pull", "Advisory issue", "SKILL.md"))

    passed, rows = summarize_tier1([result], lineup=["security"])
    text = _text(rows)

    assert not passed
    assert diagnostic in text
    assert "SkillSpector 2.11.2, scan incomplete" in text
    assert "0/1 checks passed  ·  1 incomplete" in text
    assert "security incomplete:" in text
    assert "rerun" in text
    assert "not installed" not in text
    assert "didn't run" not in text
    assert "--skip" not in text


@pytest.mark.parametrize("diagnostic", [None, "skillspector timed out after 300 seconds; partial results only"])
def test_unrecorded_scanner_version_does_not_imply_an_installation_failure(diagnostic):
    result = ValidationResult(validator_name="Security Scan")
    result.mark_scan_incomplete("skillspector")
    if diagnostic:
        result.add_error(diagnostic)

    _passed, rows = summarize_tier1([result])
    text = _text(rows)

    assert "SkillSpector incomplete" in text
    assert (diagnostic or "skillspector did not produce complete evidence") in text
    assert "not installed" not in text
    assert "didn't run" not in text
    assert "--skip" not in text


def test_partial_analysis_and_missing_scanner_keep_their_separate_recoveries():
    security = ValidationResult(validator_name="Security Scan")
    security.record_scanner("skillspector", "2.11.2")
    security.mark_scan_incomplete("skillspector")
    security.fail_gate("skillspector reports incomplete analysis: reference_unresolved at SKILL.md:4")
    secrets = ValidationResult(validator_name="Secrets Detection")
    secrets.mark_scan_incomplete("gitleaks")
    secrets.add_warning("couldn't download Gitleaks 8.30.0 (connection refused). Install with: brew install gitleaks")

    _passed, rows = summarize_tier1([security, secrets])
    text = _text(rows)

    assert "security incomplete: skillspector reports incomplete analysis: reference_unresolved at SKILL.md:4" in text
    assert "secrets didn't run: couldn't download Gitleaks 8.30.0 (connection refused)" in text
    assert "check network access to github.com, then rerun" in text
    assert "brew install gitleaks" in text
    assert "--skip security" not in text


@pytest.mark.parametrize("merged", [False, True])
def test_partial_and_missing_scanners_in_the_same_check_keep_separate_recoveries(tmp_path, merged):
    result = ValidationResult(validator_name="Code Risk Analysis", metadata={"gating": {"tier": 1, "blocking": True}})
    evidence = ValidationResult() if merged else result
    evidence.record_scanner("bandit", "1.9.4")
    evidence.mark_scan_incomplete("bandit")
    evidence.add_error("Bandit scan error: syntax error while parsing")
    evidence.mark_scan_incomplete("semgrep")
    evidence.add_warning("semgrep not installed. Install with: uv tool install semgrep")
    if merged:
        result.merge_with_prefix(evidence, "demo")

    _passed, rows = summarize_tier1([result], lineup=["code-integrity"])
    text = _text(rows)

    assert "Bandit 1.9.4, scan incomplete · Semgrep didn't run" in text
    assert "code incomplete:" in text
    assert "Bandit scan error: syntax error while parsing" in text
    assert "code incomplete: semgrep not installed" in text
    assert "bandit, semgrep not installed" not in text
    assert "code didn't run" not in text
    assert "uv tool install semgrep" in text
    assert "Code Risk Analysis was incomplete in Tier 1" in _headline(tmp_path, [result])


def test_setup_item_names_each_check_with_its_own_state():
    security = _missing_skillspector()
    code = ValidationResult(validator_name="Code Risk Analysis")
    code.record_scanner("semgrep", "1.150.0")
    code.mark_scan_incomplete("bandit")
    code.add_warning("bandit not installed. Install with: pipx install bandit")

    _passed, rows = summarize_tier1([security, code], lineup=["security", "code-integrity"])
    text = _text(rows)

    assert "security didn't run, code incomplete: skillspector, bandit not installed" in text
    assert "or skip it: --skip security,code-integrity" in text
    assert "code incomplete: Bandit" not in text


def test_missing_dependency_auditor_after_a_file_label_reads_as_setup(tmp_path, monkeypatch):
    from skillevaluator.utils.tool_runner import Tools
    from skillevaluator.validators.dependencies import DependencySecurityValidator

    monkeypatch.setattr(type(Tools.pip_audit), "is_available", property(lambda _self: False))
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\ndependencies = ["idna==3.10"]\n')
    result = DependencySecurityValidator(use_safety=False).validate(tmp_path)
    result.validator_name = "Dependency Vulnerability Audit"
    result.metadata["gating"] = {"tier": 1, "blocking": True}
    assert any(str(w).startswith("pyproject.toml: pip-audit not installed") for w in result.independent_warnings())

    _passed, rows = summarize_tier1([result], lineup=["dependency"])
    text = _text(rows)

    assert "dependency vulnerability audit didn't run: pip-audit not installed" in text
    assert "or skip it: --skip dependency" in text
    assert "incomplete: pyproject.toml" not in text
    assert "Dependency Vulnerability Audit didn't run in Tier 1" in _headline(tmp_path, [result])


@pytest.mark.parametrize(
    ("last_label", "diagnostic", "scanner_row"),
    [
        ("", "semgrep not installed.", "Semgrep didn't run"),
        ("", "pyproject.toml: semgrep not installed.", "Semgrep didn't run"),
        ("", "Semgrep scan: semgrep not installed.", "Semgrep incomplete"),
        ("", "Semgrep scan error: dependency not installed.", "Semgrep incomplete"),
        ("[invalid\nlabel] ", "semgrep not installed.", "Semgrep incomplete"),
    ],
)
def test_scanner_setup_diagnostics_after_repeated_labels(last_label, diagnostic, scanner_row):
    result = ValidationResult(validator_name="Code Risk Analysis")
    result.mark_scan_incomplete("semgrep")
    result.add_warning("[collection] [member[1]] " * 4096 + last_label + diagnostic)

    _passed, rows = summarize_tier1([result])

    assert _text([row for row in rows if row.label == "scanners"]) == scanner_row


def test_an_advisory_about_a_missing_package_cannot_become_a_scanner_installation_error():
    result = ValidationResult(validator_name="Security Scan")
    result.record_scanner("skillspector", "2.11.2")
    result.mark_scan_incomplete("skillspector")
    result.add_error("skillspector reports partial analysis: reference_unresolved")
    result.add_finding(Finding("SECURITY", Severity.MEDIUM, "dependency", "Optional client not installed", "run.py"))

    _passed, rows = summarize_tier1([result])
    text = _text(rows)

    assert "security incomplete: skillspector reports partial analysis: reference_unresolved" in text
    assert "skillspector not installed" not in text
    assert "--skip security" not in text


@pytest.mark.parametrize("reverse", [False, True])
def test_completed_member_cannot_hide_an_incomplete_scanner_run(reverse):
    partial = ValidationResult(validator_name="Security Scan")
    partial.record_scanner("skillspector", None)
    partial.mark_scan_incomplete("skillspector")
    partial.add_error("skillspector returned malformed analysis accounting")
    complete = ValidationResult(validator_name="Security Scan")
    complete.record_scanner("skillspector", "2.11.2")
    results = [partial, complete]

    _passed, rows = summarize_tier1(results[::-1] if reverse else results)
    scanner = next(row for row in rows if row.label == "scanners")

    assert _text([scanner]) == "SkillSpector 2.11.2, scan incomplete"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("blocking", [False, True])
def test_final_verdict_distinguishes_incomplete_scans_from_known_failures(tmp_path, count, blocking):
    results = []
    for _ in range(count):
        result = ValidationResult(validator_name="Security Scan", metadata={"gating": {"tier": 1, "blocking": True}})
        result.record_scanner("skillspector", "2.11.2")
        result.mark_scan_incomplete("skillspector")
        result.fail_gate("skillspector reports incomplete analysis: reference_unresolved")
        results.append(result)
    if blocking:
        results[0].add_finding(Finding("SECURITY", Severity.HIGH, "known-risk", "Known risk", "run.py"))

    text = _headline(tmp_path, results)

    if count == 1:
        assert f"Security Scan {'failed' if blocking else 'was incomplete'} in Tier 1" in text
    else:
        assert ("Tier 1: 1 failed, 1 incomplete" if blocking else "Tier 1: 2 incomplete") in text
    assert "didn't run" not in text


def _missing_skillspector():
    result = ValidationResult(validator_name="Security Scan", metadata={"gating": {"tier": 1, "blocking": True}})
    result.add_warning("skillspector not installed - skipping security scan. Install with: pipx install skillspector")
    result.mark_scan_incomplete("skillspector")
    return result


def _agent_eval(verdict, *, run_budget=False):
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        passed=False,
        metadata={"gating": {"tier": 3, "blocking": True}, "agent_eval": {"verdict": verdict}},
    )
    if run_budget:
        result.mark_scan_incomplete("Tier 3 run budget")
    return result


def _context_dedup():
    result = ValidationResult(validator_name="Context Deduplication", metadata={"gating": {"tier": 2}})
    result.mark_scan_incomplete("embedding-provider")
    return result


@pytest.mark.parametrize(
    ("make", "count", "headline"),
    [
        (_missing_skillspector, 1, "Security Scan didn't run in Tier 1"),
        (_missing_skillspector, 2, "Tier 1: 2 incomplete"),
        (lambda: _agent_eval("inconclusive"), 1, "AGENT_EVAL was incomplete in Tier 3"),
        (lambda: _agent_eval("inconclusive", run_budget=True), 1, "AGENT_EVAL was incomplete in Tier 3"),
        (lambda: _agent_eval("fail", run_budget=True), 1, "AGENT_EVAL failed in Tier 3"),
        (_context_dedup, 1, "Context Deduplication was incomplete in Tier 2"),
        (_context_dedup, 2, "Tier 2: 2 incomplete"),
    ],
    ids=[
        "missing-install",
        "two-missing-installs",
        "inconclusive-agent-eval",
        "run-budget-inconclusive",
        "run-budget-fail",
        "tier2-provider-error",
        "two-tier2-provider-errors",
    ],
)
def test_final_verdict_keeps_setup_issues_apart_from_incomplete_and_failed_checks(tmp_path, make, count, headline):
    text = _headline(tmp_path, [make() for _ in range(count)])

    assert headline in text
    for other in ("didn't run", "was incomplete", "failed in"):
        if other not in headline:
            assert other not in text


def test_incomplete_fix_quotes_only_the_first_bounded_diagnostics():
    # The dependency audit reports one warning per loosely pinned requirement, so there can be thousands.
    result = ValidationResult(validator_name="Dependency Vulnerability Audit")
    result.mark_scan_incomplete("pip-audit")
    result.add_warning("requirements.txt: " + "a" * 2000)
    for line in range(2, 201):
        result.add_warning(f"requirements.txt: line {line} requires an exact package version")

    _passed, rows = summarize_tier1([result])
    item = next(row for row in rows if row.label == "scan")
    problem = _text([item])

    assert problem.startswith("dependency vulnerability audit incomplete: requirements.txt: aaa")
    assert "a" * 300 not in problem
    assert "…; requirements.txt: line 2 requires" in problem
    assert "line 3 requires an exact package version; and 197 more" in problem
    assert "line 4 requires" not in problem
    assert len(problem) < 700
    assert "see detailed reports for full diagnostics" in _text(rows)


def test_incomplete_fix_counts_diagnostics_with_a_common_truncated_prefix():
    result = ValidationResult(validator_name="Dependency Vulnerability Audit")
    result.mark_scan_incomplete("pip-audit")
    prefix = "pyproject.toml: " + "a" * 2000
    for declaration in (1, 2, 2, 3, 4):
        result.add_warning(f"{prefix}: declaration {declaration} requires an exact pin")

    _passed, rows = summarize_tier1([result])
    problem = _text([row for row in rows if row.label == "scan"])

    assert problem.count("pyproject.toml:") == 3
    assert problem.endswith("; and 1 more")


def _not_applicable(validator_name, check, message):
    result = ValidationResult(validator_name=validator_name)
    result.add_success(check, message)
    return result


@pytest.mark.parametrize("hygiene_ran", [False, True])
def test_not_applicable_row_never_names_a_check_the_ticker_shows_as_run(hygiene_ran):
    code_risk = _not_applicable(
        "Code Risk Analysis", "code_file_discovery", "No code files found - code risk analysis is not applicable"
    )
    lint = _not_applicable("SCRIPT_LINT", "lint", "No scripts/ or tools/ directory found")
    schema = ValidationResult(validator_name="Schema & Repository Governance")
    schema.add_success("frontmatter", "Frontmatter is valid")
    results = [code_risk, lint, schema]
    if hygiene_ran:
        hygiene = ValidationResult(validator_name="Code Integrity & Hygiene")
        hygiene.add_success("hygiene", "No hygiene issues found")
        results.append(hygiene)

    _passed, rows = summarize_tier1(results, lineup=["code-integrity", "lint", "schema"])
    labels = [row.label for row in rows]
    texts = [_text([row]) for row in rows]
    not_applicable = labels.index("n/a")

    assert texts[not_applicable] == ("code, lint" if hygiene_ran else "code-integrity, lint")
    assert texts[not_applicable + 1] == "No code files found - code risk analysis is not applicable"
    ticker = texts[labels.index("checks")]
    assert ticker == ("✓ code-integrity" if hygiene_ran else "code-integrity") + " · lint · ✓ schema"
    assert f"{2 if hygiene_ran else 1}/{2 if hygiene_ran else 1} checks passed" in texts[labels.index("summary")]
