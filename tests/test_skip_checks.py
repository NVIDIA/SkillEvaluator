"""``--skip``: user-skipped Tier 1 checks do not run, stay visible, and never certify Tier 1."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.reporting.base import is_cleanly_skipped, missing_tier1_publication_checks
from skillevaluator.reporting.console_ui import summarize_tier1
from skillevaluator.tier1.commands import run_validation, runs_secrets_scan

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

SKILL = Path(__file__).resolve().parents[1] / "src/skillevaluator/tier3/reference_skills/text-analyzer"


def _by_name(results: list[ValidationResult]) -> dict[str, ValidationResult]:
    return {r.validator_name: r for r in results}


def test_skip_secrets_does_not_run_gitleaks_and_blocks_certification():
    with patch("skillevaluator.validators.secrets.SecretsValidator.validate") as validate:
        results = run_validation(SKILL, checks="code-integrity", skip="secrets")
    validate.assert_not_called()
    secrets = _by_name(results)["Secrets Detection"]
    assert is_cleanly_skipped(secrets)
    assert secrets.metadata["skip_reason"] == "skipped with --skip"
    assert not _by_name(results)["Code Risk Analysis"].metadata.get("skipped")
    assert "code-integrity" in missing_tier1_publication_checks(results)


def test_skip_whole_check_records_skipped_result():
    results = run_validation(SKILL, checks="unicode,schema", skip="unicode")
    unicode = _by_name(results)["Unicode Smuggling Detection"]
    assert is_cleanly_skipped(unicode)
    assert "unicode" in missing_tier1_publication_checks(results)


def test_skip_harness_records_skipped_result():
    with patch("skillevaluator.validators.harness.HarnessValidator.validate") as validate:
        results = run_validation(SKILL, checks="schema,harness", skip="harness")
    validate.assert_not_called()
    assert is_cleanly_skipped(_by_name(results)["Harness Compatibility"])


def test_skipped_check_is_visible_in_tier1_summary():
    results = run_validation(SKILL, checks="unicode,schema", skip="unicode")
    _ok, rows = summarize_tier1(results, lineup=["schema", "unicode"])
    text = "\n".join("".join(seg for seg, _style in row.segments) for row in rows)
    assert "\N{EN DASH} unicode" in text
    assert "1/1 checks passed" in text
    assert "  ·  1 skipped" in text
    assert "unicode  with --skip, not checked" in text


def test_skip_secrets_inside_code_integrity_counts_as_skipped():
    from skillevaluator.models.result import ValidationResult

    results = [
        ValidationResult(validator_name="Code Risk Analysis"),
        ValidationResult(
            validator_name="Secrets Detection",
            metadata={"skipped": True, "skipped_by_user": True, "skip_reason": "skipped with --skip"},
        ),
        ValidationResult(validator_name="Code Integrity & Hygiene"),
    ]
    _ok, rows = summarize_tier1(results, lineup=["code-integrity"])
    text = "\n".join("".join(seg for seg, _style in row.segments) for row in rows)
    assert "2/2 checks passed  ·  1 skipped" in text
    assert "secrets  with --skip, not checked" in text


def test_runs_secrets_scan():
    assert runs_secrets_scan(None, None)
    assert not runs_secrets_scan(None, "secrets")
    assert not runs_secrets_scan(None, "code")
    assert not runs_secrets_scan("schema,security", None)


def test_unknown_skip_is_a_usage_error():
    result = CliRunner().invoke(cli, ["validate", str(SKILL), "--tiers", "1", "--skip", "nope"])
    assert result.exit_code == 2
    assert "unknown check nope" in result.output


def test_skip_skips_gitleaks_setup():
    with patch("skillevaluator.utils.scanner_install.ensure_gitleaks") as ensure:
        from skillevaluator.cli import _offer_scanner_install

        _offer_scanner_install(None, "secrets")
    ensure.assert_not_called()


def test_skip_security_skips_skillspector_setup():
    with patch("skillevaluator.utils.scanner_install.prepare_skillspector") as prepare:
        from skillevaluator.cli import _offer_scanner_install

        _offer_scanner_install(None, "security,secrets")
    prepare.assert_not_called()


def test_tier1_summary_names_scanners_and_versions():
    from skillevaluator.models.result import ValidationResult

    security = ValidationResult(validator_name="Security Scan")
    security.record_scanner("skillspector", "2.13.3")
    code_risk = ValidationResult(validator_name="Code Risk Analysis")
    code_risk.record_scanner("bandit", "1.9.4")
    other = ValidationResult(validator_name="Code Risk Analysis")
    other.record_scanner("semgrep", None)
    code_risk.merge(other)
    secrets = ValidationResult(validator_name="Secrets Detection", passed=False)
    secrets.mark_scan_incomplete("gitleaks")
    _ok, rows = summarize_tier1([security, code_risk, secrets])
    row = next(row for row in rows if row.label == "scanners")
    assert "".join(seg for seg, _style in row.segments) == (
        "SkillSpector 2.13.3 · Bandit 1.9.4 · Semgrep unknown · Gitleaks incomplete"
    )
    assert code_risk.to_dict()["scanners"] == {"bandit": "1.9.4", "semgrep": "unknown"}


def test_skipped_scanner_is_shown_as_skipped():
    results = run_validation(SKILL, checks="code-integrity", skip="secrets")
    _ok, rows = summarize_tier1(results)
    row = next(row for row in rows if row.label == "scanners")
    assert "Gitleaks skipped" in "".join(seg for seg, _style in row.segments)
