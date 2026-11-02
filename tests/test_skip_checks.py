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


def test_skipped_check_is_visible_in_tier1_summary():
    results = run_validation(SKILL, checks="unicode,schema", skip="unicode")
    _ok, rows = summarize_tier1(results, lineup=["schema", "unicode"])
    text = "\n".join("".join(seg for seg, _style in row.segments) for row in rows)
    assert "\N{EN DASH} unicode" in text
    assert "1/1 passed  ·  1 skipped" in text
    assert "unicode  with --skip, not checked" in text


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
