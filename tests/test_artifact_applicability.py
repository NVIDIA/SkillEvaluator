# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 on a target with no supported artifact (``--type auto`` detects ``unknown``).

The skill-manifest checks (schema, version, quality) are recorded as not
applicable instead of failing with "No skills found"; the content scans still
run with their unrestricted scope, and such a run never certifies Tier 1. A
folder that holds an artifact manifest entry (``SKILL.md``,
``workflow-rules.mdc``, ``agent_plugin.yaml``/``.yml`` or
``.claude-plugin/plugin.json``) in any letter case, or a dangling symlink by
one of those names, is a broken artifact and still fails the manifest checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.cli_core import detect_content_type
from skillevaluator.constants import CONTENT_TYPE_UNKNOWN
from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting import HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.reporting.base import assess_publication, is_cleanly_skipped, missing_tier1_publication_checks
from skillevaluator.reporting.console_ui import summarize_tier1
from skillevaluator.reporting.presentation import build_final_render_context
from skillevaluator.tier1 import commands as tier1_commands
from skillevaluator.tier1.commands import run_validation

MANIFEST_VALIDATORS = {
    "schema": "Schema & Repository Governance",
    "version": "Semantic Version Validation",
    "quality": "Quality Score (4-Dimension Analysis)",
}
APPLICABILITY = {
    "applicability": "not_applicable",
    "reason_code": "unsupported_content_type",
    "detected_content_type": "unknown",
    "supported_content_types": ["skill", "rules", "workflows", "plugin"],
}


def _ordinary_repository(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Example service\n", encoding="utf-8")
    (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    workflow = root / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    assert detect_content_type(root) == CONTENT_TYPE_UNKNOWN
    return root


def _by_name(results: list[ValidationResult]) -> dict[str, ValidationResult]:
    return {result.validator_name: result for result in results}


def _rendered(rows) -> str:
    return "\n".join("".join(text for text, _style in row.segments) for row in rows)


def _row_text(row) -> str:
    return "".join(text for text, _style in row.segments)


def _broken_skill(root: Path) -> Path:
    """A folder that is meant to be an artifact but whose manifest the checks cannot read."""
    root.mkdir(parents=True)
    (root / "README.md").write_text("# Example tool\n", encoding="utf-8")
    return root


def _validate(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli, ["validate", *args, "--no-llm"])
    return result.exit_code, result.output


def _json_report(output_dir: Path) -> dict:
    return json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))


# The code-integrity validators run Bandit and Gitleaks, which a unit test
# cannot rely on: CI installs no Gitleaks, and the managed install would
# download it. Tests that select code-integrity replace them with this stub.
EXTERNAL_SCANNER_VALIDATORS = ("CodeRiskValidator", "SecretsValidator", "HygieneValidator")


class _PassingValidator:
    name = "Built-in validator"
    description = "Built-in validation"

    def __init__(self, **_kwargs: object) -> None:
        pass

    @staticmethod
    def validate(_target: Path) -> ValidationResult:
        result = ValidationResult(validator_name="Built-in validator")
        result.add_success("built_in", "Built-in validation completed")
        return result

    validate_security_only = validate
    validate_pii_only = validate


def _stub_validators(monkeypatch: pytest.MonkeyPatch, *validator_names: str) -> None:
    for validator_name in validator_names:
        monkeypatch.setattr(tier1_commands, validator_name, _PassingValidator)


def test_unknown_content_records_manifest_checks_as_not_applicable(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    ran: list[str] = []

    results = run_validation(
        repository,
        checks="schema,version,quality,pii,unicode,lint",
        content_type=CONTENT_TYPE_UNKNOWN,
        on_check=ran.append,
    )

    assert ran == ["pii", "unicode", "lint"]
    by_name = _by_name(results)
    for check_id, name in MANIFEST_VALIDATORS.items():
        result = by_name[name]
        assert is_cleanly_skipped(result)
        assert "not applicable" in result.metadata["skip_reason"]
        assert "--type" in result.metadata["skip_reason"]
        assert result.metadata["tier1_applicability"] == APPLICABILITY
        assert result.metadata["publication_evidence"]["check_id"] == check_id
        assert not result.metadata.get("skipped_by_user")
    for name in ("PII Scan", "Unicode Smuggling Detection"):
        assert by_name[name].passed
        assert not by_name[name].metadata.get("skipped")
    assert not any("No skills found" in error for result in results for error in result.errors)


def test_unknown_content_keeps_the_unrestricted_security_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    scopes: list[str | None] = []

    def fake_security_scan(self, _target: Path) -> ValidationResult:
        scopes.append(self.content_type)
        result = ValidationResult(validator_name="Security Scan")
        result.add_success("skillspector", "No security issues found")
        return result

    monkeypatch.setattr(tier1_commands.SecurityValidator, "validate_security_only", fake_security_scan)

    results = run_validation(repository, checks="schema,security", content_type=CONTENT_TYPE_UNKNOWN)

    assert scopes == [None]
    by_name = _by_name(results)
    assert by_name["Security Scan"].passed
    assert not by_name["Security Scan"].metadata.get("skipped")
    assert is_cleanly_skipped(by_name["Schema & Repository Governance"])


def test_unknown_content_still_fails_an_unknown_check_name(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")

    results = run_validation(repository, checks="schema,typo", content_type=CONTENT_TYPE_UNKNOWN)

    assert not results[0].passed
    assert "Unknown Tier 1 check(s): typo" in results[0].errors
    assert is_cleanly_skipped(_by_name(results)["Schema & Repository Governance"])


def test_no_content_type_keeps_the_historical_skill_validation(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")

    results = run_validation(repository, checks="schema", content_type=None)

    schema = _by_name(results)["Schema & Repository Governance"]
    assert not schema.passed
    assert any("No skills found" in error for error in schema.errors)


def test_unknown_content_with_a_mis_cased_manifest_still_fails_the_manifest_checks(tmp_path: Path) -> None:
    # Case-sensitive filesystems detect this folder as unknown; spec-compliant
    # agents do not load a Skill.md, so it must fail like any broken skill.
    folder = _broken_skill(tmp_path / "my-tool")
    (folder / "Skill.md").write_text(
        "---\nname: my-tool\ndescription: A tool whose manifest name has the wrong letter case.\n---\n\n# My tool\n",
        encoding="utf-8",
    )

    results = run_validation(
        folder, checks="schema,quality,pii", content_type=CONTENT_TYPE_UNKNOWN, on_check=lambda _name: None
    )

    by_check = {result.metadata["publication_evidence"]["check_id"]: result for result in results}
    assert sorted(by_check) == ["pii", "quality", "schema"]
    for check_id in ("schema", "quality"):
        assert not by_check[check_id].metadata.get("skipped")
        assert not by_check[check_id].passed
    assert not any("tier1_applicability" in result.metadata for result in results)
    data = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    assert "tier1_applicability" not in data
    assert data["overall_status"] == "failed"


def _dangling_symlink(link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to("missing-target")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _assert_manifest_check_failed(results: list[ValidationResult]) -> None:
    schema = _by_name(results)["Schema & Repository Governance"]
    assert not schema.metadata.get("skipped")
    assert not schema.passed
    assert not any("tier1_applicability" in result.metadata for result in results)


# Every manifest auto-detection looks for. A dangling symlink by one of these
# names is not detected, so the target is ``unknown``, but it is a broken
# artifact rather than a target without one.
DANGLING_MANIFESTS = [
    "SKILL.md",
    "workflow-rules.mdc",
    "agent_plugin.yaml",
    "agent_plugin.yml",
    ".claude-plugin/plugin.json",
    ".claude-plugin",
]


@pytest.mark.parametrize("manifest", DANGLING_MANIFESTS)
def test_unknown_content_with_a_dangling_manifest_symlink_still_fails_schema(tmp_path: Path, manifest: str) -> None:
    folder = _broken_skill(tmp_path / "my-tool")
    _dangling_symlink(folder / manifest)

    results = run_validation(
        folder, checks="schema,quality,pii", content_type=CONTENT_TYPE_UNKNOWN, on_check=lambda _name: None
    )

    _assert_manifest_check_failed(results)


@pytest.mark.parametrize("manifest", ["Workflow-Rules.mdc", "Agent_Plugin.YAML", ".Claude-Plugin/Plugin.JSON"])
def test_unknown_content_with_a_mis_cased_artifact_manifest_still_fails_schema(tmp_path: Path, manifest: str) -> None:
    # Auto-detection matches these names exactly, so case-sensitive
    # filesystems report such a folder as unknown.
    folder = _broken_skill(tmp_path / "my-tool")
    path = folder / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name: my-tool\n", encoding="utf-8")

    results = run_validation(
        folder, checks="schema,quality,pii", content_type=CONTENT_TYPE_UNKNOWN, on_check=lambda _name: None
    )

    _assert_manifest_check_failed(results)


def test_unknown_content_with_only_a_marketplace_manifest_is_not_applicable(tmp_path: Path) -> None:
    # A marketplace repository lists plugins; its .claude-plugin folder holds
    # no plugin.json, so it is not a broken plugin.
    repository = _ordinary_repository(tmp_path / "repository")
    marketplace = repository / ".claude-plugin" / "marketplace.json"
    marketplace.parent.mkdir()
    marketplace.write_text('{"name": "example", "plugins": []}\n', encoding="utf-8")

    results = run_validation(
        repository, checks="schema,pii", content_type=CONTENT_TYPE_UNKNOWN, on_check=lambda _name: None
    )

    assert is_cleanly_skipped(_by_name(results)["Schema & Repository Governance"])
    assert all(result.metadata["tier1_applicability"] == APPLICABILITY for result in results)


@pytest.mark.parametrize("manifest", ["SKILL.md", ".claude-plugin/plugin.json"])
def test_validate_folder_with_a_dangling_manifest_symlink_still_fails(tmp_path: Path, manifest: str) -> None:
    folder = _broken_skill(tmp_path / "my-tool")
    _dangling_symlink(folder / manifest)
    reports = tmp_path / "reports"

    exit_code, output = _validate(
        str(folder), "--tiers", "1", "--checks", "schema,quality,pii", "-r", "json", "--output-dir", str(reports)
    )

    assert exit_code == 1, output
    payload = _json_report(reports)
    assert payload["overall_status"] == "failed"
    assert "tier1_applicability" not in payload
    # A plugin manifest, even a dangling link to one, routes the folder to the plugin checks.
    schema_check = (
        "Plugin Schema & Component Checks"
        if manifest == ".claude-plugin/plugin.json"
        else "Schema & Repository Governance"
    )
    schema = next(entry for entry in payload["results"] if entry["validator"] == schema_check)
    assert schema["status"] == "failed"


@pytest.mark.parametrize(
    ("checks", "skip"),
    [("pii", None), ("security,pii,code-integrity", None), ("schema,version,quality,pii", "schema,version,quality")],
)
def test_unknown_content_records_applicability_for_any_check_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checks: str, skip: str | None
) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    # Only the PII scan runs for real: the security and code-integrity scans
    # need SkillSpector, Bandit and Gitleaks, which CI does not provide.
    _stub_validators(monkeypatch, "SecurityValidator", *EXTERNAL_SCANNER_VALIDATORS)

    results = run_validation(
        repository, checks=checks, skip=skip, content_type=CONTENT_TYPE_UNKNOWN, on_check=lambda _name: None
    )

    assert all(result.metadata["tier1_applicability"] == APPLICABILITY for result in results)
    data = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    assert data["tier1_applicability"] == APPLICABILITY
    assert data["overall_status"] == "passed"
    assert data["publication"]["eligible"] is False


def test_unknown_content_records_applicability_when_fail_fast_stops_early(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    (repository / "README.md").write_text("# Example service\n\nBilling: 4532-0151-1283-0366\n", encoding="utf-8")
    ran: list[str] = []

    results = run_validation(
        repository,
        checks="schema,pii,unicode",
        content_type=CONTENT_TYPE_UNKNOWN,
        fail_fast=True,
        on_check=ran.append,
    )

    assert ran == ["pii"]
    assert not _by_name(results)["PII Scan"].passed
    assert all(result.metadata["tier1_applicability"] == APPLICABILITY for result in results)


def test_validate_ordinary_repository_narrowed_to_content_scans_records_applicability(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    reports = tmp_path / "reports"

    exit_code, output = _validate(
        str(repository), "--tiers", "1", "--checks", "pii", "-r", "json", "--output-dir", str(reports)
    )

    assert exit_code == 0, output
    payload = _json_report(reports)
    assert payload["tier1_applicability"] == APPLICABILITY
    assert [entry["validator"] for entry in payload["results"]] == ["PII Scan"]


def test_not_applicable_results_never_certify_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _ordinary_repository(tmp_path / "repository")

    _stub_validators(
        monkeypatch,
        "SecurityValidator",
        *EXTERNAL_SCANNER_VALIDATORS,
        "UnicodeSmuggleValidator",
        "ScriptLintValidator",
        "LicenseValidator",
    )

    results = run_validation(repository, content_type=CONTENT_TYPE_UNKNOWN)

    assert all(result.passed for result in results)
    assert missing_tier1_publication_checks(results) == ("quality", "schema", "version")
    assert assess_publication(results, None).status == "incomplete"


def test_tier1_summary_counts_not_applicable_checks_apart(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    lineup = ["schema", "version", "pii", "quality"]
    results = run_validation(repository, checks=",".join(lineup), content_type=CONTENT_TYPE_UNKNOWN)

    passed, rows = summarize_tier1(results, lineup=lineup)
    text = _rendered(rows)

    assert passed is True
    # "1/1 passed" today; "1/1 checks passed" once the per-check count wording
    # changes. Counting the N/A checks would give 4/4 in either wording.
    assert re.search(r"\b1/1 (checks )?passed\b", text), text
    assert not re.search(r"\b4/4 (checks )?passed\b", text), text
    assert not re.search(r"\b\d+/4 (checks )?passed\b", text), text
    # The N/A checks are named on the n/a row only: never marked passed or
    # failed in the check ticker, and absent from every other row.
    not_applicable_rows = [row for row in rows if row.label == "n/a"]
    assert [_row_text(row) for row in not_applicable_rows] == ["schema, version, quality"]
    ticker = next(_row_text(row) for row in rows if row.label == "checks")
    assert "✓ pii" in ticker
    for check_id in MANIFEST_VALIDATORS:
        assert f"✓ {check_id}" not in ticker
        assert f"✗ {check_id}" not in ticker
        assert not any(check_id in _row_text(row) for row in rows if row.label not in {"checks", "n/a"})
    assert "not applicable" in text

    tier = build_final_render_context(results).tiers[1]
    assert (tier.passed, tier.failed, tier.skipped, tier.not_applicable) == (1, 0, 0, 3)
    assert tier.summary == "1 of 1 checks passed; 0 failed, 0 incomplete, 0 skipped, 3 N/A"


def test_reports_show_only_manifest_checks_as_not_applicable(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    results = run_validation(repository, checks="schema,quality", content_type=CONTENT_TYPE_UNKNOWN)

    data = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    assert data["tier1_applicability"] == APPLICABILITY
    assert data["overall_status"] == "passed"
    assert data["publication"]["eligible"] is False
    assert [entry["status"] for entry in data["results"]] == ["skipped", "skipped"]

    html = HTMLReporter(include_timestamp=False).render_all(results)
    body = re.sub(r"<(style|script)\b.*?</\1>", "", html, flags=re.DOTALL)
    assert "0 of 2 checks ran, 2 not applicable" in body
    assert "2 skipped" not in body
    assert "verdict-chip fail" not in body
    assert not re.search(r"\b0(\.0)?%", body)

    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    assert "| Validators Run | 0 |" in markdown
    assert "| ✅ Passed | 0 |" in markdown
    assert "FAILED" not in markdown
    assert "not applicable" in markdown


def test_json_report_omits_tier1_applicability_for_a_detected_artifact() -> None:
    result = ValidationResult(validator_name="PII Scan")
    result.add_success("pii", "No PII found")

    data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert "tier1_applicability" not in data


def test_validate_ordinary_repository_passes_and_records_applicability(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    reports = tmp_path / "reports"

    exit_code, output = _validate(
        str(repository), "--tiers", "1", "--checks", "schema,quality,pii", "-r", "json", "--output-dir", str(reports)
    )

    assert exit_code == 0, output
    assert "No skills found" not in output
    payload = _json_report(reports)
    assert payload["overall_status"] == "passed"
    assert payload["tier1_applicability"] == APPLICABILITY
    assert payload["publication"]["eligible"] is False
    statuses = {entry["validator"]: (entry["status"], entry["skip_reason"]) for entry in payload["results"]}
    assert statuses["PII Scan"] == ("passed", None)
    for name in ("Schema & Repository Governance", "Quality Score (4-Dimension Analysis)"):
        status, reason = statuses[name]
        assert status == "skipped"
        assert reason.startswith("not applicable")


def test_validate_default_tiers_on_ordinary_repository_report_n_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    # The assertions read the final panel, which wraps at the terminal width.
    monkeypatch.setenv("COLUMNS", "120")

    exit_code, output = _validate(
        str(repository), "--no-dedup", "--checks", "schema,version,pii", "--output-dir", str(tmp_path / "reports")
    )

    assert exit_code == 0, output
    assert "Tier 1: 1 of 1 checks passed; 0 failed, 0 incomplete, 0 skipped, 2 N/A" in output
    assert "Publication eligibility: Incomplete" in output
    assert "No skills found" not in output


def test_validate_ordinary_repository_with_a_card_number_still_fails(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    (repository / "README.md").write_text("# Example service\n\nBilling: 4532-0151-1283-0366\n", encoding="utf-8")
    reports = tmp_path / "reports"

    exit_code, output = _validate(
        str(repository), "--tiers", "1", "--checks", "schema,pii", "-r", "json", "--output-dir", str(reports)
    )

    assert exit_code == 1, output
    payload = _json_report(reports)
    assert payload["overall_status"] == "failed"
    pii = next(entry for entry in payload["results"] if entry["validator"] == "PII Scan")
    assert pii["status"] == "failed"
    assert any(finding["check_name"] == "credit_cards" for finding in pii["findings"])


def test_explicit_skill_type_still_reports_no_skills_found(tmp_path: Path) -> None:
    repository = _ordinary_repository(tmp_path / "repository")

    exit_code, output = _validate(
        str(repository),
        "--type",
        "skill",
        "--tiers",
        "1",
        "--checks",
        "schema,quality",
        "--output-dir",
        str(tmp_path / "reports"),
    )

    assert exit_code == 1
    assert "No skills found" in output


def test_agent_config_skill_roots_still_validate_as_a_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _ordinary_repository(tmp_path / "repository")
    skill_dirs = []
    for root in (".agents", ".claude"):
        skill = repository / root / "skills" / "sample"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: sample\ndescription: A sample skill kept under an agent config directory.\n---\n\n# Sample\n",
            encoding="utf-8",
        )
        skill_dirs.append(skill)
    catalogs: list[list[Path]] = []

    def fake_catalog(_ctx, *, skill_dirs: list[Path], **_kwargs: object) -> None:
        catalogs.append(sorted(skill_dirs))

    monkeypatch.setattr("skillevaluator.cli._validate_catalog", fake_catalog)

    exit_code, output = _validate(str(repository), "--tiers", "1", "--output-dir", str(tmp_path / "reports"))

    assert exit_code == 0, output
    assert [[path.resolve() for path in catalog] for catalog in catalogs] == [sorted(p.resolve() for p in skill_dirs)]
