# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for all reporters in the reporting module."""

import json
import re
import tempfile
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import (
    advisory_skip_result,
    agent_eval_result_from_run,
    build_agent_eval_payload,
    render_agent_eval_html_report,
)
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import (
    BenchmarkReporter,
    CLIReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    SARIFReporter,
)
from skillevaluator.tier1.commands import emit_reports


def _api_key_line() -> str:
    """Build a representative API-key finding without embedding it in source."""
    return "API" + '_KEY = "' + "sk" + "-live-" + "abc123" + '"'


def _html_report_data(output: str) -> dict:
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', output, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _html_tab(output: str, tab_id: str) -> str:
    marker = f'id="tab-{tab_id}"'
    start = output.index(marker)
    next_tab = output.find('id="tab-', start + len(marker))
    report_data = output.find('id="report-data"', start + len(marker))
    candidates = [position for position in (next_tab, report_data) if position >= 0]
    end = min(candidates) if candidates else len(output)
    return output[start:end]


def _failed_result(validator_name: str, *, severity: Severity | None = Severity.HIGH) -> ValidationResult:
    result = ValidationResult(validator_name=validator_name, validator_description="test failure")
    if severity is None:
        result.add_error("validator failed without a structured finding")
    else:
        result.add_finding(
            Finding(
                category="TEST",
                severity=severity,
                check_name="test-failure",
                message="test failure",
                file_path="SKILL.md",
            )
        )
    return result


def _similarity_result(*, severity: Severity = Severity.MEDIUM) -> ValidationResult:
    result = ValidationResult(
        validator_name="Similarity Check",
        validator_description="Detect duplicate content via embedding similarity",
    )
    result.add_finding(
        Finding(
            category="SIMILARITY",
            severity=severity,
            check_name="SIMILAR",
            message="same-name similarity match",
            file_path="team-a/shared-skill",
            metadata={
                "entry_a": "shared-skill",
                "entry_b": "shared-skill",
                "path_a": "team-a/shared-skill",
                "path_b": "team-b/shared-skill",
            },
        )
    )
    return result


@pytest.fixture
def success_result() -> ValidationResult:
    """Create a passing validation result."""
    result = ValidationResult(
        validator_name="SCHEMA",
        validator_description="Schema & Repository Governance",
    )
    result.add_success("manifest_found", "Found SKILL.md in skill directory")
    result.add_success("frontmatter_valid", "Parsed successfully", name="test-skill")
    result.add_success("structure_valid", "Valid structure: skills/test-skill/")
    result.summary.files_scanned = 3
    return result


@pytest.fixture
def failure_result() -> ValidationResult:
    """Create a failing validation result."""
    result = ValidationResult(
        validator_name="SECRETS",
        validator_description="Hardcoded Secrets Detection",
    )
    result.summary.files_scanned = 42

    aws_key = "AKIA" + "IOSFODNN7EXAMPLE"
    # Add findings using new-style Finding objects
    result.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.CRITICAL,
            check_name="aws-access-key-id",
            message="AWS Access Key ID detected",
            file_path="config/settings.py",
            line_number=15,
            line_content=f'AWS_KEY = "{aws_key}"',
            suggestion="Remove hardcoded key. Use environment variables.",
        )
    )
    result.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.HIGH,
            check_name="generic-api-key",
            message="Generic API key pattern detected",
            file_path="src/api.py",
            line_number=8,
            line_content=_api_key_line(),
            suggestion="Move API key to secrets vault.",
        )
    )
    return result


@pytest.fixture
def mixed_results(success_result: ValidationResult, failure_result: ValidationResult) -> list[ValidationResult]:
    """Create a list with both passing and failing results."""
    return [success_result, failure_result]


class TestCLIReporter:
    """Tests for CLIReporter."""

    def test_render_success(self, success_result: ValidationResult) -> None:
        """Test rendering a successful validation."""
        reporter = CLIReporter()
        output = reporter.render(success_result)

        assert "SCHEMA" in output
        assert "Validation passed" in output
        assert "manifest_found" in output
        assert "frontmatter_valid" in output

    def test_render_failure(self, failure_result: ValidationResult) -> None:
        """Test rendering a failed validation."""
        reporter = CLIReporter()
        output = reporter.render(failure_result)

        assert "SECRETS" in output
        assert "Validation failed" in output
        assert "AWS Access Key ID detected" in output
        assert "aws-access-key-id" in output

    def test_render_all_marks_skipped_live_eval_as_advisory(self) -> None:
        result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
        result.add_warning("A public LLM provider is required for live evaluation.")
        result.metadata["agent_eval"] = {
            "provenance": {
                "advisory": True,
                "reason": "skipped",
                "message": "A public LLM provider is required for live evaluation.",
            }
        }

        output = CLIReporter().render_all([result])

        assert "SKIP" in output
        assert "Required validations passed" in output
        assert "live evaluation skipped" in output

    def test_render_failure_escapes_markup_in_finding_fields(self) -> None:
        """Dynamic finding text should not be parsed as Rich markup."""
        result = ValidationResult(
            validator_name="SECURITY",
            validator_description="Security checks",
        )
        result.add_finding(
            Finding(
                category="SECURITY",
                severity=Severity.HIGH,
                check_name="mcp-least-privilege",
                message="Unexpected path [/tmp/nemo-gym-upstream]",
                file_path="skills/gym-benchmark-config/SKILL.md",
                line_number=1,
                line_content="Read [/tmp/nemo-gym-upstream] before running.",
                suggestion="Replace broad access to [/tmp/nemo-gym-upstream].",
            )
        )

        output = CLIReporter().render(result)
        plain_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        assert "[/tmp/nemo-gym-upstream]" in plain_output
        assert "Replace broad access" in plain_output

    def test_render_escapes_markup_in_legacy_errors(self) -> None:
        result = ValidationResult(
            validator_name="SIMILARITY",
            validator_description="Similarity checks",
        )
        result.add_error("Catalog value [/red] could not be parsed")

        output = CLIReporter().render(result)
        plain_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        assert "[/red]" in plain_output

    def test_failed_tier2_scan_is_reported_as_an_error_not_a_duplicate(self, tmp_path: Path, monkeypatch) -> None:
        from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
        from skillevaluator.deduplication.utils import skill_collector

        monkeypatch.setattr(skill_collector, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 2)
        skill = tmp_path / "over-budget"
        skill.mkdir()
        for name in ("SKILL.md", "a.bin", "b.bin"):
            (skill / name).write_text("content", encoding="utf-8")

        result = IntraSkillValidator().validate(skill)
        assert result.findings[0].check_name == "path_count_limit"

        output = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render_all([result]))

        assert "Errors:" in output
        assert "Skill tree contains more than 2 paths." in output
        assert "duplicates found" not in output.lower()

    def test_render_escapes_markup_in_success_details(self) -> None:
        result = ValidationResult(
            validator_name="SIMILARITY",
            validator_description="Similarity checks",
        )
        result.add_success(
            "catalog_compared",
            "Compared '[tool]' against catalog",
            target_name="[tool]",
        )

        output = CLIReporter().render(result)
        plain_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        assert "[tool]" in plain_output

    def test_render_all(self, mixed_results: list[ValidationResult]) -> None:
        """Test rendering multiple results."""
        reporter = CLIReporter()
        output = reporter.render_all(mixed_results)

        assert "SCHEMA" in output
        assert "SECRETS" in output
        assert "PASS" in output
        assert "FAIL" in output

    @pytest.mark.parametrize("severity", [Severity.MEDIUM, Severity.LOW])
    def test_render_all_shows_passing_findings_without_changing_pass_state(self, severity: Severity) -> None:
        result = _similarity_result(severity=severity)

        output = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render_all([result]))

        assert result.passed is True
        assert "same-name similarity match" in output
        assert "All validations passed" in output

    def test_render_finding_shows_both_related_paths(self) -> None:
        result = _similarity_result()

        output = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render(result))

        assert "team-a/shared-skill" in output
        assert "team-b/shared-skill" in output

    def test_render_tier3_harbor_links_and_evidence(self) -> None:
        """Tier 3 CLI output should include Harbor job, analysis, and step links."""
        output = CLIReporter().render(_tier3_harbor_result())
        plain_output = re.sub(r"\x1b\[[0-9;]*m", "", output)

        assert "Harbor logs:" in plain_output
        assert "Harbor analysis:" in plain_output
        assert "https://harbor.example.test/jobs/log-analyzer" in plain_output
        assert "View Step 9" in plain_output
        assert "log-analyzer-001?step=9" in plain_output
        assert "javascript:alert" not in plain_output

    def test_name_property(self) -> None:
        """Test reporter name."""
        reporter = CLIReporter()
        assert reporter.name == "cli"

    def test_folder_static_test_evidence_does_not_report_only_the_first_skill(self) -> None:
        result = ValidationResult(validator_name="Code Integrity & Hygiene")
        result.add_success(
            "skill-a",
            "All checks passed",
            checks=[
                {
                    "name": "test_discovery",
                    "description": "Found 1 candidate; target tests were not executed and coverage was not measured",
                    "metadata": {"test_count": 1},
                }
            ],
        )
        result.add_success(
            "skill-b",
            "All checks passed",
            checks=[
                {
                    "name": "test_discovery",
                    "description": "Found 3 candidates; target tests were not executed and coverage was not measured",
                    "metadata": {"test_count": 3},
                }
            ],
        )
        expected = "Target tests were not executed and coverage was not measured for any discovered skill"

        assert CLIReporter._static_test_evidence_message(result) == expected
        assert BenchmarkReporter._static_test_evidence_message(result) == expected


class TestBenchmarkReporter:
    def test_static_test_limitation_is_preserved_when_other_findings_exist(self) -> None:
        hygiene = ValidationResult(validator_name="Code Integrity & Hygiene")
        hygiene.add_success(
            "test_discovery",
            "Found 1 candidate; target tests were not executed and coverage was not measured",
            test_count=1,
            execution_performed=False,
            coverage_measured=False,
        )
        hygiene.metadata["benchmark_policy"] = {"tier3_required": False}
        advisory = ValidationResult(validator_name="Advisory")
        advisory.add_finding(
            Finding(
                category="ADVISORY",
                severity=Severity.LOW,
                check_name="note",
                message="Minor observation",
                file_path="SKILL.md",
            )
        )

        output = BenchmarkReporter(include_timestamp=False).render_all([hygiene, advisory])

        assert "target tests were not executed" in output.lower()
        assert "coverage was not measured" in output.lower()
        assert "Publication Recommendation" in output


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_render_success(self, success_result: ValidationResult) -> None:
        """Test JSON output for successful validation."""
        reporter = JSONReporter(include_timestamp=False)
        output = reporter.render(success_result)

        data = json.loads(output)
        assert data["validator"] == "SCHEMA"
        assert data["passed"] is True
        assert len(data["success_details"]) == 3

    def test_render_failure(self, failure_result: ValidationResult) -> None:
        """Test JSON output for failed validation."""
        reporter = JSONReporter(include_timestamp=False)
        output = reporter.render(failure_result)

        data = json.loads(output)
        assert data["validator"] == "SECRETS"
        assert data["passed"] is False
        assert len(data["findings"]) == 2
        assert data["findings"][0]["severity"] == "critical"

    def test_render_all(self, mixed_results: list[ValidationResult]) -> None:
        """Test JSON output for multiple results."""
        reporter = JSONReporter(include_timestamp=False)
        output = reporter.render_all(mixed_results)

        data = json.loads(output)
        assert data["overall_passed"] is False
        assert data["total_validators"] == 2
        assert len(data["results"]) == 2

    def test_render_all_persists_typed_rubric_execution_metadata(self) -> None:
        result = ValidationResult(
            validator_name="RUBRIC_EVAL",
            validator_description="LLM rubric evaluation",
        )
        result.metadata["rubric_eval"] = {
            "execution_status": "succeeded",
            "overall_score": 76.4,
            "overall_pass": False,
            "checks": [{"id": "scope_definition", "score": 4, "pass": False}],
        }
        result.add_error("Rubric evaluation failed")

        data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

        assert data["rubric_eval"]["execution_status"] == "succeeded"
        assert data["results"][0]["rubric_eval"] == data["rubric_eval"]

    def test_compact_output(self, success_result: ValidationResult) -> None:
        """Test compact JSON output without indentation."""
        reporter = JSONReporter(indent=None, include_timestamp=False)
        output = reporter.render(success_result)

        # Compact JSON should not have newlines (except in strings)
        assert output.count("\n") == 0

    def test_name_property(self) -> None:
        """Test reporter name."""
        reporter = JSONReporter()
        assert reporter.name == "json"


class TestSARIFReporter:
    """Tests for SARIFReporter."""

    def test_render_failure_produces_valid_sarif(self, failure_result: ValidationResult) -> None:
        reporter = SARIFReporter(include_timestamp=False)
        document = json.loads(reporter.render_all([failure_result]))

        assert document["version"] == "2.1.0"
        assert document["$schema"] == (
            "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
        )
        run = document["runs"][0]
        driver = run["tool"]["driver"]
        assert driver["name"] == "SkillEvaluator"
        assert driver["informationUri"] == "https://github.com/NVIDIA/SkillEvaluator"
        assert len(driver["rules"]) == 2
        assert len(run["results"]) == 2

        critical = next(item for item in run["results"] if item["level"] == "error")
        assert critical["ruleId"] == "SECRETS/aws-access-key-id"
        assert critical["message"]["text"] == "AWS Access Key ID detected"
        location = critical["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == "config/settings.py"
        assert location["region"]["startLine"] == 15

    def test_severity_mapping(self) -> None:
        result = ValidationResult(validator_name="TEST", validator_description="test")
        result.add_finding(
            Finding(
                category="TEST",
                severity=Severity.MEDIUM,
                check_name="medium-check",
                message="medium issue",
                file_path="SKILL.md",
                line_number=3,
            )
        )
        result.add_finding(
            Finding(
                category="TEST",
                severity=Severity.LOW,
                check_name="low-check",
                message="low issue",
                file_path="SKILL.md",
            )
        )

        run = json.loads(SARIFReporter(include_timestamp=False).render_all([result]))["runs"][0]
        levels = {item["ruleId"]: item["level"] for item in run["results"]}
        assert levels["TEST/medium-check"] == "warning"
        assert levels["TEST/low-check"] == "note"

    def test_emit_reports_writes_sarif_file(self, failure_result: ValidationResult) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            emit_reports(
                [failure_result],
                report_formats=("sarif",),
                output_dir=output_dir,
                basename="skillevaluator-security",
                announce_paths=False,
            )
            sarif_path = output_dir / "skillevaluator-security.sarif.json"
            assert sarif_path.is_file()
            document = json.loads(sarif_path.read_text(encoding="utf-8"))
            assert document["runs"][0]["results"]

    def test_name_and_extension(self) -> None:
        reporter = SARIFReporter()
        assert reporter.name == "sarif"
        assert reporter.get_file_extension() == ".sarif.json"

    def test_incomplete_scan_marks_execution_unsuccessful(self) -> None:
        result = ValidationResult(validator_name="BANDIT", validator_description="Bandit")
        result.add_error("Bandit did not return valid JSON; scan did not complete")
        result.mark_scan_incomplete("bandit")

        invocation = json.loads(SARIFReporter(include_timestamp=False).render_all([result]))["runs"][0]["invocations"][
            0
        ]
        assert invocation["executionSuccessful"] is False
        assert invocation["toolExecutionNotifications"][0]["message"]["text"] == "bandit scan did not complete"

    def test_physical_location_uri_encodes_spaces_and_skips_invalid_lines(self) -> None:
        root = Path("/workspace/my skill")
        result = ValidationResult(validator_name="QUALITY", validator_description="quality")
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.MEDIUM,
                check_name="spacing",
                message="issue",
                file_path=str(root / "SKILL.md"),
                line_number=0,
            )
        )
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.LOW,
                check_name="spacing",
                message="issue",
                file_path=str(root / "notes/readme.md"),
                line_number=3,
            )
        )

        run = json.loads(SARIFReporter(include_timestamp=False, workspace_root=root).render_all([result]))["runs"][0]
        locations = [item["locations"][0]["physicalLocation"] for item in run["results"]]
        assert locations[0]["artifactLocation"]["uri"] == "SKILL.md"
        assert "region" not in locations[0]
        assert locations[1]["artifactLocation"]["uri"] == "notes/readme.md"
        assert locations[1]["region"]["startLine"] == 3

    def test_nested_skill_path_relativized_to_repository_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        skill = repo / "skills" / "demo"
        skill.mkdir(parents=True)
        result = ValidationResult(validator_name="QUALITY", validator_description="quality")
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.MEDIUM,
                check_name="spacing",
                message="issue",
                file_path="SKILL.md",
                line_number=2,
            )
        )

        run = json.loads(
            SARIFReporter(
                include_timestamp=False,
                workspace_root=repo,
                scan_root=skill,
            ).render_all([result])
        )["runs"][0]
        uri = run["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert uri == "skills/demo/SKILL.md"

    def test_merge_catalog_sarif_combines_into_single_run(self) -> None:
        child = SARIFReporter(include_timestamp=False)
        documents = []
        for index in range(21):
            result = ValidationResult(validator_name="TEST", validator_description="test")
            result.add_finding(
                Finding(
                    category="TEST",
                    severity=Severity.LOW,
                    check_name=f"check-{index}",
                    message="issue",
                    file_path=f"skills/s{index}/SKILL.md",
                )
            )
            documents.append(json.loads(child.render_all([result])))

        from skillevaluator.reporting.sarif_reporter import merge_catalog_sarif_documents

        merged = merge_catalog_sarif_documents(documents)
        assert len(merged["runs"]) == 1
        assert len(merged["runs"][0]["results"]) == 21
        automation = merged["runs"][0]["automationDetails"]
        assert automation == {"id": "/skillevaluator/catalog"}
        assert "name" not in automation

    def test_merge_catalog_sarif_preserves_incomplete_invocation(self) -> None:
        result = ValidationResult(validator_name="BANDIT", validator_description="Bandit")
        result.add_error("Bandit scan did not complete")
        result.mark_scan_incomplete("bandit")
        child = json.loads(SARIFReporter(include_timestamp=False).render_all([result]))

        from skillevaluator.reporting.sarif_reporter import merge_catalog_sarif_documents

        merged = merge_catalog_sarif_documents([child])
        invocations = merged["runs"][0]["invocations"]
        assert invocations == child["runs"][0]["invocations"]
        assert invocations[0]["executionSuccessful"] is False
        assert invocations[0]["toolExecutionNotifications"][0]["descriptor"]["id"] == "incomplete/bandit"


def _blocking_result() -> ValidationResult:
    """Tier 1 validator result with one critical + one high finding."""
    r = ValidationResult(validator_name="SECRETS", validator_description="Secrets")
    r.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.CRITICAL,
            check_name="aws-key",
            message="key",
            file_path="a.py",
        )
    )
    r.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.HIGH,
            check_name="api-key",
            message="key",
            file_path="b.py",
        )
    )
    return r


def _advisory_dedup_result() -> ValidationResult:
    """Tier 2 (advisory) dedup result with one medium finding."""
    r = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="dedup",
    )
    r.add_finding(
        Finding(
            category="PARTIAL_OVERLAP",
            severity=Severity.MEDIUM,
            check_name="partial_overlap",
            message="overlap",
            file_path="SKILL.md",
        )
    )
    return r


def _advisory_agent_eval_result() -> ValidationResult:
    """Tier 3 (advisory) agent-eval result with one low finding."""
    r = ValidationResult(validator_name="AGENT_EVAL", validator_description="eval")
    r.add_finding(
        Finding(
            category="AGENT_EVAL",
            severity=Severity.LOW,
            check_name="some_check",
            message="advisory",
            file_path="evals/evals.json",
        )
    )
    return r


def _tier3_harbor_result() -> ValidationResult:
    """Tier 3 result with safe Harbor links and one unsafe link to sanitize."""
    r = ValidationResult(validator_name="AGENT_EVAL", validator_description="eval")
    step_url = "https://harbor.example.test/jobs/log-analyzer/tasks/_/codex/openai/gpt/trials/log-analyzer-001?step=9"
    r.metadata["agent_eval"] = {
        "schema_version": "2.0",
        "summary": {
            "schema_version": "2.0",
            "verdict": "pass",
            "skill_name": "log-analyzer",
            "best_agent": "codex",
            "agents_run": ["codex"],
            "overall_score": 0.82,
            "overall_lift": 0.12,
            "environment": "local",
            "runtime_seconds": 42.0,
            "harbor_viewer": {
                "job_url": "javascript:alert(1)",
                "analysis_url": "file:///tmp/report.html",
            },
        },
        "skill_name": "log-analyzer",
        "verdict": "pass",
        "best_agent": "codex",
        "agents_run": ["codex"],
        "environment": "local",
        "overall_score": 0.82,
        "overall_lift": 0.12,
        "composite_lift": 0.12,
        "runtime_seconds": 42.0,
        "agents": {},
        "dimensions": [],
        "evaluators": {},
        "insights": {},
        "conclusions": [],
        "recommendations": [
            {
                "title": "Tighten workflow",
                "message": "Tighten the workflow instructions for the failed retrieval case.",
                "category": "Improve",
                "severity": "warn",
                "source": "deterministic",
                "evidence": {"url": step_url, "label": "goal_accuracy"},
            },
            {
                "title": "Unsafe evidence",
                "message": "This unsafe evidence link should not render.",
                "evidence": {"url": "javascript:alert(2)", "label": "bad"},
            },
        ],
        "suggestions": ["Tighten the workflow instructions for the failed retrieval case."],
        "suggestions_v2": [
            {
                "metric": "goal_accuracy",
                "recommendation": "Tighten the workflow instructions for the failed retrieval case.",
                "harbor_evidence": {"url": step_url, "label": "goal_accuracy"},
                "evidence_refs": [],
            }
        ],
        "metric_ids": [],
        "metric_labels": {},
        "attempt_policy": {},
        "dataset": [],
        "provenance": {},
        "harbor_viewer": {
            "job_url": "https://harbor.example.test/jobs/log-analyzer",
            "analysis_url": "https://harbor.example.test/jobs/log-analyzer?tab=analysis",
            "jobs": [
                {
                    "url": "https://harbor.example.test/jobs/log-analyzer",
                    "analysis_url": "https://harbor.example.test/jobs/log-analyzer?tab=analysis",
                    "name": "log-analyzer-codex-with--20260624",
                }
            ],
            "evidence_links": [{"url": step_url, "label": "goal_accuracy", "step": 9}],
        },
    }
    r.add_success("agent_eval", "Tier 3 evaluation complete")
    r.passed = True
    return r


class TestTier3HarborCanonicalPayload:
    """Canonical Tier 3 payload should preserve Harbor report links."""

    def test_build_payload_preserves_harbor_links_and_step_evidence(self) -> None:
        step_url = (
            "https://harbor.example.test/jobs/log-analyzer/tasks/_/codex/openai/gpt/trials/log-analyzer-001?step=9"
        )
        payload = build_agent_eval_payload(
            "log-analyzer",
            {
                "codex": {
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                    "with_skill": {"security": 1.0, "accuracy": 0.62, "goal_accuracy": 0.55},
                    "without_skill": {"security": 1.0, "accuracy": 0.50, "goal_accuracy": 0.40},
                    "rewards": [
                        {
                            "trial_id": "log-analyzer-001__abc",
                            "entry_id": "log-analyzer-001",
                            "overall": 0.62,
                            "security": 1.0,
                            "accuracy": 0.62,
                            "goal_accuracy": 0.55,
                            "harbor_viewer": {
                                "job_name": "log-analyzer-codex-with--20260624",
                                "job_url": "https://harbor.example.test/jobs/log-analyzer",
                                "analysis_url": ("https://harbor.example.test/jobs/log-analyzer?tab=analysis"),
                                "trial_url": (
                                    "https://harbor.example.test/jobs/log-analyzer/tasks/_/"
                                    "codex/openai/gpt/trials/log-analyzer-001"
                                ),
                                "evidence_urls": [{"label": "goal_accuracy", "url": step_url}],
                            },
                        }
                    ],
                }
            },
            use_llm_judge=False,
        )

        assert payload is not None
        assert payload["harbor_viewer"]["job_url"].endswith("/jobs/log-analyzer")
        assert payload["harbor_viewer"]["analysis_url"].endswith("?tab=analysis")
        assert payload["harbor_viewer"]["evidence_links"][0]["label"] == "Step 9"
        assert payload["harbor_viewer"]["evidence_links"][0]["step"] == 9
        assert payload["trials"][0]["harbor_viewer"]["evidence_urls"][0]["step"] == 9
        assert payload["recommendations"][0]["evidence"]["url"] == step_url

    def test_failed_run_has_no_synthetic_score_or_top_performer(self) -> None:
        payload = build_agent_eval_payload(
            "broken-skill",
            {
                "codex": {
                    "execution_status": "failed",
                    "execution_errors": ["Scored attempt coverage is 0/1"],
                    "expected_attempts": 1,
                    "scored_attempts": 0,
                    "with_skill": {},
                    "rewards": [],
                }
            },
            use_llm_judge=False,
        )

        assert payload is not None
        assert payload["execution_status"] == "failed"
        assert payload["overall_score"] is None
        assert payload["best_agent"] == ""
        assert payload["composite_lift"] is None
        assert payload["conclusions"][0]["title"] == "Evaluation incomplete"
        assert not any("healthy" in text.lower() for text in payload["suggestions"])

    def test_advisory_skip_is_false_without_synthetic_score(self) -> None:
        result = advisory_skip_result("No public provider key", skill_name="demo")

        assert result.passed is False
        assert result.metadata["agent_eval"]["execution_status"] == "skipped"
        assert result.metadata["agent_eval"]["overall_score"] is None
        cli_output = CLIReporter().render_all([result])
        assert "SKIP" in cli_output
        assert "Validation failed" not in cli_output
        assert "Required validations passed" in cli_output

    def test_legacy_summary_without_execution_status_is_unknown(self, tmp_path: Path) -> None:
        from skillevaluator.tier3.harbor.report_data import load_agent_data

        summary = tmp_path / "codex" / "with-skill" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({"scores": {"accuracy": 1.0}, "num_trials": 1}), encoding="utf-8")

        agents = load_agent_data(tmp_path)

        assert agents["codex"]["execution_status"] == "unknown"
        assert agents["codex"]["with_skill"] == {}

    def test_failed_summary_status_survives_reader(self, tmp_path: Path) -> None:
        from skillevaluator.tier3.harbor.report_data import load_agent_data

        summary = tmp_path / "codex" / "with-skill" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "scores": {},
                    "num_trials": 0,
                    "execution_status": "failed",
                    "execution_errors": ["No Harbor job result"],
                    "expected_attempts": 1,
                    "scored_attempts": 0,
                }
            ),
            encoding="utf-8",
        )

        agents = load_agent_data(tmp_path)

        assert agents["codex"]["execution_status"] == "failed"
        assert agents["codex"]["execution_errors"] == ["No Harbor job result"]

    def test_failed_run_produces_false_agent_eval_result(self, tmp_path: Path) -> None:
        skill = tmp_path / "demo"
        skill.mkdir()
        results_root = tmp_path / "results" / "demo"
        run_id = "20260709_120000"
        run_dir = results_root / run_id
        summary = run_dir / "codex" / "with-skill" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "scores": {},
                    "num_trials": 0,
                    "execution_status": "failed",
                    "execution_errors": ["No Harbor job result"],
                    "expected_attempts": 1,
                    "scored_attempts": 0,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        (results_root / "latest").symlink_to(run_id)

        result = agent_eval_result_from_run(
            skill,
            results_dir=tmp_path / "results",
            use_llm_judge=False,
        )

        assert result is not None
        assert result.passed is False
        assert result.errors == ["No Harbor job result"]
        assert result.metadata["agent_eval"]["overall_score"] is None


class TestGatingSplit:
    """Tests for the CLI-consistent Tier 1/Tier 2 gate and advisory Tier 3."""

    @pytest.mark.parametrize(
        "validator_name",
        [
            "Similarity Check",
            "Context Deduplication",
            "Context Optimization",
            "Context Optimization Check",
            "Tier 2 Deduplication",
        ],
    )
    def test_failed_tier2_names_render_in_tier2_and_block(
        self,
        validator_name: str,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / validator_name.lower().replace(" ", "-")

        reports_passed = emit_reports(
            [_failed_result(validator_name)],
            report_formats=("html",),
            output_dir=output_dir,
            basename="report",
        )

        output = (output_dir / "report.html").read_text(encoding="utf-8")
        report_data = _html_report_data(output)
        assert reports_passed is False
        assert 'data-target-tab="tier2"' in output
        assert "<strong>0</strong>/1 checks passed" in output
        assert report_data["gating"] == {
            "blocking_tiers": ["tier1", "tier2"],
            "advisory_tiers": ["tier3"],
            "blocking": {"critical": 0, "high": 1, "medium": 0, "low": 0},
            "advisory": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "blocking_findings": 1,
            "would_block": True,
        }
        assert "Advisory (Tier 2" not in output

    def test_failed_tier2_without_structured_findings_still_blocks(self) -> None:
        output = HTMLReporter(include_timestamp=False).render_all(
            [_failed_result("Context Deduplication", severity=None)]
        )

        gating = _html_report_data(output)["gating"]
        assert gating["would_block"] is True

    def test_tier1_failure_remains_blocking(self) -> None:
        output = HTMLReporter(include_timestamp=False).render_all([_failed_result("Security Scan")])

        gating = _html_report_data(output)["gating"]
        assert gating["blocking_tiers"] == ["tier1", "tier2"]
        assert gating["blocking"]["high"] == 1
        assert gating["would_block"] is True

    def test_tier3_failure_remains_advisory(self) -> None:
        output = HTMLReporter(include_timestamp=False).render_all([_failed_result("AGENT_EVAL")])

        gating = _html_report_data(output)["gating"]
        assert gating["advisory_tiers"] == ["tier3"]
        assert gating["blocking"]["high"] == 0
        assert gating["advisory"]["high"] == 1
        assert gating["would_block"] is False

    def test_combined_report_labels_tier2_blocking_and_only_tier3_advisory(self) -> None:
        output = HTMLReporter(
            include_timestamp=False,
            tabs=[
                {"id": "tier1", "label": "Tier 1"},
                {"id": "tier2", "label": "Tier 2"},
                {"id": "tier3", "label": "Tier 3"},
            ],
        ).render_all([_failed_result("Similarity Check"), _failed_result("AGENT_EVAL")])

        assert "Blocking (Tier 1 + Tier 2)" in output
        assert "Advisory (Tier 3 agent-eval)" in output
        assert "Advisory (Tier 2" not in output
        assert "Advisory findings (dedup + agent-eval)" not in output

    def test_tier1_quality_score_warning_and_advisory_rows_render(self) -> None:
        quality = ValidationResult(validator_name="Quality Score", validator_description="Skill quality")
        quality.metadata["quality_scores"] = {
            "skill_name": "demo-skill",
            "overall_score": 72.5,
            "grade": "C",
            "skill_type": "script-based",
            "dimensions": {
                "clarity": {"score": 75.0, "weight": 0.5},
                "safety": {"score": 70.0, "weight": 0.5},
            },
            "metrics": {"total_tokens": 123, "recommended_max_tokens": 500, "script_count": 1},
        }
        warnings = ValidationResult(validator_name="Schema", validator_description="Schema checks")
        warnings.add_warning("Metadata author is recommended.")
        advisory = ValidationResult(validator_name="Code Integrity & Hygiene", validator_description="Advisory checks")
        advisory.add_finding(
            Finding(
                category="HYGIENE",
                severity=Severity.LOW,
                check_name="tests_missing",
                message="Target tests were not executed and coverage was not measured.",
                file_path="demo-skill/SKILL.md",
                suggestion="Add a representative test case.",
            )
        )

        output = HTMLReporter(include_timestamp=False).render_all([quality, warnings, advisory])
        report_data = _html_report_data(output)

        assert report_data["quality_scores"]["demo-skill"]["overall_score"] == 72.5
        assert 'data-quality-skill-name="demo-skill"' in output
        assert "Quality Score: 72.5/100" in output
        assert "Type: script-based" in output
        assert ">clarity</td>" in output
        assert '<td class="warning-text">1</td>' in output
        assert "Metadata author is recommended." in output
        assert "1 advisory finding(s)" in output
        assert "not blocking:" in output
        assert "Target tests were not executed and coverage was not measured." in output

    def test_similarity_only_html_uses_tier2_tab_and_renders_the_match(self, tmp_path: Path) -> None:
        result = _similarity_result(severity=Severity.HIGH)

        emit_reports(
            [result],
            report_formats=("html",),
            output_dir=tmp_path,
            basename="similarity-only",
        )

        output = (tmp_path / "similarity-only.html").read_text(encoding="utf-8")
        assert 'id="tab-tier1"' not in output
        tier2 = _html_tab(output, "tier2")
        assert "same-name similarity match" in tier2
        assert "Intra-skill check was not run" not in tier2

    def test_default_html_reporter_classifies_similarity_as_tier2(self) -> None:
        output = HTMLReporter(include_timestamp=False).render_all([_similarity_result(severity=Severity.HIGH)])

        assert 'id="tab-tier1"' not in output
        assert "same-name similarity match" in _html_tab(output, "tier2")

    def test_combined_html_keeps_similarity_out_of_tier1(
        self,
        success_result: ValidationResult,
        tmp_path: Path,
    ) -> None:
        similarity = _similarity_result(severity=Severity.HIGH)

        emit_reports(
            [success_result, similarity],
            report_formats=("html",),
            output_dir=tmp_path,
            basename="combined",
        )

        output = (tmp_path / "combined.html").read_text(encoding="utf-8")
        assert "same-name similarity match" not in _html_tab(output, "tier1")
        assert "same-name similarity match" in _html_tab(output, "tier2")

    def test_similarity_html_renders_both_related_paths(self) -> None:
        result = _similarity_result(severity=Severity.HIGH)

        output = HTMLReporter(
            include_timestamp=False,
            tabs=[{"id": "tier2", "label": "Tier 2"}],
        ).render_all([result])
        tier2 = _html_tab(output, "tier2")

        assert "team-a/shared-skill" in tier2
        assert "team-b/shared-skill" in tier2


class TestHTMLReporter:
    """Tests for HTMLReporter."""

    def test_render_success(self, success_result: ValidationResult) -> None:
        """Test HTML output for successful validation."""
        reporter = HTMLReporter(include_timestamp=False)
        output = reporter.render(success_result)

        assert "<!DOCTYPE html>" in output
        assert "SCHEMA" in output
        assert "Pass" in output  # Status badge uses "Pass" not "PASS"
        assert "manifest_found" in output

    def test_render_failure(self, failure_result: ValidationResult) -> None:
        """Test HTML output for failed validation."""
        reporter = HTMLReporter(include_timestamp=False)
        output = reporter.render(failure_result)

        assert "<!DOCTYPE html>" in output
        assert "SECRETS" in output
        assert "fail" in output.lower()
        assert "AWS Access Key ID detected" in output
        assert "severity-critical" in output

    def test_render_all(self, mixed_results: list[ValidationResult]) -> None:
        """Test HTML output for multiple results."""
        reporter = HTMLReporter(include_timestamp=False)
        output = reporter.render_all(mixed_results)

        assert "Validation Report" in output or "SkillEvaluator" in output
        assert "SCHEMA" in output
        assert "SECRETS" in output

    def test_render_tier3_harbor_links_and_evidence(self) -> None:
        """Tier 3 HTML should link Harbor logs, analysis, and exact trajectory steps."""
        output = HTMLReporter(include_timestamp=False).render_all([_tier3_harbor_result()])

        assert "Harbor artifacts" in output
        assert "Harbor logs" in output
        assert "Harbor analysis" in output
        assert "View Step 9" in output
        assert "log-analyzer-001?step=9" in output
        assert "javascript:alert" not in output

    def test_failed_tier3_html_renders_na_without_top_performer(self) -> None:
        payload = build_agent_eval_payload(
            "broken-skill",
            {
                "codex": {
                    "execution_status": "failed",
                    "execution_errors": ["No Harbor job result"],
                    "expected_attempts": 1,
                    "scored_attempts": 0,
                    "with_skill": {},
                    "rewards": [],
                }
            },
            use_llm_judge=False,
        )
        result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
        result.metadata["agent_eval"] = payload
        result.add_error("No Harbor job result")

        output = HTMLReporter(include_timestamp=False).render_all([result])

        assert "No successfully scored agent is available" in output
        assert "Default agent is the top performer" not in output
        assert "Best Performing Agent" in output
        assert ">N/A<" in output
        assert "performing well across evaluated dimensions" not in output

        markdown = MarkdownReporter(include_timestamp=False).render_all([result])
        assert "composite lift = N/A" in markdown
        assert "composite lift = +0.00" not in markdown

    def test_standalone_tier3_report_uses_skillevaluator_branding(self, tmp_path: Path) -> None:
        """Standalone Tier 3 HTML should use the canonical SkillEvaluator branding."""
        skill = tmp_path / "log-analyzer"
        skill.mkdir()
        results_dir = tmp_path / "20260624_134842"
        trial_dir = results_dir / "codex" / "with-skill" / "trials" / "log-analyzer-001__sample"
        trial_dir.mkdir(parents=True)
        (results_dir / "codex" / "with-skill" / "summary.json").write_text(
            json.dumps(
                {
                    "scores": {
                        "security": 1.0,
                        "skill_execution": 0.95,
                        "skill_efficiency": 0.84,
                        "accuracy": 0.92,
                        "goal_accuracy": 0.88,
                        "behavior_check": 0.90,
                    },
                    "metrics": [
                        "security",
                        "skill_execution",
                        "skill_efficiency",
                        "accuracy",
                        "goal_accuracy",
                        "behavior_check",
                    ],
                    "num_trials": 1,
                    "execution_status": "succeeded",
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "reward.json").write_text(
            json.dumps(
                {
                    "entry_id": "log-analyzer-001",
                    "security": 1.0,
                    "skill_execution": 0.95,
                    "skill_efficiency": 0.84,
                    "accuracy": 0.92,
                    "goal_accuracy": 0.88,
                    "behavior_check": 0.90,
                    "harbor_viewer": {
                        "analysis_url": "https://harbor.example.test/jobs/log-analyzer?tab=analysis",
                        "evidence_urls": [
                            {
                                "label": "skill_efficiency",
                                "url": "https://harbor.example.test/jobs/log-analyzer/trials/log-analyzer-001?step=9",
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

        report_path = render_agent_eval_html_report(
            skill,
            results_dir,
            use_llm_judge=False,
        )
        output = report_path.read_text(encoding="utf-8")

        assert "SkillEvaluator" in output
        assert "Generated by <strong>SkillEvaluator</strong>" in output
        assert "?tab=analysis" in output
        assert "?step=9" in output

    def test_standalone_failed_tier3_report_shows_failure_without_score(self, tmp_path: Path) -> None:
        skill = tmp_path / "broken-skill"
        skill.mkdir()
        summary = tmp_path / "codex" / "with-skill" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "scores": {},
                    "num_trials": 0,
                    "execution_status": "failed",
                    "execution_errors": ["Missing reward for case-001"],
                    "expected_attempts": 1,
                    "scored_attempts": 0,
                }
            ),
            encoding="utf-8",
        )

        output = render_agent_eval_html_report(
            skill,
            tmp_path,
            use_llm_judge=False,
        ).read_text(encoding="utf-8")

        assert ">N/A<" in output
        assert "No successfully scored agent" in output
        assert "Missing reward for case-001" in output
        assert "-1.00" not in output

    def test_custom_title(self, success_result: ValidationResult) -> None:
        """Test custom report title."""
        reporter = HTMLReporter(title="Custom Report Title")
        output = reporter.render(success_result)

        assert "Custom Report Title" in output

    def test_self_contained(self, success_result: ValidationResult) -> None:
        """Test that HTML is self-contained with embedded CSS."""
        reporter = HTMLReporter()
        output = reporter.render(success_result)

        assert "<style>" in output
        assert "</style>" in output

    def test_name_property(self) -> None:
        """Test reporter name."""
        reporter = HTMLReporter()
        assert reporter.name == "html"


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_render_success(self, success_result: ValidationResult) -> None:
        """Test Markdown output for successful validation."""
        reporter = MarkdownReporter(include_timestamp=False)
        output = reporter.render(success_result)

        assert "SCHEMA" in output
        assert "✅" in output or "✓" in output
        assert "manifest_found" in output

    def test_render_failure(self, failure_result: ValidationResult) -> None:
        """Test Markdown output for failed validation."""
        reporter = MarkdownReporter(include_timestamp=False)
        output = reporter.render(failure_result)

        assert "SECRETS" in output
        assert "❌" in output
        assert "AWS Access Key ID detected" in output

    def test_render_all(self, mixed_results: list[ValidationResult]) -> None:
        """Test Markdown output for multiple results."""
        reporter = MarkdownReporter(include_timestamp=False)
        output = reporter.render_all(mixed_results)

        assert "# " in output  # Header
        assert "SCHEMA" in output
        assert "SECRETS" in output
        assert "|" in output  # Table

    @pytest.mark.parametrize("severity", [Severity.MEDIUM, Severity.LOW])
    def test_render_all_shows_passing_findings_without_changing_pass_state(self, severity: Severity) -> None:
        result = _similarity_result(severity=severity)

        output = MarkdownReporter(include_timestamp=False).render_all([result])

        assert result.passed is True
        assert "**Status:** ✅ PASSED" in output
        assert "same-name similarity match" in output

    def test_render_finding_shows_both_related_paths(self) -> None:
        result = _similarity_result()

        output = MarkdownReporter(include_timestamp=False).render(result)

        assert "team-a/shared-skill" in output
        assert "team-b/shared-skill" in output

    def test_render_tier3_harbor_links_and_evidence(self) -> None:
        """Tier 3 Markdown should include compact Harbor and evidence links."""
        output = MarkdownReporter(include_timestamp=False).render_all([_tier3_harbor_result()])

        assert "**Harbor logs:** [Open Harbor logs](" in output
        assert "**Harbor analysis:** [Open Harbor analysis](" in output
        assert "https://harbor.example.test/jobs/log-analyzer" in output
        assert "Evidence: [View Step 9](" in output
        assert "log-analyzer-001?step=9" in output
        assert "javascript:alert" not in output

    def test_details_section(self, failure_result: ValidationResult) -> None:
        """Test expandable details section."""
        reporter = MarkdownReporter(include_details=True)
        output = reporter.render(failure_result)

        assert "<details>" in output
        assert "</details>" in output

    def test_max_findings_limit(self, failure_result: ValidationResult) -> None:
        """Test max findings limit."""
        reporter = MarkdownReporter(max_findings_shown=1)
        output = reporter.render(failure_result)

        # Should show truncation message
        assert "more" in output.lower()

    def test_name_property(self) -> None:
        """Test reporter name."""
        reporter = MarkdownReporter()
        assert reporter.name == "markdown"


class TestReporterSaveMethod:
    """Tests for the save method across all reporters."""

    def test_json_save(self, success_result: ValidationResult) -> None:
        """Test saving JSON report to file."""
        reporter = JSONReporter(include_timestamp=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            reporter.save([success_result], output_path)
            content = output_path.read_text(encoding="utf-8")
            data = json.loads(content)
            assert data["overall_passed"] is True
        finally:
            output_path.unlink()

    def test_html_save(self, success_result: ValidationResult) -> None:
        """Test saving HTML report to file."""
        reporter = HTMLReporter()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = Path(f.name)

        try:
            reporter.save([success_result], output_path)
            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
        finally:
            output_path.unlink()

    def test_markdown_save(self, success_result: ValidationResult) -> None:
        """Test saving Markdown report to file."""
        reporter = MarkdownReporter()

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_path = Path(f.name)

        try:
            reporter.save([success_result], output_path)
            content = output_path.read_text(encoding="utf-8")
            assert "#" in content
        finally:
            output_path.unlink()


class TestFindingBackwardCompatibility:
    """Tests for backward compatibility with string severity."""

    def test_finding_with_string_severity(self) -> None:
        """Test that Finding accepts string severity."""
        finding = Finding(
            category="LICENSE",
            severity="HIGH",  # String instead of Severity enum
            check_name="blocked_license",
            message="License blocked",
            file_path="LICENSE",
        )

        # Should convert to Severity enum
        assert finding.severity == Severity.HIGH

    def test_finding_with_enum_severity(self) -> None:
        """Test that Finding accepts Severity enum."""
        finding = Finding(
            category="LICENSE",
            severity=Severity.HIGH,
            check_name="blocked_license",
            message="License blocked",
            file_path="LICENSE",
        )

        assert finding.severity == Severity.HIGH

    def test_legacy_add_finding(self) -> None:
        """Test legacy add_finding(tag, severity, message) signature."""
        result = ValidationResult(validator_name="TEST")
        result.add_finding("SECRET", Severity.HIGH, "Test secret found")

        assert not result.passed
        assert len(result.errors) == 1
        assert "[SECRET-HIGH]" in result.errors[0]


class TestValidationResultMethods:
    """Tests for ValidationResult methods."""

    def test_add_success(self) -> None:
        """Test add_success method."""
        result = ValidationResult(validator_name="TEST")
        result.add_success("check1", "Check passed", extra="data")

        assert result.passed
        assert len(result.success_details) == 1
        assert result.success_details[0].check_name == "check1"
        assert result.success_details[0].metadata["extra"] == "data"
        assert result.summary.checks_performed == 1

    def test_add_structured_finding(self) -> None:
        """Test add_structured_finding method."""
        result = ValidationResult(validator_name="TEST")
        finding = Finding(
            category="TEST",
            severity=Severity.HIGH,
            check_name="test_check",
            message="Test message",
            file_path="test.py",
        )
        result.add_structured_finding(finding, is_error=True)

        assert not result.passed
        assert len(result.findings) == 1
        assert len(result.errors) == 1

    def test_merge_results(self) -> None:
        """Test merging two ValidationResults."""
        result1 = ValidationResult(validator_name="TEST1")
        result1.add_error("Error 1")

        result2 = ValidationResult(validator_name="TEST2")
        result2.add_warning("Warning 1")

        result1.merge(result2)

        assert not result1.passed
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1


class TestHTMLReporterHeroHelpers:
    """Hero-card helpers introduced for the combined Tier 1+2+3 report.

    These helpers exist so the hero card on the report's first page can:

    - Show ``skills/<name>`` in the title instead of leaking the full
      filesystem / repo path (the full path stays clickable in the header).
    - Render one tile per *tier that actually ran*, with each tile's stats
      sourced from only that tier's results (no leakage from global counts).

    Each test below pins one of those guarantees in isolation so future
    template / pipeline tweaks can't silently regress hero behavior.
    """


class TestTopIssuesDeepLink:
    """Top-Issues rows must deep-link to the actual finding card.

    Originally the executive-summary pills called ``scrollToSkill`` which
    only opened the skill card and landed the user on the quality-score
    panel (the first thing rendered inside the card). The fix:

    1. Each finding gets a stable ``issue_key`` shared with the matching
       top-issue row so JS can find the right card to scroll to.
    2. Skill pills inside the executive summary scroll to that finding.
    3. A separate ``Q`` chip is rendered next to each pill for users who
       actually want to inspect the quality score.

    The tests below pin those three guarantees so a future template
    refactor can't silently regress the wiring.
    """

    @staticmethod
    def _failing_finding_result(skill_name: str = "skill-x") -> ValidationResult:
        result = ValidationResult(validator_name="SECRETS")
        result.add_finding(
            Finding(
                category="SECRET",
                severity=Severity.HIGH,
                check_name="generic-api-key",
                message="Generic API key pattern detected",
                file_path=f"{skill_name}/src/api.py",
                line_number=8,
                line_content=_api_key_line(),
                suggestion="Move API key to secrets vault.",
            )
        )
        return result
