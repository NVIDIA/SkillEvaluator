# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for security validators: Secrets, Dependencies, CodeRisk.

Tests validator logic and graceful handling when external tools are unavailable.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.utils.tool_runner import Severity, Tools
from skillevaluator.validators.base import ValidationResult
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.secrets import SecretsValidator

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_tool_unavailable():
    """Context manager to mock a tool as unavailable."""

    def _mock(tool_name: str):
        return patch.object(getattr(Tools, tool_name), "_path", None)

    return _mock


@pytest.fixture
def mock_tool_available():
    """Context manager to mock a tool as available."""

    def _mock(tool_name: str, path: str = "/usr/bin/tool"):
        return patch.object(getattr(Tools, tool_name), "_path", path)

    return _mock


# =============================================================================
# SECRETS VALIDATOR TESTS
# =============================================================================


class TestSecretsValidator:
    """Tests for SecretsValidator (Gitleaks integration)."""

    def test_init_default(self):
        """Test default initialization."""
        validator = SecretsValidator()
        assert validator.name == "Secrets Detection"
        assert validator.scan_git_history is False

    def test_init_with_git_history(self):
        """Test initialization with git history scanning enabled."""
        validator = SecretsValidator(scan_git_history=True)
        assert validator.scan_git_history is True

    def test_validate_no_gitleaks_installed(self, sample_skill_dir: Path, mock_tool_unavailable):
        """Test graceful handling when gitleaks is not installed."""
        validator = SecretsValidator()

        with mock_tool_unavailable("gitleaks"):
            result = validator.validate(sample_skill_dir)

        assert result.status == "incomplete"
        assert not result.passed
        assert any("gitleaks not installed" in w for w in result.warnings)

    def test_validate_clean_skill(self, sample_skill_dir: Path, mock_tool_available):
        """Test validation of a clean skill directory."""
        validator = SecretsValidator()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.error_message = None

        with (
            mock_tool_available("gitleaks", "/usr/bin/gitleaks"),
            patch.object(Tools.gitleaks, "run", return_value=mock_result),
        ):
            result = validator.validate(sample_skill_dir)

        assert result.passed
        assert any("No secrets detected" in m for m in result.messages)

    def test_validate_skill_with_secrets(self, skill_with_security_issues: Path, mock_tool_available):
        """Test detection of secrets in a skill."""
        validator = SecretsValidator()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 10
        mock_result.error_message = None
        reported_secret = "sk" + "-" + "12345" + "67890"
        mock_result.stdout = """[
            {
                "RuleID": "generic-api-key",
                "Description": "Generic API Key",
                "File": "script.py",
                "StartLine": 5,
                "Secret": "__REPORTED_SECRET__",
                "Tags": ["api", "key"]
            }
        ]""".replace("__REPORTED_SECRET__", reported_secret)

        with (
            mock_tool_available("gitleaks", "/usr/bin/gitleaks"),
            patch.object(Tools.gitleaks, "run", return_value=mock_result) as mock_run,
        ):
            result = validator.validate(skill_with_security_issues)

        assert not result.passed
        assert any("SECRET" in e for e in result.errors)
        args = mock_run.call_args.args[0]
        assert args[args.index("--exit-code") + 1] == "10"

    def test_severity_mapping(self):
        """Test severity determination from rule tags."""
        validator = SecretsValidator()

        # Critical severities
        assert validator._determine_severity("aws-key", ["aws", "key"]) == Severity.CRITICAL
        assert validator._determine_severity("private-key", ["private-key"]) == Severity.CRITICAL

        # High severities
        assert validator._determine_severity("generic-api-key", ["api"]) == Severity.HIGH

        # Default to high for unknown
        assert validator._determine_severity("unknown-rule", []) == Severity.HIGH


# =============================================================================
# DEPENDENCY SECURITY VALIDATOR TESTS
# =============================================================================


class TestDependencySecurityValidator:
    """Tests for DependencySecurityValidator (pip-audit + Safety)."""

    def test_init_default(self):
        """Test default initialization."""
        validator = DependencySecurityValidator()
        assert validator.name == "Dependency Vulnerability Audit"
        assert validator.use_safety is True
        assert validator.fail_on_medium is False

    def test_init_custom(self):
        """Test custom initialization."""
        validator = DependencySecurityValidator(use_safety=False, fail_on_medium=True)
        assert validator.use_safety is False
        assert validator.fail_on_medium is True

    def test_validate_no_dependency_files(self, sample_skill_dir: Path):
        """Test validation when no dependency files exist."""
        validator = DependencySecurityValidator()
        result = validator.validate(sample_skill_dir)

        assert result.passed
        assert any("No dependency files found" in m for m in result.messages)

    def test_validate_with_requirements(self, tmp_path: Path, mock_tool_available):
        """Test validation with requirements.txt."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill
---
# Test
""")
        (skill_dir / "requirements.txt").write_text("requests>=2.28.0\nflask>=2.0.0\n")

        validator = DependencySecurityValidator(use_safety=False)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.stdout = '{"dependencies": []}'
        mock_result.error_message = None

        with (
            mock_tool_available("pip_audit", "/usr/bin/pip-audit"),
            patch.object(Tools.pip_audit, "run", return_value=mock_result),
        ):
            result = validator.validate(skill_dir)

        assert result.passed

    def test_validate_with_vulnerabilities(self, tmp_path: Path, mock_tool_available):
        """Test detection of vulnerable dependencies."""
        skill_dir = tmp_path / "vuln-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: vuln-skill
description: Vulnerable skill
---
# Test
""")
        (skill_dir / "requirements.txt").write_text("pyyaml==5.3\n")

        validator = DependencySecurityValidator(use_safety=False)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 1
        mock_result.error_message = None
        mock_result.stdout = """{"dependencies": [
            {
                "name": "pyyaml",
                "version": "5.3",
                "vulns": [
                    {
                        "id": "PYSEC-2020-123",
                        "fix_versions": ["5.4"]
                    }
                ]
            }
        ]}"""

        with (
            mock_tool_available("pip_audit", "/usr/bin/pip-audit"),
            patch.object(Tools.pip_audit, "run", return_value=mock_result),
        ):
            result = validator.validate(skill_dir)

        assert any("CVE" in e or "PYSEC" in e for e in result.errors + result.warnings)

    def test_no_tools_installed(self, tmp_path: Path, mock_tool_unavailable):
        """Test graceful handling when no audit tools installed."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test
---
# Test
""")
        (skill_dir / "requirements.txt").write_text("requests>=2.28.0\n")

        validator = DependencySecurityValidator()

        with mock_tool_unavailable("pip_audit"), mock_tool_unavailable("safety"):
            result = validator.validate(skill_dir)

        assert result.passed
        assert any("pip-audit not installed" in w for w in result.warnings)


# =============================================================================
# CODE RISK VALIDATOR TESTS
# =============================================================================


class TestCodeRiskValidator:
    """Tests for CodeRiskValidator (Bandit + Semgrep)."""

    def test_init_default(self):
        """Test default initialization."""
        validator = CodeRiskValidator()
        assert validator.name == "Code Risk Analysis"
        assert validator.use_semgrep is True
        assert validator.exclude_tests is True
        assert validator.fail_on_low is False

    def test_init_custom(self):
        """Test custom initialization."""
        validator = CodeRiskValidator(use_semgrep=False, exclude_tests=False, fail_on_low=True)
        assert validator.use_semgrep is False
        assert validator.exclude_tests is False
        assert validator.fail_on_low is True

    def test_validate_no_code_files(self, sample_skill_dir: Path):
        """Test validation when no code files exist."""
        validator = CodeRiskValidator()
        result = validator.validate(sample_skill_dir)

        assert result.passed
        assert any("No code files found" in m for m in result.messages)

    def test_validate_no_tools_installed(self, skill_with_security_issues: Path, mock_tool_unavailable):
        """Test graceful handling when no analysis tools installed."""
        validator = CodeRiskValidator()

        with mock_tool_unavailable("bandit"), mock_tool_unavailable("semgrep"):
            result = validator.validate(skill_with_security_issues)

        assert result.status == "incomplete"
        assert not result.passed
        assert result.metadata["incomplete_scans"] == ["bandit", "semgrep"]
        assert any("bandit not installed" in w for w in result.warnings)

    def test_validate_bandit_clean(self, skill_with_security_issues: Path, mock_tool_available):
        """Test Bandit analysis with no findings."""
        validator = CodeRiskValidator(use_semgrep=False)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.stdout = '{"results": [], "metrics": {}}'
        mock_result.error_message = None

        with (
            mock_tool_available("bandit", "/usr/bin/bandit"),
            patch.object(Tools.bandit, "run", return_value=mock_result),
        ):
            result = validator.validate(skill_with_security_issues)

        assert result.passed

    def test_validate_bandit_findings(self, skill_with_security_issues: Path, mock_tool_available):
        """Test Bandit analysis with security findings."""
        validator = CodeRiskValidator(use_semgrep=False)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 1
        mock_result.error_message = None
        mock_result.stdout = """{
            "results": [
                {
                    "test_id": "B102",
                    "test_name": "exec_used",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "issue_text": "Use of exec detected.",
                    "filename": "script.py",
                    "line_number": 12,
                    "issue_cwe": {"id": 78}
                }
            ],
            "metrics": {}
        }"""

        with (
            mock_tool_available("bandit", "/usr/bin/bandit"),
            patch.object(Tools.bandit, "run", return_value=mock_result),
        ):
            result = validator.validate(skill_with_security_issues)

        assert not result.passed
        assert any("BANDIT" in e for e in result.errors)

    def test_validate_bandit_crash_is_not_a_clean_pass(self, skill_with_security_issues: Path, mock_tool_available):
        """A bandit usage error (exit 2, empty stdout) must warn, not report a clean scan."""
        validator = CodeRiskValidator(use_semgrep=False)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 2
        mock_result.stdout = ""
        mock_result.stderr = "usage: bandit [-h] [-r] ... error: unrecognized arguments"
        mock_result.error_message = None

        with (
            mock_tool_available("bandit", "/usr/bin/bandit"),
            patch.object(Tools.bandit, "run", return_value=mock_result),
        ):
            result = validator.validate(skill_with_security_issues)

        assert not any("No security issues found" in m for m in result.messages)
        assert any("Bandit" in w and "exit code 2" in w for w in result.warnings)

    def test_semgrep_default_command_uses_packaged_offline_rules(
        self, skill_with_security_issues: Path, mock_tool_available
    ):
        """The default Semgrep config must work while metrics remain disabled."""
        validator = CodeRiskValidator()

        semgrep_result = MagicMock()
        semgrep_result.success = True
        semgrep_result.exit_code = 0
        semgrep_result.stdout = '{"results": [], "errors": []}'
        semgrep_result.stderr = ""
        semgrep_result.error_message = None

        with (
            mock_tool_available("semgrep", "/usr/bin/semgrep"),
            patch.object(Tools.semgrep, "run", return_value=semgrep_result) as semgrep_run,
        ):
            validator._run_semgrep(skill_with_security_issues)

        args = semgrep_run.call_args.args[0]
        config_path = Path(args[args.index("--config") + 1])
        assert config_path.name == "semgrep_rules.yaml"
        assert config_path.is_file()
        assert not str(config_path).startswith(("p/", "r/"))
        assert args[args.index("--metrics") + 1] == "off"
        assert "--disable-version-check" in args

    def test_validate_semgrep_timeout_is_not_a_clean_pass(self, skill_with_security_issues: Path, mock_tool_available):
        """A Semgrep execution failure must fail validation."""
        validator = CodeRiskValidator()

        bandit_result = MagicMock()
        bandit_result.success = True
        bandit_result.exit_code = 0
        bandit_result.stdout = '{"results": [], "metrics": {}}'
        bandit_result.error_message = None

        semgrep_result = MagicMock()
        semgrep_result.success = False
        semgrep_result.exit_code = -1
        semgrep_result.stdout = ""
        semgrep_result.stderr = ""
        semgrep_result.error_message = "Semgrep timed out after 180 seconds"

        with (
            mock_tool_available("bandit", "/usr/bin/bandit"),
            mock_tool_available("semgrep", "/usr/bin/semgrep"),
            patch.object(Tools.bandit, "run", return_value=bandit_result),
            patch.object(Tools.semgrep, "run", return_value=semgrep_result),
        ):
            result = validator.validate(skill_with_security_issues)

        assert not result.passed
        assert "Semgrep timed out after 180 seconds" in result.errors

    def test_validate_semgrep_crash_is_not_a_clean_pass(self, skill_with_security_issues: Path, mock_tool_available):
        """A Semgrep fatal error must fail validation, not report a clean scan."""
        validator = CodeRiskValidator()

        bandit_result = MagicMock()
        bandit_result.success = True
        bandit_result.exit_code = 0
        bandit_result.stdout = '{"results": [], "metrics": {}}'
        bandit_result.error_message = None

        semgrep_result = MagicMock()
        semgrep_result.success = True
        semgrep_result.exit_code = 2
        semgrep_result.stdout = ""
        semgrep_result.stderr = ""
        semgrep_result.error_message = None

        with (
            mock_tool_available("bandit", "/usr/bin/bandit"),
            mock_tool_available("semgrep", "/usr/bin/semgrep"),
            patch.object(Tools.bandit, "run", return_value=bandit_result),
            patch.object(Tools.semgrep, "run", return_value=semgrep_result),
        ):
            result = validator.validate(skill_with_security_issues)

        assert not result.passed
        assert not any("Semgrep: No" in m for m in result.messages)
        assert any("Semgrep" in e and "exit code 2" in e for e in result.errors)
        assert any("no diagnostic output" in e for e in result.errors)

    @pytest.mark.parametrize(
        ("output", "expected_error"),
        (
            pytest.param("", "did not return valid json", id="empty-output"),
            pytest.param("not json", "did not return valid json", id="malformed-json"),
            pytest.param("{}", "missing required 'results' list", id="empty-object"),
            pytest.param(
                json.dumps({"results": {}, "errors": []}),
                "field 'results' must be a list",
                id="non-list-results",
            ),
            pytest.param(
                json.dumps({"results": []}),
                "missing required 'errors' list",
                id="missing-errors",
            ),
            pytest.param(
                json.dumps({"results": [], "errors": {}}),
                "field 'errors' must be a list",
                id="non-list-errors",
            ),
            pytest.param(
                json.dumps({"results": [None], "errors": []}),
                "entries must be objects",
                id="invalid-result-entry",
            ),
        ),
    )
    def test_semgrep_rejects_untrustworthy_json_reports(self, output: str, expected_error: str) -> None:
        result = ValidationResult()

        CodeRiskValidator()._process_semgrep_output(output, result)

        assert not result.passed
        assert any(expected_error in error.lower() for error in result.errors)
        assert not any("Semgrep: No" in message for message in result.messages)

    def test_semgrep_errors_without_findings_fail_validation(self) -> None:
        result = ValidationResult()
        output = json.dumps(
            {
                "results": [],
                "errors": [{"message": "invalid security-audit configuration"}],
            }
        )

        CodeRiskValidator()._process_semgrep_output(output, result)

        assert not result.passed
        assert any("invalid security-audit configuration" in error for error in result.errors)
        assert not any("Semgrep: No" in message for message in result.messages)

    def test_semgrep_errors_and_findings_are_both_reported(self) -> None:
        result = ValidationResult()
        output = json.dumps(
            {
                "results": [
                    {
                        "check_id": "python.lang.security.audit.dangerous-system-call",
                        "path": "script.py",
                        "start": {"line": 10},
                        "extra": {
                            "severity": "ERROR",
                            "message": "Dangerous system call detected",
                            "metadata": {"cwe": ["CWE-78"]},
                        },
                    }
                ],
                "errors": [{"message": "one file could not be parsed"}],
            }
        )

        CodeRiskValidator()._process_semgrep_output(output, result)

        assert not result.passed
        assert any("one file could not be parsed" in error for error in result.errors)
        assert any(
            finding.check_name == "python.lang.security.audit.dangerous-system-call" for finding in result.findings
        )

    def test_semgrep_accepts_valid_clean_report(self) -> None:
        result = ValidationResult()

        CodeRiskValidator()._process_semgrep_output('{"results": [], "errors": []}', result)

        assert result.passed
        assert not result.errors
        assert result.messages == ["Semgrep: No security issues found"]

    def test_validate_semgrep_findings(self, skill_with_security_issues: Path, mock_tool_available):
        """Test Semgrep analysis with findings."""
        validator = CodeRiskValidator()

        bandit_result = MagicMock()
        bandit_result.success = True
        bandit_result.exit_code = 0
        bandit_result.stdout = '{"results": [], "metrics": {}}'
        bandit_result.error_message = None

        semgrep_result = MagicMock()
        semgrep_result.success = True
        semgrep_result.exit_code = 0
        semgrep_result.error_message = None
        semgrep_result.stdout = """{
            "results": [
                {
                    "check_id": "python.lang.security.audit.dangerous-system-call",
                    "path": "script.py",
                    "start": {"line": 10},
                    "extra": {
                        "severity": "ERROR",
                        "message": "Dangerous system call detected",
                        "metadata": {"cwe": ["CWE-78"]}
                    }
                }
            ],
            "errors": []
        }"""

        with (
            mock_tool_available("bandit", "/usr/bin/bandit"),
            mock_tool_available("semgrep", "/usr/bin/semgrep"),
            patch.object(Tools.bandit, "run", return_value=bandit_result),
            patch.object(Tools.semgrep, "run", return_value=semgrep_result),
        ):
            result = validator.validate(skill_with_security_issues)

        all_findings = result.errors + result.warnings
        assert any("SEMGREP" in finding for finding in all_findings)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSecurityValidatorsIntegration:
    """Integration tests for security validators working together."""

    def test_all_validators_handle_missing_skill_md(self, tmp_path: Path):
        """Test all validators handle directories without SKILL.md."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        validators = [
            SecretsValidator(),
            DependencySecurityValidator(),
            CodeRiskValidator(),
        ]

        for validator in validators:
            result = validator.validate(empty_dir)
            # Should not crash
            assert isinstance(result.passed, bool)

    def test_validators_export_from_init(self):
        """Test that new validators are exported from __init__."""
        from skillevaluator.validators import (
            CodeRiskValidator,
            DependencySecurityValidator,
            SecretsValidator,
        )

        assert SecretsValidator is not None
        assert DependencySecurityValidator is not None
        assert CodeRiskValidator is not None

    def test_validators_have_consistent_interface(self):
        """Test all validators implement ValidatorBase interface."""
        validators = [
            SecretsValidator(),
            DependencySecurityValidator(),
            CodeRiskValidator(),
        ]

        for validator in validators:
            assert hasattr(validator, "name")
            assert hasattr(validator, "description")
            assert hasattr(validator, "validate")
            assert callable(validator.validate)


# =============================================================================
# TOOL RUNNER TESTS
# =============================================================================


class TestToolRunner:
    """Tests for tool_runner utilities."""

    def test_severity_is_error(self):
        """Test Severity.is_error() logic."""
        assert Severity.CRITICAL.is_error() is True
        assert Severity.HIGH.is_error() is True
        assert Severity.MEDIUM.is_error() is False
        assert Severity.MEDIUM.is_error(fail_on_medium=True) is True
        assert Severity.LOW.is_error() is False
        assert Severity.LOW.is_error(fail_on_low=True) is True

    def test_external_tool_availability(self):
        """Test ExternalTool availability detection."""
        import sys

        from skillevaluator.utils.tool_runner import ExternalTool

        # Non-existent tool
        fake_tool = ExternalTool("FakeTool", "this-tool-does-not-exist-xyz")
        assert fake_tool.is_available is False
        assert fake_tool.path is None

        # The interpreter running the test must be available on every platform.
        python_tool = ExternalTool("Python", sys.executable)
        assert python_tool.is_available is True
        assert python_tool.path is not None

    def test_tools_registry(self):
        """Test Tools registry has expected tools."""
        assert hasattr(Tools, "gitleaks")
        assert hasattr(Tools, "pip_audit")
        assert hasattr(Tools, "safety")
        assert hasattr(Tools, "bandit")
        assert hasattr(Tools, "semgrep")

    def test_validation_result_add_finding(self):
        """Test ValidationResult.add_finding() method."""
        from skillevaluator.validators.base import ValidationResult

        result = ValidationResult()

        # Test critical severity becomes error
        result.add_finding("TEST", Severity.CRITICAL, "Critical issue")
        assert not result.passed
        assert any("TEST-CRITICAL" in e for e in result.errors)

        # Test low severity becomes warning
        result2 = ValidationResult()
        result2.add_finding("TEST", Severity.LOW, "Minor issue")
        assert result2.passed
        assert any("TEST-LOW" in w for w in result2.warnings)


# =============================================================================
# STRUCTURED FINDINGS TESTS
# =============================================================================


class TestStructuredFindings:
    """Tests for the new Finding dataclass and structured finding support."""

    def test_finding_dataclass_creation(self):
        """Test that Finding dataclass can be created with all fields."""
        from skillevaluator.validators.base import Finding

        finding = Finding(
            category="PII",
            severity="HIGH",
            check_name="emails",
            message="Email address requires review",
            file_path="SKILL.md",
            line_number=7,
            line_content="  author: John <john@gmail.com>",
            suggestion="Use a placeholder like user@example.com",
        )

        from skillevaluator.models import Severity

        assert finding.category == "PII"
        assert finding.severity == Severity.HIGH
        assert finding.check_name == "emails"
        assert finding.message == "Email address requires review"
        assert finding.file_path == "SKILL.md"
        assert finding.line_number == 7
        assert finding.line_content == "  author: John <john@gmail.com>"
        assert finding.suggestion == "Use a placeholder like user@example.com"

    def test_finding_dataclass_optional_fields(self):
        """Test Finding dataclass with optional fields as None."""
        from skillevaluator.validators.base import Finding

        finding = Finding(
            category="SCHEMA",
            severity="MEDIUM",
            check_name="naming",
            message="Invalid naming convention",
            file_path="my-skill/",
        )

        assert finding.line_number is None
        assert finding.line_content is None
        assert finding.suggestion is None

    def test_add_structured_finding_as_error(self):
        """Test add_structured_finding() creates both Finding and legacy error."""
        from skillevaluator.validators.base import Finding, ValidationResult

        result = ValidationResult()

        finding = Finding(
            category="PII",
            severity="HIGH",
            check_name="emails",
            message="Email address requires review",
            file_path="SKILL.md",
            line_number=10,
            line_content="email: test@gmail.com",
            suggestion="Use a placeholder email",
        )

        result.add_structured_finding(finding, is_error=True)

        # Check structured finding was added
        assert len(result.findings) == 1
        assert result.findings[0] is finding

        # Check legacy error was created
        assert not result.passed
        assert len(result.errors) == 1
        assert "[PII-HIGH]" in result.errors[0]
        assert "SKILL.md:10" in result.errors[0]

    def test_add_structured_finding_as_warning(self):
        """Test add_structured_finding() with is_error=False creates warning."""
        from skillevaluator.validators.base import Finding, ValidationResult

        result = ValidationResult()

        finding = Finding(
            category="PII",
            severity="LOW",
            check_name="mac_addresses",
            message="MAC address detected",
            file_path="README.md",
            line_number=20,
        )

        result.add_structured_finding(finding, is_error=False)

        # Check it's a warning, not an error
        assert result.passed
        assert len(result.findings) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 0

    def test_validation_result_merge_includes_findings(self):
        """Test that merge() combines findings from both results."""
        from skillevaluator.validators.base import Finding, ValidationResult

        result1 = ValidationResult()
        result2 = ValidationResult()

        finding1 = Finding(
            category="PII",
            severity="HIGH",
            check_name="emails",
            message="Email 1",
            file_path="file1.md",
        )
        finding2 = Finding(
            category="PII",
            severity="HIGH",
            check_name="emails",
            message="Email 2",
            file_path="file2.md",
        )

        result1.add_structured_finding(finding1)
        result2.add_structured_finding(finding2)

        result1.merge(result2)

        assert len(result1.findings) == 2
        assert result1.findings[0].message == "Email 1"
        assert result1.findings[1].message == "Email 2"

    def test_validation_result_merge_with_prefix_for_findings(self):
        """Test merge_with_prefix() prefixes finding file paths."""
        from skillevaluator.validators.base import Finding, ValidationResult

        result1 = ValidationResult()
        result2 = ValidationResult()

        finding = Finding(
            category="PII",
            severity="HIGH",
            check_name="emails",
            message="Email found",
            file_path="SKILL.md",
            line_number=5,
        )
        result2.add_structured_finding(finding)

        result1.merge_with_prefix(result2, "my-skill")

        assert len(result1.findings) == 1
        assert "[my-skill]" in result1.findings[0].file_path

    def test_finding_exported_from_validators_init(self):
        """Test Finding is exported from validators __init__."""
        from skillevaluator.validators import Finding

        assert Finding is not None
        # Can create instance
        f = Finding(category="TEST", severity="LOW", check_name="test", message="Test", file_path="test.py")
        assert f.category == "TEST"


class TestPIIValidatorStructuredFindings:
    """Tests for PII validator creating structured findings."""

    def test_pii_scan_creates_structured_findings(self, tmp_path: Path):
        """Test that PII scan creates Finding objects with full context."""
        from skillevaluator.validators.security import SecurityValidator

        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill with PII
metadata:
  author: John Doe <john@gmail.com>
---
# Test Skill

Contact: user@personal-email.org
""")

        validator = SecurityValidator(use_llm=False)
        result = validator.validate_pii_only(skill_dir)

        # Should have found PII
        assert not result.passed
        assert len(result.findings) > 0

        # Check first finding has all context
        from skillevaluator.models import Severity

        finding = result.findings[0]
        assert finding.category == "PII"
        assert finding.severity in (Severity.HIGH, Severity.CRITICAL, Severity.MEDIUM, Severity.LOW)
        assert finding.check_name is not None
        assert finding.message is not None
        assert finding.file_path is not None
        assert finding.line_number is not None
        assert finding.line_content is not None  # Should have actual line
        assert finding.suggestion is not None  # Should have fix suggestion

    def test_pii_findings_include_suggestion(self, tmp_path: Path):
        """Test that PII findings include actionable suggestions."""
        from skillevaluator.validators.security import SecurityValidator

        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        # Create file with personal path
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test
---
# Test

File is at /Users/johndoe/Documents/project
""")

        validator = SecurityValidator(use_llm=False)
        result = validator.validate_pii_only(skill_dir)

        assert len(result.findings) > 0
        finding = result.findings[0]

        # Should have a suggestion for personal paths
        assert finding.suggestion is not None
        assert "HOME" in finding.suggestion or "generic" in finding.suggestion.lower()

    def test_backward_compatibility_errors_list(self, tmp_path: Path):
        """Test that legacy errors list is still populated for backward compatibility."""
        from skillevaluator.validators.security import SecurityValidator

        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Compatibility fixture for legacy error output
---
# Test

File is at /Users/johndoe/Documents/project
""")

        validator = SecurityValidator(use_llm=False)
        result = validator.validate_pii_only(skill_dir)

        # Both findings AND errors should be populated
        assert len(result.findings) > 0
        assert len(result.errors) > 0

        # Error string should contain standard format
        assert any("[PII-" in e for e in result.errors)
