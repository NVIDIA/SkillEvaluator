# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code Risk Analysis Validator using Bandit and Semgrep.

Performs static code analysis to detect security vulnerabilities,
code quality issues, and potential bugs via CWE/OWASP patterns.
"""

from __future__ import annotations

from pathlib import Path

from skillevaluator.config import CONFIG_DIR
from skillevaluator.constants import SCAN_EXCLUDED_DIRS, SCAN_EXCLUDED_FILES
from skillevaluator.utils.tool_runner import Severity, Tools, parse_json_output
from skillevaluator.validators.base import Finding, ValidationResult, ValidatorBase, iter_scannable_files


def _semgrep_file_excludes() -> list[str]:
    """Return generated file patterns for Semgrep's case-sensitive excludes."""
    excludes = set(SCAN_EXCLUDED_FILES)
    for name in SCAN_EXCLUDED_FILES:
        path = Path(name)
        if path.suffix:
            excludes.add(f"{path.stem.upper()}{path.suffix}")
    return sorted(excludes)


class CodeRiskValidator(ValidatorBase):
    """Static code analysis for security risks.

    Combines Bandit's Python checks with a packaged, high-confidence Semgrep
    baseline for command/dynamic-code execution, Python SQL interpolation,
    and download-to-shell patterns.
    """

    def __init__(
        self,
        use_semgrep: bool = True,
        fail_on_low: bool = False,
        exclude_tests: bool = True,
    ):
        """Initialize code risk validator.

        Args:
            use_semgrep: Run Semgrep for additional coverage
            fail_on_low: Treat LOW severity as errors
            exclude_tests: Skip test files and directories
        """
        self.use_semgrep = use_semgrep
        self.fail_on_low = fail_on_low
        self.exclude_tests = exclude_tests

    @property
    def name(self) -> str:
        return "Code Risk Analysis"

    @property
    def description(self) -> str:
        return "Static code analysis using Bandit and packaged Semgrep rules"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run code risk analysis on skill(s) at path."""
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Analyzing code risk for",
        )

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run static analysis on a single skill directory."""
        result = ValidationResult()

        file_counts = self._count_code_files(skill_path)
        if not any(file_counts.values()):
            result.add_message("No code files found - skipping code risk analysis")
            return result

        result.add_message(
            f"Found {file_counts['py']} Python, {file_counts['sh']} Shell, "
            f"{file_counts['js']} JavaScript/TypeScript files"
        )

        if file_counts["py"]:
            result.merge(self._run_bandit(skill_path))

        if self.use_semgrep:
            result.merge(self._run_semgrep(skill_path))

        return result

    def _count_code_files(self, skill_path: Path) -> dict[str, int]:
        """Count code files by type.

        Files under Tier 1 artifact directories (``evals/``, ``results/``,
        ``versions/`` and dot-prefixed variants) are excluded so that
        evaluation snapshots do not skew the counts that decide whether
        bandit / semgrep should run.
        """
        return {
            "py": len(iter_scannable_files(skill_path, {".py"})),
            "sh": len(iter_scannable_files(skill_path, {".sh"})),
            "js": len(
                iter_scannable_files(
                    skill_path,
                    {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"},
                )
            ),
        }

    def _run_bandit(self, skill_path: Path) -> ValidationResult:
        """Run Bandit security linter on Python code."""
        result = ValidationResult()

        if not Tools.bandit.is_available:
            result.add_warning(f"bandit not installed. {Tools.bandit.get_install_hint()}")
            result.mark_scan_incomplete("bandit")
            return result

        # Tier 1 artifact dirs (evals/, results/, versions/, ...) hold
        # snapshot copies of the live skill; scanning them produces
        # duplicate findings. Bandit takes a single comma-separated
        # ``--exclude`` arg that combines the test exclusion (when on)
        # with the artifact exclusion list.
        excludes: list[str] = []
        if self.exclude_tests:
            excludes.extend(["tests", "test", "*_test.py", "test_*.py"])
        excludes.extend(sorted(SCAN_EXCLUDED_DIRS))

        # resolve() so a skill dir named like an option token (e.g.
        # "--exclude=*") cannot be parsed as a scanner flag.
        args = ["-r", str(skill_path.resolve()), "-f", "json", "-q", "-ll"]
        args.extend(["--exclude", ",".join(excludes)])

        tool_result = Tools.bandit.run(args, timeout=120)

        if tool_result.error_message:
            result.add_warning(tool_result.error_message)
            result.mark_scan_incomplete("bandit")
        elif tool_result.exit_code in (0, 1):
            # Exit code 0 = clean, 1 = findings; both emit JSON on stdout
            self._process_bandit_output(tool_result.stdout, result, exit_code=tool_result.exit_code)
        else:
            error_msg = tool_result.stderr or tool_result.stdout
            result.add_warning(f"Bandit error (exit code {tool_result.exit_code}): {error_msg[:200]}")
            result.mark_scan_incomplete("bandit")

        return result

    def _process_bandit_output(
        self,
        output: str,
        result: ValidationResult,
        *,
        exit_code: int | None = None,
    ) -> None:
        """Parse Bandit JSON output and report issues."""
        data = parse_json_output(output)
        if not isinstance(data, dict):
            result.add_error("Bandit did not return valid JSON; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return

        issues = data.get("results")
        if not isinstance(issues, list) or not all(isinstance(issue, dict) for issue in issues):
            result.add_error("Bandit JSON report is missing a valid 'results' list; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return

        errors = data.get("errors", [])
        if not isinstance(errors, list):
            result.add_error("Bandit JSON field 'errors' must be a list; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return
        if errors:
            for error in errors[:3]:
                detail = error.get("reason") or error.get("message") if isinstance(error, dict) else str(error)
                result.add_error(f"Bandit scan error: {str(detail or 'Unknown error')[:200]}")
            result.mark_scan_incomplete("bandit")

        if exit_code == 1 and not issues:
            result.add_error("Bandit findings exit code 1 did not include findings; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return
        if exit_code == 0 and issues:
            result.add_error("Bandit returned findings with clean exit code 0; scanner status is inconsistent")
            result.mark_scan_incomplete("bandit")

        if not issues:
            if not errors:
                result.add_message("Bandit: No security issues found")
            return

        # Summarize by severity
        counts = self._count_by_severity(issues)
        result.add_message(f"Bandit found: {counts['HIGH']} HIGH, {counts['MEDIUM']} MEDIUM, {counts['LOW']} LOW")

        for issue in issues:
            self._report_bandit_issue(result, issue)

    def _count_by_severity(self, issues: list[dict]) -> dict[str, int]:
        """Count issues by severity level."""
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in issues:
            sev = issue.get("issue_severity", "LOW").upper()
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _report_bandit_issue(self, result: ValidationResult, issue: dict) -> None:
        """Report a single Bandit finding."""
        severity_str = issue.get("issue_severity", "MEDIUM").upper()
        confidence = issue.get("issue_confidence", "MEDIUM").upper()
        severity = self._map_bandit_severity(severity_str)

        test_id = issue.get("test_id", "B000")
        test_name = issue.get("test_name", "unknown")
        issue_text = issue.get("issue_text", "Security issue detected")
        filename = issue.get("filename", "unknown")
        line = issue.get("line_number", 0)
        cwe_id = issue.get("issue_cwe", {}).get("id", "")

        cwe_ref = f" (CWE-{cwe_id})" if cwe_id else ""
        message = f"{issue_text}{cwe_ref}"

        # HIGH severity or MEDIUM with decent confidence are errors
        is_error = severity_str == "HIGH" or (severity_str == "MEDIUM" and confidence in ("HIGH", "MEDIUM"))

        result.add_structured_finding(
            Finding(
                category="BANDIT",
                severity=severity,
                check_name=f"{test_id}:{test_name}",
                message=message,
                file_path=filename,
                line_number=line,
                suggestion=f"Review {test_name} ({test_id}){cwe_ref}",
                metadata={"cwe": cwe_id} if cwe_id else {},
            ),
            is_error=is_error,
        )

    def _map_bandit_severity(self, severity_str: str) -> Severity:
        """Map Bandit severity string to Severity enum."""
        mapping = {
            "HIGH": Severity.HIGH,
            "MEDIUM": Severity.MEDIUM,
            "LOW": Severity.LOW,
        }
        return mapping.get(severity_str.upper(), Severity.MEDIUM)

    def _run_semgrep(self, skill_path: Path) -> ValidationResult:
        """Run Semgrep security scanner."""
        result = ValidationResult()

        if not Tools.semgrep.is_available:
            result.add_warning(f"semgrep not installed. {Tools.semgrep.get_install_hint()}")
            result.mark_scan_incomplete("semgrep")
            return result

        rules_path = (CONFIG_DIR / "semgrep_rules.yaml").resolve()
        args = [
            "scan",
            "--config",
            str(rules_path),
            "--json",
            "--quiet",
            "--metrics",
            "off",
            "--disable-version-check",
            "--no-rewrite-rule-ids",
            "--error",
        ]
        if self.exclude_tests:
            args.extend(["--exclude", "tests", "--exclude", "test"])
        for d in sorted(SCAN_EXCLUDED_DIRS):
            args.extend(["--exclude", d])
        for f in _semgrep_file_excludes():
            args.extend(["--exclude", f])
        args.append(str(skill_path.resolve()))

        tool_result = Tools.semgrep.run(args, timeout=180)

        if tool_result.error_message:
            result.add_error(tool_result.error_message)
            result.mark_scan_incomplete("semgrep")
        elif tool_result.exit_code in (0, 1):
            # Exit code 0 = clean, 1 = findings; both emit JSON on stdout
            self._process_semgrep_output(tool_result.stdout, result, exit_code=tool_result.exit_code)
        else:
            error_msg = tool_result.stderr or tool_result.stdout or "no diagnostic output"
            result.add_error(f"Semgrep error (exit code {tool_result.exit_code}): {error_msg[:200]}")
            result.mark_scan_incomplete("semgrep")

        return result

    def _process_semgrep_output(
        self,
        output: str,
        result: ValidationResult,
        *,
        exit_code: int | None = None,
    ) -> None:
        """Parse Semgrep JSON output and report findings."""
        data = parse_json_output(output)
        if data is None:
            result.add_error("Semgrep did not return valid JSON; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        if not isinstance(data, dict):
            result.add_error("Semgrep JSON output was not an object; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return

        if "results" not in data:
            result.add_error("Semgrep JSON report is missing required 'results' list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        findings = data["results"]
        if not isinstance(findings, list):
            result.add_error("Semgrep JSON field 'results' must be a list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        if not all(isinstance(finding, dict) for finding in findings):
            result.add_error("Semgrep JSON 'results' entries must be objects; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return

        if "errors" not in data:
            result.add_error("Semgrep JSON report is missing required 'errors' list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        scan_errors = data["errors"]
        if not isinstance(scan_errors, list):
            result.add_error("Semgrep JSON field 'errors' must be a list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        if not all(isinstance(error, dict) for error in scan_errors):
            result.add_error("Semgrep JSON 'errors' entries must be objects; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return

        # Report scan errors (limit to avoid noise)
        for error in scan_errors[:3]:
            message = str(error.get("message") or error.get("type") or "Unknown scan error")
            result.add_error(f"Semgrep scan error: {message[:100]}")
        if scan_errors:
            result.mark_scan_incomplete("semgrep")

        if exit_code == 1 and not findings:
            result.add_error("Semgrep findings exit code 1 did not include findings; scan did not complete")
            result.mark_scan_incomplete("semgrep")
        elif exit_code == 0 and findings:
            result.add_error("Semgrep returned findings with clean exit code 0; scanner status is inconsistent")
            result.mark_scan_incomplete("semgrep")

        if not findings:
            if not scan_errors:
                result.add_message("Semgrep: No security issues found")
            return

        result.add_message(f"Semgrep found {len(findings)} issue(s)")

        for finding in findings:
            self._report_semgrep_finding(result, finding)

    def _report_semgrep_finding(self, result: ValidationResult, finding: dict) -> None:
        """Report a single Semgrep finding."""
        rule_id = finding.get("check_id", "unknown")
        extra = finding.get("extra", {})

        severity_str = extra.get("severity", "WARNING").upper()
        severity = Severity.HIGH if severity_str == "ERROR" else Severity.MEDIUM

        message = extra.get("message", "Issue detected")[:100]
        path = finding.get("path", "unknown")
        line = finding.get("start", {}).get("line", 0)

        # Build reference string for CWE/OWASP
        metadata = extra.get("metadata", {})
        refs = self._build_reference_string(metadata)

        result.add_structured_finding(
            Finding(
                category="SEMGREP",
                severity=severity,
                check_name=rule_id,
                message=message,
                file_path=path,
                line_number=line,
                suggestion=refs.strip(" ()") if refs else None,
                metadata=metadata,
            ),
            is_error=severity == Severity.HIGH,
        )

    def _build_reference_string(self, metadata: dict) -> str:
        """Build CWE/OWASP reference string from metadata."""
        refs = []
        if cwe := metadata.get("cwe", []):
            refs.append(f"CWE: {', '.join(cwe[:2])}")
        if owasp := metadata.get("owasp", []):
            refs.append(f"OWASP: {', '.join(owasp[:2])}")
        return f" ({'; '.join(refs)})" if refs else ""
