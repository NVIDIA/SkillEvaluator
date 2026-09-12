# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency Security Validator using pip-audit and Safety.

Scans Python dependencies for known CVE vulnerabilities by querying
the PyPI Advisory Database, OSV, and PyUp.io Safety DB.
"""

from __future__ import annotations

import math
from pathlib import Path

from skillevaluator.utils.tool_runner import Severity, ToolResult, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators.base import ValidationResult, ValidatorBase

_PIP_AUDIT_COMPLETED_EXIT_CODES = frozenset({0, 1})
_PIP_AUDIT_SEVERITIES = frozenset(severity.value for severity in Severity)


class DependencySecurityValidator(ValidatorBase):
    """Scan dependencies for CVE vulnerabilities.

    Uses pip-audit (primary) and Safety (secondary) to detect known
    security issues in Python dependencies from requirements.txt,
    pyproject.toml, and setup.py files.
    """

    def __init__(self, use_safety: bool = True, fail_on_medium: bool = False):
        """Initialize dependency validator.

        Args:
            use_safety: Run Safety check for additional coverage
            fail_on_medium: Treat MEDIUM severity as errors
        """
        self.use_safety = use_safety
        self.fail_on_medium = fail_on_medium

    @property
    def name(self) -> str:
        return "Dependency Vulnerability Audit"

    @property
    def description(self) -> str:
        return "Scan dependencies for CVE vulnerabilities using pip-audit and Safety"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run dependency audit on skill(s) at path."""
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Auditing dependencies for",
        )

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Audit dependencies for a single skill directory."""
        result = ValidationResult()

        # Find dependency files
        dep_files = self._find_dependency_files(skill_path)
        if not dep_files["requirements"] and not dep_files["pyproject"]:
            result.add_success(
                check_name="dependency_file_discovery",
                message="No dependency files found - vulnerability audit is not applicable",
            )
            return result

        # Audit requirements.txt files
        for req_file in dep_files["requirements"]:
            result.merge(self._audit_requirements(req_file))

        # Audit pyproject.toml
        if dep_files["pyproject"]:
            result.merge(self._audit_pyproject(dep_files["pyproject"]))

        return result

    def _find_dependency_files(self, skill_path: Path) -> dict:
        """Locate dependency files in skill directory."""
        return {
            "requirements": list(skill_path.glob("requirements*.txt")),
            "pyproject": skill_path / "pyproject.toml" if (skill_path / "pyproject.toml").exists() else None,
        }

    def _audit_requirements(self, req_file: Path) -> ValidationResult:
        """Audit a requirements.txt file."""
        result = ValidationResult()
        result.add_message(f"Auditing {req_file.name}")

        result.merge(self._run_pip_audit_on_file(req_file))

        if self.use_safety and Tools.safety.is_available:
            result.merge(self._run_safety(req_file))

        return result

    def _audit_pyproject(self, pyproject: Path) -> ValidationResult:
        """Audit pyproject.toml dependencies via local environment."""
        result = ValidationResult()
        result.add_message(f"Auditing {pyproject.name}")

        if not Tools.pip_audit.is_available:
            result.add_warning(f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}")
            result.mark_scan_incomplete("pip-audit")
            return result

        tool_result = Tools.pip_audit.run(
            ["--local", "--format", "json", "--progress-spinner", "off"],
            cwd=pyproject.parent,
            timeout=180,
        )

        self._process_pip_audit_result(tool_result, result, pyproject.name)

        return result

    def _run_pip_audit_on_file(self, req_file: Path) -> ValidationResult:
        """Run pip-audit on a requirements file."""
        result = ValidationResult()

        if not Tools.pip_audit.is_available:
            result.add_warning(f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}")
            result.mark_scan_incomplete("pip-audit")
            return result

        tool_result = Tools.pip_audit.run(
            ["-r", str(req_file), "--format", "json", "--progress-spinner", "off"],
            timeout=180,
        )

        self._process_pip_audit_result(tool_result, result, req_file.name)

        return result

    def _process_pip_audit_result(
        self,
        tool_result: ToolResult,
        result: ValidationResult,
        source: str,
    ) -> None:
        """Accept a pip-audit run only when its process state can be trusted."""
        if tool_result.success is not True or tool_result.error_message:
            detail = tool_result.error_message or "pip-audit did not complete"
            result.add_warning(f"{source}: {detail}")
            result.mark_scan_incomplete("pip-audit")
            return

        exit_code = tool_result.exit_code
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code not in _PIP_AUDIT_COMPLETED_EXIT_CODES
        ):
            result.add_warning(f"{source}: pip-audit exited with unexpected code {exit_code}; scan did not complete")
            result.mark_scan_incomplete("pip-audit")
            return

        self._process_pip_audit(tool_result.stdout, result, source, exit_code=exit_code)

    def _process_pip_audit(
        self,
        output: str,
        result: ValidationResult,
        source: str,
        *,
        exit_code: int,
    ) -> None:
        """Parse a completed pip-audit process and report trustworthy evidence."""
        report = self._validated_pip_audit_report(output)
        if report is None:
            result.add_warning(f"{source}: pip-audit returned a malformed JSON report; scan did not complete")
            result.mark_scan_incomplete("pip-audit")
            return

        dependencies, skipped_count = report

        vuln_count = 0
        for dep in dependencies:
            pkg_name = dep["name"]
            pkg_version = dep["version"]

            for vuln in dep["vulns"]:
                vuln_count += 1
                self._report_vulnerability(
                    result,
                    pkg_name=pkg_name,
                    pkg_version=pkg_version,
                    vuln_id=vuln["id"],
                    fix_versions=vuln["fix_versions"],
                    severity=self._get_vuln_severity(vuln),
                )

        if skipped_count:
            result.add_warning(
                f"{source}: pip-audit skipped {skipped_count} dependency(ies); scan coverage is incomplete"
            )
            result.mark_scan_incomplete("pip-audit")
            return

        expected_exit_code = 1 if vuln_count else 0
        if exit_code != expected_exit_code:
            result.add_warning(
                f"{source}: pip-audit exit code {exit_code} contradicts its JSON report; scan did not complete"
            )
            result.mark_scan_incomplete("pip-audit")
            return

        status = f"Found {vuln_count} vulnerability(ies)" if vuln_count else "No vulnerabilities found"
        result.add_success(
            check_name="pip_audit",
            message=f"{source}: {status} (pip-audit)",
            source=source,
            vulnerability_count=vuln_count,
        )

    @staticmethod
    def _validated_pip_audit_report(output: str) -> tuple[list[dict], int] | None:
        """Return resolved dependencies and skipped count for canonical pip-audit JSON."""
        try:
            data = parse_json_output(output)
        except (RecursionError, ValueError):
            return None
        if not isinstance(data, dict):
            return None

        dependencies = data.get("dependencies")
        if not isinstance(dependencies, list):
            return None
        if "fixes" in data and not isinstance(data["fixes"], list):
            return None

        resolved: list[dict] = []
        skipped_count = 0
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                return None

            name = dependency.get("name")
            if not isinstance(name, str) or not name.strip():
                return None

            if "skip_reason" in dependency:
                skip_reason = dependency["skip_reason"]
                if not isinstance(skip_reason, str) or not skip_reason.strip():
                    return None
                skipped_count += 1
                continue

            version = dependency.get("version")
            vulnerabilities = dependency.get("vulns")
            if not isinstance(version, str) or not version.strip() or not isinstance(vulnerabilities, list):
                return None

            for vulnerability in vulnerabilities:
                if not DependencySecurityValidator._valid_pip_audit_vulnerability(vulnerability):
                    return None
            resolved.append(dependency)

        return resolved, skipped_count

    @staticmethod
    def _valid_pip_audit_vulnerability(vulnerability: object) -> bool:
        """Validate fields consumed from one pip-audit vulnerability entry."""
        if not isinstance(vulnerability, dict):
            return False

        vuln_id = vulnerability.get("id")
        fix_versions = vulnerability.get("fix_versions")
        if not isinstance(vuln_id, str) or not vuln_id.strip() or not isinstance(fix_versions, list):
            return False
        if any(not isinstance(version, str) or not version.strip() for version in fix_versions):
            return False

        severity = vulnerability.get("severity")
        if severity is not None and (
            not isinstance(severity, str) or severity.strip().casefold() not in _PIP_AUDIT_SEVERITIES
        ):
            return False
        aliases = vulnerability.get("aliases", [])
        if not isinstance(aliases, list):
            return False
        for alias in aliases:
            if isinstance(alias, str):
                if not alias.strip():
                    return False
                continue
            if not isinstance(alias, dict):
                return False
            cvss = alias.get("cvss")
            score = cvss.get("score") if isinstance(cvss, dict) else None
            if (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or (isinstance(score, float) and not math.isfinite(score))
                or not 0.0 <= score <= 10.0
            ):
                return False

        description = vulnerability.get("description")
        return description is None or isinstance(description, str)

    def _run_safety(self, req_file: Path) -> ValidationResult:
        """Run Safety check for supplementary coverage."""
        result = ValidationResult()

        tool_result = Tools.safety.run(
            ["check", "-r", str(req_file), "--output", "json"],
            timeout=60,
        )

        # Safety errors are non-critical (pip-audit is primary)
        if tool_result.success:
            self._process_safety(tool_result.stdout, result)

        return result

    def _process_safety(self, output: str, result: ValidationResult) -> None:
        """Parse Safety output and report additional findings."""
        data = parse_json_output(output)
        if not data:
            return

        # Handle varying Safety output formats
        vulnerabilities = data.get("vulnerabilities", data if isinstance(data, list) else [])

        for vuln in vulnerabilities:
            if not isinstance(vuln, dict):
                continue

            pkg_name = vuln.get("package_name", vuln.get("name", "unknown"))
            severity_str = vuln.get("severity", "medium").lower()
            severity = Severity(severity_str) if severity_str in Severity else Severity.MEDIUM

            vuln_id = vuln.get("vulnerability_id", vuln.get("id", "Unknown"))
            advisory = vuln.get("advisory", "")[:100]

            # Safety findings are supplementary - only critical as errors
            result.add_finding(
                tag="SAFETY",
                severity=severity,
                message=f"{pkg_name}: {vuln_id} - {advisory}",
                fail_on_medium=False,
            )

    def _report_vulnerability(
        self,
        result: ValidationResult,
        *,
        pkg_name: str,
        pkg_version: str,
        vuln_id: str,
        fix_versions: list[str],
        severity: Severity,
    ) -> None:
        """Report a single vulnerability finding."""
        fix_hint = f" -> upgrade to {fix_versions[0]}" if fix_versions else ""
        message = f"{pkg_name}=={pkg_version}: {vuln_id}{fix_hint}"

        result.add_finding(
            tag="CVE",
            severity=severity,
            message=message,
            fail_on_medium=self.fail_on_medium,
        )

    def _get_vuln_severity(self, vuln: dict) -> Severity:
        """Extract severity from vulnerability data."""
        # Explicit severity field
        if "severity" in vuln:
            sev = vuln["severity"].strip().casefold()
            try:
                return Severity(sev)
            except ValueError:
                pass

        # CVSS score from aliases
        for alias in vuln.get("aliases", []):
            if isinstance(alias, dict) and "cvss" in alias:
                score = alias.get("cvss", {}).get("score", 0)
                return cvss_to_severity(score)

        # Default based on ID prefix (PYSEC/GHSA are usually important)
        vuln_id = vuln.get("id", "")
        if vuln_id.startswith(("PYSEC", "GHSA")):
            return Severity.HIGH

        return Severity.MEDIUM
