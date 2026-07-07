# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency Security Validator using pip-audit and Safety.

Scans Python dependencies for known CVE vulnerabilities by querying
the PyPI Advisory Database, OSV, and PyUp.io Safety DB.
"""

from __future__ import annotations

from pathlib import Path

from skillevaluator.utils.tool_runner import Severity, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators.base import ValidationResult, ValidatorBase


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
            result.add_message("No dependency files found - skipping vulnerability audit")
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
            return result

        tool_result = Tools.pip_audit.run(
            ["--local", "--format", "json", "--progress-spinner", "off"],
            cwd=pyproject.parent,
            timeout=180,
        )

        if tool_result.error_message:
            result.add_warning(tool_result.error_message)
        else:
            self._process_pip_audit(tool_result.stdout, result, pyproject.name)

        return result

    def _run_pip_audit_on_file(self, req_file: Path) -> ValidationResult:
        """Run pip-audit on a requirements file."""
        result = ValidationResult()

        if not Tools.pip_audit.is_available:
            result.add_warning(f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}")
            return result

        tool_result = Tools.pip_audit.run(
            ["-r", str(req_file), "--format", "json", "--progress-spinner", "off"],
            timeout=180,
        )

        if tool_result.error_message:
            result.add_warning(f"{req_file.name}: {tool_result.error_message}")
        else:
            self._process_pip_audit(tool_result.stdout, result, req_file.name)

        return result

    def _process_pip_audit(self, output: str, result: ValidationResult, source: str) -> None:
        """Parse pip-audit output and report vulnerabilities."""
        data = parse_json_output(output, on_error="No known vulnerabilities found")
        if data is None:
            return

        # Handle both list and dict output formats
        dependencies = data if isinstance(data, list) else data.get("dependencies", [])

        vuln_count = 0
        for dep in dependencies:
            if not isinstance(dep, dict):
                continue

            pkg_name = dep.get("name", "unknown")
            pkg_version = dep.get("version", "unknown")

            for vuln in dep.get("vulns", []):
                vuln_count += 1
                self._report_vulnerability(
                    result,
                    pkg_name=pkg_name,
                    pkg_version=pkg_version,
                    vuln_id=vuln.get("id", "Unknown"),
                    fix_versions=vuln.get("fix_versions", []),
                    severity=self._get_vuln_severity(vuln),
                )

        status = f"Found {vuln_count} vulnerability(ies)" if vuln_count else "No vulnerabilities found"
        result.add_message(f"{source}: {status} (pip-audit)")

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
            sev = vuln["severity"].lower()
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
