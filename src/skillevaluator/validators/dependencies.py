# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency Security Validator using pip-audit and Safety.

Scans Python dependencies for known CVE vulnerabilities by querying
the PyPI Advisory Database, OSV, and PyUp.io Safety DB.
"""

from __future__ import annotations

import json
import tempfile
import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot
from skillevaluator.utils.tool_runner import Severity, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators.base import ValidationResult, ValidatorBase


class DependencySecurityValidator(ValidatorBase):
    """Scan dependencies for CVE vulnerabilities.

    Uses pip-audit (primary) and Safety (secondary) to detect known
    security issues in Python dependencies from requirements.txt,
    and static PEP 621 pyproject.toml declarations. Pyproject coverage is
    limited to directly declared exact versions; transitive packages require
    a separately supplied requirements file.
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
        coverages = []

        def audit_target(target: Path) -> ValidationResult:
            result = self._validate_single_skill(target)
            coverage = result.metadata.get("dependency_coverage")
            if isinstance(coverage, dict):
                relative_target = target.relative_to(skill_path).as_posix()
                coverages.append({"target": relative_target, **coverage})
                result.add_message(
                    f"{relative_target}/pyproject.toml coverage: "
                    f"{len(coverage['requested_pins'])} requested pin(s), "
                    f"{len(coverage['audited_pins'])} audited pin(s); "
                    "direct declarations only, transitive dependencies not audited"
                )
            return result

        result = self._validate_folder_or_skill(
            skill_path,
            audit_target,
            action_description="Auditing dependencies for",
        )
        if coverages:
            result.metadata["dependency_coverage"] = coverages
        return result

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
            "pyproject": skill_path / "pyproject.toml"
            if (skill_path / "pyproject.toml").exists() or (skill_path / "pyproject.toml").is_symlink()
            else None,
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
        """Audit declared pins without installing the target or running build code."""
        result = ValidationResult()
        result.metadata["dependency_coverage"] = {
            "source": "pyproject.toml",
            "scope": "direct declared pins (all optional groups), scanner-platform markers",
            "declarations": None,
            "requested_pins": [],
            "audited_pins": [],
            "transitive_dependencies": "not audited",
        }
        result.add_message(
            "Auditing pyproject.toml: direct declared pins, including optional groups; no transitive audit"
        )

        def incomplete(message: str) -> None:
            result.mark_scan_incomplete("pip-audit")
            result.add_warning(f"pyproject.toml: {message}")

        try:
            with SecureRoot(pyproject.parent) as root:
                raw, _ = root.read_bytes(Path(pyproject.name), 1024 * 1024)
            data = tomllib.loads(raw.decode("utf-8"))
        except (OSError, SecurePathError, UnicodeError, tomllib.TOMLDecodeError):
            incomplete("cannot read a bounded regular TOML file")
            return result

        project = data.get("project")
        if not isinstance(project, dict):
            incomplete("no static PEP 621 project metadata; supply pinned requirements.txt")
            return result
        dynamic = project.get("dynamic", [])
        if not isinstance(dynamic, list) or any(name in dynamic for name in ("dependencies", "optional-dependencies")):
            incomplete("dynamic dependency metadata is unsupported; supply pinned requirements.txt")
        tool = data.get("tool", {})
        if isinstance(tool, dict) and "poetry" in tool:
            incomplete("Poetry dependency metadata is not audited; supply pinned requirements.txt")

        groups = {"dependencies": project.get("dependencies", [])}
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            incomplete("optional-dependencies must be a table of requirement lists")
        else:
            groups.update({f"optional group {name}": values for name, values in optional.items()})
        pins: set[str] = set()
        declaration_count = 0
        for group, declarations in groups.items():
            if not isinstance(declarations, list):
                incomplete(f"{group} must be a list")
                continue
            for index, declaration in enumerate(declarations):
                declaration_count += 1
                if declaration_count > 4096:
                    incomplete("dependency declaration limit (4096) exceeded")
                    return result
                # Never include arbitrary declaration text (e.g. URL credentials) in diagnostics.
                label = f"{group}, declaration {index + 1}"
                try:
                    if not isinstance(declaration, str) or "\n" in declaration or "\r" in declaration:
                        raise InvalidRequirement("invalid declaration")
                    requirement = Requirement(declaration)
                except InvalidRequirement:
                    incomplete(f"{label}: invalid requirement")
                    continue
                if requirement.url:
                    incomplete(f"{label}: URL/path dependencies are unsupported")
                    continue
                if requirement.marker:
                    # Audit every optional group, evaluating `extra` in its group context.
                    extra = group.removeprefix("optional group ") if group.startswith("optional group ") else ""
                    if not requirement.marker.evaluate({"extra": extra}):
                        result.add_message(f"pyproject.toml: {label} skipped (marker does not match scanner platform)")
                        continue
                specifiers = list(requirement.specifier)
                if len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
                    incomplete(f"{label}: an exact version pin is required; supply resolved requirements.txt")
                    continue
                pins.add(f"{requirement.name}=={specifiers[0].version}")

        result.metadata["dependency_coverage"].update(
            {"declarations": declaration_count, "requested_pins": sorted(pins)}
        )
        if not pins:
            result.add_message("pyproject.toml: no applicable exact pins to audit")
            return result

        if not Tools.pip_audit.is_available:
            incomplete(f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}")
            return result

        with tempfile.TemporaryDirectory(prefix="skillevaluator-dependencies-") as temporary_dir:
            requirements = Path(temporary_dir) / "requirements.txt"
            requirements.write_text("\n".join(sorted(pins)) + "\n", encoding="utf-8")
            tool_result = Tools.pip_audit.run(
                [
                    "-r",
                    str(requirements),
                    "--no-deps",
                    "--disable-pip",
                    "--format",
                    "json",
                    "--progress-spinner",
                    "off",
                ],
                timeout=180,
            )

        if tool_result.error_message or tool_result.exit_code not in (0, 1):
            incomplete("pip-audit did not complete successfully")
            return result
        try:
            audit = json.loads(tool_result.stdout)
            dependencies = audit if isinstance(audit, list) else audit["dependencies"]
            if not isinstance(dependencies, list):
                raise ValueError("invalid inventory")
            returned_pins = set()
            for dep in dependencies:
                if not isinstance(dep, dict) or not isinstance(dep.get("vulns"), list):
                    raise ValueError("invalid dependency")
                returned_pins.add((canonicalize_name(dep["name"]), Version(dep["version"])))
                if any(
                    not isinstance(vuln, dict)
                    or not isinstance(vuln.get("id"), str)
                    or not isinstance(vuln.get("fix_versions", []), list)
                    or any(not isinstance(version, str) for version in vuln.get("fix_versions", []))
                    or ("severity" in vuln and not isinstance(vuln["severity"], str))
                    or not isinstance(vuln.get("aliases", []), list)
                    or any(not isinstance(alias, str) for alias in vuln.get("aliases", []))
                    for vuln in dep["vulns"]
                ):
                    raise ValueError("invalid advisory")
            expected_pins = {
                (canonicalize_name(requirement.name), Version(next(iter(requirement.specifier)).version))
                for pin in pins
                for requirement in [Requirement(pin)]
            }
            if returned_pins != expected_pins:
                raise ValueError("incomplete inventory")
            if tool_result.exit_code == 1 and not any(dep["vulns"] for dep in dependencies):
                raise ValueError("unexpected finding exit")
        except (KeyError, TypeError, ValueError, RecursionError):
            incomplete("pip-audit returned incomplete or invalid target evidence")
            return result
        self._process_pip_audit(tool_result.stdout, result, pyproject.name)
        result.metadata["dependency_coverage"]["audited_pins"] = sorted(pins)

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
