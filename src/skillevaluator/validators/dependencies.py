# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency Security Validator using pip-audit and Safety.

Audits the Python dependencies a skill *declares* in ``requirements*.txt`` and
``pyproject.toml`` against the PyPI Advisory Database, OSV, and PyUp.io Safety
DB.

Advisory applicability is asserted only for exactly pinned declarations
(``name==version``). Those pins are copied into a normalized temporary
requirements file and audited with ``pip-audit --no-deps --disable-pip``, so the
audit never installs, builds, or resolves the declared packages and never sees
index options or other pip flags from the audited files. Floating, ranged, bare,
URL, and unparseable declarations cannot be matched to one released version;
each produces an informational ``dependency-version-unverified`` finding rather
than being silently dropped or audited against an unrelated environment.
"""

from __future__ import annotations

import re
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES
from skillevaluator.models.result import Finding
from skillevaluator.utils.secure_fs import SecurePathError, secure_read_path_text
from skillevaluator.utils.tool_runner import Severity, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators.base import ValidationResult, ValidatorBase

# Dependency manifests are read through a bounded, no-follow secure read.
MAX_DEPENDENCY_FILE_BYTES = CONTENT_DEDUP_MAX_FILE_BYTES
# Per-file cap on individual ``dependency-version-unverified`` findings; the
# remainder is summarized in one message so a huge manifest cannot flood reports.
MAX_UNVERIFIED_FINDINGS_PER_FILE = 100
UNVERIFIED_CHECK_NAME = "dependency-version-unverified"

# PEP 508 distribution name, optional extras, then the rest of the requirement.
_REQUIREMENT_RE = re.compile(
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[(?P<extras>[^\]]*)\])?\s*(?P<rest>.*)",
    re.DOTALL,
)
_SPECIFIER_RE = re.compile(r"\s*(?P<op>~=|===|==|!=|<=|>=|<|>)\s*(?P<version>[^\s,;]+)\s*")
# Conservative PEP 440 public/local version (no wildcards). Anything else is
# treated as unverifiable rather than handed to pip-audit.
_EXACT_VERSION_RE = re.compile(
    r"(?:\d+!)?\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
    re.IGNORECASE,
)
# Trailing per-requirement pip options such as ``--hash=sha256:...``.
_TRAILING_OPTIONS_RE = re.compile(r"\s+--?[A-Za-z]")


@dataclass(frozen=True)
class DependencyDeclaration:
    """One direct dependency declaration and the evidence available for it."""

    raw: str
    name: str | None
    line_number: int | None
    role: str
    exact_version: str | None

    @property
    def audit_line(self) -> str | None:
        """Normalized ``name==version`` line, or ``None`` when not exactly pinned."""
        if self.name is None or self.exact_version is None:
            return None
        return f"{self.name}=={self.exact_version}"


def canonicalize_package_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_dependency_declaration(raw: str, *, line_number: int | None, role: str) -> DependencyDeclaration:
    """Parse one PEP 508 requirement string without installing or resolving it.

    Only a single ``==`` specifier with a plain PEP 440 version counts as an
    exact pin. Direct URL references, wildcards, ``===`` arbitrary equality,
    ranges, and malformed strings are returned with ``exact_version=None``.
    """
    text = raw.strip()
    match = _REQUIREMENT_RE.fullmatch(text)
    if match is None:
        return DependencyDeclaration(text, None, line_number, role, None)
    name = canonicalize_package_name(match.group("name"))
    rest = match.group("rest").strip()
    if rest.startswith("@"):
        return DependencyDeclaration(text, name, line_number, role, None)
    specifier_text = rest.split(";", 1)[0].strip()
    if specifier_text.startswith("(") and specifier_text.endswith(")"):
        specifier_text = specifier_text[1:-1].strip()
    if not specifier_text:
        return DependencyDeclaration(text, name, line_number, role, None)
    specifiers = []
    for part in specifier_text.split(","):
        spec = _SPECIFIER_RE.fullmatch(part)
        if spec is None:
            return DependencyDeclaration(text, name, line_number, role, None)
        specifiers.append((spec.group("op"), spec.group("version")))
    exact = None
    if len(specifiers) == 1 and specifiers[0][0] == "==" and _EXACT_VERSION_RE.fullmatch(specifiers[0][1]):
        exact = specifiers[0][1]
    return DependencyDeclaration(text, name, line_number, role, exact)


def parse_requirements_text(text: str, *, role: str) -> tuple[list[DependencyDeclaration], int]:
    """Parse a requirements file into declarations.

    Returns ``(declarations, skipped_option_lines)``. Option lines (``-r``,
    ``-c``, ``-e``, ``--index-url``, ...) are never interpreted or followed;
    trailing per-requirement options such as ``--hash`` are stripped.
    """
    declarations: list[DependencyDeclaration] = []
    skipped_options = 0
    logical = ""
    start_line: int | None = None
    lines = text.splitlines()
    for index, physical in enumerate(lines, start=1):
        if start_line is None:
            start_line = index
        stripped = physical.rstrip()
        if stripped.endswith("\\"):
            logical += stripped[:-1] + " "
            if index < len(lines):
                continue
        else:
            logical += stripped
        line = logical.strip()
        line_number = start_line
        logical = ""
        start_line = None
        if line.startswith("#"):
            continue
        line = re.split(r"\s#", line, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            skipped_options += 1
            continue
        option = _TRAILING_OPTIONS_RE.search(line)
        if option is not None:
            line = line[: option.start()].strip()
        declarations.append(parse_dependency_declaration(line, line_number=line_number, role=role))
    return declarations, skipped_options


def parse_pyproject_declarations(data: dict) -> list[DependencyDeclaration]:
    """Collect direct dependency declarations from a parsed ``pyproject.toml``.

    Reads ``[project].dependencies``, ``[project.optional-dependencies]``, and
    Poetry's ``[tool.poetry.dependencies]`` (whose bare versions are exact).
    """
    declarations: list[DependencyDeclaration] = []
    project = data.get("project")
    if isinstance(project, dict):
        declarations.extend(_parse_string_group(project.get("dependencies"), "runtime"))
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for group, values in optional.items():
                declarations.extend(_parse_string_group(values, f"optional:{group}"))

    tool = data.get("tool")
    poetry = tool.get("poetry") if isinstance(tool, dict) else None
    poetry_deps = poetry.get("dependencies") if isinstance(poetry, dict) else None
    if isinstance(poetry_deps, dict):
        for name, constraint in poetry_deps.items():
            if not isinstance(name, str) or name.lower() == "python":
                continue
            declarations.append(_parse_poetry_declaration(name, constraint))
    return declarations


def _parse_string_group(values: object, role: str) -> list[DependencyDeclaration]:
    if not isinstance(values, list):
        return []
    return [
        parse_dependency_declaration(value, line_number=None, role=role) for value in values if isinstance(value, str)
    ]


def _parse_poetry_declaration(name: str, constraint: object) -> DependencyDeclaration:
    """Map a Poetry constraint to a declaration; only bare/``==`` versions are exact."""
    if isinstance(constraint, dict):
        constraint = constraint.get("version", "*")
    text = constraint.strip() if isinstance(constraint, str) else "*"
    if text in {"", "*"}:
        return parse_dependency_declaration(name, line_number=None, role="poetry")
    if _EXACT_VERSION_RE.fullmatch(text):
        return parse_dependency_declaration(f"{name}=={text}", line_number=None, role="poetry")
    declaration = parse_dependency_declaration(f"{name}{text}", line_number=None, role="poetry")
    if declaration.name is None:
        # Caret/tilde Poetry ranges are not PEP 440; keep the package visible.
        package = parse_dependency_declaration(name, line_number=None, role="poetry")
        return DependencyDeclaration(f"{name} {text}", package.name, None, "poetry", None)
    return declaration


class DependencySecurityValidator(ValidatorBase):
    """Scan declared dependencies for CVE vulnerabilities.

    Uses pip-audit (primary) and Safety (secondary) against the exactly pinned
    declarations in ``requirements*.txt`` and ``pyproject.toml``. Nothing is
    installed, and the evaluator's own Python environment is never audited.
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
        return "Scan declared, exactly pinned dependencies for CVE vulnerabilities using pip-audit and Safety"

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
            "requirements": sorted(skill_path.glob("requirements*.txt")),
            "pyproject": skill_path / "pyproject.toml" if (skill_path / "pyproject.toml").exists() else None,
        }

    def _read_dependency_file(self, path: Path, result: ValidationResult) -> str | None:
        """Read a dependency manifest with a bounded, no-follow secure read."""
        try:
            return secure_read_path_text(path, MAX_DEPENDENCY_FILE_BYTES)
        except SecurePathError as exc:
            result.add_warning(f"{path.name}: refusing to read dependency file: {exc}")
        except OSError as exc:
            result.add_warning(f"{path.name}: could not read dependency file: {exc}")
        return None

    def _audit_requirements(self, req_file: Path) -> ValidationResult:
        """Audit the exactly pinned declarations of a requirements file."""
        result = ValidationResult()
        result.add_message(f"Auditing {req_file.name}")

        text = self._read_dependency_file(req_file, result)
        if text is None:
            return result
        declarations, skipped_options = parse_requirements_text(text, role=f"requirements:{req_file.name}")
        if skipped_options:
            result.add_message(
                f"{req_file.name}: skipped {skipped_options} option line(s) (-r/-c/-e/--index-url ...); "
                "included files and editable installs are not followed"
            )
        result.merge(self._audit_declarations(req_file, declarations))
        return result

    def _audit_pyproject(self, pyproject: Path) -> ValidationResult:
        """Audit the dependencies declared by ``pyproject.toml``.

        The declared dependencies are parsed from the target file; the Python
        environment running SkillEvaluator is never audited in their place.
        """
        result = ValidationResult()
        result.add_message(f"Auditing {pyproject.name}")

        text = self._read_dependency_file(pyproject, result)
        if text is None:
            return result
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            result.add_warning(f"{pyproject.name}: could not parse dependency declarations: {exc}")
            return result

        declarations = parse_pyproject_declarations(data)
        if not declarations:
            result.add_message(f"{pyproject.name}: no dependency declarations found")
            return result
        result.merge(self._audit_declarations(pyproject, declarations))
        return result

    def _audit_declarations(self, source: Path, declarations: list[DependencyDeclaration]) -> ValidationResult:
        """Flag unverifiable declarations, then audit exact pins without resolution."""
        result = ValidationResult()
        exact: list[DependencyDeclaration] = []
        unverified: list[DependencyDeclaration] = []
        for declaration in declarations:
            (exact if declaration.audit_line else unverified).append(declaration)

        for declaration in unverified[:MAX_UNVERIFIED_FINDINGS_PER_FILE]:
            self._report_unverified(result, source, declaration)
        if len(unverified) > MAX_UNVERIFIED_FINDINGS_PER_FILE:
            result.add_message(
                f"{source.name}: {len(unverified) - MAX_UNVERIFIED_FINDINGS_PER_FILE} more unpinned "
                "declaration(s) not listed individually"
            )

        if not exact:
            result.add_message(f"{source.name}: no exactly pinned dependencies to audit")
            return result

        batches = self._exact_batches(exact)
        with tempfile.TemporaryDirectory(prefix="skillevaluator-pip-audit-") as temp_dir:
            audit_files = self._write_audit_files(Path(temp_dir), batches)
            if Tools.pip_audit.is_available:
                for audit_file in audit_files:
                    result.merge(self._run_pip_audit_on_file(audit_file, source=source, cwd=Path(temp_dir)))
            else:
                result.add_warning(f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}")

            if self.use_safety and Tools.safety.is_available:
                for audit_file in audit_files:
                    result.merge(self._run_safety(audit_file))
        return result

    def _report_unverified(self, result: ValidationResult, source: Path, declaration: DependencyDeclaration) -> None:
        label = declaration.name or declaration.raw[:80]
        result.add_finding(
            Finding(
                category="DEPENDENCY",
                severity=Severity.INFO,
                check_name=UNVERIFIED_CHECK_NAME,
                message=(
                    f"{label}: cannot audit a floating version ('{declaration.raw[:120]}' is not an exact "
                    "'==' pin); vulnerability applicability was not asserted"
                ),
                file_path=source.name,
                line_number=declaration.line_number,
                suggestion="Pin an exact version (name==x.y.z) or audit a lockfile, then rerun the dependency audit.",
                metadata={
                    "package_name": declaration.name,
                    "declared_constraint": declaration.raw[:200],
                    "dependency_role": declaration.role,
                    "resolution_status": "unverified",
                },
            )
        )

    @staticmethod
    def _exact_batches(exact: list[DependencyDeclaration]) -> list[list[str]]:
        """Group exact pins so no batch names one package twice.

        ``pip-audit --no-deps`` rejects duplicate package names with different
        pins (e.g. marker-split pins); each conflicting pin gets its own batch.
        """
        batches: list[dict[str, str]] = []
        for declaration in exact:
            line = declaration.audit_line
            if line is None or declaration.name is None:
                continue
            for batch in batches:
                existing = batch.get(declaration.name)
                if existing is None or existing == line:
                    batch[declaration.name] = line
                    break
            else:
                batches.append({declaration.name: line})
        return [sorted(batch.values()) for batch in batches]

    @staticmethod
    def _write_audit_files(temp_dir: Path, batches: list[list[str]]) -> list[Path]:
        """Write normalized ``name==version`` requirement files for pip-audit."""
        files = []
        for index, lines in enumerate(batches):
            audit_file = temp_dir / f"requirements-{index}.txt"
            audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            files.append(audit_file)
        return files

    def _run_pip_audit_on_file(self, audit_file: Path, *, source: Path, cwd: Path) -> ValidationResult:
        """Run pip-audit on a normalized pinned requirements file.

        ``--no-deps --disable-pip`` audits exactly the listed pins without
        creating a virtual environment, invoking pip, or building packages.
        """
        result = ValidationResult()
        tool_result = Tools.pip_audit.run(
            [
                "-r",
                str(audit_file),
                "--no-deps",
                "--disable-pip",
                "--format",
                "json",
                "--progress-spinner",
                "off",
            ],
            cwd=cwd,
            timeout=180,
        )

        if tool_result.error_message:
            result.add_warning(f"{source.name}: {tool_result.error_message}")
        elif tool_result.exit_code != 0 and parse_json_output(tool_result.stdout) is None:
            detail = (tool_result.stderr or "").strip().splitlines()
            reason = detail[-1][:300] if detail else f"exit code {tool_result.exit_code}"
            result.add_warning(f"{source.name}: pip-audit failed: {reason}")
        else:
            self._process_pip_audit(tool_result.stdout, result, source.name)

        return result

    def _process_pip_audit(self, output: str, result: ValidationResult, source: str) -> None:
        """Parse pip-audit output and report vulnerabilities."""
        data = parse_json_output(output, on_error="No known vulnerabilities found")
        if data is None:
            return

        # Handle both list and dict output formats
        dependencies = data if isinstance(data, list) else data.get("dependencies", [])

        vuln_count = 0
        seen: set[tuple[str, str, str]] = set()
        for dep in dependencies:
            if not isinstance(dep, dict):
                continue

            pkg_name = dep.get("name", "unknown")
            pkg_version = dep.get("version", "unknown")

            for vuln in dep.get("vulns", []):
                # pip-audit can list the same advisory twice for one package.
                key = (str(pkg_name), str(pkg_version), str(vuln.get("id", "Unknown")))
                if key in seen:
                    continue
                seen.add(key)
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
