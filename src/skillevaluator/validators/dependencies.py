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

For plugins (inside a plugin tree scope) the audit also covers npm
(``package.json`` / ``package-lock.json``) and container images (MCP
``docker|podman|nerdctl run`` commands and ``Dockerfile`` ``FROM`` lines); see
:mod:`skillevaluator.validators.dependency_ecosystems`. A per-ecosystem summary
is recorded as ``metadata['plugin']['cve_summary']``. In a plugin run the audit
never passes silently: a manifest, lockfile, or Dockerfile that cannot be read
or parsed, and a scanner run that produces no evidence (pip-audit included),
make the ecosystem INCOMPLETE.
"""

from __future__ import annotations

import json
import re
import tempfile
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES, SCAN_EXCLUDED_DIRS
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files, secure_read_path_text
from skillevaluator.utils.structured_data import (
    MAX_STRUCTURED_DEPTH,
    StructuredDataError,
    StructuredDataSyntaxError,
    load_bounded_json,
    preflight_json_structure,
)
from skillevaluator.utils.tool_runner import Severity, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators import dependency_ecosystems as eco
from skillevaluator.validators.base import ValidationResult, ValidatorBase
from skillevaluator.validators.mcp_static import (
    EXACT_PEP440_VERSION_RE,
    exact_pypi_version,
    is_local_spec,
    is_remote_pypi_spec,
    mcp_container_image,
    parse_mcp_runner,
)
from skillevaluator.validators.plugin_tree import active_plugin_tree, is_plugin_tree_root, plugin_tree_exclusions

if TYPE_CHECKING:
    from skillevaluator.validators.policy import ValidationPolicy

# Dependency manifests are read through a bounded, no-follow secure read.
MAX_DEPENDENCY_FILE_BYTES = CONTENT_DEDUP_MAX_FILE_BYTES
# Per-file cap on individual ``dependency-version-unverified`` findings; the
# remainder is summarized in one message so a huge manifest cannot flood reports.
MAX_UNVERIFIED_FINDINGS_PER_FILE = 100
UNVERIFIED_CHECK_NAME = eco.UNVERIFIED_CHECK_NAME
# Bounded discovery of npm manifests and Dockerfiles below one scanned directory.
MAX_ECOSYSTEM_FILES = 64
MAX_ECOSYSTEM_DISCOVERED_PATHS = 20_000
# npm lockfiles list every installed package, so they get lockfile-sized bounds
# (a larger byte cap and collection/token budgets that cover MAX_NPM_PACKAGES)
# instead of the 1,024-entry budget of load_bounded_json. A lockfile over these
# bounds makes the npm audit INCOMPLETE; it is never skipped silently.
MAX_LOCKFILE_BYTES = 8 * 1024 * 1024
MAX_LOCKFILE_COLLECTION_ITEMS = 4 * eco.MAX_NPM_PACKAGES
MAX_LOCKFILE_TOKENS = 2_000_000
# ``incomplete_scans`` names of the plugin CVE audit.
PIP_AUDIT_SCAN = "pip-audit"
NPM_AUDIT_SCAN = "npm-audit"
CONTAINER_AUDIT_SCAN = "container-image-audit"

# PEP 508 distribution name, optional extras, then the rest of the requirement.
_REQUIREMENT_RE = re.compile(
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[(?P<extras>[^\]]*)\])?\s*(?P<rest>.*)",
    re.DOTALL,
)
_SPECIFIER_RE = re.compile(r"\s*(?P<op>~=|===|==|!=|<=|>=|<|>)\s*(?P<version>[^\s,;]+)\s*")
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
    if len(specifiers) == 1 and specifiers[0][0] == "==" and EXACT_PEP440_VERSION_RE.fullmatch(specifiers[0][1]):
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
    if EXACT_PEP440_VERSION_RE.fullmatch(text):
        return parse_dependency_declaration(f"{name}=={text}", line_number=None, role="poetry")
    declaration = parse_dependency_declaration(f"{name}{text}", line_number=None, role="poetry")
    if declaration.name is None:
        # Caret/tilde Poetry ranges are not PEP 440; keep the package visible.
        package = parse_dependency_declaration(name, line_number=None, role="poetry")
        return DependencyDeclaration(f"{name} {text}", package.name, None, "poetry", None)
    return declaration


def _reject_json_constant(value: str) -> object:
    raise StructuredDataSyntaxError(f"Input is not strict JSON ({value})")


def _load_npm_lockfile(text: str) -> Any:
    """Parse an npm lockfile under lockfile-sized bounds.

    The text is lexically bounded (depth, collection size, token count, string
    length) before ``json.loads`` materializes it. Raises
    :class:`~skillevaluator.utils.structured_data.StructuredDataError` (or
    ``ValueError``) when the lockfile is over a bound or is not strict JSON.
    """
    preflight_json_structure(
        text,
        max_depth=MAX_STRUCTURED_DEPTH,
        max_tokens=MAX_LOCKFILE_TOKENS,
        max_collection_items=MAX_LOCKFILE_COLLECTION_ITEMS,
    )
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise StructuredDataSyntaxError(f"Input is not valid JSON: {exc}") from exc


def _python_runner_declaration(spec: str) -> DependencyDeclaration | None:
    """A ``uvx``/``pipx run`` package spec as a declaration; local paths are skipped.

    Its version is exact when the MCP pinning check calls the spec pinned: both
    use :func:`~skillevaluator.validators.mcp_static.exact_pypi_version`, which
    also reads uv's ``pkg@1.2.3`` spelling of ``pkg==1.2.3``. Git and URL specs
    are kept as unverifiable.
    """
    text = spec.strip()
    if not text or is_local_spec(text):
        return None
    if is_remote_pypi_spec(text):
        return DependencyDeclaration(text, None, None, "mcp", None)
    declaration = parse_dependency_declaration(text, line_number=None, role="mcp")
    return replace(declaration, exact_version=exact_pypi_version(text))


class DependencySecurityValidator(ValidatorBase):
    """Scan declared dependencies for CVE vulnerabilities.

    Uses pip-audit (primary) and Safety (secondary) against the exactly pinned
    declarations in ``requirements*.txt`` and ``pyproject.toml``. Nothing is
    installed, and the evaluator's own Python environment is never audited.
    """

    def __init__(
        self,
        use_safety: bool = True,
        fail_on_medium: bool = False,
        *,
        policy: ValidationPolicy | None = None,
        resolve_endpoints: bool = False,
    ):
        """Initialize dependency validator.

        Args:
            use_safety: Run Safety check for additional coverage
            fail_on_medium: Treat MEDIUM severity as errors
            policy: Validation policy; ``mcp.allowed_private_hosts`` admits private
                container registries, and ``endpoints.resolve`` resolves registry names
            resolve_endpoints: Resolve container registry names (``--resolve-endpoints``)
        """
        self.use_safety = use_safety
        self.fail_on_medium = fail_on_medium
        self.allowed_private_hosts: tuple[str, ...] = tuple(policy.mcp_allowed_private_hosts) if policy else ()
        self.resolve_endpoints = resolve_endpoints or bool(policy is not None and policy.resolve_endpoints)
        self._summary: dict[str, dict[str, Any]] = {}
        self._mcp_cache: dict[Path, list[Any]] = {}

    @property
    def name(self) -> str:
        return "Dependency Vulnerability Audit"

    @property
    def description(self) -> str:
        return "Scan declared, exactly pinned dependencies for CVE vulnerabilities using pip-audit and Safety"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run dependency audit on skill(s) at path."""
        self._summary = {name: eco.empty_ecosystem_summary() for name in ("python", "npm", "container")}
        self._mcp_cache = {}
        result = self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Auditing dependencies for",
        )
        if active_plugin_tree() is not None:
            # Plugin runs record a per-ecosystem CVE summary for the plugin report sections.
            result.metadata.setdefault("plugin", {})["cve_summary"] = {
                "method": "declared_exact_pins",
                "ecosystems": self._summary,
            }
        return result

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Audit dependencies for a single skill directory (plus npm and images inside a plugin)."""
        result = self._audit_python(skill_path)
        if active_plugin_tree() is not None and skill_path.is_dir():
            result.merge(self._audit_npm(skill_path))
            if is_plugin_tree_root(skill_path):
                result.merge(self._audit_mcp_packages(skill_path))
            result.merge(self._audit_containers(skill_path))
        return result

    def _audit_python(self, skill_path: Path) -> ValidationResult:
        """Audit the Python declarations of one directory."""
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
        result.merge(self._audit_declarations(req_file.name, declarations))
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
        result.merge(self._audit_declarations(pyproject.name, declarations))
        return result

    def _audit_declarations(self, source: str, declarations: list[DependencyDeclaration]) -> ValidationResult:
        """Flag unverifiable declarations, then audit exact pins without resolution.

        ``source`` is the declaring file's POSIX path relative to the audited
        directory; it names the file in findings and messages.
        """
        result = ValidationResult()
        exact: list[DependencyDeclaration] = []
        unverified: list[DependencyDeclaration] = []
        for declaration in declarations:
            (exact if declaration.audit_line else unverified).append(declaration)

        for declaration in unverified[:MAX_UNVERIFIED_FINDINGS_PER_FILE]:
            result.add_finding(
                eco.unverified_finding(
                    declaration.name or declaration.raw[:80],
                    declaration.raw,
                    source,
                    ecosystem="python",
                    role=declaration.role,
                    kind="version",
                    line_number=declaration.line_number,
                )
            )
        if len(unverified) > MAX_UNVERIFIED_FINDINGS_PER_FILE:
            result.add_message(
                f"{source}: {len(unverified) - MAX_UNVERIFIED_FINDINGS_PER_FILE} more unpinned "
                "declaration(s) not listed individually"
            )

        outcome: eco.AuditOutcome | None = None
        audited = 0
        if exact:
            outcome, audited = self._audit_exact_pins(result, source, exact)
        else:
            result.add_message(f"{source}: no exactly pinned dependencies to audit")
        eco.record_outcome(
            self._summary.setdefault("python", eco.empty_ecosystem_summary()),
            outcome,
            declarations=len(declarations),
            audited=audited,
            unverified=len(unverified),
            partial=True,
        )
        if outcome is not None and outcome.status == "incomplete" and active_plugin_tree() is not None:
            # A plugin run never passes without evidence, as for the npm and container
            # audits; standalone skills keep the warning.
            result.mark_scan_incomplete(PIP_AUDIT_SCAN)
        return result

    def _audit_exact_pins(
        self, result: ValidationResult, source: str, exact: list[DependencyDeclaration]
    ) -> tuple[eco.AuditOutcome, int]:
        """Audit exact pins with pip-audit (and Safety); return the outcome and how many pins pip-audit audited.

        A batch that produced no evidence (pip-audit missing, timeout, offline,
        crash, no JSON report) makes the outcome INCOMPLETE with its error; the
        batches that ran keep their evidence.
        """
        outcome = eco.AuditOutcome()
        batches = self._exact_batches(exact)
        audited_lines: set[str] = set()
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="skillevaluator-pip-audit-") as temp_dir:
            audit_files = self._write_audit_files(Path(temp_dir), batches)
            if Tools.pip_audit.is_available:
                for lines, audit_file in zip(batches, audit_files, strict=True):
                    batch_result, error = self._run_pip_audit_on_file(
                        audit_file, source=source, cwd=Path(temp_dir), outcome=outcome
                    )
                    result.merge(batch_result)
                    if error is None:
                        audited_lines.update(lines)
                    else:
                        errors.append(error)
            else:
                error = f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}"
                result.add_warning(error)
                errors.append(error)

            if self.use_safety and Tools.safety.is_available:
                for audit_file in audit_files:
                    result.merge(self._run_safety(audit_file))
        if audited_lines:
            outcome.scanner = PIP_AUDIT_SCAN
        if errors:
            outcome.status = "incomplete"
            outcome.error = f"{source}: {'; '.join(dict.fromkeys(errors))}"
        return outcome, sum(1 for declaration in exact if declaration.audit_line in audited_lines)

    @staticmethod
    def _exact_batches(exact: list[DependencyDeclaration]) -> list[list[str]]:
        """Group exact pins into ``name==version`` batches that name each package once.

        ``pip-audit --no-deps`` rejects duplicate package names with different
        pins (e.g. marker-split pins); each conflicting pin gets its own batch.
        """
        pins = [(declaration.name, declaration.audit_line) for declaration in exact if declaration.audit_line]
        return [sorted(batch.values()) for batch in eco.split_conflicting_pins(pins)]

    @staticmethod
    def _write_audit_files(temp_dir: Path, batches: list[list[str]]) -> list[Path]:
        """Write normalized ``name==version`` requirement files for pip-audit."""
        files = []
        for index, lines in enumerate(batches):
            audit_file = temp_dir / f"requirements-{index}.txt"
            audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            files.append(audit_file)
        return files

    def _run_pip_audit_on_file(
        self, audit_file: Path, *, source: str, cwd: Path, outcome: eco.AuditOutcome
    ) -> tuple[ValidationResult, str | None]:
        """Run pip-audit on a normalized pinned requirements file.

        ``--no-deps --disable-pip`` audits exactly the listed pins without
        creating a virtual environment, invoking pip, or building packages.
        Vulnerabilities are tallied on *outcome*. Returns the result and, when
        the run produced no evidence, the error.
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

        error: str | None = None
        if tool_result.error_message:
            error = tool_result.error_message
        elif tool_result.exit_code != 0 and parse_json_output(tool_result.stdout) is None:
            detail = (tool_result.stderr or "").strip().splitlines()
            reason = detail[-1][: eco.MAX_ERROR_CHARS] if detail else f"exit code {tool_result.exit_code}"
            error = f"pip-audit failed: {reason}"
        elif not self._process_pip_audit(tool_result.stdout, result, source, outcome):
            error = "pip-audit produced no JSON report"
        if error is not None:
            result.add_warning(f"{source}: {error}")
        return result, error

    def _process_pip_audit(self, output: str, result: ValidationResult, source: str, outcome: eco.AuditOutcome) -> bool:
        """Parse pip-audit output and report vulnerabilities; ``False`` when there is no report."""
        data = parse_json_output(output, on_error="No known vulnerabilities found")
        if data is None:
            return False

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
                    outcome,
                    pkg_name=pkg_name,
                    pkg_version=pkg_version,
                    vuln_id=vuln.get("id", "Unknown"),
                    fix_versions=vuln.get("fix_versions", []),
                    severity=self._get_vuln_severity(vuln),
                )

        status = f"Found {vuln_count} vulnerability(ies)" if vuln_count else "No vulnerabilities found"
        result.add_message(f"{source}: {status} (pip-audit)")
        return True

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
        outcome: eco.AuditOutcome,
        *,
        pkg_name: str,
        pkg_version: str,
        vuln_id: str,
        fix_versions: list[str],
        severity: Severity,
    ) -> None:
        """Report a single vulnerability finding and tally it on *outcome*."""
        fix_hint = f" -> upgrade to {fix_versions[0]}" if fix_versions else ""
        message = f"{pkg_name}=={pkg_version}: {vuln_id}{fix_hint}"
        outcome.count(severity)
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

    # ------------------------------------------------------------------ #
    # npm and container images (plugin runs only)                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _discover(directory: Path, selected) -> list[PurePosixPath]:
        """No-follow discovery below *directory*, skipping bundled-skill subtrees it does not own.

        Returns every match; callers audit the first ``MAX_ECOSYSTEM_FILES`` and mark the rest INCOMPLETE.
        """
        owned = plugin_tree_exclusions(directory)
        files = discover_secure_files(
            directory,
            selected=selected,
            excluded_dirs=SCAN_EXCLUDED_DIRS,
            max_paths=MAX_ECOSYSTEM_DISCOVERED_PATHS,
            allow_context_alias=False,
        )
        found: list[PurePosixPath] = []
        for file in files:
            parts = file.relative_path.parts
            if any(parts[: len(prefix)] == prefix for prefix in owned):
                continue
            found.append(PurePosixPath(*parts))
        return sorted(found)

    @staticmethod
    def _source_label(directory: Path, rel: PurePosixPath) -> str:
        """The plugin-relative path of *rel* for messages and warnings.

        Findings take *rel* itself: the plugin-tree walk rebases a bundled
        skill's finding paths onto the skill directory, so a plugin-relative
        finding path would name the skill directory twice.
        """
        tree = active_plugin_tree()
        if tree is not None:
            try:
                prefix = Path(directory).absolute().relative_to(Path(tree.root).absolute())
            except ValueError:
                prefix = Path()
            return (PurePosixPath(*prefix.parts) / rel).as_posix()
        return rel.as_posix()

    @staticmethod
    def _read_npm_manifest(directory: Path, rel: PurePosixPath) -> tuple[Any, str | None]:
        """Parse one npm manifest; ``(data, None)``, or ``(None, reason)`` when it cannot be read safely.

        Lockfiles get lockfile-sized bounds; ``package.json`` keeps the manifest bounds.
        """
        lockfile = rel.name in eco.LOCKFILE_NAMES
        try:
            with SecureRoot(directory) as root:
                raw, _metadata = root.read_bytes(
                    Path(*rel.parts), MAX_LOCKFILE_BYTES if lockfile else MAX_DEPENDENCY_FILE_BYTES
                )
            text = raw.decode("utf-8-sig")
            return (_load_npm_lockfile(text) if lockfile else load_bounded_json(text)), None
        except (SecurePathError, StructuredDataError, OSError, UnicodeError, ValueError) as exc:
            return None, str(exc)[: eco.MAX_ERROR_CHARS]

    def _record_unaudited(
        self,
        result: ValidationResult,
        summary: dict[str, Any],
        label: str,
        error: str,
        *,
        scan_name: str,
        declarations: int = 0,
        new_source: bool = True,
    ) -> None:
        """Record a source (or part of one) that could not be audited: INCOMPLETE with the error, never a silent pass."""
        outcome = eco.AuditOutcome(status="incomplete", error=error)
        self._apply_outcome(result, outcome, label, scan_name=scan_name)
        eco.record_outcome(summary, outcome, declarations=declarations, audited=0, unverified=0, new_source=new_source)

    def _audit_npm(self, directory: Path) -> ValidationResult:
        """Audit exact npm pins from lockfiles (or package.json when a directory has no lockfile).

        A manifest that cannot be read or parsed within its bounds makes the npm
        audit INCOMPLETE. The directory's next manifest (another lockfile, then
        ``package.json`` with direct dependencies only) is audited in its place.
        """
        result = ValidationResult()
        summary = self._summary.setdefault("npm", eco.empty_ecosystem_summary())
        names = {eco.PACKAGE_JSON, *eco.LOCKFILE_NAMES}
        try:
            manifests = self._discover(directory, lambda relative: relative.name in names)
        except (SecurePathError, ValueError) as exc:
            label = self._source_label(directory, PurePosixPath())
            self._record_unaudited(
                result, summary, label, f"npm manifest discovery failed: {exc}", scan_name=NPM_AUDIT_SCAN
            )
            return result
        if len(manifests) > MAX_ECOSYSTEM_FILES:
            self._record_unaudited(
                result,
                summary,
                self._source_label(directory, PurePosixPath()),
                f"{len(manifests)} npm manifests found; only the first {MAX_ECOSYSTEM_FILES} were audited",
                scan_name=NPM_AUDIT_SCAN,
            )
            manifests = manifests[:MAX_ECOSYSTEM_FILES]
        by_dir: dict[PurePosixPath, list[PurePosixPath]] = {}
        for rel in manifests:
            by_dir.setdefault(rel.parent, []).append(rel)
        for rels in by_dir.values():
            lockfiles = [rel for rel in rels if rel.name in eco.LOCKFILE_NAMES]
            candidates = [*lockfiles, *(rel for rel in rels if rel.name == eco.PACKAGE_JSON)]
            unreadable: list[str] = []
            for rel in candidates:
                label = self._source_label(directory, rel)
                data, error = self._read_npm_manifest(directory, rel)
                if error is not None:
                    self._record_unaudited(
                        result,
                        summary,
                        label,
                        f"could not read npm manifest safely: {error}",
                        scan_name=NPM_AUDIT_SCAN,
                    )
                    unreadable.append(label)
                    continue
                if unreadable:
                    result.add_message(f"{label}: audited in place of unreadable {', '.join(unreadable)}")
                self._audit_npm_source(
                    result,
                    summary,
                    label,
                    data,
                    lockfile=rel.name in eco.LOCKFILE_NAMES,
                    file_path=rel.as_posix(),
                )
                break
        return result

    def _audit_npm_source(
        self,
        result: ValidationResult,
        summary: dict[str, Any],
        label: str,
        data: Any,
        *,
        lockfile: bool,
        file_path: str,
    ) -> None:
        """Audit the exact pins of one parsed lockfile or ``package.json``; floating versions are unverified.

        A manifest with more than ``MAX_NPM_PACKAGES`` packages is audited up to the
        cap, and the rest make the npm audit INCOMPLETE.
        """
        declarations, total = eco.read_npm_declarations(data, lockfile=lockfile)
        result.add_message(f"Auditing {label} ({total} npm declaration(s))")
        self._audit_npm_declarations(result, summary, label, declarations, total=total, file_path=file_path)

    def _audit_npm_declarations(
        self,
        result: ValidationResult,
        summary: dict[str, Any],
        label: str,
        declarations: list[eco.NpmDeclaration],
        *,
        total: int,
        file_path: str,
    ) -> None:
        """Audit npm declarations; *label* names the source in messages, *file_path* in findings."""
        if total > len(declarations):
            self._record_unaudited(
                result,
                summary,
                label,
                f"{total} npm packages declared; only the first {len(declarations)} were audited",
                scan_name=NPM_AUDIT_SCAN,
                declarations=total - len(declarations),
                new_source=False,
            )
        exact = [(d.name, d.exact_version) for d in declarations if d.exact_version]
        unverified = [d for d in declarations if not d.exact_version]
        for declaration in unverified[: eco.MAX_UNVERIFIED_PER_SOURCE]:
            result.add_finding(
                eco.unverified_finding(
                    declaration.name,
                    declaration.raw,
                    file_path,
                    ecosystem="npm",
                    role=declaration.role,
                    kind="npm version",
                )
            )
        if len(unverified) > eco.MAX_UNVERIFIED_PER_SOURCE:
            result.add_message(
                f"{label}: {len(unverified) - eco.MAX_UNVERIFIED_PER_SOURCE} more floating npm declaration(s) "
                "not listed individually"
            )
        if not exact:
            eco.record_outcome(summary, None, declarations=len(declarations), audited=0, unverified=len(unverified))
            return
        outcome = eco.audit_npm_pins([(name, version) for name, version in exact if version], source=file_path)
        self._apply_outcome(result, outcome, label, scan_name=NPM_AUDIT_SCAN)
        if outcome.unaudited:
            self._record_unaudited(
                result,
                summary,
                label,
                f"{outcome.unaudited} more exact npm pins were not audited (the cap is {eco.MAX_NPM_PACKAGES})",
                scan_name=NPM_AUDIT_SCAN,
                new_source=False,
            )
        eco.record_outcome(
            summary,
            outcome,
            declarations=len(declarations),
            audited=len(exact) - outcome.unaudited,
            unverified=len(unverified),
        )

    def _audit_containers(self, directory: Path) -> ValidationResult:
        """Audit exact container images from MCP run commands (plugin root) and Dockerfiles."""
        result = ValidationResult()
        summary = self._summary.setdefault("container", eco.empty_ecosystem_summary())
        # (image, label for messages, file path for findings); MCP images are only read at the plugin root.
        images: list[tuple[eco.ImageDeclaration, str, str]] = []
        if is_plugin_tree_root(directory):
            for image, label in self._mcp_images(directory):
                images.append((eco.image_declaration(image, "mcp"), label, label))
        try:
            dockerfiles = self._discover(directory, lambda relative: eco.is_dockerfile_name(relative.name))
        except (SecurePathError, ValueError) as exc:
            label = self._source_label(directory, PurePosixPath())
            self._record_unaudited(
                result, summary, label, f"Dockerfile discovery failed: {exc}", scan_name=CONTAINER_AUDIT_SCAN
            )
            dockerfiles = []
        if len(dockerfiles) > MAX_ECOSYSTEM_FILES:
            self._record_unaudited(
                result,
                summary,
                self._source_label(directory, PurePosixPath()),
                f"{len(dockerfiles)} Dockerfiles found; only the first {MAX_ECOSYSTEM_FILES} were audited",
                scan_name=CONTAINER_AUDIT_SCAN,
            )
            dockerfiles = dockerfiles[:MAX_ECOSYSTEM_FILES]
        for rel in dockerfiles:
            label = self._source_label(directory, rel)
            try:
                with SecureRoot(directory) as root:
                    text = root.read_text(Path(*rel.parts), MAX_DEPENDENCY_FILE_BYTES)
            except (SecurePathError, OSError, UnicodeError) as exc:
                self._record_unaudited(
                    result,
                    summary,
                    label,
                    f"could not read Dockerfile safely: {exc}",
                    scan_name=CONTAINER_AUDIT_SCAN,
                )
                continue
            images.extend((declaration, label, rel.as_posix()) for declaration in eco.parse_dockerfile_images(text))
        unique: dict[str, tuple[eco.ImageDeclaration, str, str]] = {}
        for declaration, label, file_path in images:
            unique.setdefault(declaration.image, (declaration, label, file_path))
        if len(unique) > eco.MAX_IMAGES:
            self._record_unaudited(
                result,
                summary,
                self._source_label(directory, PurePosixPath()),
                f"{len(unique)} container images found; only the first {eco.MAX_IMAGES} were audited",
                scan_name=CONTAINER_AUDIT_SCAN,
            )
        for declaration, label, file_path in list(unique.values())[: eco.MAX_IMAGES]:
            if not declaration.exact:
                result.add_finding(
                    eco.unverified_finding(
                        declaration.image,
                        declaration.image,
                        file_path,
                        ecosystem="container",
                        role=declaration.role,
                        kind="container image",
                    )
                )
                eco.record_outcome(summary, None, declarations=1, audited=0, unverified=1)
                continue
            result.add_message(f"Auditing container image {declaration.image} ({label})")
            outcome = eco.audit_image(
                declaration.image,
                source=file_path,
                allowed_hosts=self.allowed_private_hosts,
                resolve=self.resolve_endpoints,
            )
            self._apply_outcome(result, outcome, label, scan_name=CONTAINER_AUDIT_SCAN)
            eco.record_outcome(summary, outcome, declarations=1, audited=1, unverified=0)
        return result

    def _mcp_images(self, root: Path) -> list[tuple[str, str]]:
        """Container images launched by the plugin's MCP servers (every declared form, no validation)."""
        images: list[tuple[str, str]] = []
        seen_images: set[str] = set()
        for declaration in self._mcp_declarations(root):
            image = mcp_container_image(declaration.config)
            # Manifests that share an MCP file (Claude Code and Codex both load .mcp.json) list an image once.
            if image and image not in seen_images:
                seen_images.add(image)
                images.append((image, self._mcp_label(declaration)))
        return images

    @staticmethod
    def _mcp_label(declaration: Any) -> str:
        return f"{declaration.file} (mcpServers['{declaration.name}'])"

    def _mcp_declarations(self, root: Path) -> list[Any]:
        """Every MCP server the plugin's clients launch, read from the plugin's component inventory.

        Mirrors the schema check (``build_plugin_inventory``): the selected
        manifest's format profile picks its default MCP files (``.mcp.json`` for
        Claude Code and Codex, ``mcp.json`` for Cursor and Agent Plugins) and the
        component fields it reads, and every additional supported manifest in the
        root contributes its servers too, because the client that loads it
        launches them.
        """
        key = Path(root).absolute()
        if key not in self._mcp_cache:
            self._mcp_cache[key] = self._collect_mcp_declarations(root)
        return self._mcp_cache[key]

    @staticmethod
    def _collect_mcp_declarations(root: Path) -> list[Any]:
        from skillevaluator.plugin_components import plugin_inventory_for_root

        # An oversize or non-UTF-8 client manifest is read leniently: its client still launches its servers.
        inventory = plugin_inventory_for_root(root)
        return inventory.all_mcp_declarations() if inventory is not None else []

    def _audit_mcp_packages(self, root: Path) -> ValidationResult:
        """CVE-audit the packages MCP package runners install (``npx``/``bunx``/``pnpm dlx``, ``uvx``/``pipx run``).

        Runner argv is read by :func:`~skillevaluator.validators.mcp_static.parse_mcp_runner`,
        the reader the pinning check uses. Exact npm specs (``pkg@1.2.3``) join
        the npm audit and exact PyPI specs (``pkg==1.2.3`` or ``pkg@1.2.3``,
        including ``uvx --with`` requirements) the pip-audit batch, with the role
        ``mcp``, one audit per MCP config file; floating specs get the INFO
        ``dependency-version-unverified`` finding. Local paths are skipped.
        """
        result = ValidationResult()
        npm: dict[str, list[eco.NpmDeclaration]] = {}
        pypi: dict[str, list[DependencyDeclaration]] = {}
        for declaration in self._mcp_declarations(root):
            invocation = parse_mcp_runner(declaration.config)
            if invocation is None:
                continue
            # One audit per MCP config file, so a file with many servers runs each scanner once.
            label = str(declaration.file)
            for spec in invocation.npm_specs:
                npm_declaration = eco.npm_spec_declaration(spec, "mcp")
                if npm_declaration is not None:
                    npm.setdefault(label, []).append(npm_declaration)
            if invocation.ecosystem == "pypi":
                for spec in invocation.specs:
                    python_declaration = _python_runner_declaration(spec)
                    if python_declaration is not None:
                        pypi.setdefault(label, []).append(python_declaration)
        summary = self._summary.setdefault("npm", eco.empty_ecosystem_summary())
        for label, declarations in npm.items():
            result.add_message(f"Auditing {label} ({len(declarations)} npm package(s) an MCP runner installs)")
            self._audit_npm_declarations(result, summary, label, declarations, total=len(declarations), file_path=label)
        for label, declarations in pypi.items():
            result.merge(self._audit_declarations(label, declarations))
        return result

    @staticmethod
    def _apply_outcome(result: ValidationResult, outcome: eco.AuditOutcome, label: str, *, scan_name: str) -> None:
        for finding in outcome.findings:
            result.add_finding(finding)
        if outcome.omitted:
            result.add_message(f"{label}: {outcome.omitted} more vulnerability finding(s) not listed individually")
        if outcome.status == "incomplete":
            result.add_warning(f"{label}: dependency audit incomplete: {outcome.error}")
            result.mark_scan_incomplete(scan_name)
        else:
            total = sum(outcome.vulnerabilities.values())
            status = f"Found {total} vulnerability(ies)" if total else "No vulnerabilities found"
            result.add_message(f"{label}: {status} ({outcome.scanner})")
