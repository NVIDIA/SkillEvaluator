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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.models.result import Finding
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files, secure_read_path_text
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json
from skillevaluator.utils.tool_runner import Severity, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators import dependency_ecosystems as eco
from skillevaluator.validators.base import ValidationResult, ValidatorBase
from skillevaluator.validators.mcp_static import (
    is_local_spec,
    is_pep440_version,
    is_remote_pypi_spec,
    mcp_container_image,
    parse_mcp_runner,
    pypi_pin,
    specifier_pin,
)
from skillevaluator.validators.plugin_tree import (
    active_plugin_tree,
    is_plugin_tree_root,
    plugin_relative_dir,
    plugin_tree_exclusions,
)

if TYPE_CHECKING:
    from skillevaluator.validators.policy import ValidationPolicy

# Dependency manifests are read through a bounded, no-follow secure read.
MAX_DEPENDENCY_FILE_BYTES = CONTENT_DEDUP_MAX_FILE_BYTES
UNVERIFIED_CHECK_NAME = eco.UNVERIFIED_CHECK_NAME
# A Python advisory (pip-audit or Safety), and an exact pin the scanner could not audit.
PYTHON_VULN_CHECK = "python-vulnerability"
NOT_AUDITED_CHECK_NAME = "dependency-not-audited"
# At most this many npm manifests and Dockerfiles below one scanned directory are
# audited. Their discovery walk is bounded like the whole-plugin walk that runs
# before it (PLUGIN_TREE_MAX_DISCOVERED_PATHS).
MAX_ECOSYSTEM_FILES = 64
# npm lockfiles list every installed package, so they get lockfile-sized bounds
# (a larger byte cap and collection/token budgets that cover MAX_NPM_PACKAGES)
# instead of the 1,024-entry budget of load_bounded_json. A lockfile over these
# bounds makes the npm audit INCOMPLETE; it is never skipped silently.
MAX_LOCKFILE_BYTES = 8 * 1024 * 1024
MAX_LOCKFILE_COLLECTION_ITEMS = 4 * eco.MAX_NPM_PACKAGES
MAX_LOCKFILE_TOKENS = 2_000_000
# A parsed JSON document has at most one more value or key than it has preflight
# tokens, so this graph budget refuses no lockfile that the token budget admits.
MAX_LOCKFILE_NODES = MAX_LOCKFILE_TOKENS + 1
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


@dataclass(frozen=True)
class _Discovered:
    """The files one ecosystem audits below a directory, or why they could not be discovered."""

    files: tuple[PurePosixPath, ...] = ()
    error: str | None = None


# npm manifests the audit reads, and the yarn, pnpm, and bun lockfiles it reports as not audited.
_NPM_MANIFEST_NAMES = frozenset({eco.PACKAGE_JSON, *eco.LOCKFILE_NAMES, *eco.UNSUPPORTED_LOCKFILE_NAMES})


def _is_npm_manifest(relative: PurePath) -> bool:
    return relative.name in _NPM_MANIFEST_NAMES


def _is_dockerfile(relative: PurePath) -> bool:
    return eco.is_dockerfile_name(relative.name)


def canonicalize_package_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_dependency_declaration(raw: str, *, line_number: int | None, role: str) -> DependencyDeclaration:
    """Parse one PEP 508 requirement string without installing or resolving it.

    Only a single ``==`` or ``===`` specifier that names one valid PEP 440
    version, in any spelling PEP 440 accepts, is an exact pin
    (:func:`~skillevaluator.validators.mcp_static.specifier_pin`). Direct URL
    references, wildcards, a ``===`` string that is not a version, ranges, and
    malformed strings are returned with ``exact_version=None``.
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
    pin = specifier_pin(*specifiers[0]) if len(specifiers) == 1 else None
    exact = pin.version if pin is not None and pin.auditable else None
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
    if is_pep440_version(text):
        return parse_dependency_declaration(f"{name}=={text}", line_number=None, role="poetry")
    declaration = parse_dependency_declaration(f"{name}{text}", line_number=None, role="poetry")
    if declaration.name is None:
        # Caret/tilde Poetry ranges are not PEP 440; keep the package visible.
        package = parse_dependency_declaration(name, line_number=None, role="poetry")
        return DependencyDeclaration(f"{name} {text}", package.name, None, "poetry", None)
    return declaration


_TIER2_WORD_RE = re.compile(r"\bTier 2 ")


def read_failure_reason(exc: BaseException) -> str:
    """Why a dependency file could not be read, in the words of this Tier 1 check.

    The shared no-follow reader was written for Tier 2 inputs and names Tier 2
    in its messages ("Selected Tier 2 file is not valid UTF-8"); this audit is
    a Tier 1 check, so that word is dropped and a bad encoding is said plainly.
    """
    if isinstance(exc, UnicodeError) or (isinstance(exc, SecurePathError) and exc.code == "invalid_text_encoding"):
        return "the file is not valid UTF-8 text"
    return _TIER2_WORD_RE.sub("", str(exc))[: eco.MAX_ERROR_CHARS]


def _safety_json(output: str) -> Any:
    """The JSON report in Safety's stdout. Safety 3.x prints deprecation banners before and after it."""
    data = parse_json_output(output)
    if data is not None:
        return data
    decoder = json.JSONDecoder()
    for match in re.finditer(r"^[ \t]*[\[{]", output or "", re.MULTILINE):
        try:
            value, _end = decoder.raw_decode(output, match.end() - 1)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict | list):
            return value
    return None


def _python_runner_declaration(spec: str) -> DependencyDeclaration | None:
    """A ``uvx``/``pipx run`` package spec as a declaration; local paths are skipped.

    The MCP pinning check reads the same pin
    (:func:`~skillevaluator.validators.mcp_static.pypi_pin`, which also reads
    uv's ``pkg@1.2.3`` spelling of ``pkg==1.2.3``): a pinned spec is audited
    when its version is a valid PEP 440 version. A ``===`` pin of any other
    string, and git and URL specs, are kept as unverifiable.
    """
    text = spec.strip()
    if not text or is_local_spec(text):
        return None
    if is_remote_pypi_spec(text):
        return DependencyDeclaration(text, None, None, "mcp", None)
    declaration = parse_dependency_declaration(text, line_number=None, role="mcp")
    pin = pypi_pin(text)
    return replace(declaration, exact_version=pin.version if pin is not None and pin.auditable else None)


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
        self._advisories = eco.AdvisorySeverityLookup()
        self._npm_registry = eco.NpmRegistryCheck()

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
        self._advisories = eco.AdvisorySeverityLookup()
        self._npm_registry = eco.NpmRegistryCheck()
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
            npm_manifests, dockerfiles = self._discover_ecosystem_files(skill_path)
            result.merge(self._audit_npm(skill_path, npm_manifests))
            if is_plugin_tree_root(skill_path):
                result.merge(self._audit_mcp_packages(skill_path))
            result.merge(self._audit_containers(skill_path, dockerfiles))
        return result

    def _audit_python(self, skill_path: Path) -> ValidationResult:
        """Audit the Python declarations of one directory."""
        result = ValidationResult()

        # Find dependency files
        dep_files = self._find_dependency_files(skill_path)
        if not dep_files["requirements"] and not dep_files["pyproject"]:
            result.add_message("No dependency files found for the Python audit (requirements*.txt, pyproject.toml)")
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
        """Read a dependency manifest with a bounded, no-follow secure read.

        A manifest that cannot be read is never skipped silently: in a plugin
        run it makes the Python audit INCOMPLETE (see :meth:`_python_unaudited`).
        """
        try:
            return secure_read_path_text(path, MAX_DEPENDENCY_FILE_BYTES)
        except SecurePathError as exc:
            if exc.code == "invalid_text_encoding":
                self._python_unaudited(result, path.name, read_failure_reason(exc))
            else:
                self._python_unaudited(
                    result, path.name, f"refusing to read dependency file: {read_failure_reason(exc)}"
                )
        except OSError as exc:
            self._python_unaudited(result, path.name, f"could not read dependency file: {exc}")
        return None

    def _python_unaudited(self, result: ValidationResult, label: str, reason: str) -> None:
        """A Python manifest that could not be read or parsed: INCOMPLETE in a plugin run, a warning otherwise.

        Plugin runs never pass without evidence, as for an npm lockfile or a
        Dockerfile that cannot be read; standalone skills keep the warning.
        """
        message = f"{label}: Python dependencies were not audited: {reason}"
        result.add_warning(message)
        summary = self._summary.setdefault("python", eco.empty_ecosystem_summary())
        if active_plugin_tree() is None:
            summary["sources"] += 1
            return
        outcome = eco.AuditOutcome(status="incomplete", error=message)
        eco.record_outcome(summary, outcome, declarations=0, audited=0, unverified=0)
        result.mark_scan_incomplete(PIP_AUDIT_SCAN)

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
            self._python_unaudited(result, pyproject.name, f"could not parse dependency declarations: {exc}")
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
        directory; it names the file in findings, messages and summary errors:
        the manifest name inside the scanned directory (bundled-skill paths are
        rebased onto the plugin root later), or the root-relative MCP manifest.
        """
        result = ValidationResult()
        exact: list[DependencyDeclaration] = []
        unverified: list[DependencyDeclaration] = []
        for declaration in declarations:
            (exact if declaration.audit_line else unverified).append(declaration)

        for declaration in unverified[: eco.MAX_UNVERIFIED_PER_SOURCE]:
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
        if len(unverified) > eco.MAX_UNVERIFIED_PER_SOURCE:
            result.add_message(
                f"{source}: {len(unverified) - eco.MAX_UNVERIFIED_PER_SOURCE} more unpinned "
                "declaration(s) not listed individually"
            )

        outcome: eco.AuditOutcome | None = None
        audited = not_audited = 0
        if exact:
            outcome, audited, not_audited = self._audit_exact_pins(result, source, exact)
        else:
            result.add_message(f"{source}: no exactly pinned dependencies to audit")
        summary = self._summary.setdefault("python", eco.empty_ecosystem_summary())
        eco.record_outcome(
            summary,
            outcome,
            declarations=len(declarations),
            audited=audited,
            unverified=len(unverified) + not_audited,
            partial=True,
        )
        self._mark_unverified(summary, not_audited)
        if outcome is not None and outcome.status == "incomplete" and active_plugin_tree() is not None:
            # A plugin run never passes without evidence, as for the npm and container
            # audits; standalone skills keep the warning.
            result.mark_scan_incomplete(PIP_AUDIT_SCAN)
        return result

    def _audit_exact_pins(
        self, result: ValidationResult, source: str, exact: list[DependencyDeclaration]
    ) -> tuple[eco.AuditOutcome, int, int]:
        """Audit exact pins with pip-audit (and Safety).

        Returns the outcome, how many pins pip-audit audited, and how many it
        could not audit. Vulnerabilities from both scanners are tallied on the
        outcome; they share ``reported``, so an advisory both report is listed
        once. A batch that produced no evidence (pip-audit missing, timeout,
        offline, crash, no JSON report) makes the outcome INCOMPLETE with its
        error; the batches that ran keep their evidence. A pin that pip-audit
        skips (a package or version PyPI does not know, such as a private
        package) is not audited either: it gets a MEDIUM ``dependency-not-audited``
        finding with pip-audit's reason.
        """
        outcome = eco.AuditOutcome()
        batches = self._exact_batches(exact)
        audited_lines: set[str] = set()
        skipped: dict[str, str] = {}
        errors: list[str] = []
        reported: set[tuple[str, str, str]] = set()
        with tempfile.TemporaryDirectory(prefix="skillevaluator-pip-audit-") as temp_dir:
            audit_files = self._write_audit_files(Path(temp_dir), batches)
            if Tools.pip_audit.is_available:
                for lines, audit_file in zip(batches, audit_files, strict=True):
                    batch_result, error, batch_skipped = self._run_pip_audit_on_file(
                        audit_file,
                        cwd=Path(temp_dir),
                        file_path=source,
                        exact=exact,
                        reported=reported,
                        outcome=outcome,
                    )
                    result.merge(batch_result)
                    if error is None:
                        audited_lines.update(lines)
                        skipped.update(batch_skipped)
                    else:
                        errors.append(error)
            else:
                error = f"pip-audit not installed. {Tools.pip_audit.get_install_hint()}"
                result.add_warning(error)
                errors.append(error)

            if self.use_safety and Tools.safety.is_available:
                for audit_file in audit_files:
                    result.merge(
                        self._run_safety(audit_file, file_path=source, exact=exact, reported=reported, outcome=outcome)
                    )
        not_audited = [
            declaration
            for declaration in exact
            if declaration.audit_line in audited_lines and declaration.name in skipped
        ]
        for declaration in not_audited[: eco.MAX_UNVERIFIED_PER_SOURCE]:
            self._report_not_audited(result, source, declaration, skipped[declaration.name or ""])
        audited = sum(1 for declaration in exact if declaration.audit_line in audited_lines) - len(not_audited)
        if audited:
            outcome.scanner = PIP_AUDIT_SCAN
        if errors:
            outcome.status = "incomplete"
            outcome.error = f"{source}: {'; '.join(dict.fromkeys(errors))}"
        return outcome, audited, len(not_audited)

    @staticmethod
    def _mark_unverified(summary: dict[str, Any], not_audited: int) -> None:
        """Exact pins were declared but none could be audited: ``unverified`` (Not audited), never ``audited``."""
        if not_audited and summary["audited"] == 0 and summary["status"] in {"no_exact", "audited"}:
            summary["status"] = "unverified"

    @staticmethod
    def _report_not_audited(
        result: ValidationResult, location: str, declaration: DependencyDeclaration, reason: str
    ) -> None:
        """An exact pin pip-audit skipped (for example, not on PyPI): applicability was never checked."""
        result.add_finding(
            Finding(
                category="DEPENDENCY",
                severity=Severity.MEDIUM,
                check_name=NOT_AUDITED_CHECK_NAME,
                message=(
                    f"{declaration.audit_line}: pip-audit could not audit this pin ({reason[:160]}); "
                    "vulnerability applicability was not asserted"
                ),
                file_path=location,
                line_number=declaration.line_number,
                suggestion=(
                    "Check the package name and version. A private or internal package needs its own "
                    "vulnerability audit."
                ),
                metadata={
                    "ecosystem": "python",
                    "package_name": declaration.name,
                    "package_version": declaration.exact_version,
                    "dependency_role": declaration.role,
                    "resolution_status": "not_audited",
                    "skip_reason": reason[: eco.MAX_ERROR_CHARS],
                    "scanner": "pip-audit",
                },
            )
        )

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
        self,
        audit_file: Path,
        *,
        cwd: Path,
        file_path: str,
        exact: list[DependencyDeclaration],
        reported: set[tuple[str, str, str]],
        outcome: eco.AuditOutcome,
    ) -> tuple[ValidationResult, str | None, dict[str, str]]:
        """Run pip-audit on a normalized pinned requirements file.

        ``--no-deps --disable-pip`` audits exactly the listed pins without
        creating a virtual environment, invoking pip, or building packages.
        Vulnerabilities are tallied on *outcome*. Returns the result, the error
        when the run produced no evidence, and the packages pip-audit skipped
        (name -> ``skip_reason``).
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
        skipped: dict[str, str] = {}
        if tool_result.error_message:
            error = tool_result.error_message
        elif tool_result.exit_code != 0 and parse_json_output(tool_result.stdout) is None:
            detail = (tool_result.stderr or "").strip().splitlines()
            reason = detail[-1][: eco.MAX_ERROR_CHARS] if detail else f"exit code {tool_result.exit_code}"
            error = f"pip-audit failed: {reason}"
        else:
            processed = self._process_pip_audit(
                tool_result.stdout, result, file_path, exact=exact, reported=reported, outcome=outcome
            )
            if processed is None:
                error = "pip-audit produced no JSON report"
            else:
                skipped = processed
        if error is not None:
            result.add_warning(f"{file_path}: {error}")
        return result, error, skipped

    def _process_pip_audit(
        self,
        output: str,
        result: ValidationResult,
        file_path: str,
        *,
        exact: Sequence[DependencyDeclaration] = (),
        reported: set[tuple[str, str, str]] | None = None,
        outcome: eco.AuditOutcome,
    ) -> dict[str, str] | None:
        """Parse pip-audit output into ``python-vulnerability`` findings, tallied on *outcome*.

        Returns the packages pip-audit skipped (canonical name -> ``skip_reason``,
        for example a pin PyPI does not know), or ``None`` when there is no report.
        """
        data = parse_json_output(output, on_error="No known vulnerabilities found")
        if data is None:
            return None

        # Handle both list and dict output formats
        dependencies = data if isinstance(data, list) else data.get("dependencies", [])
        roles = {declaration.name: declaration.role for declaration in exact if declaration.name}
        reported = reported if reported is not None else set()

        vuln_count = 0
        skipped: dict[str, str] = {}
        for dep in dependencies if isinstance(dependencies, list) else []:
            if not isinstance(dep, dict):
                continue

            pkg_name = str(dep.get("name", "unknown"))
            skip_reason = dep.get("skip_reason")
            if skip_reason and not dep.get("vulns"):
                skipped[canonicalize_package_name(pkg_name)] = str(skip_reason)
                continue
            pkg_version = str(dep.get("version", "unknown"))

            for vuln in dep.get("vulns", []) or []:
                if not isinstance(vuln, dict):
                    continue
                vuln_id = str(vuln.get("id", "Unknown"))
                aliases = [str(alias) for alias in vuln.get("aliases", []) or [] if isinstance(alias, str)]
                # pip-audit can list the same advisory twice for one pin. The version is in the
                # key: one file can pin a package twice (marker-split pins), and each pin is audited.
                key = (canonicalize_package_name(pkg_name), pkg_version, vuln_id)
                if key in reported:
                    continue
                reported.add(key)
                reported.update((key[0], pkg_version, alias) for alias in aliases)
                vuln_count += 1
                fix_versions = vuln.get("fix_versions", [])
                self._report_vulnerability(
                    result,
                    outcome,
                    file_path=file_path,
                    pkg_name=pkg_name,
                    pkg_version=pkg_version,
                    vuln_id=vuln_id,
                    aliases=aliases,
                    fix_versions=[str(item) for item in fix_versions] if isinstance(fix_versions, list) else [],
                    severity=self._get_vuln_severity(vuln),
                    scanner="pip-audit",
                    role=roles.get(canonicalize_package_name(pkg_name)),
                )

        status = f"Found {vuln_count} vulnerability(ies)" if vuln_count else "No vulnerabilities found"
        if skipped:
            status += f"; {len(skipped)} pin(s) could not be audited"
        result.add_message(f"{file_path}: {status} (pip-audit)")
        return skipped

    def _run_safety(
        self,
        req_file: Path,
        *,
        file_path: str | None = None,
        exact: Sequence[DependencyDeclaration] = (),
        reported: set[tuple[str, str, str]] | None = None,
        outcome: eco.AuditOutcome,
    ) -> ValidationResult:
        """Run Safety check for supplementary coverage (advisories pip-audit did not report)."""
        result = ValidationResult()

        tool_result = Tools.safety.run(
            ["check", "-r", str(req_file), "--output", "json"],
            timeout=60,
        )

        # Safety is secondary (pip-audit is primary): a failed run is a warning, never a gate.
        if tool_result.error_message:
            result.add_warning(f"Safety did not run: {tool_result.error_message[:200]}")
        elif tool_result.success:
            self._process_safety(
                tool_result.stdout,
                result,
                file_path=file_path or req_file.name,
                exact=exact,
                reported=reported,
                outcome=outcome,
            )

        return result

    def _process_safety(
        self,
        output: str,
        result: ValidationResult,
        *,
        file_path: str = "requirements.txt",
        exact: Sequence[DependencyDeclaration] = (),
        reported: set[tuple[str, str, str]] | None = None,
        outcome: eco.AuditOutcome,
    ) -> None:
        """Parse Safety JSON (2.x list or 3.x report, banners around it allowed) into findings, tallied on *outcome*.

        An advisory pip-audit already reported (same package, version, and id or
        CVE) is not repeated. Output with no JSON report is a warning.
        """
        data = _safety_json(output)
        if data is None:
            result.add_warning("Safety produced no JSON report; its advisories were not used")
            return

        if isinstance(data, list):
            vulnerabilities = data
        else:
            vulnerabilities = data.get("vulnerabilities", [])
        roles = {declaration.name: declaration.role for declaration in exact if declaration.name}
        reported = reported if reported is not None else set()
        pins = {declaration.name: declaration.exact_version for declaration in exact if declaration.name}

        for vuln in vulnerabilities if isinstance(vulnerabilities, list) else []:
            if isinstance(vuln, list) and len(vuln) >= 5:
                # Safety 2.x legacy rows: [package, spec, version, advisory, id, ...]
                vuln = {
                    "package_name": vuln[0],
                    "analyzed_version": vuln[2],
                    "advisory": vuln[3],
                    "vulnerability_id": vuln[4],
                }
            if not isinstance(vuln, dict):
                continue

            pkg_name = str(vuln.get("package_name", vuln.get("name", "unknown")))
            canonical = canonicalize_package_name(pkg_name)
            vuln_id = str(vuln.get("vulnerability_id", vuln.get("id", "Unknown")))
            cve = vuln.get("CVE") or vuln.get("cve")
            ids = [vuln_id, *([str(cve)] if isinstance(cve, str) and cve else [])]
            version = str(vuln.get("analyzed_version") or pins.get(canonical) or "unknown")
            if any((canonical, version, item) in reported for item in ids):
                continue
            reported.update((canonical, version, item) for item in ids)
            advisory = vuln.get("advisory")
            fixed = vuln.get("fixed_versions")
            self._report_vulnerability(
                result,
                outcome,
                file_path=file_path,
                pkg_name=pkg_name,
                pkg_version=version,
                vuln_id=f"SAFETY-{vuln_id}" if vuln_id.isdigit() else vuln_id,
                aliases=ids[1:],
                fix_versions=[str(item) for item in fixed] if isinstance(fixed, list) else [],
                severity=self._safety_severity(vuln.get("severity"), ids),
                scanner="safety",
                role=roles.get(canonical),
                summary=str(advisory)[:100] if isinstance(advisory, str) else "",
            )

    def _safety_severity(self, value: Any, ids: list[str]) -> eco.AdvisorySeverity:
        """Safety's own severity (a word, or a 3.x ``{cvss_v3: {base_severity, base_score}}``), else the advisory."""
        if isinstance(value, str) and value.strip().lower() in {"critical", "high", "medium", "low"}:
            return eco.AdvisorySeverity(Severity(value.strip().lower()), "safety")
        if isinstance(value, dict):
            for key in ("cvss_v3", "cvssv3", "cvss_v2"):
                block = value.get(key)
                if not isinstance(block, dict):
                    continue
                word = str(block.get("base_severity") or "").strip().lower()
                if word in {"critical", "high", "medium", "low"}:
                    return eco.AdvisorySeverity(Severity(word), f"safety:{key}")
                score = block.get("base_score")
                if isinstance(score, int | float) and score > 0:
                    return eco.AdvisorySeverity(cvss_to_severity(float(score)), f"safety:{key}")
        return self._advisories.severity(ids)

    def _report_vulnerability(
        self,
        result: ValidationResult,
        outcome: eco.AuditOutcome,
        *,
        file_path: str,
        pkg_name: str,
        pkg_version: str,
        vuln_id: str,
        aliases: list[str],
        fix_versions: list[str],
        severity: eco.AdvisorySeverity,
        scanner: str,
        role: str | None = None,
        summary: str = "",
    ) -> None:
        """Report one Python advisory as a structured ``python-vulnerability`` finding and tally it on *outcome*.

        The severity comes from the advisory data. When it cannot be known, the
        finding is HIGH and says so, rather than inventing a severity silently.
        """
        fix_hint = f" -> upgrade to {fix_versions[0]}" if fix_versions else ""
        detail = f" ({summary})" if summary else ""
        level = severity.severity
        unknown = ""
        if level is None:
            level = Severity.HIGH
            unknown = " (advisory severity unknown; reported as HIGH)"
        outcome.count(level)
        metadata: dict[str, Any] = {
            "ecosystem": "python",
            "package_name": pkg_name,
            "package_version": pkg_version,
            "vulnerability_id": vuln_id,
            "aliases": aliases[:8],
            "fix_versions": fix_versions[:8],
            "scanner": scanner,
            "severity_source": severity.source if severity.severity is not None else "unknown",
        }
        if severity.severity is None:
            metadata["severity_unknown_reason"] = severity.source[: eco.MAX_ERROR_CHARS]
        if role:
            metadata["dependency_role"] = role
        result.add_finding(
            Finding(
                category="DEPENDENCY",
                severity=level,
                check_name=PYTHON_VULN_CHECK,
                message=f"{pkg_name}=={pkg_version}: {vuln_id}{detail}{fix_hint}{unknown}",
                file_path=file_path,
                suggestion="Upgrade to a fixed version and rerun the dependency audit." if fix_versions else None,
                metadata=metadata,
            ),
            fail_on_medium=self.fail_on_medium,
        )

    def _get_vuln_severity(self, vuln: dict) -> eco.AdvisorySeverity:
        """Severity of a pip-audit advisory: an explicit field when present, else the advisory data.

        pip-audit's JSON states no severity, so the advisory record (the GitHub
        advisory alias first) is looked up; ``severity=None`` when that fails.
        """
        explicit = vuln.get("severity")
        if isinstance(explicit, str) and explicit.strip().lower() in {"critical", "high", "medium", "low"}:
            return eco.AdvisorySeverity(Severity(explicit.strip().lower()), "scanner")
        for alias in vuln.get("aliases", []) or []:
            if isinstance(alias, dict) and isinstance(alias.get("cvss"), dict):
                score = alias["cvss"].get("score")
                if isinstance(score, int | float) and score > 0:
                    return eco.AdvisorySeverity(cvss_to_severity(float(score)), "scanner:cvss")
        ids = [str(vuln.get("id", "")), *(alias for alias in vuln.get("aliases", []) or [] if isinstance(alias, str))]
        return self._advisories.severity(ids)

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
            max_paths=PLUGIN_TREE_MAX_DISCOVERED_PATHS,
            allow_context_alias=False,
        )
        found: list[PurePosixPath] = []
        for file in files:
            parts = file.relative_path.parts
            if any(parts[: len(prefix)] == prefix for prefix in owned):
                continue
            found.append(PurePosixPath(*parts))
        return sorted(found)

    def _discover_ecosystem_files(self, directory: Path) -> tuple[_Discovered, _Discovered]:
        """The npm manifests and the Dockerfiles below *directory*, found in one no-follow walk.

        Discovery fails closed on a selected entry that is not a single regular
        file (a directory named ``Dockerfile``, a hard-linked ``package.json``),
        and such a failure belongs to one ecosystem. After any failure each
        ecosystem is therefore discovered on its own, so the other is still
        audited.
        """
        try:
            found = self._discover(directory, lambda relative: _is_npm_manifest(relative) or _is_dockerfile(relative))
        except (SecurePathError, ValueError):
            return self._discover_one(directory, _is_npm_manifest), self._discover_one(directory, _is_dockerfile)
        return (
            _Discovered(tuple(rel for rel in found if _is_npm_manifest(rel))),
            _Discovered(tuple(rel for rel in found if _is_dockerfile(rel))),
        )

    def _discover_one(self, directory: Path, selected: Callable[[PurePath], bool]) -> _Discovered:
        try:
            return _Discovered(tuple(self._discover(directory, selected)))
        except (SecurePathError, ValueError) as exc:
            return _Discovered(error=read_failure_reason(exc))

    @staticmethod
    def _source_label(directory: Path, rel: PurePosixPath) -> str:
        """The plugin-relative path of *rel* for messages and warnings.

        Findings take *rel* itself: the plugin-tree walk rebases a bundled
        skill's finding paths onto the skill directory, so a plugin-relative
        finding path would name the skill directory twice.
        """
        prefix = plugin_relative_dir(directory)
        return (rel if prefix is None else prefix / rel).as_posix()

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
            if not lockfile:
                return load_bounded_json(text), None
            data = load_bounded_json(
                text,
                max_tokens=MAX_LOCKFILE_TOKENS,
                max_collection_items=MAX_LOCKFILE_COLLECTION_ITEMS,
                max_nodes=MAX_LOCKFILE_NODES,
            )
            return data, None
        except (SecurePathError, StructuredDataError, OSError, UnicodeError, ValueError) as exc:
            return None, read_failure_reason(exc)

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

    def _audit_npm(self, directory: Path, discovered: _Discovered) -> ValidationResult:
        """Audit exact npm pins from lockfiles (or package.json when a directory has no lockfile).

        A manifest that cannot be read or parsed within its bounds makes the npm
        audit INCOMPLETE. The directory's next manifest (another lockfile, then
        ``package.json`` with direct dependencies only) is audited in its place.
        """
        result = ValidationResult()
        summary = self._summary.setdefault("npm", eco.empty_ecosystem_summary())
        if discovered.error is not None:
            label = self._source_label(directory, PurePosixPath())
            self._record_unaudited(
                result, summary, label, f"npm manifest discovery failed: {discovered.error}", scan_name=NPM_AUDIT_SCAN
            )
            return result
        manifests = list(discovered.files)
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
            if not lockfiles:
                # yarn and pnpm lockfiles pin exact versions this audit cannot read: never a silent pass.
                for rel in (rel for rel in rels if rel.name in eco.UNSUPPORTED_LOCKFILE_NAMES):
                    self._record_unaudited(
                        result,
                        summary,
                        self._source_label(directory, rel),
                        f"{rel.name} is not supported by the npm audit, so the versions it pins were not audited "
                        "(commit a package-lock.json, or audit this lockfile separately)",
                        scan_name=NPM_AUDIT_SCAN,
                    )
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
                    finding_path=rel.as_posix(),
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
        finding_path: str | None = None,
    ) -> None:
        """Audit the exact pins of one parsed lockfile or ``package.json``; floating versions are unverified.

        A manifest with more than ``MAX_NPM_PACKAGES`` packages is audited up to the
        cap, and the rest make the npm audit INCOMPLETE.
        """
        declarations, total = eco.read_npm_declarations(data, lockfile=lockfile)
        result.add_message(f"Auditing {label} ({total} npm declaration(s))")
        self._audit_npm_declarations(result, summary, label, declarations, total=total, finding_path=finding_path)

    def _audit_npm_declarations(
        self,
        result: ValidationResult,
        summary: dict[str, Any],
        label: str,
        declarations: list[eco.NpmDeclaration],
        *,
        total: int,
        finding_path: str | None = None,
    ) -> None:
        """Audit npm declarations of one source.

        ``label`` names the source in messages (plugin-root relative);
        ``finding_path`` is the path findings carry, relative to the scanned
        directory, because a bundled skill's finding paths are rebased onto the
        plugin root afterwards (a plugin-relative path would get the skill
        folder twice).
        """
        path = finding_path or label
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
        # An exact pin from a git, file, or path source, or one the public registry does not know, is
        # not audited (the public audit cannot speak for it): MEDIUM dependency-not-audited, as for Python.
        off_registry: list[tuple[eco.NpmDeclaration, str | None]] = [
            (d, None) for d in declarations if d.exact_version and d.not_from_registry
        ]
        unknown_reasons: list[str] = []
        if Tools.osv_scanner.is_available or Tools.npm.is_available:
            for declaration in declarations:
                if not declaration.exact_version or declaration.not_from_registry:
                    continue
                if not declaration.needs_registry_check:
                    continue
                exists, error = self._npm_registry.exists(declaration.name, declaration.exact_version)
                if exists is False:
                    off_registry.append((declaration, "the public npm registry does not have this package version"))
                elif exists is None and error:
                    unknown_reasons.append(error)
        skipped = {id(d) for d, _reason in off_registry}
        not_audited = len(off_registry)
        exact = [(d.name, d.exact_version) for d in declarations if d.exact_version and id(d) not in skipped]
        unverified = [d for d in declarations if not d.exact_version]
        for declaration in unverified[: eco.MAX_UNVERIFIED_PER_SOURCE]:
            result.add_finding(
                eco.unverified_finding(
                    declaration.name,
                    declaration.raw,
                    path,
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
        for declaration, reason in off_registry[: eco.MAX_UNVERIFIED_PER_SOURCE]:
            result.add_finding(eco.npm_not_audited_finding(declaration, path, reason))
        if len(off_registry) > eco.MAX_UNVERIFIED_PER_SOURCE:
            result.add_message(
                f"{label}: {len(off_registry) - eco.MAX_UNVERIFIED_PER_SOURCE} more npm package(s) that could "
                "not be audited not listed individually"
            )
        if unknown_reasons:
            result.add_message(
                f"{label}: {len(unknown_reasons)} npm pin(s) were not checked against the public registry "
                f"({unknown_reasons[0][:160]}); they were audited by name and version"
            )
        if not exact:
            eco.record_outcome(
                summary, None, declarations=len(declarations), audited=0, unverified=len(unverified) + not_audited
            )
            self._mark_unverified(summary, not_audited)
            return
        outcome = eco.audit_npm_pins([(name, version) for name, version in exact if version], source=path)
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
            unverified=len(unverified) + not_audited,
        )
        if outcome.status != "incomplete":
            self._mark_unverified(summary, not_audited)

    def _audit_containers(self, directory: Path, discovered: _Discovered) -> ValidationResult:
        """Audit exact container images from MCP run commands (plugin root) and Dockerfiles.

        Each image keeps a message label (the MCP server or Dockerfile) and the
        file path its findings carry: the MCP file or the Dockerfile, never a
        label, so every report can point at a real file.
        """
        result = ValidationResult()
        summary = self._summary.setdefault("container", eco.empty_ecosystem_summary())
        # (declaration, message label, finding path, MCP server name)
        images: list[tuple[eco.ImageDeclaration, str, str, str | None]] = []
        if is_plugin_tree_root(directory):
            for image, label, path, server in self._mcp_images(directory):
                images.append((eco.image_declaration(image, "mcp"), label, path, server))
        if discovered.error is not None:
            label = self._source_label(directory, PurePosixPath())
            self._record_unaudited(
                result,
                summary,
                label,
                f"Dockerfile discovery failed: {discovered.error}",
                scan_name=CONTAINER_AUDIT_SCAN,
            )
        dockerfiles = list(discovered.files)
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
                    f"could not read Dockerfile safely: {read_failure_reason(exc)}",
                    scan_name=CONTAINER_AUDIT_SCAN,
                )
                continue
            images.extend(
                (declaration, label, rel.as_posix(), None) for declaration in eco.parse_dockerfile_images(text)
            )
        unique: dict[str, tuple[eco.ImageDeclaration, str, str, str | None]] = {}
        for entry in images:
            unique.setdefault(entry[0].image, entry)
        if len(unique) > eco.MAX_IMAGES:
            self._record_unaudited(
                result,
                summary,
                self._source_label(directory, PurePosixPath()),
                f"{len(unique)} container images found; only the first {eco.MAX_IMAGES} were audited",
                scan_name=CONTAINER_AUDIT_SCAN,
            )
        for declaration, label, path, server in list(unique.values())[: eco.MAX_IMAGES]:
            if not declaration.exact:
                finding = eco.unverified_finding(
                    declaration.image,
                    declaration.image,
                    path,
                    ecosystem="container",
                    role=declaration.role,
                    kind="container image",
                )
                if server is not None:
                    finding.metadata["mcp_server"] = server
                result.add_finding(finding)
                eco.record_outcome(summary, None, declarations=1, audited=0, unverified=1)
                continue
            result.add_message(f"Auditing container image {declaration.image} ({label})")
            outcome = eco.audit_image(
                declaration.image,
                source=path,
                allowed_hosts=self.allowed_private_hosts,
                resolve=self.resolve_endpoints,
            )
            if server is not None:
                for finding in outcome.findings:
                    finding.metadata["mcp_server"] = server
            self._apply_outcome(result, outcome, label, scan_name=CONTAINER_AUDIT_SCAN)
            eco.record_outcome(summary, outcome, declarations=1, audited=1, unverified=0)
        return result

    def _mcp_images(self, root: Path) -> list[tuple[str, str, str, str]]:
        """Images the plugin's MCP servers launch: ``(image, label, MCP file, server name)`` (no validation)."""
        images: list[tuple[str, str, str, str]] = []
        seen_images: set[str] = set()
        for declaration in self._mcp_declarations(root):
            image = mcp_container_image(declaration.config)
            # Manifests that share an MCP file (Claude Code and Codex both load .mcp.json) list an image once.
            if image and image not in seen_images:
                seen_images.add(image)
                images.append((image, self._mcp_label(declaration), str(declaration.file), str(declaration.name)))
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
        the npm audit and exact PyPI specs (``pkg==1.2.3`` or ``pkg@1.2.3``) the
        pip-audit batch, with the role ``mcp``, one audit per MCP config file.
        The PyPI specs include the ``--with`` requirements that ``uvx``, ``uv
        tool run``, and ``uv run`` install next to the server
        (``RunnerInvocation.specs`` lists ``with_specs`` too). Floating specs, and
        each ``--with-requirements`` file (its packages are not read), get the
        INFO ``dependency-version-unverified`` finding. Local paths are skipped.
        Each finding names the MCP server that installs the package
        (``metadata["mcp_server"]``, plus ``mcp_servers`` when several do), as
        container-image findings do.
        """
        result = ValidationResult()
        npm: dict[str, list[eco.NpmDeclaration]] = {}
        pypi: dict[str, list[DependencyDeclaration]] = {}
        # (MCP file, ecosystem, package name) -> the servers that install it, in declaration order.
        servers: dict[tuple[str, str, str], list[str]] = {}
        for declaration in self._mcp_declarations(root):
            invocation = parse_mcp_runner(declaration.config)
            if invocation is None:
                continue
            # One audit per MCP config file, so a file with many servers runs each scanner once.
            label = str(declaration.file)
            keys: list[tuple[str, str, str]] = []
            for spec in invocation.npm_specs:
                npm_declaration = eco.npm_spec_declaration(spec, "mcp")
                if npm_declaration is not None:
                    npm.setdefault(label, []).append(npm_declaration)
                    keys.append((label, "npm", npm_declaration.name))
            if invocation.ecosystem == "pypi":
                python_declarations = [_python_runner_declaration(spec) for spec in invocation.specs]
                # A --with-requirements file's packages are not read, so it is reported unverified rather than skipped.
                python_declarations.extend(
                    DependencyDeclaration(f"--with-requirements {requirements}", None, None, "mcp", None)
                    for requirements in invocation.requirement_files
                )
                for python_declaration in python_declarations:
                    if python_declaration is not None:
                        pypi.setdefault(label, []).append(python_declaration)
                        name = python_declaration.name or python_declaration.raw
                        keys.append((label, "python", canonicalize_package_name(name)))
            for key in keys:
                names = servers.setdefault(key, [])
                if str(declaration.name) not in names:
                    names.append(str(declaration.name))
        summary = self._summary.setdefault("npm", eco.empty_ecosystem_summary())
        for label, declarations in npm.items():
            sub = ValidationResult()
            sub.add_message(f"Auditing {label} ({len(declarations)} npm package(s) an MCP runner installs)")
            self._audit_npm_declarations(sub, summary, label, declarations, total=len(declarations))
            self._attribute_mcp_servers(sub, label, "npm", servers)
            result.merge(sub)
        for label, declarations in pypi.items():
            sub = self._audit_declarations(label, declarations)
            self._attribute_mcp_servers(sub, label, "python", servers)
            result.merge(sub)
        return result

    @staticmethod
    def _attribute_mcp_servers(
        result: ValidationResult, label: str, ecosystem: str, servers: dict[tuple[str, str, str], list[str]]
    ) -> None:
        """Name the MCP server(s) that install each finding's package."""
        for finding in result.findings:
            package = finding.metadata.get("package_name") if isinstance(finding.metadata, dict) else None
            if not isinstance(package, str):
                continue
            name = canonicalize_package_name(package) if ecosystem == "python" else package
            names = servers.get((label, ecosystem, name))
            if not names:
                continue
            finding.metadata["mcp_server"] = names[0]
            if len(names) > 1:
                finding.metadata["mcp_servers"] = list(names)

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
