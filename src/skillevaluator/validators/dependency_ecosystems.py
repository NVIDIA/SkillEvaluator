# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""npm and container-image vulnerability audit for plugin dependency checks.

Declarations are parsed from plugin-controlled files that were read through a
bounded, no-follow secure read:

* ``package-lock.json`` / ``npm-shrinkwrap.json`` (lockfile v1, v2, v3) and, when
  a directory has no lockfile, ``package.json`` (``dependencies``,
  ``optionalDependencies``, ``devDependencies``);
* container images from ``docker|podman|nerdctl run`` MCP commands and from
  ``FROM`` lines in ``Dockerfile`` / ``Containerfile`` files.

Only exact versions (``1.2.3``) and exact images (a digest, or an exact version
tag) are auditable; every floating declaration gets the INFO
``dependency-version-unverified`` finding instead. Plugin files are never handed
to a scanner: exact npm pins are written into a synthesized, name-and-version-only
lockfile in a temporary directory (no ``resolved`` URLs, no ``.npmrc``, no
install scripts), and image references are validated before they reach argv.

Scanners follow the optional-tool pattern: OSV-Scanner when installed, otherwise
``npm audit --package-lock-only`` for npm, and Grype or Trivy for images. When no
scanner can produce evidence the check is marked INCOMPLETE with an install hint;
it never passes silently.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillevaluator.models.result import Finding, Severity
from skillevaluator.utils.tool_runner import ExternalTool, Tools, cvss_to_severity, parse_json_output
from skillevaluator.validators.mcp_static import (
    exact_npm_version,
    is_exact_container_image,
    is_local_spec,
    is_remote_npm_spec,
    split_npm_spec,
)

UNVERIFIED_CHECK_NAME = "dependency-version-unverified"
NPM_VULN_CHECK = "npm-vulnerability"
CONTAINER_VULN_CHECK = "container-vulnerability"
MAX_NPM_PACKAGES = 5_000
MAX_IMAGES = 32
MAX_VULN_FINDINGS_PER_SOURCE = 200
# Per-source cap on individual dependency-version-unverified findings; the rest are
# summarized in one message, so a huge manifest cannot flood reports.
MAX_UNVERIFIED_PER_SOURCE = 100
# An ecosystem summary keeps at most MAX_SUMMARY_ERRORS scanner errors, and an
# error message is cut to MAX_ERROR_CHARS.
MAX_SUMMARY_ERRORS = 8
MAX_ERROR_CHARS = 300
NPM_AUDIT_TIMEOUT = 180
OSV_TIMEOUT = 300
IMAGE_SCAN_TIMEOUT = 600
LOCKFILE_NAMES = ("package-lock.json", "npm-shrinkwrap.json")
PACKAGE_JSON = "package.json"
SEVERITY_KEYS = ("critical", "high", "medium", "low", "info")

_NPM_NAME_RE = re.compile(r"^(?:@[A-Za-z0-9._~-]+/)?[A-Za-z0-9._~-]{1,214}$")
_IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,511}$")
_SEVERITY_WORDS = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "important": Severity.HIGH,
    "moderate": Severity.MEDIUM,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "negligible": Severity.LOW,
    "info": Severity.INFO,
    "unknown": Severity.MEDIUM,
}


# --------------------------------------------------------------------------- #
# Declarations                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NpmDeclaration:
    name: str
    raw: str
    role: str
    exact_version: str | None


@dataclass(frozen=True)
class ImageDeclaration:
    image: str
    role: str
    exact: bool


def read_npm_declarations(data: Any, *, lockfile: bool) -> tuple[list[NpmDeclaration], int]:
    """The first ``MAX_NPM_PACKAGES`` declarations of a parsed manifest, and how many it declares in total.

    A ``package.json`` gives its ``dependencies``, ``optionalDependencies``, and
    ``devDependencies``; a lockfile its resolved packages (v1 ``dependencies``,
    v2/v3 ``packages``).

    A total above the returned count means the manifest was cut at the cap; the
    caller must record that, so a package past the cap is never dropped silently.
    """
    declarations: list[NpmDeclaration] = []
    total = 0
    for declaration in _package_lock_entries(data) if lockfile else _package_json_entries(data):
        total += 1
        if len(declarations) < MAX_NPM_PACKAGES:
            declarations.append(declaration)
    return declarations, total


def _package_json_entries(data: Any) -> Iterator[NpmDeclaration]:
    if not isinstance(data, dict):
        return
    for section in ("dependencies", "optionalDependencies", "devDependencies"):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for name, spec in block.items():
            text = spec if isinstance(spec, str) else json.dumps(spec)[:120]
            exact = exact_npm_version(text) if isinstance(spec, str) else None
            if isinstance(spec, str) and spec.startswith("npm:") and "@" in spec[4:].lstrip("@"):
                alias_name, _sep, alias_version = spec[4:].rpartition("@")
                if alias_name and _NPM_NAME_RE.match(alias_name):
                    name, exact = alias_name, exact_npm_version(alias_version)
            if not _NPM_NAME_RE.match(str(name)):
                exact = None
            yield NpmDeclaration(str(name), text[:200], section, exact)


def _package_lock_entries(data: Any) -> Iterator[NpmDeclaration]:
    """Every resolved package; the input is already bounded (lockfile byte, token, and collection caps)."""
    if not isinstance(data, dict):
        return
    packages = data.get("packages")
    if isinstance(packages, dict):
        for key, entry in packages.items():
            if not key or not isinstance(entry, dict) or entry.get("link") is True:
                continue
            name = entry.get("name") if isinstance(entry.get("name"), str) else key.rsplit("node_modules/", 1)[-1]
            version = entry.get("version")
            if not isinstance(version, str):
                continue
            exact = exact_npm_version(version) if _NPM_NAME_RE.match(name) else None
            role = "lockfile:dev" if entry.get("dev") is True else "lockfile"
            yield NpmDeclaration(name, f"{name}@{version}"[:200], role, exact)
        return
    stack: list[tuple[str, Any]] = list(_items(data.get("dependencies")))
    while stack:
        name, entry = stack.pop()
        if not isinstance(entry, dict):
            continue
        version = entry.get("version")
        if isinstance(version, str):
            exact = exact_npm_version(version) if _NPM_NAME_RE.match(name) else None
            yield NpmDeclaration(name, f"{name}@{version}"[:200], "lockfile", exact)
        stack.extend(_items(entry.get("dependencies")))


def _items(value: Any) -> list[tuple[str, Any]]:
    return [(str(key), item) for key, item in value.items()] if isinstance(value, dict) else []


def image_declaration(image: str, role: str) -> ImageDeclaration:
    text = image.strip()
    exact = bool(_IMAGE_REF_RE.match(text)) and "$" not in text and is_exact_container_image(text)
    return ImageDeclaration(text[:512], role, exact)


def parse_dockerfile_images(text: str) -> list[ImageDeclaration]:
    """Distinct base images named by ``FROM`` lines (build-stage references and ``scratch`` are skipped).

    At most ``MAX_IMAGES + 1`` are returned, so a caller can tell that the file names more than it audits.
    """
    logical: list[str] = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        logical.append(current + stripped)
        current = ""
    if current:
        logical.append(current)
    stages: set[str] = set()
    seen: set[str] = set()
    images: list[ImageDeclaration] = []
    for line in logical:
        tokens = line.split()
        if not tokens or tokens[0].upper() != "FROM":
            continue
        args = [token for token in tokens[1:] if not token.startswith("--")]
        if not args:
            continue
        image = args[0]
        if len(args) >= 3 and args[1].upper() == "AS":
            stages.add(args[2].lower())
        if image.lower() in stages or image.lower() == "scratch":
            continue
        declaration = image_declaration(image, "dockerfile")
        if declaration.image in seen:
            continue
        seen.add(declaration.image)
        images.append(declaration)
        if len(images) > MAX_IMAGES:
            break
    return images


# --------------------------------------------------------------------------- #
# MCP package runners                                                         #
# --------------------------------------------------------------------------- #
def npm_spec_declaration(spec: str, role: str) -> NpmDeclaration | None:
    """An npm runner spec (``pkg``, ``@scope/pkg@1.2.3``, git or URL) as a declaration; ``None`` for a local path.

    Its version is exact when the MCP pinning check calls the spec pinned: both
    use :func:`~skillevaluator.validators.mcp_static.exact_npm_version`.
    """
    text = spec.strip()
    if not text or is_local_spec(text):
        return None
    if is_remote_npm_spec(text):
        return NpmDeclaration(text[:214], text[:200], role, None)
    name, version = split_npm_spec(text)
    exact = exact_npm_version(version) if version and _NPM_NAME_RE.match(name) else None
    return NpmDeclaration(name, text[:200], role, exact)


# --------------------------------------------------------------------------- #
# Container registries                                                        #
# --------------------------------------------------------------------------- #
def image_registry(image: str) -> str | None:
    """The registry ``host[:port]`` named by an image reference, or ``None`` for Docker Hub (the default).

    Docker's reference rules: the part before the first ``/`` is a registry when
    it contains ``.`` or ``:``, is ``localhost``, or has an upper-case letter.
    """
    first, slash, _rest = image.split("@", 1)[0].partition("/")
    if slash and ("." in first or ":" in first or first == "localhost" or first != first.lower()):
        return first
    return None


def image_registry_problem(
    image: str, *, allowed_hosts: Iterable[str] = (), resolve: bool = False
) -> tuple[str | None, bool]:
    """``(why the image's registry must not be contacted, whether the policy allowlists it)``.

    The registry host goes through the same endpoint policy as MCP URLs: a
    cloud-metadata address is always refused; a loopback, private, link-local,
    or other non-public host is refused unless ``mcp.allowed_private_hosts``
    covers it. With ``resolve`` (``--resolve-endpoints``) a registry name is
    also resolved, and every answer is classified the same way.
    """
    from skillevaluator.validators import endpoint_resolution as er

    registry = image_registry(image)
    if registry is None:
        return None, False
    host, _colon, port_text = registry.rpartition(":") if ":" in registry else (registry, "", "")
    port = int(port_text) if port_text.isdigit() else 443
    try:
        verdict = er.classify_host(host, port, allowed_hosts, resolve=er.dns_resolver() if resolve else None)
    except (OSError, UnicodeError) as exc:
        return f"registry host {host!r} could not be resolved ({type(exc).__name__})", False
    blocked = verdict.blocked
    if verdict.static is not None:
        if blocked is None:
            return None, True
        if blocked[0] == "metadata":
            return f"registry {registry} is a cloud instance-metadata endpoint, which is never contacted", False
        return (
            f"registry {registry} is a {verdict.static.reason} address; allow it with mcp.allowed_private_hosts "
            "to audit images from it",
            False,
        )
    if blocked is not None and blocked[0] == "metadata":
        return f"registry {registry} resolves to a cloud instance-metadata address ({blocked[2]})", False
    if blocked is not None:
        return (
            f"registry {registry} resolves to {er.describe_address(blocked)} ({blocked[2] or 'unclassified'}); "
            "allow it with mcp.allowed_private_hosts to audit images from it",
            False,
        )
    return None, bool(verdict.non_public)


def is_dockerfile_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {"dockerfile", "containerfile"}
        or lowered.startswith(("dockerfile.", "containerfile."))
        or lowered.endswith(".dockerfile")
    )


# --------------------------------------------------------------------------- #
# Outcomes and findings                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class AuditOutcome:
    """Evidence from one ecosystem audit of one source."""

    scanner: str | None = None
    status: str = "audited"  # audited | incomplete
    error: str | None = None
    findings: list[Finding] = field(default_factory=list)
    vulnerabilities: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SEVERITY_KEYS, 0))
    omitted: int = 0
    # Exact pins past MAX_NPM_PACKAGES that were not audited; the caller records them as INCOMPLETE.
    unaudited: int = 0

    def count(self, severity: Severity) -> None:
        """Tally one vulnerability of ``severity``."""
        self.vulnerabilities[severity.value] = self.vulnerabilities.get(severity.value, 0) + 1

    def add(self, finding: Finding) -> None:
        """Tally one vulnerability and keep its finding (the first ``MAX_VULN_FINDINGS_PER_SOURCE``)."""
        self.count(finding.severity)
        if len(self.findings) < MAX_VULN_FINDINGS_PER_SOURCE:
            self.findings.append(finding)
        else:
            self.omitted += 1


_UNVERIFIED_SUGGESTIONS = {
    "python": "Pin an exact version (name==x.y.z) or audit a lockfile, then rerun the dependency audit.",
    "npm": "Pin an exact version (or an image digest), or commit a lockfile, then rerun the dependency audit.",
    "container": "Pin the image by digest (image@sha256:...) or an exact version tag, then rerun the dependency audit.",
}
_DEFAULT_UNVERIFIED_SUGGESTION = (
    "Pin an exact version or digest, or commit a lockfile, then rerun the dependency audit."
)


def unverified_finding(
    label: str,
    raw: str,
    source: str,
    *,
    ecosystem: str,
    role: str,
    kind: str,
    line_number: int | None = None,
) -> Finding:
    return Finding(
        category="DEPENDENCY",
        severity=Severity.INFO,
        check_name=UNVERIFIED_CHECK_NAME,
        message=(
            f"{label}: cannot audit a floating {kind} ('{raw[:120]}' is not an exact version or digest); "
            "vulnerability applicability was not asserted"
        ),
        file_path=source,
        line_number=line_number,
        suggestion=_UNVERIFIED_SUGGESTIONS.get(ecosystem, _DEFAULT_UNVERIFIED_SUGGESTION),
        metadata={
            "ecosystem": ecosystem,
            "package_name": label,
            "declared_constraint": raw[:200],
            "dependency_role": role,
            "resolution_status": "unverified",
        },
    )


def _severity_from_word(value: Any) -> Severity | None:
    if isinstance(value, str):
        return _SEVERITY_WORDS.get(value.strip().lower())
    return None


def _severity_from_score(value: Any) -> Severity | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return cvss_to_severity(score) if score > 0 else None


def _vuln_finding(
    *,
    check: str,
    ecosystem: str,
    source: str,
    package: str,
    version: str,
    vuln_id: str,
    severity: Severity,
    summary: str,
    fixed: str | None,
    scanner: str,
    aliases: Iterable[str] = (),
) -> Finding:
    fix = f" -> fixed in {fixed}" if fixed else ""
    detail = f" ({summary[:120]})" if summary else ""
    return Finding(
        category="DEPENDENCY",
        severity=severity,
        check_name=check,
        message=f"{package}@{version}: {vuln_id}{detail}{fix}",
        file_path=source,
        suggestion="Upgrade to a fixed version and rerun the dependency audit." if fixed else None,
        metadata={
            "ecosystem": ecosystem,
            "package_name": package,
            "package_version": version,
            "vulnerability_id": vuln_id,
            "aliases": list(aliases)[:8],
            "scanner": scanner,
        },
    )


# --------------------------------------------------------------------------- #
# Scanner output parsers                                                      #
# --------------------------------------------------------------------------- #
def parse_osv_output(data: Any, *, ecosystem: str, source: str, check: str, outcome: AuditOutcome) -> None:
    """OSV-Scanner JSON: ``results[].packages[].{package, vulnerabilities, groups}``."""
    for result in data.get("results", []) if isinstance(data, dict) else []:
        for package in result.get("packages", []) if isinstance(result, dict) else []:
            if not isinstance(package, dict):
                continue
            info = package.get("package") if isinstance(package.get("package"), dict) else {}
            name = str(info.get("name", "unknown"))
            version = str(info.get("version", "unknown"))
            vulns = {
                str(v.get("id")): v for v in package.get("vulnerabilities", []) if isinstance(v, dict) and v.get("id")
            }
            groups = [g for g in package.get("groups", []) if isinstance(g, dict)] or [
                {"ids": [vuln_id]} for vuln_id in vulns
            ]
            for group in groups:
                ids = [str(item) for item in group.get("ids", []) if item]
                if not ids:
                    continue
                primary = next((vuln_id for vuln_id in ids if vuln_id in vulns), ids[0])
                vuln = vulns.get(primary, {})
                severity = _severity_from_score(group.get("max_severity"))
                if severity is None:
                    database = vuln.get("database_specific") if isinstance(vuln.get("database_specific"), dict) else {}
                    severity = _severity_from_word(database.get("severity")) or Severity.MEDIUM
                outcome.add(
                    _vuln_finding(
                        check=check,
                        ecosystem=ecosystem,
                        source=source,
                        package=name,
                        version=version,
                        vuln_id=primary,
                        severity=severity,
                        summary=str(vuln.get("summary") or ""),
                        fixed=None,
                        scanner="osv-scanner",
                        aliases=[vuln_id for vuln_id in ids if vuln_id != primary],
                    )
                )


def parse_npm_audit_output(data: Any, pins: dict[str, str], *, source: str, outcome: AuditOutcome) -> str | None:
    """``npm audit --json`` (report v2, or legacy v1 ``advisories``). Returns an error message, if any."""
    if not isinstance(data, dict):
        return "npm audit produced no JSON report"
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("summary") or error.get("code") or "npm audit failed")[:MAX_ERROR_CHARS]
    vulnerabilities = data.get("vulnerabilities")
    if isinstance(vulnerabilities, dict):
        for name, entry in vulnerabilities.items():
            if not isinstance(entry, dict):
                continue
            for via in entry.get("via", []):
                if not isinstance(via, dict):
                    continue  # a transitive reference to another vulnerable package
                advisory = str(via.get("url") or via.get("source") or "advisory")
                vuln_id = advisory.rstrip("/").rsplit("/", 1)[-1] if "://" in advisory else advisory
                severity = (
                    (
                        _severity_from_score((via.get("cvss") or {}).get("score"))
                        if isinstance(via.get("cvss"), dict)
                        else None
                    )
                    or _severity_from_word(via.get("severity"))
                    or Severity.MEDIUM
                )
                outcome.add(
                    _vuln_finding(
                        check=NPM_VULN_CHECK,
                        ecosystem="npm",
                        source=source,
                        package=str(via.get("name") or name),
                        version=pins.get(str(via.get("name") or name), "unknown"),
                        vuln_id=vuln_id,
                        severity=severity,
                        summary=str(via.get("title") or ""),
                        fixed=None,
                        scanner="npm audit",
                    )
                )
        return None
    advisories = data.get("advisories")
    if isinstance(advisories, dict):
        for advisory in advisories.values():
            if not isinstance(advisory, dict):
                continue
            name = str(advisory.get("module_name") or "unknown")
            outcome.add(
                _vuln_finding(
                    check=NPM_VULN_CHECK,
                    ecosystem="npm",
                    source=source,
                    package=name,
                    version=pins.get(name, "unknown"),
                    vuln_id=str(advisory.get("github_advisory_id") or advisory.get("id") or "advisory"),
                    severity=_severity_from_word(advisory.get("severity")) or Severity.MEDIUM,
                    summary=str(advisory.get("title") or ""),
                    fixed=advisory.get("patched_versions")
                    if isinstance(advisory.get("patched_versions"), str)
                    else None,
                    scanner="npm audit",
                )
            )
        return None
    if "auditReportVersion" in data or "metadata" in data:
        return None
    return "npm audit produced an unrecognized report"


def parse_grype_output(data: Any, *, image: str, source: str, outcome: AuditOutcome) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("matches"), list):
        return False
    for match in data["matches"]:
        if not isinstance(match, dict):
            continue
        vuln = match.get("vulnerability") if isinstance(match.get("vulnerability"), dict) else {}
        artifact = match.get("artifact") if isinstance(match.get("artifact"), dict) else {}
        fix = vuln.get("fix") if isinstance(vuln.get("fix"), dict) else {}
        versions = fix.get("versions") if isinstance(fix.get("versions"), list) else []
        outcome.add(
            _vuln_finding(
                check=CONTAINER_VULN_CHECK,
                ecosystem="container",
                source=source,
                package=f"{image} {artifact.get('name', 'unknown')}",
                version=str(artifact.get("version", "unknown")),
                vuln_id=str(vuln.get("id", "unknown")),
                severity=_severity_from_word(vuln.get("severity")) or Severity.MEDIUM,
                summary="",
                fixed=str(versions[0]) if versions else None,
                scanner="grype",
            )
        )
    return True


def parse_trivy_output(data: Any, *, image: str, source: str, outcome: AuditOutcome) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("Results", []), list):
        return False
    for result in data.get("Results") or []:
        if not isinstance(result, dict):
            continue
        for vuln in result.get("Vulnerabilities") or []:
            if not isinstance(vuln, dict):
                continue
            outcome.add(
                _vuln_finding(
                    check=CONTAINER_VULN_CHECK,
                    ecosystem="container",
                    source=source,
                    package=f"{image} {vuln.get('PkgName', 'unknown')}",
                    version=str(vuln.get("InstalledVersion", "unknown")),
                    vuln_id=str(vuln.get("VulnerabilityID", "unknown")),
                    severity=_severity_from_word(vuln.get("Severity")) or Severity.MEDIUM,
                    summary=str(vuln.get("Title") or ""),
                    fixed=str(vuln["FixedVersion"]) if vuln.get("FixedVersion") else None,
                    scanner="trivy",
                )
            )
    return True


# --------------------------------------------------------------------------- #
# Runners                                                                     #
# --------------------------------------------------------------------------- #
def split_conflicting_pins(pins: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    """Group ``(package, pin)`` pairs so no group pins one package twice.

    Scanners that audit a flat set of pins (a synthesized npm lockfile,
    ``pip-audit --no-deps``) reject one package pinned to two versions, as
    marker-split or per-directory pins can be, so each conflicting pin goes to
    the first group that does not pin its package yet.
    """
    batches: list[dict[str, str]] = []
    for package, pin in pins:
        for batch in batches:
            if batch.get(package) in (None, pin):
                batch[package] = pin
                break
        else:
            batches.append({package: pin})
    return batches


def _write_npm_project(directory: Path, batch: dict[str, str]) -> Path:
    """Write a synthesized name/version-only project (no resolved URLs, scripts, or registries)."""
    dependencies = dict(sorted(batch.items()))
    (directory / "package.json").write_text(
        json.dumps(
            {"name": "skillevaluator-npm-audit", "version": "0.0.0", "private": True, "dependencies": dependencies}
        ),
        encoding="utf-8",
    )
    packages: dict[str, Any] = {
        "": {"name": "skillevaluator-npm-audit", "version": "0.0.0", "dependencies": dependencies}
    }
    for name, version in dependencies.items():
        packages[f"node_modules/{name}"] = {"version": version}
    lockfile = directory / "package-lock.json"
    lockfile.write_text(
        json.dumps(
            {
                "name": "skillevaluator-npm-audit",
                "version": "0.0.0",
                "lockfileVersion": 3,
                "requires": True,
                "packages": packages,
            }
        ),
        encoding="utf-8",
    )
    return lockfile


# Layered over the caller's environment: npm still needs PATH (for node), HOME, proxy, and registry settings.
# Offline mode (npm_config_offline, NPM_CONFIG_OFFLINE, or offline=true in an .npmrc) makes npm print an empty,
# clean-looking report without asking the registry, so it is turned off here and by --no-offline in argv, which
# takes precedence over every environment spelling and every .npmrc.
_NPM_AUDIT_ENV = {
    "npm_config_update_notifier": "false",
    "npm_config_fund": "false",
    "npm_config_ignore_scripts": "true",
    "npm_config_audit": "true",
    "npm_config_offline": "false",
    "NO_UPDATE_NOTIFIER": "1",
}
NPM_AUDIT_ARGS = ("audit", "--package-lock-only", "--json", "--ignore-scripts", "--no-offline")


def _run_osv_lockfile(tool: ExternalTool, lockfile: Path, *, source: str, outcome: AuditOutcome) -> str | None:
    run = tool.run(["--format", "json", "--lockfile", str(lockfile)], cwd=lockfile.parent, timeout=OSV_TIMEOUT)
    if run.error_message:
        return run.error_message
    data = parse_json_output(run.stdout)
    if run.exit_code not in (0, 1) or not isinstance(data, dict):
        detail = (run.stderr or "").strip().splitlines()
        return f"osv-scanner failed: {detail[-1][:MAX_ERROR_CHARS] if detail else f'exit code {run.exit_code}'}"
    parse_osv_output(data, ecosystem="npm", source=source, check=NPM_VULN_CHECK, outcome=outcome)
    return None


def _run_npm_audit(
    tool: ExternalTool, directory: Path, pins: dict[str, str], *, source: str, outcome: AuditOutcome
) -> str | None:
    run = tool.run(
        list(NPM_AUDIT_ARGS),
        cwd=directory,
        timeout=NPM_AUDIT_TIMEOUT,
        env=_NPM_AUDIT_ENV,
    )
    if run.error_message:
        return run.error_message
    data = parse_json_output(run.stdout)
    if not isinstance(data, dict):
        detail = (run.stderr or "").strip().splitlines()
        reason = (
            f"exit code {run.exit_code}: {detail[-1][:MAX_ERROR_CHARS]}" if detail else f"exit code {run.exit_code}"
        )
        return f"npm audit produced no JSON report ({reason})"
    return parse_npm_audit_output(data, pins, source=source, outcome=outcome)


def audit_npm_pins(pins: list[tuple[str, str]], *, source: str) -> AuditOutcome:
    """Audit exact npm pins with OSV-Scanner, else ``npm audit``; INCOMPLETE when neither works.

    At most ``MAX_NPM_PACKAGES`` distinct pins are audited; ``unaudited`` on the
    outcome counts the rest, so the caller can mark the audit INCOMPLETE.
    """
    outcome = AuditOutcome()
    distinct = list(dict.fromkeys(pins))
    unique = distinct[:MAX_NPM_PACKAGES]
    outcome.unaudited = len(distinct) - len(unique)
    candidates: list[tuple[str, ExternalTool]] = [
        (label, tool)
        for label, tool in (("osv-scanner", Tools.osv_scanner), ("npm audit", Tools.npm))
        if tool.is_available
    ]
    if not candidates:
        outcome.status = "incomplete"
        outcome.error = (
            f"no npm vulnerability scanner is installed. {Tools.osv_scanner.get_install_hint()} "
            f"(or install the npm CLI for 'npm audit')"
        )
        return outcome
    errors: list[str] = []
    for label, tool in candidates:
        attempt = AuditOutcome(scanner=label)
        failed: str | None = None
        with tempfile.TemporaryDirectory(prefix="skillevaluator-npm-audit-") as temp_dir:
            for index, batch in enumerate(split_conflicting_pins(unique)):
                batch_dir = Path(temp_dir) / f"batch-{index}"
                batch_dir.mkdir()
                lockfile = _write_npm_project(batch_dir, batch)
                if label == "osv-scanner":
                    failed = _run_osv_lockfile(tool, lockfile, source=source, outcome=attempt)
                else:
                    failed = _run_npm_audit(tool, batch_dir, batch, source=source, outcome=attempt)
                if failed:
                    break
        if not failed:
            attempt.unaudited = outcome.unaudited
            return attempt
        errors.append(f"{label}: {failed}")
    outcome.status = "incomplete"
    outcome.error = "; ".join(errors)[:600]
    return outcome


# Registry logins Grype and Trivy read from the environment: their own login
# variables, Docker's DOCKER_AUTH_CONFIG, and Podman's auth file.
_REGISTRY_LOGIN_ENV = frozenset(
    {"TRIVY_USERNAME", "TRIVY_PASSWORD", "TRIVY_PASSWORD_STDIN", "TRIVY_REGISTRY_TOKEN"}
    | {"DOCKER_AUTH_CONFIG", "REGISTRY_AUTH_FILE"}
)
_REGISTRY_LOGIN_ENV_PREFIXES = ("GRYPE_REGISTRY_AUTH", "SYFT_REGISTRY_AUTH")


def _scanner_env_without_logins(docker_config: str) -> dict[str, str]:
    """The scanner environment for a registry the user does not trust: no registry logins.

    ``docker_config`` holds an empty ``config.json``. The registry client falls
    back to Podman's auth file when it finds no Docker config, so an empty
    directory alone would still offer the Podman login.
    """
    (Path(docker_config) / "config.json").write_text('{"auths": {}}\n', encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _REGISTRY_LOGIN_ENV and not key.upper().startswith(_REGISTRY_LOGIN_ENV_PREFIXES)
    }
    env["DOCKER_CONFIG"] = docker_config
    return env


def audit_image(image: str, *, source: str, allowed_hosts: Iterable[str] = (), resolve: bool = False) -> AuditOutcome:
    """Audit one exact image with OSV-Scanner, Grype, or Trivy (first that produces evidence).

    The image's registry host is checked first (:func:`image_registry_problem`):
    a refused registry is never handed to a scanner, and the audit is
    INCOMPLETE with the reason. Grype and Trivy run with an empty Docker config
    and without the registry login variables (:data:`_REGISTRY_LOGIN_ENV`), so
    the user's registry credentials are not offered to a plugin-chosen registry
    (an allowlisted private registry keeps them).
    """
    outcome = AuditOutcome()
    if not _IMAGE_REF_RE.match(image):
        outcome.status = "incomplete"
        outcome.error = "image reference has unexpected characters; it was not passed to a scanner"
        return outcome
    problem, trusted = image_registry_problem(image, allowed_hosts=allowed_hosts, resolve=resolve)
    if problem is not None:
        outcome.status = "incomplete"
        outcome.error = f"{problem}; the image was not passed to a scanner"
        return outcome
    with tempfile.TemporaryDirectory(prefix="skillevaluator-docker-config-") as docker_config:
        return _scan_image(
            image, source=source, registry_env=None if trusted else _scanner_env_without_logins(docker_config)
        )


def _scan_image(image: str, *, source: str, registry_env: dict[str, str] | None) -> AuditOutcome:
    """Run the first scanner that produces evidence; ``registry_env`` is the complete Grype/Trivy environment."""
    outcome = AuditOutcome()
    attempts: list[tuple[str, ExternalTool, list[str], int]] = [
        ("osv-scanner", Tools.osv_scanner, ["scan", "image", "--format", "json", image], OSV_TIMEOUT),
        ("grype", Tools.grype, [f"registry:{image}", "-o", "json", "-q"], IMAGE_SCAN_TIMEOUT),
        (
            "trivy",
            Tools.trivy,
            ["image", "--format", "json", "--quiet", "--scanners", "vuln", "--image-src", "remote", image],
            IMAGE_SCAN_TIMEOUT,
        ),
    ]
    available = [attempt for attempt in attempts if attempt[1].is_available]
    if not available:
        outcome.status = "incomplete"
        outcome.error = (
            "no container vulnerability scanner is installed. Install Grype (brew install grype) or Trivy "
            "(brew install trivy), or OSV-Scanner v2 with Docker"
        )
        return outcome
    errors: list[str] = []
    for label, tool, args, timeout in available:
        attempt = AuditOutcome(scanner=label)
        # Grype and Trivy read the registry directly; OSV-Scanner goes through the docker CLI and its contexts.
        if label in {"grype", "trivy"} and registry_env is not None:
            run = tool.run(args, timeout=timeout, env=registry_env, replace_env=True)
        else:
            run = tool.run(args, timeout=timeout)
        if run.error_message:
            errors.append(f"{label}: {run.error_message}")
            continue
        data = parse_json_output(run.stdout)
        if label == "osv-scanner":
            ok = run.exit_code in (0, 1) and isinstance(data, dict)
            if ok:
                parse_osv_output(
                    data, ecosystem="container", source=source, check=CONTAINER_VULN_CHECK, outcome=attempt
                )
        elif label == "grype":
            ok = run.exit_code == 0 and parse_grype_output(data, image=image, source=source, outcome=attempt)
        else:
            ok = run.exit_code == 0 and parse_trivy_output(data, image=image, source=source, outcome=attempt)
        if ok:
            return attempt
        detail = (run.stderr or "").strip().splitlines()
        errors.append(f"{label}: {detail[-1][:200] if detail else f'exit code {run.exit_code}'}")
    outcome.status = "incomplete"
    outcome.error = "; ".join(errors)[:600]
    return outcome


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
def empty_ecosystem_summary() -> dict[str, Any]:
    return {
        "sources": 0,
        "declarations": 0,
        "audited": 0,
        "unverified": 0,
        "scanners": [],
        "status": "not_found",
        "vulnerabilities": dict.fromkeys(SEVERITY_KEYS, 0),
        "errors": [],
    }


def record_outcome(
    summary: dict[str, Any],
    outcome: AuditOutcome | None,
    *,
    declarations: int,
    audited: int,
    unverified: int,
    new_source: bool = True,
    partial: bool = False,
) -> None:
    """Fold one source's evidence into an ecosystem summary (status: audited, incomplete, no_exact).

    ``new_source=False`` adds to the source already counted (for example, the
    packages of a lockfile that were past the audit cap). An INCOMPLETE
    outcome adds no audited packages or vulnerabilities unless ``partial`` is
    set: pip-audit audits a source in batches, and the batches that ran keep
    their evidence when another batch fails.
    """
    if new_source:
        summary["sources"] += 1
    summary["declarations"] += declarations
    summary["unverified"] += unverified
    if outcome is None:
        if summary["status"] == "not_found":
            summary["status"] = "no_exact"
        return
    if outcome.scanner and outcome.scanner not in summary["scanners"]:
        summary["scanners"].append(outcome.scanner)
    if outcome.status == "incomplete":
        summary["status"] = "incomplete"
        if outcome.error and len(summary["errors"]) < MAX_SUMMARY_ERRORS:
            summary["errors"].append(outcome.error[:MAX_ERROR_CHARS])
        if not partial:
            return
    summary["audited"] += audited
    for key, value in outcome.vulnerabilities.items():
        summary["vulnerabilities"][key] = summary["vulnerabilities"].get(key, 0) + value
    if summary["status"] in {"not_found", "no_exact"}:
        summary["status"] = "audited"
