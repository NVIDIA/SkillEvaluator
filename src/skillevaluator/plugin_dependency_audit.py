# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, opt-in vulnerability collectors for a verified component tree.

Collection is entirely offline. Only ``run_dependency_audits`` invokes scanners,
and only sanitized exact versions or image references reach those tools. Scanner
processes use throwaway homes and config files, with no inherited login, proxy,
package-manager or Docker context configuration. Nothing here is registered with
public validators or CLI configuration.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json
from skillevaluator.utils.tool_runner import ExternalTool, ToolResult, Tools
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.endpoint_resolution import EndpointTarget
from skillevaluator.validators.mcp_static import (
    classify_endpoint_host,
    exact_npm_version,
    is_exact_container_image,
    is_local_spec,
    parse_mcp_runner,
    pypi_pin,
    split_npm_spec,
)
from skillevaluator.validators.url_policy import report_value

if TYPE_CHECKING:
    from skillevaluator.plugin_paths import PluginRootReader

MAX_SOURCE_FILES = 64
MAX_INPUT_FILES = 10000
MAX_DECLARATIONS = 5000
MAX_IMAGES = 32
MAX_FILE_BYTES = 1024 * 1024
MAX_LOCKFILE_BYTES = 8 * 1024 * 1024
MAX_FINDINGS = 200
MAX_REPORT_BYTES = 8 * 1024 * 1024
UNSUPPORTED_LOCKFILES = frozenset({"yarn.lock", "pnpm-lock.yaml", "bun.lock", "bun.lockb"})
NPM_LOCKFILES = ("npm-shrinkwrap.json", "package-lock.json")
_NPM_NAME = re.compile(r"(?:@[a-zA-Z0-9~][a-zA-Z0-9._~-]*/)?[a-zA-Z0-9~][a-zA-Z0-9._~-]{0,213}\Z")
_IMAGE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/:@+-]{0,511}\Z")


@dataclass(frozen=True)
class AuditSource:
    """Normalized exact versions belonging to one declared source."""

    ecosystem: str
    source: str
    pins: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ImageDeclaration:
    image: str
    source: str


@dataclass(frozen=True)
class AuditProblem:
    ecosystem: str
    source: str
    reason: str


@dataclass
class AuditInputs:
    python_sources: list[AuditSource] = field(default_factory=list)
    npm_sources: list[AuditSource] = field(default_factory=list)
    images: list[ImageDeclaration] = field(default_factory=list)
    errors: list[AuditProblem] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.errors


def _problem(inputs: AuditInputs, ecosystem: str, source: str, reason: str) -> None:
    # Reasons are fixed descriptions, never target text or exceptions.
    problem = AuditProblem(ecosystem, source, reason)
    if problem not in inputs.errors and len(inputs.errors) < MAX_SOURCE_FILES * 4:
        inputs.errors.append(problem)


def _python_pin(text: str, *, extra: str = "", runner: bool = False) -> tuple[str, str] | None:
    if runner:
        pin = pypi_pin(text)
        if pin is None or not pin.auditable:
            raise ValueError("not a registry pin")
        # uv's pkg@version spelling is normalized before PEP 508 parsing.
        if "@" in text and "==" not in text and "===" not in text:
            text = text.rsplit("@", 1)[0] + "==" + pin.version
    requirement = Requirement(text)
    if requirement.url:
        raise ValueError("not a registry pin")
    if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
        return None
    specs = list(requirement.specifier)
    if len(specs) != 1 or specs[0].operator not in {"==", "==="} or "*" in specs[0].version:
        raise ValueError("not an exact pin")
    return canonicalize_name(requirement.name), str(Version(specs[0].version))


def _python_source(inputs: AuditInputs, source: str, text: str, *, pyproject: bool) -> None:
    declarations: list[tuple[Any, str]] = []
    if pyproject:
        data = tomllib.loads(text)
        project = data.get("project")
        if not isinstance(project, dict):
            raise ValueError("static PEP 621 metadata required")
        dynamic = project.get("dynamic", [])
        if not isinstance(dynamic, list) or any(key in dynamic for key in ("dependencies", "optional-dependencies")):
            _problem(inputs, "python", source, "dynamic Python metadata is unsupported")
        if isinstance(data.get("tool"), dict) and "poetry" in data["tool"]:
            _problem(inputs, "python", source, "Poetry metadata is unsupported")
        groups = [(project.get("dependencies", []), "")]
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            _problem(inputs, "python", source, "invalid optional dependency groups")
        else:
            groups.extend((values, name) for name, values in optional.items())
        for values, extra in groups:
            if not isinstance(values, list):
                _problem(inputs, "python", source, "invalid dependency list")
            else:
                declarations.extend((value, extra) for value in values)
    else:
        for line in re.sub(r"\\\r?\n", " ", text).splitlines():
            line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"\s+--hash=sha(?:256|384|512):[a-fA-F0-9]+(?=\s|$)", "", line)
            declarations.append((line, ""))
    if len(declarations) > MAX_DECLARATIONS:
        _problem(inputs, "python", source, "dependency declaration bound exceeded")
    pins = set()
    for raw, extra in declarations[:MAX_DECLARATIONS]:
        try:
            if not isinstance(raw, str) or "\n" in raw or "\r" in raw:
                raise ValueError("invalid requirement")
            pin = _python_pin(raw, extra=extra)
            if pin:
                pins.add(pin)
        except (InvalidRequirement, KeyError, TypeError, ValueError, RecursionError):
            _problem(inputs, "python", source, "declaration requires an exact registry version; source not audited")
    inputs.python_sources.append(AuditSource("python", source, tuple(sorted(pins))))


def _npm_pin(name: Any, version: Any) -> tuple[str, str]:
    if not isinstance(name, str) or not _NPM_NAME.fullmatch(name) or not isinstance(version, str):
        raise ValueError("invalid npm declaration")
    if version.startswith("npm:"):
        name, version = split_npm_spec(version[4:])
        if not _NPM_NAME.fullmatch(name):
            raise ValueError("invalid npm alias")
    exact = exact_npm_version(version) if version is not None else None
    if exact is None:
        raise ValueError("not exact npm version")
    return name, exact


def _npm_source(inputs: AuditInputs, source: str, data: Any, *, lockfile: bool) -> None:
    if not isinstance(data, dict):
        raise ValueError("invalid npm manifest")
    entries = []
    if lockfile:
        version = data.get("lockfileVersion")
        if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2, 3}:
            raise ValueError("unsupported npm lockfile")
        if version in {2, 3}:
            packages = data.get("packages")
            if not isinstance(packages, dict):
                raise ValueError("missing lockfile packages")
            for key, entry in packages.items():
                if not key:
                    continue
                if not isinstance(key, str) or not isinstance(entry, dict):
                    raise ValueError("invalid lockfile entry")
                if entry.get("link") is True or "node_modules/" not in key:
                    continue
                entries.append((entry.get("name", key.rsplit("node_modules/", 1)[-1]), entry))
        else:
            dependencies = data.get("dependencies")
            if not isinstance(dependencies, dict):
                raise ValueError("missing lockfile dependencies")
            stack = list(dependencies.items())
            while stack:
                name, entry = stack.pop()
                if not isinstance(entry, dict):
                    raise ValueError("invalid lockfile entry")
                entries.append((name, entry))
                nested = entry.get("dependencies", {})
                if not isinstance(nested, dict):
                    raise ValueError("invalid nested lockfile entries")
                stack.extend(nested.items())
    else:
        for section in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
            block = data.get(section, {})
            if not isinstance(block, dict):
                raise ValueError("invalid npm dependency group")
            entries.extend((name, {"version": version}) for name, version in block.items())
    if len(entries) > MAX_DECLARATIONS:
        _problem(inputs, "npm", source, "dependency declaration bound exceeded")
    pins = set()
    for name, entry in entries[:MAX_DECLARATIONS]:
        try:
            resolved = entry.get("resolved")
            if resolved is not None:
                if not isinstance(resolved, str):
                    raise ValueError("invalid registry source")
                parsed = urlsplit(resolved)
                if (
                    parsed.scheme not in {"http", "https"}
                    or parsed.hostname not in {"registry.npmjs.org", "registry.yarnpkg.com"}
                    or parsed.username is not None
                    or parsed.password is not None
                ):
                    raise ValueError("non-public registry source")
            pins.add(_npm_pin(name, entry.get("version")))
        except (TypeError, ValueError):
            _problem(inputs, "npm", source, "declaration requires an exact public registry version; source not audited")
    inputs.npm_sources.append(AuditSource("npm", source, tuple(sorted(pins))))


def _add_image(inputs: AuditInputs, image: str, source: str) -> None:
    if not _IMAGE.fullmatch(image) or not is_exact_container_image(image):
        _problem(inputs, "container", source, "image requires an exact version or digest; source not audited")
        return
    if len(inputs.images) >= MAX_IMAGES:
        _problem(inputs, "container", source, "container image bound exceeded")
        return
    entry = ImageDeclaration(image, source)
    if entry not in inputs.images:
        inputs.images.append(entry)


def _dockerfile(inputs: AuditInputs, source: str, text: str) -> None:
    stages: set[str] = set()
    for line in re.sub(r"\\\r?\n", " ", text).splitlines():
        tokens = line.split()
        if not tokens or tokens[0].upper() != "FROM":
            continue
        args = tokens[1:]
        if args and args[0].startswith("--platform="):
            args = args[1:]
        if not args:
            _problem(inputs, "container", source, "invalid Dockerfile FROM instruction")
            continue
        image = args[0]
        if image.lower() != "scratch" and image.lower() not in stages:
            _add_image(inputs, image, source)
        if len(args) == 3 and args[1].upper() == "AS":
            stages.add(args[2].lower())
        elif len(args) != 1:
            _problem(inputs, "container", source, "unsupported Dockerfile FROM instruction")


def collect_audit_inputs(
    reader: PluginRootReader, files: Iterable[PurePosixPath], *, mcp: Iterable[Any] = ()
) -> AuditInputs:
    """Read selected verified paths once, with no network or target execution."""
    inputs = AuditInputs()
    sources: dict[PurePosixPath, str] = {}
    for index, raw in enumerate(files):
        if index >= MAX_INPUT_FILES:
            for ecosystem in ("python", "npm", "container"):
                _problem(inputs, ecosystem, ".", "source enumeration bound exceeded")
            break
        path = PurePosixPath(raw)
        name = path.name.lower()
        ecosystem = (
            "python"
            if (name.startswith("requirements") and name.endswith(".txt")) or name == "pyproject.toml"
            else "npm"
            if name in {"package.json", *NPM_LOCKFILES, *UNSUPPORTED_LOCKFILES}
            else "container"
            if name in {"dockerfile", "containerfile"}
            or name.startswith(("dockerfile.", "containerfile."))
            or name.endswith(".dockerfile")
            else None
        )
        if ecosystem:
            if len(sources) >= MAX_SOURCE_FILES:
                _problem(inputs, ecosystem, str(path), "dependency source file bound exceeded")
            else:
                sources[path] = ecosystem
    for path, ecosystem in sorted(sources.items()):
        source = path.as_posix()
        # A supported lockfile controls this directory; do not also audit its package.json.
        if path.name == "package.json" and any(path.with_name(name) in sources for name in NPM_LOCKFILES):
            continue
        if path.name in UNSUPPORTED_LOCKFILES:
            if not any(path.with_name(name) in sources for name in NPM_LOCKFILES):
                _problem(inputs, "npm", source, "unsupported package-manager lockfile")
            continue
        try:
            limit = MAX_LOCKFILE_BYTES if path.name in NPM_LOCKFILES else MAX_FILE_BYTES
            text = reader.read_text(path, limit, config=True)
            if ecosystem == "python":
                _python_source(inputs, source, text, pyproject=path.name == "pyproject.toml")
            elif ecosystem == "container":
                _dockerfile(inputs, source, text)
            else:
                data = load_bounded_json(text, max_tokens=250000, max_nodes=250001, max_collection_items=20000)
                _npm_source(inputs, source, data, lockfile=path.name in NPM_LOCKFILES)
        except (OSError, SecurePathError, UnicodeError, TypeError, ValueError, StructuredDataError, RecursionError):
            _problem(inputs, ecosystem, source, "cannot read or parse a bounded supported dependency source")
    for index, declaration in enumerate(mcp):
        if index >= MAX_SOURCE_FILES:
            for ecosystem in ("python", "npm", "container"):
                _problem(inputs, ecosystem, ".", "MCP declaration bound exceeded")
            break
        source = declaration.file
        invocation = parse_mcp_runner(declaration.config)
        if invocation is None:
            continue
        if invocation.requirement_files:
            _problem(inputs, "python", source, "MCP requirement-file references are not audited")
        if invocation.ecosystem == "container":
            for image in invocation.specs:
                _add_image(inputs, image, source)
        elif invocation.ecosystem == "pypi":
            pins = []
            for spec in invocation.specs:
                if is_local_spec(spec):
                    continue
                try:
                    pin = _python_pin(spec, runner=True)
                    if pin:
                        pins.append(pin)
                except (InvalidRequirement, KeyError, ValueError, TypeError):
                    _problem(inputs, "python", source, "MCP package requires an exact registry version")
            inputs.python_sources.append(AuditSource("python", source, tuple(pins)))
        elif invocation.npm_specs:
            pins = []
            for spec in invocation.npm_specs:
                if is_local_spec(spec):
                    continue
                try:
                    name, version = split_npm_spec(spec)
                    pins.append(_npm_pin(name, version))
                except (ValueError, TypeError):
                    _problem(inputs, "npm", source, "MCP package requires an exact registry version")
            inputs.npm_sources.append(AuditSource("npm", source, tuple(pins)))
    return inputs


def collect_endpoint_targets(
    mcp: Iterable[Any], hooks: Iterable[Any], *, allowed_hosts: tuple[str, ...] = ()
) -> tuple[EndpointTarget, ...]:
    """Preserve every MCP, OAuth metadata, and HTTP hook source attribution."""
    targets = []
    for declaration in mcp:
        config = declaration.config
        if not isinstance(config, dict):
            continue
        oauth = config.get("oauth")
        for field_name, value in (
            ("url", config.get("url")),
            ("oauth.authServerMetadataUrl", oauth.get("authServerMetadataUrl") if isinstance(oauth, dict) else None),
        ):
            if isinstance(value, str) and value:
                targets.append(
                    EndpointTarget(
                        value, "mcp", f"{declaration.name}.{field_name}", declaration.file, allowed_hosts=allowed_hosts
                    )
                )
    for hook in hooks:
        if isinstance(hook.url, str) and hook.url:
            targets.append(EndpointTarget(hook.url, "hook", hook.id, hook.file, allowed_hosts=allowed_hosts))
    return tuple(targets)


def _scanner_env(home: Path) -> dict[str, str]:
    """A strict process allowlist: no auth, proxies, host configs or contexts."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "LC_CTYPE"}
    }
    docker = home / "docker"
    docker.mkdir()
    (docker / "config.json").write_text('{"auths":{}}', encoding="utf-8")
    auth = home / "registry-auth.json"
    auth.write_text('{"auths":{}}', encoding="utf-8")
    npmrc = home / "npmrc"
    npmrc.write_text("registry=https://registry.npmjs.org\nignore-scripts=true\noffline=false\n", encoding="utf-8")
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(home / "config"),
        XDG_CACHE_HOME=str(home / "cache"),
        DOCKER_CONFIG=str(docker),
        REGISTRY_AUTH_FILE=str(auth),
        NPM_CONFIG_USERCONFIG=str(npmrc),
        NPM_CONFIG_GLOBALCONFIG=str(npmrc),
        NPM_CONFIG_CACHE=str(home / "npm-cache"),
        NPM_CONFIG_OFFLINE="false",
        NPM_CONFIG_IGNORE_SCRIPTS="true",
        NPM_CONFIG_UPDATE_NOTIFIER="false",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        NETRC=str(home / "empty-netrc"),
    )
    (home / "empty-netrc").touch()
    return env


def _batches(pins: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    batches: list[dict[str, str]] = []
    for name, version in sorted(set(pins)):
        batch = next((batch for batch in batches if name not in batch), None)
        if batch is None:
            batch = {}
            batches.append(batch)
        batch[name] = version
    return batches


def _run(tool: Any, args: list[str], directory: Path, env: dict[str, str], timeout: int) -> ToolResult:
    return tool.run(args, cwd=directory, env=env, replace_env=True, timeout=timeout, log_command=False)


def _report(run: ToolResult, allowed_codes: tuple[int, ...]) -> dict[str, Any]:
    if (
        run.success is not True
        or run.error_message
        or isinstance(run.exit_code, bool)
        or run.exit_code not in allowed_codes
        or len(run.stdout.encode("utf-8")) > MAX_REPORT_BYTES
    ):
        raise ValueError("scanner failed or report exceeded bounds")
    data = load_bounded_json(run.stdout, max_tokens=250000, max_nodes=250001, max_collection_items=20000)
    if not isinstance(data, dict):
        raise ValueError("invalid report")
    return data


def _advisory(ecosystem: str, source: str, name: str, version: str, advisory: str, severity: str) -> Finding:
    words = {
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
    if severity.lower() not in words or not all(isinstance(s, str) and s.strip() for s in (name, version, advisory)):
        raise ValueError("invalid advisory")
    return Finding(
        category="CVE",
        severity=words[severity.lower()],
        check_name=f"{ecosystem}-vulnerability",
        message=f"{report_value(name)} {report_value(version)}: {report_value(advisory)}",
        file_path=source,
        metadata={
            "ecosystem": ecosystem,
            "package": report_value(name),
            "version": report_value(version),
            "vulnerability_id": report_value(advisory),
        },
    )


def _npm_report(data: dict[str, Any], pins: dict[str, str], source: str, exit_code: int) -> list[Finding]:
    if data.get("error") or data.get("auditReportVersion") != 2:
        raise ValueError("unsupported npm report")
    vulnerabilities = data.get("vulnerabilities")
    metadata = data.get("metadata")
    if not isinstance(vulnerabilities, dict) or not isinstance(metadata, dict):
        raise ValueError("invalid npm report")
    counts = metadata.get("vulnerabilities")
    deps = metadata.get("dependencies")
    if not isinstance(counts, dict) or not isinstance(deps, dict):
        raise ValueError("missing npm report totals")
    for key in ("info", "low", "moderate", "high", "critical", "total"):
        if isinstance(counts.get(key), bool) or not isinstance(counts.get(key), int) or counts[key] < 0:
            raise ValueError("invalid npm totals")
    if counts["total"] != sum(counts[key] for key in ("info", "low", "moderate", "high", "critical")):
        raise ValueError("contradictory npm totals")
    if (
        isinstance(deps.get("total"), bool)
        or not isinstance(deps.get("total"), int)
        or deps["total"] < len(pins)
        or bool(vulnerabilities) != bool(counts["total"])
        or exit_code != (1 if counts["total"] else 0)
    ):
        raise ValueError("contradictory npm evidence")
    findings = []
    for name, entry in vulnerabilities.items():
        if (
            name not in pins
            or not isinstance(entry, dict)
            or not isinstance(entry.get("via"), list)
            or not entry["via"]
        ):
            raise ValueError("invalid npm advisory")
        for via in entry["via"]:
            if isinstance(via, str):
                if via not in vulnerabilities:
                    raise ValueError("unknown transitive advisory")
                continue
            if not isinstance(via, dict):
                raise ValueError("invalid npm advisory")
            package = via.get("name", name)
            if package not in pins:
                raise ValueError("unexpected package in advisory")
            advisory = via.get("url") or str(via.get("source", ""))
            findings.append(_advisory("npm", source, package, pins[package], advisory, via.get("severity", "unknown")))
    if vulnerabilities and not findings:
        raise ValueError("no concrete npm vulnerability evidence")
    return findings


def _image_report(data: dict[str, Any], *, scanner: str, source: str) -> list[Finding]:
    findings = []
    if scanner == "grype":
        if not isinstance(data.get("matches"), list):
            raise ValueError("missing Grype matches")
        for match in data["matches"]:
            if (
                not isinstance(match, dict)
                or not isinstance(match.get("vulnerability"), dict)
                or not isinstance(match.get("artifact"), dict)
            ):
                raise ValueError("invalid Grype match")
            vuln, package = match["vulnerability"], match["artifact"]
            findings.append(
                _advisory(
                    "container",
                    source,
                    package.get("name"),
                    package.get("version"),
                    vuln.get("id"),
                    vuln.get("severity", "unknown"),
                )
            )
    else:
        if not isinstance(data.get("Results"), list):
            raise ValueError("missing Trivy results")
        for target in data["Results"]:
            if not isinstance(target, dict) or not isinstance(target.get("Vulnerabilities", []), list):
                raise ValueError("invalid Trivy result")
            for vuln in target.get("Vulnerabilities", []):
                if not isinstance(vuln, dict):
                    raise ValueError("invalid Trivy advisory")
                findings.append(
                    _advisory(
                        "container",
                        source,
                        vuln.get("PkgName"),
                        vuln.get("InstalledVersion"),
                        vuln.get("VulnerabilityID"),
                        vuln.get("Severity", "unknown"),
                    )
                )
    return findings


def run_dependency_audits(
    reader: PluginRootReader,
    inputs: AuditInputs,
    *,
    python: bool = True,
    npm: bool = True,
    container: bool = True,
    tools: Mapping[str, Any] | Any | None = None,
) -> ValidationResult:
    """Explicitly audit normalized inputs, preserving incomplete source coverage."""
    result = ValidationResult()
    enabled = {"python": python, "npm": npm, "container": container}
    defaults = {"pip_audit": Tools.pip_audit}

    def tool(name: str) -> Any:
        if tools is not None:
            if isinstance(tools, Mapping):
                return tools.get(name)
            return getattr(tools, name, None)
        if name not in defaults:
            defaults[name] = ExternalTool(name, name)
        return defaults[name]

    summary = {
        name: {
            "status": "not_found" if active else "disabled",
            "sources": 0,
            "audited": 0,
            "vulnerabilities": 0,
            "errors": [],
        }
        for name, active in enabled.items()
    }
    result.metadata["plugin"] = {"cve_summary": {"method": "declared_exact_pins", "ecosystems": summary}}

    def incomplete(ecosystem: str, source: str, reason: str) -> None:
        row = summary[ecosystem]
        row["status"] = "incomplete"
        if reason not in row["errors"] and len(row["errors"]) < 8:
            row["errors"].append(reason)
        result.mark_scan_incomplete(
            {"python": "pip-audit", "npm": "npm-audit", "container": "container-image-audit"}[ecosystem]
        )
        if len(result.findings) < MAX_FINDINGS:
            result.add_finding(
                Finding(
                    category="CVE",
                    severity=Severity.INFO,
                    check_name="dependency-not-audited",
                    message=reason,
                    file_path=reader.display(source),
                    metadata={"ecosystem": ecosystem},
                )
            )

    for error in inputs.errors:
        if enabled[error.ecosystem]:
            incomplete(error.ecosystem, error.source, error.reason)
    for source in [*inputs.python_sources, *inputs.npm_sources]:
        ecosystem = source.ecosystem
        if not enabled[ecosystem]:
            continue
        row = summary[ecosystem]
        row["sources"] += 1
        if row["status"] == "not_found":
            row["status"] = "audited"
        scanner = tool("pip_audit" if ecosystem == "python" else "npm")
        if source.pins and (scanner is None or not scanner.is_available):
            incomplete(ecosystem, source.source, "required vulnerability scanner is unavailable")
            continue
        try:
            if len(source.pins) > MAX_DECLARATIONS:
                raise ValueError("dependency pin bound exceeded")
            for name, version in source.pins:
                checked = _python_pin(f"{name}=={version}") if ecosystem == "python" else _npm_pin(name, version)
                if checked != (name, version):
                    raise ValueError("inputs must contain normalized registry pins")
        except (InvalidRequirement, TypeError, ValueError):
            incomplete(ecosystem, source.source, "audit inputs contain an invalid registry pin")
            continue
        for pins in _batches(source.pins):
            try:
                with tempfile.TemporaryDirectory(prefix="skillevaluator-private-audit-") as scratch:
                    directory = Path(scratch)
                    home = directory / "home"
                    home.mkdir()
                    env = _scanner_env(home)
                    if ecosystem == "python":
                        requirements = directory / "requirements.txt"
                        requirements.write_text(
                            "".join(f"{name}=={version}\n" for name, version in pins.items()), encoding="utf-8"
                        )
                        run = _run(
                            scanner,
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
                            directory,
                            env,
                            180,
                        )
                        _report(run, (0, 1))
                        validator = DependencySecurityValidator(use_safety=False)
                        parsed = validator._validated_pip_audit_report(run.stdout)
                        if (
                            parsed is None
                            or parsed[1]
                            or {(canonicalize_name(dep["name"]), Version(dep["version"])) for dep in parsed[0]}
                            != {(name, Version(version)) for name, version in pins.items()}
                        ):
                            raise ValueError("missing pin evidence")
                        batch = ValidationResult()
                        validator._process_pip_audit_result(run, batch, source.source)
                        if batch.is_incomplete:
                            raise ValueError("untrustworthy Python evidence")
                        findings = batch.findings
                        for finding in findings:
                            finding.file_path = reader.display(source.source)
                    else:
                        project = {
                            "name": "skillevaluator-private-audit",
                            "version": "0.0.0",
                            "private": True,
                            "dependencies": pins,
                        }
                        (directory / "package.json").write_text(json.dumps(project), encoding="utf-8")
                        packages = {
                            "": {"dependencies": pins},
                            **{f"node_modules/{name}": {"version": version} for name, version in pins.items()},
                        }
                        (directory / "package-lock.json").write_text(
                            json.dumps(
                                {
                                    "name": project["name"],
                                    "version": "0.0.0",
                                    "lockfileVersion": 3,
                                    "requires": True,
                                    "packages": packages,
                                }
                            ),
                            encoding="utf-8",
                        )
                        run = _run(
                            scanner,
                            [
                                "audit",
                                "--package-lock-only",
                                "--json",
                                "--ignore-scripts",
                                "--no-offline",
                                "--registry=https://registry.npmjs.org",
                            ],
                            directory,
                            env,
                            180,
                        )
                        findings = _npm_report(_report(run, (0, 1)), pins, reader.display(source.source), run.exit_code)
                row["audited"] += len(pins)
                row["vulnerabilities"] += len(findings)
                for finding in findings[: max(0, MAX_FINDINGS - len(result.findings))]:
                    result.add_finding(finding)
                if len(findings) > MAX_FINDINGS:
                    incomplete(ecosystem, source.source, "vulnerability finding bound exceeded")
            except (OSError, UnicodeError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
                incomplete(
                    ecosystem, source.source, "scanner returned incomplete or untrustworthy vulnerability evidence"
                )
    if container:
        for declaration in inputs.images:
            row = summary["container"]
            row["sources"] += 1
            if row["status"] == "not_found":
                row["status"] = "audited"
            if not _IMAGE.fullmatch(declaration.image) or not is_exact_container_image(declaration.image):
                incomplete("container", declaration.source, "audit inputs contain an invalid exact image reference")
                continue
            first = declaration.image.split("/", 1)[0]
            registry = (
                first
                if "/" in declaration.image
                and ("." in first or ":" in first or first == "localhost" or first != first.lower())
                else "registry-1.docker.io"
            )
            try:
                parsed_registry = urlsplit("https://" + registry)
                host = parsed_registry.hostname or ""
                _ = parsed_registry.port
                if not host or classify_endpoint_host(host) is not None:
                    raise ValueError("non-public registry")
            except ValueError:
                incomplete(
                    "container", declaration.source, "container registry is not a public endpoint; it was not contacted"
                )
                continue
            audited = False
            for name in ("grype", "trivy"):
                scanner = tool(name)
                if scanner is None or not scanner.is_available:
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix="skillevaluator-private-image-") as scratch:
                        directory = Path(scratch)
                        home = directory / "home"
                        home.mkdir()
                        env = _scanner_env(home)
                        args = (
                            [f"registry:{declaration.image}", "-o", "json", "-q"]
                            if name == "grype"
                            else [
                                "image",
                                "--format",
                                "json",
                                "--quiet",
                                "--scanners",
                                "vuln",
                                "--image-src",
                                "remote",
                                declaration.image,
                            ]
                        )
                        run = _run(scanner, args, directory, env, 600)
                        findings = _image_report(
                            _report(run, (0,)), scanner=name, source=reader.display(declaration.source)
                        )
                    row["audited"] += 1
                    row["vulnerabilities"] += len(findings)
                    for finding in findings[: max(0, MAX_FINDINGS - len(result.findings))]:
                        result.add_finding(finding)
                    if len(findings) > MAX_FINDINGS:
                        incomplete("container", declaration.source, "vulnerability finding bound exceeded")
                    audited = True
                    break
                except (OSError, UnicodeError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
                    continue
            if not audited:
                incomplete(
                    "container", declaration.source, "no container scanner produced trustworthy vulnerability evidence"
                )
    return result
