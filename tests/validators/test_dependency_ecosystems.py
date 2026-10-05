# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""npm and container-image CVE audit for plugins (scanners are faked; nothing is installed or fetched)."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators import dependency_ecosystems as eco
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.plugin_tree import plugin_tree_scope


class FakeTool:
    def __init__(self, name: str, responses: list[ToolResult] | None = None, *, available: bool = True) -> None:
        self.name = name
        self.command = name
        self.responses = list(responses or [])
        self.available = available
        self.calls: list[dict[str, Any]] = []

    @property
    def is_available(self) -> bool:
        return self.available

    def get_install_hint(self) -> str:
        return f"install {self.name}"

    def run(self, args: list[str], **kwargs: Any) -> ToolResult:
        files = {}
        cwd = kwargs.get("cwd")
        if cwd is not None:
            files = {path.name: path.read_text() for path in Path(cwd).iterdir() if path.is_file()}
        self.calls.append({"args": list(args), "files": files, **kwargs})
        if not self.responses:
            return ToolResult(True, "{}", "", 0)
        return self.responses.pop(0)


def _ok(payload: Any, exit_code: int = 0) -> ToolResult:
    return ToolResult(True, json.dumps(payload), "", exit_code)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeTool]:
    fakes = {name: FakeTool(name, available=False) for name in ("osv_scanner", "npm", "grype", "trivy")}
    for name, fake in fakes.items():
        monkeypatch.setattr(Tools, name, fake)
    return fakes


OSV_REPORT = {
    "results": [
        {
            "packages": [
                {
                    "package": {"name": "lodash", "version": "4.17.20", "ecosystem": "npm"},
                    "vulnerabilities": [
                        {"id": "GHSA-35jh-r3h4-6jhm", "aliases": ["CVE-2021-23337"], "summary": "Command Injection"}
                    ],
                    "groups": [{"ids": ["GHSA-35jh-r3h4-6jhm", "CVE-2021-23337"], "max_severity": "7.2"}],
                }
            ]
        }
    ]
}
NPM_AUDIT_REPORT = {
    "auditReportVersion": 2,
    "vulnerabilities": {
        "lodash": {
            "name": "lodash",
            "severity": "high",
            "via": [
                {
                    "source": 1,
                    "name": "lodash",
                    "title": "Command Injection",
                    "url": "https://github.com/advisories/GHSA-35jh-r3h4-6jhm",
                    "severity": "high",
                    "cvss": {"score": 7.2},
                }
            ],
        },
        "wrapper": {"name": "wrapper", "severity": "high", "via": ["lodash"]},
    },
}


# --------------------------------------------------------------------------- #
# Parsers                                                                     #
# --------------------------------------------------------------------------- #
def test_package_json_exact_and_floating_versions() -> None:
    declarations = eco.parse_package_json(
        {
            "dependencies": {"lodash": "4.17.20", "left-pad": "^1.3.0", "alias": "npm:real-pkg@2.0.1"},
            "devDependencies": {"jest": "=29.7.0", "git-dep": "github:org/repo"},
        }
    )
    exact = {d.name: d.exact_version for d in declarations}
    assert exact == {"lodash": "4.17.20", "left-pad": None, "real-pkg": "2.0.1", "jest": "29.7.0", "git-dep": None}


def test_package_lock_v3_and_v1() -> None:
    v3 = {
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "root"},
            "node_modules/a": {"version": "1.0.0"},
            "node_modules/a/node_modules/b": {"version": "2.0.0", "dev": True},
            "node_modules/linked": {"link": True, "resolved": "../x"},
            "node_modules/local": {"version": "file:../local"},
        },
    }
    assert [(d.name, d.exact_version) for d in eco.parse_package_lock(v3)] == [
        ("a", "1.0.0"),
        ("b", "2.0.0"),
        ("local", None),
    ]
    v1 = {"dependencies": {"x": {"version": "1.2.3", "dependencies": {"y": {"version": "0.1.0"}}}}}
    assert sorted((d.name, d.exact_version) for d in eco.parse_package_lock(v1)) == [("x", "1.2.3"), ("y", "0.1.0")]


def test_dockerfile_images_skip_stages_and_scratch() -> None:
    text = (
        "# comment\nARG BASE=node\nFROM --platform=linux/amd64 node:20.11.1 AS build\n"
        "FROM build\nFROM scratch\nFROM ${BASE}:20\nFROM python@sha256:" + "a" * 64 + "\nFROM ubuntu\n"
    )
    images = [(image.image, image.exact) for image in eco.parse_dockerfile_images(text)]
    assert images == [
        ("node:20.11.1", True),
        ("${BASE}:20", False),
        ("python@sha256:" + "a" * 64, True),
        ("ubuntu", False),
    ]


def test_image_refs_that_look_like_flags_are_never_exact() -> None:
    assert eco.image_declaration("--privileged", "mcp").exact is False
    assert eco.image_declaration("ghcr.io/org/img:1.2.3", "mcp").exact is True


def test_unverified_finding_for_another_ecosystem_gets_the_default_suggestion() -> None:
    """Regression: an ecosystem without its own suggestion raised KeyError."""
    finding = eco.unverified_finding("serde", "^1", "Cargo.toml", ecosystem="cargo", role="runtime", kind="crate")
    assert finding.severity == Severity.INFO
    assert finding.check_name == eco.UNVERIFIED_CHECK_NAME
    assert finding.suggestion == eco._DEFAULT_UNVERIFIED_SUGGESTION
    assert finding.metadata["ecosystem"] == "cargo"


def test_npm_audit_parser_skips_transitive_via_entries() -> None:
    outcome = eco.AuditOutcome(scanner="npm audit")
    error = eco.parse_npm_audit_output(NPM_AUDIT_REPORT, {"lodash": "4.17.20"}, source="package.json", outcome=outcome)
    assert error is None
    [finding] = outcome.findings
    assert finding.check_name == "npm-vulnerability"
    assert finding.severity == Severity.HIGH
    assert finding.message.startswith("lodash@4.17.20: GHSA-35jh-r3h4-6jhm")
    assert (
        eco.parse_npm_audit_output(
            {"error": {"code": "ENOTFOUND", "summary": "offline"}}, {}, source="x", outcome=outcome
        )
        == "offline"
    )


def test_grype_and_trivy_parsers() -> None:
    outcome = eco.AuditOutcome(scanner="grype")
    grype = {
        "matches": [
            {
                "vulnerability": {"id": "CVE-1", "severity": "Critical", "fix": {"versions": ["1.1"]}},
                "artifact": {"name": "openssl", "version": "1.0"},
            }
        ]
    }
    assert eco.parse_grype_output(grype, image="img:1", source="Dockerfile", outcome=outcome)
    trivy = {
        "Results": [
            {
                "Vulnerabilities": [
                    {"VulnerabilityID": "CVE-2", "PkgName": "zlib", "InstalledVersion": "1", "Severity": "LOW"}
                ]
            }
        ]
    }
    assert eco.parse_trivy_output(trivy, image="img:1", source="Dockerfile", outcome=outcome)
    assert [f.severity for f in outcome.findings] == [Severity.CRITICAL, Severity.LOW]
    assert outcome.vulnerabilities["critical"] == 1


# --------------------------------------------------------------------------- #
# Scanner selection                                                           #
# --------------------------------------------------------------------------- #
def test_osv_scanner_is_preferred_and_gets_a_synthesized_lockfile(tools: dict[str, FakeTool]) -> None:
    tools["osv_scanner"].available = True
    tools["osv_scanner"].responses = [_ok(OSV_REPORT, exit_code=1)]
    tools["npm"].available = True
    outcome = eco.audit_npm_pins([("lodash", "4.17.20")], source="package.json")
    assert outcome.scanner == "osv-scanner"
    assert outcome.status == "audited"
    [finding] = outcome.findings
    assert finding.severity == Severity.HIGH
    assert finding.metadata["aliases"] == ["CVE-2021-23337"]
    [call] = tools["osv_scanner"].calls
    assert call["args"][:3] == ["--format", "json", "--lockfile"]
    lock = json.loads(call["files"]["package-lock.json"])
    assert lock["packages"]["node_modules/lodash"] == {"version": "4.17.20"}
    assert "resolved" not in json.dumps(lock)
    assert tools["npm"].calls == []


def test_npm_audit_fallback_is_package_lock_only(tools: dict[str, FakeTool]) -> None:
    tools["npm"].available = True
    tools["npm"].responses = [_ok(NPM_AUDIT_REPORT, exit_code=1)]
    outcome = eco.audit_npm_pins([("lodash", "4.17.20")], source="package.json")
    assert outcome.scanner == "npm audit"
    [call] = tools["npm"].calls
    assert call["args"] == ["audit", "--package-lock-only", "--json", "--ignore-scripts", "--no-offline"]
    assert call["env"]["npm_config_ignore_scripts"] == "true"
    assert call["env"]["npm_config_offline"] == "false"
    assert set(call["files"]) == {"package.json", "package-lock.json"}


def test_npm_audit_offline_error_is_incomplete(tools: dict[str, FakeTool]) -> None:
    tools["npm"].available = True
    tools["npm"].responses = [_ok({"error": {"code": "ENOTFOUND", "summary": "request to registry failed"}}, 1)]
    outcome = eco.audit_npm_pins([("lodash", "4.17.20")], source="package.json")
    assert outcome.status == "incomplete"
    assert "registry failed" in (outcome.error or "")


def test_npm_audit_without_json_reports_exit_code_and_stderr(tools: dict[str, FakeTool]) -> None:
    tools["npm"].available = True
    tools["npm"].responses = [ToolResult(True, "", "env: node: No such file or directory", 127)]
    outcome = eco.audit_npm_pins([("lodash", "4.17.20")], source="package.json")
    assert outcome.status == "incomplete"
    assert "exit code 127" in (outcome.error or "")
    assert "node: No such file or directory" in (outcome.error or "")


def test_no_npm_scanner_is_incomplete_with_install_hint(tools: dict[str, FakeTool]) -> None:
    outcome = eco.audit_npm_pins([("lodash", "4.17.20")], source="package.json")
    assert outcome.status == "incomplete"
    assert "install osv_scanner" in (outcome.error or "")


def test_image_scanners_fall_back_in_order(tools: dict[str, FakeTool]) -> None:
    tools["osv_scanner"].available = True
    tools["osv_scanner"].responses = [ToolResult(True, "", "docker not found", 127)]
    tools["grype"].available = True
    tools["grype"].responses = [_ok({"matches": []})]
    outcome = eco.audit_image("node:20.11.1", source="Dockerfile")
    assert outcome.scanner == "grype"
    assert outcome.status == "audited"
    assert tools["grype"].calls[0]["args"] == ["registry:node:20.11.1", "-o", "json", "-q"]


def test_no_image_scanner_is_incomplete(tools: dict[str, FakeTool]) -> None:
    outcome = eco.audit_image("node:20.11.1", source="Dockerfile")
    assert outcome.status == "incomplete"
    assert "Grype" in (outcome.error or "")


# --------------------------------------------------------------------------- #
# Validator integration                                                       #
# --------------------------------------------------------------------------- #
def _plugin(root: Path, files: dict[str, Any]) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    manifest = {
        "name": "demo",
        "mcpServers": {
            "db": {"command": "docker", "args": ["run", "-i", "--rm", "ghcr.io/example/db-mcp@sha256:" + "b" * 64]},
            "floating": {"command": "podman", "args": ["run", "example/tool:latest"]},
        },
    }
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest))
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content))
    return root


def _dependency_result(root: Path) -> ValidationResult:
    [result] = run_validation(root, checks="dependency", content_type=CONTENT_TYPE_PLUGIN)
    return result


def test_plugin_dependency_audit_covers_npm_and_images(tmp_path: Path, tools: dict[str, FakeTool]) -> None:
    tools["osv_scanner"].available = True
    tools["osv_scanner"].responses = [_ok(OSV_REPORT, 1), _ok({"results": []}), _ok({"results": []})]
    root = _plugin(
        tmp_path / "demo",
        {
            "servers/api/package.json": {"dependencies": {"lodash": "4.17.20", "left-pad": "^1.3.0"}},
            "docker/Dockerfile": "FROM node:20.11.1\nFROM alpine:latest\n",
        },
    )
    result = _dependency_result(root)
    checks = [(f.check_name, f.severity) for f in result.findings]
    assert ("npm-vulnerability", Severity.HIGH) in checks
    unverified = [f for f in result.findings if f.check_name == "dependency-version-unverified"]
    assert {f.metadata["ecosystem"] for f in unverified} == {"npm", "container"}
    assert {f.metadata["package_name"] for f in unverified} == {"left-pad", "example/tool:latest", "alpine:latest"}
    assert not result.passed
    summary = result.metadata["plugin"]["cve_summary"]["ecosystems"]
    assert summary["npm"]["audited"] == 1
    assert summary["npm"]["unverified"] == 1
    assert summary["npm"]["scanners"] == ["osv-scanner"]
    assert summary["npm"]["vulnerabilities"]["high"] == 1
    assert summary["container"]["audited"] == 2
    assert summary["container"]["status"] == "audited"
    image_calls = [call["args"] for call in tools["osv_scanner"].calls if call["args"][:2] == ["scan", "image"]]
    assert sorted(args[-1] for args in image_calls) == ["ghcr.io/example/db-mcp@sha256:" + "b" * 64, "node:20.11.1"]


def test_plugin_dependency_audit_without_scanners_is_incomplete(tmp_path: Path, tools: dict[str, FakeTool]) -> None:
    root = _plugin(
        tmp_path / "demo",
        {"package-lock.json": {"lockfileVersion": 3, "packages": {"node_modules/a": {"version": "1.0.0"}}}},
    )
    result = _dependency_result(root)
    assert result.is_incomplete
    assert set(result.incomplete_scans) == {"npm-audit", "container-image-audit"}
    assert result.metadata["plugin"]["cve_summary"]["ecosystems"]["npm"]["status"] == "incomplete"


def test_lockfile_takes_precedence_over_package_json(tmp_path: Path, tools: dict[str, FakeTool]) -> None:
    tools["osv_scanner"].available = True
    root = _plugin(
        tmp_path / "demo",
        {
            "package.json": {"dependencies": {"a": "^1.0.0"}},
            "package-lock.json": {"lockfileVersion": 3, "packages": {"node_modules/a": {"version": "1.0.4"}}},
        },
    )
    result = _dependency_result(root)
    assert not [f for f in result.findings if f.metadata.get("package_name") == "a"]
    lock = json.loads(tools["osv_scanner"].calls[0]["files"]["package-lock.json"])
    assert lock["packages"]["node_modules/a"]["version"] == "1.0.4"


def test_standalone_skill_audit_ignores_npm_and_images(tmp_path: Path, tools: dict[str, FakeTool]) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: A skill.\n---\nBody\n")
    (skill / "package.json").write_text(json.dumps({"dependencies": {"lodash": "4.17.20"}}))
    (skill / "Dockerfile").write_text("FROM node:20.11.1\n")
    result = DependencySecurityValidator().validate(skill)
    assert not result.is_incomplete
    assert "plugin" not in result.metadata
    assert all(call == [] for call in (tool.calls for tool in tools.values()))


def test_bundled_skill_npm_and_image_findings_name_the_skill_directory_once(
    tmp_path: Path, tools: dict[str, FakeTool]
) -> None:
    """Regression: npm and image findings in a bundled skill pointed at skills/foo/skills/foo/package.json."""
    tools["osv_scanner"].available = True
    tools["osv_scanner"].responses = [_ok(OSV_REPORT, 1), _ok({"results": []})]
    root = _bare_plugin(
        tmp_path / "demo",
        {
            "skills/foo/SKILL.md": "---\nname: foo\ndescription: A bundled skill.\n---\nBody\n",
            "skills/foo/package.json": {"dependencies": {"lodash": "4.17.20", "left-pad": "^1.3.0"}},
            "skills/foo/Dockerfile": "FROM node:20.11.1\nFROM alpine:latest\n",
        },
    )
    result = _dependency_result(root)
    assert sorted((f.check_name, f.metadata["package_name"], f.file_path) for f in result.findings) == [
        ("dependency-version-unverified", "alpine:latest", "[foo] skills/foo/Dockerfile"),
        ("dependency-version-unverified", "left-pad", "[foo] skills/foo/package.json"),
        ("npm-vulnerability", "lodash", "[foo] skills/foo/package.json"),
    ]
    assert [call["args"][-1] for call in tools["osv_scanner"].calls if call["args"][:2] == ["scan", "image"]] == [
        "node:20.11.1"
    ]


# --------------------------------------------------------------------------- #
# Never passes silently                                                       #
# --------------------------------------------------------------------------- #
_IMAGE = "ghcr.io/example/db-mcp:1.2.3"
_DOCKER_SERVER = {"type": "stdio", "command": "docker", "args": ["run", "-i", "--rm", _IMAGE]}
_AP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
_AP_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"


def _bare_plugin(root: Path, files: dict[str, Any]) -> Path:
    """A plugin without MCP servers, so only the files under test feed the audit."""
    for rel, content in {".claude-plugin/plugin.json": {"name": "demo"}, **files}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content if isinstance(content, str) else json.dumps(content))
    return root


def _lockfile(count: int) -> dict[str, Any]:
    packages: dict[str, Any] = {"": {"name": "srv", "version": "1.0.0"}}
    for index in range(count):
        packages[f"node_modules/pkg{index}"] = {
            "version": "1.0.0",
            "resolved": f"https://registry.npmjs.org/pkg{index}/-/pkg{index}-1.0.0.tgz",
            "integrity": "sha512-abc",
        }
    return {"name": "srv", "lockfileVersion": 3, "packages": packages}


def _summary(result: ValidationResult, ecosystem: str) -> dict[str, Any]:
    return result.metadata["plugin"]["cve_summary"]["ecosystems"][ecosystem]


def test_lockfile_larger_than_the_manifest_budget_is_audited(tmp_path: Path, tools: dict[str, FakeTool]) -> None:
    """Regression: a 1,100-package lockfile exceeded the 1,024-entry budget and was skipped with only a warning."""
    tools["osv_scanner"].available = True
    root = _bare_plugin(
        tmp_path / "demo",
        {"server/package-lock.json": _lockfile(1100), "server/package.json": {"dependencies": {"pkg1": "^1.0.0"}}},
    )
    result = _dependency_result(root)
    assert not result.is_incomplete
    npm = _summary(result, "npm")
    assert (npm["status"], npm["declarations"], npm["audited"]) == ("audited", 1100, 1100)
    [call] = tools["osv_scanner"].calls
    assert len(json.loads(call["files"]["package-lock.json"])["packages"]) == 1101


@pytest.mark.parametrize(
    ("lockfile", "limits", "reason"),
    [
        (b'{"lockfileVersion": 3, "packages": {', {}, "not valid JSON"),
        (b'{"lockfileVersion": 3, "name": "caf\xe9"}', {}, "can't decode"),
        (json.dumps(_lockfile(3)).encode(), {"MAX_LOCKFILE_BYTES": 64}, "64-byte limit"),
        (json.dumps(_lockfile(20)).encode(), {"MAX_LOCKFILE_COLLECTION_ITEMS": 10}, "collection size exceeds 10"),
    ],
    ids=["corrupt", "not-utf8", "over-byte-cap", "over-collection-cap"],
)
def test_unreadable_lockfile_is_incomplete_and_falls_back_to_package_json(
    tmp_path: Path,
    tools: dict[str, FakeTool],
    monkeypatch: pytest.MonkeyPatch,
    lockfile: bytes,
    limits: dict[str, int],
    reason: str,
) -> None:
    from skillevaluator.validators import dependencies

    for name, value in limits.items():
        monkeypatch.setattr(dependencies, name, value)
    tools["osv_scanner"].available = True
    root = _bare_plugin(
        tmp_path / "demo",
        {"server/package-lock.json": lockfile, "server/package.json": {"dependencies": {"lodash": "4.17.20"}}},
    )
    result = _dependency_result(root)
    assert result.incomplete_scans == ["npm-audit"]
    assert not result.passed
    npm = _summary(result, "npm")
    assert npm["status"] == "incomplete"
    [error] = npm["errors"]
    assert error.startswith("could not read npm manifest safely:")
    assert reason in error
    assert any(w.startswith("server/package-lock.json: dependency audit incomplete:") for w in result.warnings)
    # The directory's package.json is audited in the lockfile's place.
    assert "server/package.json: audited in place of unreadable server/package-lock.json" in result.messages
    [call] = tools["osv_scanner"].calls
    assert json.loads(call["files"]["package-lock.json"])["packages"]["node_modules/lodash"] == {"version": "4.17.20"}
    assert npm["audited"] == 1


@pytest.mark.parametrize(
    "files",
    [
        {"server/package-lock.json": b"{not json"},
        {"server/package.json": b'{"dependencies": '},
    ],
    ids=["lockfile-only", "package-json-only"],
)
def test_unreadable_npm_manifest_without_fallback_is_incomplete(
    tmp_path: Path, tools: dict[str, FakeTool], files: dict[str, bytes]
) -> None:
    tools["osv_scanner"].available = True
    result = _dependency_result(_bare_plugin(tmp_path / "demo", files))
    assert result.incomplete_scans == ["npm-audit"]
    assert _summary(result, "npm")["status"] == "incomplete"
    assert tools["osv_scanner"].calls == []


def test_unreadable_dockerfile_is_incomplete(tmp_path: Path, tools: dict[str, FakeTool]) -> None:
    tools["osv_scanner"].available = True
    root = _bare_plugin(tmp_path / "demo", {"docker/Dockerfile": b"# caf\xe9\nFROM node:20.11.1\n"})
    result = _dependency_result(root)
    assert result.incomplete_scans == ["container-image-audit"]
    container = _summary(result, "container")
    assert container["status"] == "incomplete"
    assert container["errors"][0].startswith("could not read Dockerfile safely:")
    assert any(w.startswith("docker/Dockerfile: dependency audit incomplete:") for w in result.warnings)


def test_ecosystem_discovery_failure_is_incomplete(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.utils.secure_fs import SecurePathError

    def _fail(_directory: Path, _selected: Any) -> list:
        raise SecurePathError("path_count_limit", "tree exceeds the path limit")

    monkeypatch.setattr(DependencySecurityValidator, "_discover", staticmethod(_fail))
    result = _dependency_result(_bare_plugin(tmp_path / "demo", {}))
    assert set(result.incomplete_scans) == {"npm-audit", "container-image-audit"}
    assert "npm manifest discovery failed" in _summary(result, "npm")["errors"][0]
    assert "Dockerfile discovery failed" in _summary(result, "container")["errors"][0]


def test_npm_manifests_beyond_the_discovery_cap_are_incomplete(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: manifests past the discovery cap were dropped silently, so the audit could pass."""
    tools["osv_scanner"].available = True
    monkeypatch.setattr("skillevaluator.validators.dependencies.MAX_ECOSYSTEM_FILES", 1)
    root = _bare_plugin(
        tmp_path / "demo",
        {f"servers/{name}/package.json": {"dependencies": {"lodash": "4.17.20"}} for name in ("a", "b")},
    )
    result = _dependency_result(root)
    assert result.incomplete_scans == ["npm-audit"]
    npm = _summary(result, "npm")
    assert npm["status"] == "incomplete"
    assert "2 npm manifests found; only the first 1 were audited" in npm["errors"]


def test_dockerfiles_and_images_beyond_their_caps_are_incomplete(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: Dockerfiles and images past their caps were dropped silently, so the audit could pass."""
    tools["grype"].available = True
    tools["grype"].responses = [_ok({"matches": []}) for _ in range(4)]
    monkeypatch.setattr("skillevaluator.validators.dependencies.MAX_ECOSYSTEM_FILES", 1)
    root = _bare_plugin(tmp_path / "demo", {f"{name}/Dockerfile": "FROM node:20.11.1\n" for name in ("a", "b")})
    container = _summary(_dependency_result(root), "container")
    assert container["status"] == "incomplete"
    assert "2 Dockerfiles found; only the first 1 were audited" in container["errors"]

    monkeypatch.setattr("skillevaluator.validators.dependencies.MAX_ECOSYSTEM_FILES", 64)
    monkeypatch.setattr(eco, "MAX_IMAGES", 2)
    tools["grype"].calls.clear()
    # A repeated image counts once toward the cap; three distinct images exceed it.
    dockerfile = "FROM node:20.11.1\nFROM node:20.11.1\nFROM alpine:3.19.1\nFROM debian:12.5\n"
    root = _bare_plugin(tmp_path / "images", {"docker/Dockerfile": dockerfile})
    result = _dependency_result(root)
    assert result.incomplete_scans == ["container-image-audit"]
    container = _summary(result, "container")
    assert "3 container images found; only the first 2 were audited" in container["errors"]
    assert [call["args"][0] for call in tools["grype"].calls] == ["registry:node:20.11.1", "registry:alpine:3.19.1"]


@pytest.mark.parametrize(
    "files",
    [
        {".claude-plugin/plugin.json": {"name": "demo", "mcpServers": {"db": _DOCKER_SERVER}}},
        {".cursor-plugin/plugin.json": {"name": "demo"}, "mcp.json": {"mcpServers": {"db": _DOCKER_SERVER}}},
        {".cursor-plugin/plugin.json": {"name": "demo", "mcpServers": {"db": _DOCKER_SERVER}}},
        {
            "plugin.json": {"$schema": _AP_SCHEMA, "name": "demo"},
            "mcp.json": {"$schema": _AP_MCP_SCHEMA, "mcpServers": {"db": _DOCKER_SERVER}},
        },
        {
            ".codex-plugin/plugin.json": {"name": "demo", "mcpServers": "./.mcp.json"},
            ".mcp.json": {"mcpServers": {"db": _DOCKER_SERVER}},
        },
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            "plugin.json": {"$schema": _AP_SCHEMA, "name": "demo"},
            "mcp.json": {"$schema": _AP_MCP_SCHEMA, "mcpServers": {"db": _DOCKER_SERVER}},
        },
    ],
    ids=["claude", "cursor-mcp-json", "cursor-inline", "agent-plugins", "codex", "additional-agent-plugins"],
)
def test_mcp_images_follow_each_manifest_format(
    tmp_path: Path, tools: dict[str, FakeTool], files: dict[str, Any]
) -> None:
    """Regression: the root mcp.json of Cursor and Agent Plugins plugins was never read for container images."""
    tools["grype"].available = True
    tools["grype"].responses = [_ok({"matches": []})]
    root = tmp_path / "demo"
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(content))
    result = _dependency_result(root)
    assert [call["args"] for call in tools["grype"].calls] == [[f"registry:{_IMAGE}", "-o", "json", "-q"]]
    container = _summary(result, "container")
    assert (container["sources"], container["audited"], container["status"]) == (1, 1, "audited")


def test_mcp_file_shared_by_two_manifests_lists_its_image_once(tmp_path: Path) -> None:
    root = _bare_plugin(
        tmp_path / "demo",
        {".codex-plugin/plugin.json": {"name": "demo"}, ".mcp.json": {"mcpServers": {"db": _DOCKER_SERVER}}},
    )
    assert DependencySecurityValidator()._mcp_images(root) == [(_IMAGE, ".mcp.json (mcpServers['db'])")]


def _fake_pip_audit(monkeypatch: pytest.MonkeyPatch, *responses: ToolResult) -> FakeTool:
    fake = FakeTool("pip-audit", list(responses))
    monkeypatch.setattr(Tools, "pip_audit", fake)
    monkeypatch.setattr(Tools, "safety", FakeTool("safety", available=False))
    return fake


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (ToolResult(False, "", "", -1, "pip-audit timed out after 180 seconds"), "pip-audit timed out"),
        (ToolResult(True, "", "ERROR: network unreachable\n", 1), "pip-audit failed: ERROR: network unreachable"),
        (ToolResult(True, "", "", 0), "pip-audit produced no JSON report"),
    ],
    ids=["timeout", "offline", "no-report"],
)
def test_pip_audit_failure_makes_the_plugin_python_audit_incomplete(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch, response: ToolResult, error: str
) -> None:
    """Regression: a pip-audit timeout or offline run produced a clean 'Audited' Python row with zeros."""
    fake = _fake_pip_audit(monkeypatch, response)
    result = _dependency_result(_bare_plugin(tmp_path / "demo", {"requirements.txt": "requests==2.19.0\n"}))
    assert len(fake.calls) == 1
    assert result.incomplete_scans == ["pip-audit"]
    assert not result.passed
    python = _summary(result, "python")
    assert (python["status"], python["audited"], python["scanners"]) == ("incomplete", 0, [])
    assert python["errors"] == [f"requirements.txt: {response.error_message or error}"]
    assert any(error in warning for warning in result.warnings)


def test_pip_audit_evidence_keeps_the_plugin_python_audit_complete(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_pip_audit(monkeypatch, _ok({"dependencies": [{"name": "requests", "version": "2.19.0", "vulns": []}]}))
    result = _dependency_result(_bare_plugin(tmp_path / "demo", {"requirements.txt": "requests==2.19.0\n"}))
    assert not result.is_incomplete
    python = _summary(result, "python")
    assert (python["status"], python["audited"], python["scanners"], python["errors"]) == (
        "audited",
        1,
        ["pip-audit"],
        [],
    )


def test_one_failed_pip_audit_batch_counts_only_the_audited_pins(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_pip_audit(
        monkeypatch, _ok({"dependencies": []}), ToolResult(False, "", "", -1, "pip-audit timed out after 180 seconds")
    )
    requirements = "numpy==1.26.4; python_version < '3.13'\nnumpy==2.1.0; python_version >= '3.13'\n"
    result = _dependency_result(_bare_plugin(tmp_path / "demo", {"requirements.txt": requirements}))
    assert [call["files"] for call in fake.calls] == [
        {"requirements-0.txt": "numpy==1.26.4\n", "requirements-1.txt": "numpy==2.1.0\n"}
    ] * 2
    assert result.incomplete_scans == ["pip-audit"]
    python = _summary(result, "python")
    assert (python["status"], python["declarations"], python["audited"], python["scanners"]) == (
        "incomplete",
        2,
        1,
        ["pip-audit"],
    )


def test_failed_pip_audit_batch_keeps_the_vulnerabilities_of_the_batch_that_ran(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "dependencies": [
            {"name": "numpy", "version": "1.26.4", "vulns": [{"id": "PYSEC-0000-1", "fix_versions": ["1.26.5"]}]}
        ]
    }
    _fake_pip_audit(monkeypatch, _ok(report), ToolResult(False, "", "", -1, "pip-audit timed out after 180 seconds"))
    requirements = "numpy==1.26.4; python_version < '3.13'\nnumpy==2.1.0; python_version >= '3.13'\n"
    result = _dependency_result(_bare_plugin(tmp_path / "demo", {"requirements.txt": requirements}))
    python = _summary(result, "python")
    assert (python["status"], python["audited"], python["vulnerabilities"]["high"]) == ("incomplete", 1, 1)
    assert python["errors"] == ["requirements.txt: pip-audit timed out after 180 seconds"]


def test_one_source_reports_a_repeated_pip_audit_error_once(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    offline = ToolResult(True, "", "ERROR: network unreachable\n", 1)
    _fake_pip_audit(monkeypatch, offline, offline)
    requirements = "numpy==1.26.4; python_version < '3.13'\nnumpy==2.1.0; python_version >= '3.13'\n"
    result = _dependency_result(_bare_plugin(tmp_path / "demo", {"requirements.txt": requirements}))
    python = _summary(result, "python")
    assert (python["status"], python["audited"], python["scanners"]) == ("incomplete", 0, [])
    assert python["errors"] == ["requirements.txt: pip-audit failed: ERROR: network unreachable"]


def test_conflicting_pins_are_split_into_batches_that_name_each_package_once() -> None:
    pins = [("a", "1.0.0"), ("b", "2.0.0"), ("a", "3.0.0"), ("a", "1.0.0"), ("a", "4.0.0")]
    assert eco.split_conflicting_pins(pins) == [{"a": "1.0.0", "b": "2.0.0"}, {"a": "3.0.0"}, {"a": "4.0.0"}]


def test_incomplete_outcome_counts_audited_packages_only_when_partial() -> None:
    outcome = eco.AuditOutcome(scanner="pip-audit", status="incomplete", error="x: timed out")
    outcome.count(Severity.HIGH)
    whole = eco.empty_ecosystem_summary()
    eco.record_outcome(whole, outcome, declarations=2, audited=1, unverified=0)
    partial = eco.empty_ecosystem_summary()
    eco.record_outcome(partial, outcome, declarations=2, audited=1, unverified=0, partial=True)
    assert (whole["status"], whole["audited"], whole["vulnerabilities"]["high"]) == ("incomplete", 0, 0)
    assert (partial["status"], partial["audited"], partial["vulnerabilities"]["high"]) == ("incomplete", 1, 1)
    assert whole["errors"] == partial["errors"] == ["x: timed out"]


def test_each_directory_is_walked_once_for_npm_manifests_and_dockerfiles(
    tmp_path: Path, tools: dict[str, FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    walked: list[str] = []
    discover = DependencySecurityValidator._discover

    def _counting(directory: Path, selected: Any) -> list:
        walked.append(Path(directory).name)
        return discover(directory, selected)

    monkeypatch.setattr(DependencySecurityValidator, "_discover", staticmethod(_counting))
    tools["osv_scanner"].available = True
    root = _bare_plugin(
        tmp_path / "demo",
        {
            "skills/foo/SKILL.md": "---\nname: foo\ndescription: A bundled skill.\n---\nBody\n",
            "skills/foo/package.json": {"dependencies": {"lodash": "4.17.20"}},
            "Dockerfile": "FROM node:20.11.1\n",
        },
    )
    result = _dependency_result(root)
    assert sorted(walked) == ["demo", "foo"]
    assert (_summary(result, "npm")["audited"], _summary(result, "container")["audited"]) == (1, 1)


def test_dockerfile_discovery_failure_still_audits_the_npm_manifests(
    tmp_path: Path, tools: dict[str, FakeTool]
) -> None:
    """A directory named like a Dockerfile fails Dockerfile discovery only; the npm audit still runs."""
    tools["osv_scanner"].available = True
    root = _bare_plugin(
        tmp_path / "demo",
        {"server/package.json": {"dependencies": {"lodash": "4.17.20"}}, "Dockerfile/README.md": "not a Dockerfile\n"},
    )
    result = _dependency_result(root)
    assert result.incomplete_scans == ["container-image-audit"]
    assert "Dockerfile discovery failed" in _summary(result, "container")["errors"][0]
    npm = _summary(result, "npm")
    assert (npm["status"], npm["audited"]) == ("audited", 1)


def test_source_labels_are_plugin_relative(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    skill = root / "skills" / "foo"
    skill.mkdir(parents=True)
    label = DependencySecurityValidator._source_label
    with plugin_tree_scope(root, [skill]):
        assert label(skill, PurePosixPath("package.json")) == "skills/foo/package.json"
        assert label(root, PurePosixPath("docker/Dockerfile")) == "docker/Dockerfile"
        assert label(root, PurePosixPath()) == "."
        assert label(tmp_path / "elsewhere", PurePosixPath("package.json")) == "package.json"
    assert label(skill, PurePosixPath("package.json")) == "package.json"
