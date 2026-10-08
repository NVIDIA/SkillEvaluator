# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks 10 and 12, review follow-ups (proof M5, M25, H14, L17).

Fixtures copy the shapes of check-12 ``pos-01`` (advisory lookups), ``edge-03`` (unreadable manifests),
``edge-08`` (pins the registry does not know) and ``pos-03`` (MCP runner packages). Scanners, the advisory
lookup, and the npm registry check are fakes: nothing is installed or fetched.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators import dependency_ecosystems as eco


class _FakeTool:
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
        cwd = kwargs.get("cwd")
        files = {p.name: p.read_text() for p in Path(cwd).iterdir() if p.is_file()} if cwd else {}
        self.calls.append({"args": list(args), "files": files})
        if not self.responses:
            return ToolResult(True, '{"dependencies": []}', "", 0)
        return self.responses.pop(0)


def _ok(payload: Any, exit_code: int = 0) -> ToolResult:
    return ToolResult(True, json.dumps(payload), "", exit_code)


@pytest.fixture
def scanners(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeTool]:
    fakes = {
        "pip_audit": _FakeTool("pip-audit"),
        "safety": _FakeTool("safety", available=False),
        "npm": _FakeTool("npm"),
        **{name: _FakeTool(name, available=False) for name in ("osv_scanner", "grype", "trivy")},
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(Tools, name, fake)
    monkeypatch.setattr(eco, "fetch_osv_record", lambda _id: {"database_specific": {"severity": "HIGH"}}, raising=False)
    return fakes


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake public npm registry: ``known`` holds the name@version pins it has; every request is recorded."""
    state: dict[str, Any] = {"known": set(), "asked": [], "error": None}

    def fetch(name: str, version: str) -> bool:
        state["asked"].append(f"{name}@{version}")
        if state["error"] is not None:
            raise state["error"]
        return f"{name}@{version}" in state["known"]

    monkeypatch.setattr(eco, "fetch_npm_version", fetch, raising=False)
    return state


def _plugin(root: Path, files: dict[str, Any]) -> Path:
    for rel, content in {".claude-plugin/plugin.json": {"name": "demo"}, **files}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _dependency_result(root: Path) -> ValidationResult:
    [result] = run_validation(root, checks="dependency", content_type=CONTENT_TYPE_PLUGIN)
    return result


def _checks(result: ValidationResult, check: str) -> list[Finding]:
    return [finding for finding in result.findings if finding.check_name == check]


def _ecosystem(result: ValidationResult, name: str) -> dict[str, Any]:
    return result.metadata["plugin"]["cve_summary"]["ecosystems"][name]


def _npm_report(*vulns: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "auditReportVersion": 2,
        "vulnerabilities": {
            name: {
                "via": [
                    {"name": name, "title": f"{name} advisory", "url": f"https://example.test/{ghsa}", "severity": sev}
                ]
            }
            for name, ghsa, sev in vulns
        },
    }


# --------------------------------------------------------------------------- #
# M5: Tier 2 plugin context dedup counts each bundled-skill severity once     #
# --------------------------------------------------------------------------- #
def test_tier2_plugin_context_dedup_counts_each_severity_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """merge_with_prefix adds severity counts; the Tier 2 loop added them a second time (2 skills: medium 4, low 4)."""
    from skillevaluator.tier2 import commands

    class _FakeIntraSkill:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def validate(self, _skill_dir: Path) -> ValidationResult:
            result = ValidationResult(validator_name="Context Deduplication")
            for severity, check in ((Severity.HIGH, "dup-high"), (Severity.LOW, "dup-low")):
                result.add_finding(
                    Finding(
                        category="CONTENT_DEDUP",
                        severity=severity,
                        check_name=check,
                        message=check,
                        file_path="SKILL.md",
                    )
                )
            return result

    monkeypatch.setattr(commands, "IntraSkillValidator", _FakeIntraSkill)
    plugin = _plugin(
        tmp_path / "plugin",
        {f"skills/{name}/SKILL.md": f"---\nname: {name}\ndescription: Skill {name}.\n---\n# {name}\n" for name in "ab"},
    )

    [aggregate] = commands.run_plugin_skill_context_dedup(plugin)

    by_severity = dict.fromkeys(Severity, 0)
    for finding in aggregate.findings:
        by_severity[finding.severity] += 1
    assert {f.check_name for f in aggregate.findings} == {"dup-high", "dup-low"}
    assert by_severity[Severity.MEDIUM] == 2 and by_severity[Severity.LOW] == 2  # HIGH is capped at MEDIUM
    summary = aggregate.summary
    assert (summary.critical_count, summary.high_count, summary.medium_count, summary.low_count) == (
        by_severity[Severity.CRITICAL],
        by_severity[Severity.HIGH],
        by_severity[Severity.MEDIUM],
        by_severity[Severity.LOW],
    )


# --------------------------------------------------------------------------- #
# M25 / H14: the OSV severity lookup gives up after a network failure          #
# --------------------------------------------------------------------------- #
_THREE_ADVISORIES = [
    {"id": f"PYSEC-2099-{n}", "fix_versions": ["9.9"], "aliases": [f"GHSA-aaaa-bbbb-000{n}", f"CVE-2099-000{n}"]}
    for n in (1, 2, 3)
]


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out")), ConnectionResetError("reset")],
    ids=["timeout", "urlerror", "connection-reset"],
)
def test_advisory_lookup_stops_after_the_first_network_failure(error: Exception) -> None:
    """Black-holed api.osv.dev: 6 advisories x 3 ids x 10 s took 183 s. Now one failure ends the lookups."""
    calls: list[str] = []

    def fetch(vuln_id: str) -> Any:
        calls.append(vuln_id)
        raise error

    lookup = eco.AdvisorySeverityLookup(fetch)
    severities = [lookup.severity([v["id"], *v["aliases"]]) for v in _THREE_ADVISORIES]

    assert len(calls) == 1
    assert all(item.severity is None for item in severities)
    assert len({item.source for item in severities}) == 1
    assert "advisory lookup failed" in severities[-1].source


def test_advisory_lookup_keeps_going_after_a_404_for_one_id() -> None:
    calls: list[str] = []

    def fetch(vuln_id: str) -> Any:
        calls.append(vuln_id)
        if vuln_id.startswith("GHSA-"):
            raise urllib.error.HTTPError(f"https://osv.example/{vuln_id}", 404, "Not Found", {}, None)
        return {"severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]}

    lookup = eco.AdvisorySeverityLookup(fetch)
    severities = [lookup.severity([v["id"], *v["aliases"]]) for v in _THREE_ADVISORIES]

    assert [item.severity for item in severities] == [Severity.CRITICAL] * 3
    assert len(calls) == 6  # GHSA 404, then PYSEC, per advisory


def test_advisory_lookup_stops_at_the_run_time_budget() -> None:
    now = [0.0]
    calls: list[str] = []

    def fetch(vuln_id: str) -> Any:
        calls.append(vuln_id)
        now[0] += eco.ADVISORY_LOOKUP_BUDGET / 2 + 1  # slow, but each answer arrives
        return {"database_specific": {"severity": "HIGH"}}

    lookup = eco.AdvisorySeverityLookup(fetch, clock=lambda: now[0])
    severities = [lookup.severity([v["aliases"][0]]) for v in _THREE_ADVISORIES]

    assert len(calls) == 2
    assert [item.severity for item in severities[:2]] == [Severity.HIGH, Severity.HIGH]
    assert severities[2].severity is None and "budget" in severities[2].source


def test_pos01_blocked_advisory_service_costs_one_lookup_per_run(
    tmp_path: Path, scanners: dict[str, _FakeTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fetch(vuln_id: str) -> Any:
        calls.append(vuln_id)
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(eco, "fetch_osv_record", fetch)
    scanners["pip_audit"].responses = [
        _ok({"dependencies": [{"name": "requests", "version": "2.19.0", "vulns": _THREE_ADVISORIES}]}, 1)
    ]

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": "requests==2.19.0\n"}))

    cves = _checks(result, "python-vulnerability")
    assert len(cves) == 3 and len(calls) == 1
    assert all(f.severity == Severity.HIGH and "severity unknown" in f.message for f in cves)
    assert all("timed out" in f.metadata["severity_unknown_reason"] for f in cves)


# --------------------------------------------------------------------------- #
# H14 / M25: one advisory on two marker-split pins is reported for each pin   #
# --------------------------------------------------------------------------- #
_DUAL_PINS = "foo==1.0 ; python_version < '3.8'\nfoo==2.0 ; python_version >= '3.8'\n"


def _pip_audit_for_each_pin(tool: _FakeTool) -> None:
    def run(args: list[str], **kwargs: Any) -> ToolResult:
        requirements = Path(args[args.index("-r") + 1]).read_text(encoding="utf-8").split()
        tool.calls.append({"args": list(args), "pins": requirements})
        dependencies = []
        for line in requirements:
            name, _sep, version = line.partition("==")
            vuln = {"id": "PYSEC-2099-1", "fix_versions": ["9.9"], "aliases": ["GHSA-aaaa-bbbb-cccc"]}
            dependencies.append({"name": name, "version": version, "vulns": [vuln]})
        return _ok({"dependencies": dependencies}, 1)

    tool.run = run  # type: ignore[method-assign]


def test_one_advisory_on_two_marker_split_pins_is_reported_for_each_pin(
    tmp_path: Path, scanners: dict[str, _FakeTool]
) -> None:
    """The dedup key had no version, so the second pin's advisory was dropped (base: two errors; branch: one)."""
    _pip_audit_for_each_pin(scanners["pip_audit"])

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": _DUAL_PINS}))

    assert len(scanners["pip_audit"].calls) == 2  # each conflicting pin gets its own batch
    cves = _checks(result, "python-vulnerability")
    assert sorted(f.metadata["package_version"] for f in cves) == ["1.0", "2.0"]


def test_safety_dedup_keeps_the_version(tmp_path: Path, scanners: dict[str, _FakeTool]) -> None:
    """Safety repeats pip-audit's advisory only for the same pin; the other pin's report is kept."""
    scanners["pip_audit"].responses = [
        _ok(
            {
                "dependencies": [
                    {
                        "name": "foo",
                        "version": "1.0",
                        "vulns": [{"id": "PYSEC-2099-1", "fix_versions": [], "aliases": ["CVE-2099-0001"]}],
                    }
                ]
            },
            1,
        ),
        _ok({"dependencies": [{"name": "foo", "version": "2.0", "vulns": []}]}),
    ]
    scanners["safety"].available = True
    safety_rows = [
        {"package_name": "foo", "analyzed_version": version, "vulnerability_id": "70001", "CVE": "CVE-2099-0001"}
        for version in ("1.0", "2.0")
    ]
    scanners["safety"].responses = [_ok({"vulnerabilities": [row]}) for row in safety_rows]

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": _DUAL_PINS}))

    cves = sorted(
        (f.metadata["package_version"], f.metadata["scanner"]) for f in _checks(result, "python-vulnerability")
    )
    assert cves == [("1.0", "pip-audit"), ("2.0", "safety")]


# --------------------------------------------------------------------------- #
# L17: the Tier 1 audit never says "Tier 2"                                   #
# --------------------------------------------------------------------------- #
def test_edge03_unreadable_manifests_use_tier1_wording(tmp_path: Path, scanners: dict[str, _FakeTool]) -> None:
    """check-12 edge-03 said 'Selected Tier 2 file is not valid UTF-8: Dockerfile.bad' five times."""
    scanners["grype"].available = True
    plugin = _plugin(
        tmp_path / "plugin",
        {
            "Dockerfile.bad": b"FROM python:3.12\n# caf\xe9\n",
            "requirements-dev.txt": b"requests==2.19.0  # caf\xe9\n",
            "servers/b/package-lock.json": b'{"lockfileVersion": 3, "name": "caf\xe9"}',
        },
    )

    result = _dependency_result(plugin)

    serialized = json.dumps(result.to_dict())
    assert "Tier 2" not in serialized
    for name in ("container", "npm"):
        [error] = _ecosystem(result, name)["errors"]
        assert error.endswith("the file is not valid UTF-8 text"), error
    assert any(
        "requirements-dev.txt" in error and "not valid UTF-8" in error
        for error in _ecosystem(result, "python")["errors"]
    )


# --------------------------------------------------------------------------- #
# M25 (npm half): a pin the public registry does not know is not "audited"    #
# --------------------------------------------------------------------------- #
_PRIVATE = "@example-private-zz/private-pkg"


def _lock(packages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"name": "js", "lockfileVersion": 3, "packages": {"": {"name": "js"}, **packages}}


def test_edge08_npm_pin_the_registry_does_not_know_is_not_audited(
    tmp_path: Path, scanners: dict[str, _FakeTool], registry: dict[str, Any]
) -> None:
    """check-12 edge-08: 'npm Audited (1 audited, 0 unverified, none)' for a private package the registry lacks."""
    lock = _lock({f"node_modules/{_PRIVATE}": {"version": "1.0.0"}})

    result = _dependency_result(_plugin(tmp_path / "plugin", {"servers/js/package-lock.json": lock}))

    assert registry["asked"] == [f"{_PRIVATE}@1.0.0"]
    [finding] = _checks(result, "dependency-not-audited")
    assert finding.severity == Severity.MEDIUM
    assert finding.metadata["ecosystem"] == "npm"
    assert finding.metadata["package_name"] == _PRIVATE
    assert finding.file_path == "servers/js/package-lock.json"
    npm = _ecosystem(result, "npm")
    assert (npm["status"], npm["audited"], npm["unverified"]) == ("unverified", 0, 1)
    assert scanners["npm"].calls == []  # the private name is not sent to the public audit


def test_npm_pins_the_registry_has_are_audited_and_registry_evidence_skips_the_check(
    tmp_path: Path, scanners: dict[str, _FakeTool], registry: dict[str, Any]
) -> None:
    """pos-02 and neg-01 lockfiles are hand-written (no resolved URL or integrity hash) and must stay audited."""
    registry["known"] = {"lodash@4.17.20"}
    scanners["npm"].responses = [_ok(_npm_report(("lodash", "GHSA-35jh", "high")))]
    lock = _lock(
        {
            "node_modules/lodash": {"version": "4.17.20"},
            "node_modules/minimist": {
                "version": "1.2.8",
                "resolved": "https://registry.npmjs.org/minimist/-/minimist-1.2.8.tgz",
                "integrity": "sha512-example",
            },
            "node_modules/ms": {"version": "2.1.3", "integrity": "sha512-example"},
            "node_modules/minimist/node_modules/inner": {"version": "1.0.0", "inBundle": True},
            "node_modules/git-dep": {"version": "1.0.0", "resolved": "git+ssh://git@example.com/acme/git-dep.git#abc"},
            "packages/workspace-pkg": {"name": "workspace-pkg", "version": "1.0.0"},
        }
    )

    result = _dependency_result(_plugin(tmp_path / "plugin", {"servers/js/package-lock.json": lock}))

    assert registry["asked"] == ["lodash@4.17.20"]
    [call] = scanners["npm"].calls
    audited = json.loads(call["files"]["package.json"])["dependencies"]
    assert sorted(audited) == ["inner", "lodash", "minimist", "ms"]
    [not_audited] = _checks(result, "dependency-not-audited")
    assert not_audited.metadata["package_name"] == "git-dep" and "git source" in not_audited.message
    npm = _ecosystem(result, "npm")
    assert (npm["status"], npm["declarations"], npm["audited"], npm["unverified"]) == ("audited", 5, 4, 1)
    assert [f.severity for f in _checks(result, "npm-vulnerability")] == [Severity.HIGH]


def test_npm_registry_check_stops_after_a_network_failure(
    tmp_path: Path, scanners: dict[str, _FakeTool], registry: dict[str, Any]
) -> None:
    registry["error"] = urllib.error.URLError(TimeoutError("timed out"))
    scanners["npm"].responses = [_ok(_npm_report())]
    lock = _lock({f"node_modules/pkg-{n}": {"version": "1.0.0"} for n in range(3)})

    result = _dependency_result(_plugin(tmp_path / "plugin", {"servers/js/package-lock.json": lock}))

    assert len(registry["asked"]) == 1
    assert _ecosystem(result, "npm")["audited"] == 3  # unknown is not "missing": audited by name and version
    assert any("not checked against the public registry" in message for message in result.messages)


# --------------------------------------------------------------------------- #
# M25 / L17: an MCP runner's package findings name the server                 #
# --------------------------------------------------------------------------- #
def test_pos03_mcp_runner_package_findings_name_their_server(
    tmp_path: Path, scanners: dict[str, _FakeTool], registry: dict[str, Any]
) -> None:
    """check-12 pos-03: python-vulnerability mcp==1.8.0 had file_path '.mcp.json' and no mcp_server."""
    registry["known"] = {"@modelcontextprotocol/server-filesystem@0.6.2"}
    scanners["pip_audit"].responses = [
        _ok(
            {
                "dependencies": [
                    {
                        "name": "mcp",
                        "version": "1.8.0",
                        "vulns": [{"id": "GHSA-3qhf-m339-9g5v", "fix_versions": ["1.9.4"], "aliases": []}],
                    }
                ]
            },
            1,
        )
    ]
    scanners["npm"].responses = [_ok(_npm_report(("@modelcontextprotocol/server-filesystem", "GHSA-hc55", "high")))]
    servers = {
        "fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem@0.6.2", "/tmp/demo"]},
        "pysdk": {"command": "uvx", "args": ["--from", "mcp==1.8.0", "mcp-demo-server"]},
        "memory": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
        "local": {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server/index.js"]},
    }

    result = _dependency_result(_plugin(tmp_path / "plugin", {".mcp.json": {"mcpServers": servers}}))

    attributed = {
        (f.check_name, f.metadata.get("package_name"), f.metadata.get("mcp_server"), f.file_path)
        for f in result.findings
        if f.category == "DEPENDENCY"
    }
    assert attributed == {
        ("python-vulnerability", "mcp", "pysdk", ".mcp.json"),
        ("npm-vulnerability", "@modelcontextprotocol/server-filesystem", "fs", ".mcp.json"),
        ("dependency-version-unverified", "@modelcontextprotocol/server-memory", "memory", ".mcp.json"),
    }
