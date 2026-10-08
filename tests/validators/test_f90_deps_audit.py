# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 12 (known vulnerabilities): Python advisories are real findings with their real severity.

Fixtures copy the shapes of the proof examples check-12 ``edge-10`` (policy override), ``pos-01`` (Python CVEs
in every report), ``edge-09`` (broken Python manifests), ``edge-08`` (pins PyPI does not know), ``edge-04``
(yarn and pnpm lockfiles), ``edge-02``/``pos-04`` (MCP container image) and ``pos-01`` ``out-real-safety``
(Safety 3.x). Scanners and the advisory lookup are fakes: nothing is installed or fetched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators import dependency_ecosystems as eco
from skillevaluator.validators.policy import ValidationPolicy, apply_policy, load_policy_file


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
        self.calls.append({"args": list(args), **kwargs})
        if not self.responses:
            return ToolResult(True, '{"dependencies": []}', "", 0)
        return self.responses.pop(0)


def _ok(payload: Any, exit_code: int = 0) -> ToolResult:
    return ToolResult(True, json.dumps(payload), "", exit_code)


# The five requests==2.19.0 advisories pip-audit reported in check-12 edge-10 (ids from the saved run).
_REQUESTS_VULNS = [
    {"id": "PYSEC-2018-28", "fix_versions": ["2.20.0"], "aliases": ["GHSA-x84v-xcm2-53pg", "CVE-2018-18074"]},
    {"id": "PYSEC-2023-74", "fix_versions": ["2.31.0"], "aliases": ["GHSA-j8r2-6x86-q33q", "CVE-2023-32681"]},
    {"id": "PYSEC-2026-1873", "fix_versions": ["2.32.0"], "aliases": ["GHSA-9wx4-h78v-vm56"]},
    {"id": "PYSEC-2026-1872", "fix_versions": ["2.32.4"], "aliases": ["GHSA-9hjg-9r4m-mvj7"]},
    {"id": "PYSEC-2026-2275", "fix_versions": ["2.33.0"], "aliases": ["GHSA-gc5v-m9x4-r6x2"]},
]


@pytest.fixture
def scanners(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeTool]:
    fakes = {
        "pip_audit": _FakeTool("pip-audit"),
        "safety": _FakeTool("safety", available=False),
        **{name: _FakeTool(name, available=False) for name in ("osv_scanner", "npm", "grype", "trivy")},
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(Tools, name, fake)
    return fakes


@pytest.fixture
def advisories(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake OSV records by id (GitHub advisories carry ``database_specific.severity``); unknown ids fail."""
    records: dict[str, Any] = {}
    asked: list[str] = []

    def fetch(vuln_id: str) -> Any:
        asked.append(vuln_id)
        if vuln_id not in records:
            raise OSError("404 Not Found")
        return records[vuln_id]

    monkeypatch.setattr(eco, "fetch_osv_record", fetch, raising=False)
    records["__asked__"] = asked
    return records


def _plugin(root: Path, files: dict[str, Any], *, manifest: str = ".claude-plugin/plugin.json") -> Path:
    for rel, content in {manifest: {"name": "demo"}, **files}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _dependency_result(root: Path, **kwargs: Any) -> ValidationResult:
    [result] = run_validation(root, checks="dependency", content_type=CONTENT_TYPE_PLUGIN, **kwargs)
    return result


def _python(result: ValidationResult) -> dict[str, Any]:
    return result.metadata["plugin"]["cve_summary"]["ecosystems"]["python"]


def _checks(result: ValidationResult, check: str) -> list[Finding]:
    return [finding for finding in result.findings if finding.check_name == check]


# --------------------------------------------------------------------------- #
# H14: Python CVEs survive a policy override and reach every report           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "manifest", [".claude-plugin/plugin.json", ".codex-plugin/plugin.json"], ids=["claude", "codex"]
)
def test_edge10_unrelated_policy_override_keeps_python_cves_blocking(
    tmp_path: Path, scanners: dict[str, _FakeTool], advisories: dict[str, Any], manifest: str
) -> None:
    """Proof H14, check-12 edge-10: lowering only dependency-version-unverified turned exit 1 into exit 0."""
    scanners["pip_audit"].responses = [
        _ok({"dependencies": [{"name": "requests", "version": "2.19.0", "vulns": _REQUESTS_VULNS}]}, 1)
    ]
    for vuln in _REQUESTS_VULNS:
        advisories[vuln["aliases"][0]] = {"database_specific": {"severity": "HIGH"}}
    plugin = _plugin(tmp_path / "plugin", {"requirements.txt": "requests==2.19.0\nflask\n"}, manifest=manifest)
    policy = tmp_path / "policy.yaml"
    policy.write_text("severity_overrides:\n  DEPENDENCY.dependency-version-unverified: low\n", encoding="utf-8")
    out = tmp_path / "out"

    run = CliRunner().invoke(
        cli,
        [
            "validate",
            str(plugin),
            "--tiers",
            "1",
            "--no-llm",
            "--checks",
            "schema,dependency",
            "--policy",
            str(policy),
            "-r",
            "json,sarif,html,markdown",
            "-o",
            str(out),
        ],
    )

    assert run.exit_code == 1, run.output
    report_path = next(path for path in out.glob("*.json") if not path.name.endswith(".sarif.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dependency = next(r for r in report["results"] if r["validator"] == "Dependency Vulnerability Audit")
    cves = [f for f in dependency["findings"] if f["check_name"] == "python-vulnerability"]
    assert sorted(f["metadata"]["vulnerability_id"] for f in cves) == sorted(v["id"] for v in _REQUESTS_VULNS)
    assert {f["severity"] for f in cves} == {"high"}
    assert {f["file_path"] for f in cves} == {"requirements.txt"}
    assert report["severity_counts"]["high"] == 5
    assert not dependency["passed"]
    # The override still applied to the finding it names.
    unverified = [f for f in dependency["findings"] if f["check_name"] == "dependency-version-unverified"]
    assert [f["severity"] for f in unverified] == ["low"]
    # pos-01: SARIF, BENCHMARK.md and HTML carry the Python CVEs too.
    sarif = json.loads(next(out.glob("*.sarif.json")).read_text(encoding="utf-8"))
    sarif_ids = [
        result["properties"]["metadata"]["vulnerability_id"]
        for sarif_run in sarif["runs"]
        for result in sarif_run["results"]
        if result.get("properties", {}).get("metadata", {}).get("ecosystem") == "python"
        and "vulnerability_id" in result["properties"]["metadata"]
    ]
    assert sorted(sarif_ids) == sorted(v["id"] for v in _REQUESTS_VULNS)
    benchmark = (out / "BENCHMARK.md").read_text(encoding="utf-8")
    assert "PYSEC-2023-74" in benchmark
    html = next(out.glob("*.html")).read_text(encoding="utf-8")
    assert all(v["id"] in html for v in _REQUESTS_VULNS)


def test_policy_override_keeps_plain_string_errors() -> None:
    """Proof H14 cause: apply_policy rebuilt errors from findings only, erasing plain-string errors."""
    result = ValidationResult(validator_name="Dependency Vulnerability Audit")
    result.add_error("No skills found in demo. Expected SKILL.md files in skill directories.")
    result.add_finding(
        Finding(
            category="DEPENDENCY",
            severity=Severity.INFO,
            check_name="dependency-version-unverified",
            message="flask: cannot audit a floating version",
            file_path="requirements.txt",
        )
    )
    policy = ValidationPolicy(severity_overrides={"DEPENDENCY.dependency-version-unverified": Severity.LOW})

    [applied] = apply_policy([result], policy)

    assert not applied.passed
    assert applied.errors == ["No skills found in demo. Expected SKILL.md files in skill directories."]
    assert applied.summary.errors == 1
    assert applied.findings[0].severity == Severity.LOW
    assert applied.warnings == ["[DEPENDENCY-LOW] flask: cannot audit a floating version in requirements.txt"]


def test_policy_file_override_shape_is_the_edge10_shape(tmp_path: Path) -> None:
    """The edge-10 policy file loads to the override the unit test above uses."""
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("severity_overrides:\n  DEPENDENCY.dependency-version-unverified: low\n", encoding="utf-8")
    policy = load_policy_file(policy_file)
    assert policy.severity_for("DEPENDENCY", "dependency-version-unverified", Severity.INFO) == Severity.LOW


# --------------------------------------------------------------------------- #
# M25: severity from advisory data; broken manifests; unknown pins            #
# --------------------------------------------------------------------------- #
def test_pos01_python_advisory_severity_comes_from_the_github_advisory(
    tmp_path: Path, scanners: dict[str, _FakeTool], advisories: dict[str, Any]
) -> None:
    """Proof M25, check-12 pos-01: PYSEC-2021-142 is CRITICAL and PYSEC-2023-74 MODERATE, but both were HIGH."""
    scanners["pip_audit"].responses = [
        _ok(
            {
                "dependencies": [
                    {
                        "name": "pyyaml",
                        "version": "5.3.1",
                        "vulns": [
                            {
                                "id": "PYSEC-2021-142",
                                "fix_versions": ["5.4"],
                                "aliases": ["CVE-2020-14343", "GHSA-8q59-q68h-6hv4"],
                            }
                        ],
                    },
                    {
                        "name": "requests",
                        "version": "2.19.0",
                        "vulns": [
                            {
                                "id": "PYSEC-2023-74",
                                "fix_versions": ["2.31.0"],
                                "aliases": ["CVE-2023-32681", "GHSA-j8r2-6x86-q33q"],
                            }
                        ],
                    },
                ]
            },
            1,
        )
    ]
    advisories["GHSA-8q59-q68h-6hv4"] = {
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
        "database_specific": {"severity": "CRITICAL"},
    }
    advisories["GHSA-j8r2-6x86-q33q"] = {
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N"}],
        "database_specific": {"severity": "MODERATE"},
    }

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": "pyyaml==5.3.1\nrequests==2.19.0\n"}))

    severities = {f.metadata["vulnerability_id"]: f.severity for f in _checks(result, "python-vulnerability")}
    assert severities == {"PYSEC-2021-142": Severity.CRITICAL, "PYSEC-2023-74": Severity.MEDIUM}
    # The GitHub advisory alias is asked first: PYSEC records carry no severity.
    assert advisories["__asked__"] == ["GHSA-8q59-q68h-6hv4", "GHSA-j8r2-6x86-q33q"]
    assert _python(result)["vulnerabilities"]["critical"] == 1
    assert _python(result)["vulnerabilities"]["medium"] == 1
    assert result.summary.critical_count == 1


def test_unknown_advisory_severity_is_reported_high_and_says_so(
    tmp_path: Path, scanners: dict[str, _FakeTool], advisories: dict[str, Any]
) -> None:
    scanners["pip_audit"].responses = [
        _ok({"dependencies": [{"name": "requests", "version": "2.19.0", "vulns": [_REQUESTS_VULNS[0]]}]}, 1)
    ]

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": "requests==2.19.0\n"}))

    [finding] = _checks(result, "python-vulnerability")
    assert finding.severity == Severity.HIGH
    assert "advisory severity unknown" in finding.message
    assert finding.metadata["severity_source"] == "unknown"
    assert "404" in finding.metadata["severity_unknown_reason"]


def test_cvss3_base_score_matches_the_specification_examples() -> None:
    assert eco.cvss3_base_score("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") == 9.8
    assert eco.cvss3_base_score("CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N") == 6.1
    assert eco.cvss3_base_score("CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:N") == 0.0
    assert eco.cvss3_base_score("CVSS:4.0/AV:N") is None
    assert eco.severity_from_osv_record(
        {"severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N"}]}
    ) == (Severity.MEDIUM)


def test_advisory_lookup_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(eco.OSV_API_URL_ENV, "off")
    lookup = eco.AdvisorySeverityLookup()
    severity = lookup.severity(["GHSA-j8r2-6x86-q33q"])
    assert severity.severity is None
    assert "turned off" in severity.source


def test_edge09_broken_python_manifests_make_the_audit_incomplete(
    tmp_path: Path, scanners: dict[str, _FakeTool]
) -> None:
    """Proof M25, check-12 edge-09: a non-UTF-8 requirements file and bad TOML passed with exit 0."""
    plugin = _plugin(
        tmp_path / "plugin",
        {
            "requirements.txt": b"requests==2.19.0  # caf\xe9\n",
            "skills/yaml-tool/SKILL.md": "---\nname: yaml-tool\ndescription: YAML helper.\n---\n# YAML tool\n",
            "skills/yaml-tool/pyproject.toml": '[project]\nname = "yaml-tool"\ndependencies = ["pyyaml==5.3.1"\n',
        },
    )

    result = _dependency_result(plugin)

    assert result.incomplete_scans == ["pip-audit"]
    assert not result.passed
    python = _python(result)
    assert python["status"] == "incomplete"
    assert any("requirements.txt" in error and "not valid UTF-8" in error for error in python["errors"])
    assert any("pyproject.toml" in error and "could not parse" in error for error in python["errors"])
    assert not any("Tier 2" in warning for warning in result.warnings)
    assert scanners["pip_audit"].calls == []


def test_edge09_shape_in_a_standalone_skill_stays_a_warning(tmp_path: Path, scanners: dict[str, _FakeTool]) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: Demo.\n---\n# Skill\n", encoding="utf-8")
    (skill / "requirements.txt").write_bytes(b"requests==2.19.0  # caf\xe9\n")

    [result] = run_validation(skill, checks="dependency")

    assert not result.is_incomplete
    assert any("not valid UTF-8" in warning for warning in result.warnings)


def test_edge08_pins_pypi_does_not_know_are_not_counted_as_audited(
    tmp_path: Path, scanners: dict[str, _FakeTool]
) -> None:
    """Proof M25, check-12 edge-08: pip-audit's skip_reason was ignored, so unknown pins looked audited and clean."""
    reason = "Dependency not found on PyPI and could not be audited: example-private-pkg-zz (1.0.0)"
    scanners["pip_audit"].responses = [
        _ok(
            {
                "dependencies": [
                    {"name": "example-private-pkg-zz", "skip_reason": reason},
                    {"name": "requests", "skip_reason": "Dependency not found on PyPI: requests (99.0.0)"},
                ]
            }
        )
    ]

    result = _dependency_result(
        _plugin(tmp_path / "plugin", {"requirements.txt": "example-private-pkg-zz==1.0.0\nrequests==99.0.0\n"})
    )

    not_audited = _checks(result, "dependency-not-audited")
    assert sorted(f.metadata["package_name"] for f in not_audited) == ["example-private-pkg-zz", "requests"]
    assert {f.severity for f in not_audited} == {Severity.MEDIUM}
    assert all(f.file_path == "requirements.txt" for f in not_audited)
    python = _python(result)
    assert (python["audited"], python["unverified"], python["status"]) == (0, 2, "unverified")
    assert not any(message.endswith("No vulnerabilities found (pip-audit)") for message in result.messages)


# --------------------------------------------------------------------------- #
# L17: Safety 3.x, MCP image paths, yarn and pnpm lockfiles                   #
# --------------------------------------------------------------------------- #
_SAFETY3_STDOUT = (
    "+==============================================================================+\n"
    "  DEPRECATED: this command (`check`) has been DEPRECATED.\n"
    "+==============================================================================+\n"
    + json.dumps(
        {
            "report_meta": {"scan_target": "files"},
            "vulnerabilities": [
                {
                    "vulnerability_id": "58755",
                    "package_name": "requests",
                    "analyzed_version": "2.19.0",
                    "advisory": "Requests leaks Proxy-Authorization headers to destination servers.",
                    "CVE": "CVE-2023-32681",
                    "severity": None,
                    "fixed_versions": [],
                },
                {
                    "vulnerability_id": "71064",
                    "package_name": "requests",
                    "analyzed_version": "2.19.0",
                    "advisory": "Requests session verify=False persists.",
                    "CVE": "CVE-2024-35195",
                    "severity": {"cvss_v3": {"base_score": 5.6, "base_severity": "MEDIUM"}},
                    "fixed_versions": ["2.32.0"],
                },
            ],
        },
        indent=2,
    )
    + "\n+==============================================================================+\n"
)


def test_safety3_report_with_banners_and_null_severity_contributes(
    tmp_path: Path, scanners: dict[str, _FakeTool], advisories: dict[str, Any]
) -> None:
    """Proof L17, check-12 pos-01 out-real-safety: the banner hid the JSON, and severity null would crash."""
    scanners["safety"].available = True
    scanners["safety"].responses = [ToolResult(True, _SAFETY3_STDOUT, "", 64)]
    scanners["pip_audit"].responses = [
        _ok({"dependencies": [{"name": "requests", "version": "2.19.0", "vulns": [_REQUESTS_VULNS[1]]}]}, 1)
    ]
    advisories["GHSA-j8r2-6x86-q33q"] = {"database_specific": {"severity": "MODERATE"}}

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": "requests==2.19.0\n"}))

    cves = _checks(result, "python-vulnerability")
    # CVE-2023-32681 is pip-audit's PYSEC-2023-74 alias, so Safety does not repeat it.
    assert [(f.metadata["scanner"], f.metadata["vulnerability_id"], f.severity) for f in cves] == [
        ("pip-audit", "PYSEC-2023-74", Severity.MEDIUM),
        ("safety", "SAFETY-71064", Severity.MEDIUM),
    ]
    assert cves[1].metadata["aliases"] == ["CVE-2024-35195"]


def test_safety_output_without_json_is_a_warning(tmp_path: Path, scanners: dict[str, _FakeTool]) -> None:
    scanners["safety"].available = True
    scanners["safety"].responses = [ToolResult(True, "Safety needs an account. Run safety auth.\n", "", 1)]

    result = _dependency_result(_plugin(tmp_path / "plugin", {"requirements.txt": "requests==2.19.0\n"}))

    assert any("Safety produced no JSON report" in warning for warning in result.warnings)
    assert result.passed


def test_mcp_image_findings_point_at_the_mcp_file(tmp_path: Path, scanners: dict[str, _FakeTool]) -> None:
    """Proof L17, check-12 pos-04/edge-02: the file path was ".mcp.json (mcpServers['pinned'])"."""
    image = "ghcr.io/example/db-mcp:1.2.3"
    scanners["grype"].available = True
    scanners["grype"].responses = [
        _ok(
            {
                "matches": [
                    {
                        "vulnerability": {"id": "CVE-2024-0001", "severity": "High", "fix": {"versions": []}},
                        "artifact": {"name": "openssl", "version": "3.0.0"},
                    }
                ]
            }
        )
    ]
    servers = {
        "pinned": {"command": "docker", "args": ["run", "-i", "--rm", image]},
        "floating": {"command": "docker", "args": ["run", "-i", "--rm", "example/tool:latest"]},
    }

    result = _dependency_result(_plugin(tmp_path / "plugin", {".mcp.json": {"mcpServers": servers}}))

    findings = _checks(result, "container-vulnerability") + [
        f for f in _checks(result, "dependency-version-unverified") if f.metadata.get("ecosystem") == "container"
    ]
    assert {(f.file_path, f.metadata["mcp_server"]) for f in findings} == {
        (".mcp.json", "pinned"),
        (".mcp.json", "floating"),
    }


@pytest.mark.parametrize("lockfile", ["yarn.lock", "pnpm-lock.yaml"])
def test_edge04_yarn_and_pnpm_lockfiles_make_the_npm_audit_incomplete(
    tmp_path: Path, scanners: dict[str, _FakeTool], lockfile: str
) -> None:
    """Proof L17, check-12 edge-04: a yarn.lock pinning minimist 1.2.5 gave no message at all, exit 0."""
    files: dict[str, Any] = {f"servers/js/{lockfile}": 'minimist@^1.2.0:\n  version "1.2.5"\n'}
    if lockfile == "pnpm-lock.yaml":
        files["servers/js/package.json"] = {"dependencies": {"minimist": "^1.2.0"}}

    result = _dependency_result(_plugin(tmp_path / "plugin", files))

    assert "npm-audit" in result.incomplete_scans
    npm = result.metadata["plugin"]["cve_summary"]["ecosystems"]["npm"]
    assert npm["status"] == "incomplete"
    assert any(lockfile in error and "not supported" in error for error in npm["errors"])


def test_npm_lockfile_beside_a_yarn_lock_is_audited_without_incomplete(
    tmp_path: Path, scanners: dict[str, _FakeTool]
) -> None:
    scanners["npm"].available = True
    scanners["npm"].responses = [_ok({"auditReportVersion": 2, "vulnerabilities": {}})]
    lock = {
        "name": "js",
        "lockfileVersion": 3,
        "packages": {"": {"name": "js"}, "node_modules/minimist": {"version": "1.2.8"}},
    }
    result = _dependency_result(
        _plugin(tmp_path / "plugin", {"servers/js/package-lock.json": lock, "servers/js/yarn.lock": "# yarn\n"})
    )
    assert "npm-audit" not in result.incomplete_scans
