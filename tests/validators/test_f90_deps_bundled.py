# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks 10 and 12 (proof M5): bundled skills keep their findings and every bundled skill is audited.

Fixtures copy the shapes of check-10 ``e10-bundled-skill-medium-dropped``, check-12
``edge-06-critical-in-first-skill-stops-later-audits`` and the skeptic's ``x3-npm-moderate-low``.
Scanners and the advisory lookup are fakes: nothing is installed or fetched.
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
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.plugin_tree import plugin_tree_scope


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


def _lock(**packages: str) -> dict[str, Any]:
    return {
        "name": "demo",
        "lockfileVersion": 3,
        "packages": {"": {"name": "demo"}, **{f"node_modules/{n}": {"version": v} for n, v in packages.items()}},
    }


def _skill_md(name: str) -> str:
    return f"---\nname: {name}\ndescription: Bundled skill {name} for the dependency audit.\n---\n# {name}\n"


def _plugin(root: Path, files: dict[str, Any]) -> Path:
    for rel, content in {".claude-plugin/plugin.json": {"name": "demo"}, **files}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


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


def test_x3_passing_bundled_skill_keeps_its_medium_and_low_findings(
    tmp_path: Path, scanners: dict[str, _FakeTool]
) -> None:
    """Proof M5, skeptic x3: npm moderate/low in a bundled skill vanished; the same lockfile at the root kept them."""
    scanners["npm"].responses = [
        _ok(_npm_report(("word-wrap", "GHSA-j8xg", "moderate"), ("cookie", "GHSA-pxg6", "low")))
    ]
    plugin = _plugin(
        tmp_path / "plugin",
        {
            "skills/js-skill/SKILL.md": _skill_md("js-skill"),
            "skills/js-skill/package-lock.json": _lock(**{"word-wrap": "1.2.3", "cookie": "0.6.0"}),
        },
    )

    [result] = run_validation(plugin, checks="dependency", content_type=CONTENT_TYPE_PLUGIN)

    assert result.passed
    findings = {(f.severity, f.file_path) for f in result.findings if f.check_name == "npm-vulnerability"}
    # One skill prefix, and the path below it is the real file (no skills/js-skill/skills/js-skill/...).
    assert findings == {
        (Severity.MEDIUM, "[js-skill] skills/js-skill/package-lock.json"),
        (Severity.LOW, "[js-skill] skills/js-skill/package-lock.json"),
    }
    assert (result.summary.medium_count, result.summary.low_count) == (1, 1)
    assert any(detail.check_name == "js-skill" for detail in result.success_details)


def test_edge06_critical_in_first_skill_does_not_stop_later_audits(
    tmp_path: Path, scanners: dict[str, _FakeTool]
) -> None:
    """Proof M5, check-12 edge-06: skills/b-second (requests, lodash) was never audited after a-first's CRITICAL."""
    scanners["npm"].responses = [
        _ok(_npm_report(("minimist", "GHSA-xvch", "critical"))),
        _ok(_npm_report(("lodash", "GHSA-35jh", "high"))),
    ]
    scanners["pip_audit"].responses = [
        _ok(
            {
                "dependencies": [
                    {
                        "name": "requests",
                        "version": "2.19.0",
                        "vulns": [{"id": "PYSEC-2018-28", "fix_versions": ["2.20.0"]}],
                    }
                ]
            },
            1,
        )
    ]
    plugin = _plugin(
        tmp_path / "plugin",
        {
            "skills/a-first/SKILL.md": _skill_md("a-first"),
            "skills/a-first/package-lock.json": _lock(minimist="1.2.5"),
            "skills/b-second/SKILL.md": _skill_md("b-second"),
            "skills/b-second/requirements.txt": "requests==2.19.0\n",
            "skills/b-second/package-lock.json": _lock(lodash="4.17.20"),
        },
    )
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
            "-r",
            "json",
            "-o",
            str(out),
        ],
    )

    assert run.exit_code == 1, run.output
    report = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    dependency = next(r for r in report["results"] if r["validator"] == "Dependency Vulnerability Audit")
    paths = sorted({f["file_path"] for f in dependency["findings"]})
    assert paths == [
        "[a-first] skills/a-first/package-lock.json",
        "[b-second] skills/b-second/package-lock.json",
        "[b-second] skills/b-second/requirements.txt",
    ]
    assert len(scanners["pip_audit"].calls) == 1
    # severity_counts include the bundled skills' findings (the base reported critical 0 here).
    all_findings = [f for r in report["results"] for f in r["findings"]]
    for severity in ("critical", "high", "medium", "low"):
        assert report["severity_counts"][severity] == sum(1 for f in all_findings if f["severity"] == severity)
    assert report["severity_counts"]["critical"] == 1


class _MediumValidator(ValidatorBase):
    """A validator that reports one MEDIUM finding per directory with a ``notes.md`` (a stand-in for SkillSpector)."""

    @property
    def name(self) -> str:
        return "Fake Content Scan"

    @property
    def description(self) -> str:
        return "Reports MEDIUM findings"

    def validate(self, skill_path: Path) -> ValidationResult:
        return self._validate_folder_or_skill(skill_path, self._one, action_description="Scanning")

    @staticmethod
    def _one(path: Path) -> ValidationResult:
        result = ValidationResult()
        if (path / "notes.md").is_file():
            result.add_finding(
                Finding(
                    category="SECURITY",
                    severity=Severity.MEDIUM,
                    check_name="E1",
                    message="External Transmission",
                    file_path="notes.md",
                    line_number=31,
                )
            )
        result.add_success("scan", "scanned")
        return result


def test_e10_passing_bundled_skill_keeps_a_medium_content_finding(tmp_path: Path) -> None:
    """Proof M5, check-10 e10: the command's MEDIUM was kept, the same MEDIUM in skills/poster was dropped."""
    plugin = _plugin(
        tmp_path / "plugin",
        {
            "skills/poster/SKILL.md": _skill_md("poster"),
            "skills/poster/notes.md": "curl -X POST -d @report.json https://api.example.test/\n",
            "notes.md": "curl -X POST -d @report.json https://api.example.test/\n",
        },
    )

    with plugin_tree_scope(plugin, [plugin / "skills" / "poster"]):
        result = _MediumValidator().validate(plugin)

    assert sorted(f.file_path for f in result.findings) == ["[poster] skills/poster/notes.md", "notes.md"]
    assert result.summary.medium_count == 2
    assert result.passed


def test_bundled_skill_schema_findings_are_counted_in_the_severity_summary(tmp_path: Path) -> None:
    """Proof M5: merge_with_prefix copied findings without counting them, so severity_counts missed them."""
    plugin = _plugin(
        tmp_path / "plugin",
        {"skills/broken/SKILL.md": "---\nname: broken\n---\n# Broken\n"},
    )

    result = PluginSchemaValidator().validate(plugin)

    bundled = [f for f in result.findings if f.file_path.startswith("[broken]")]
    assert bundled
    for severity, counted in (
        (Severity.CRITICAL, result.summary.critical_count),
        (Severity.HIGH, result.summary.high_count),
        (Severity.MEDIUM, result.summary.medium_count),
        (Severity.LOW, result.summary.low_count),
    ):
        assert counted == sum(1 for f in result.findings if f.severity == severity), severity
