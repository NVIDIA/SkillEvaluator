# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency advisories are structured findings bound to the pin that declared them."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skillevaluator.reporting import JSONReporter
from skillevaluator.reporting.sarif_reporter import SARIFReporter
from skillevaluator.utils.tool_runner import Severity, ToolResult, Tools
from skillevaluator.validators.base import ValidationResult
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.policy import ValidationPolicy, apply_policy

PYYAML_ADVISORY = {"id": "PYSEC-2020-176", "fix_versions": ["5.4"], "aliases": ["CVE-2020-14343"]}


def _skill(tmp_path: Path, files: dict[str, str]) -> Path:
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo\n---\n", encoding="utf-8")
    for name, text in files.items():
        (skill / name).write_text(text, encoding="utf-8")
    return skill


@pytest.fixture
def pip_audit(monkeypatch):
    """Answer each pip-audit run from the pins it receives, with advisories by package name."""
    advisories: dict[str, list[dict]] = {}
    calls: list[list[str]] = []
    monkeypatch.setattr(Tools.pip_audit, "_path", "/usr/bin/pip-audit")

    def run(args, **kwargs):
        assert "--no-deps" in args and "--disable-pip" in args
        pins = Path(args[args.index("-r") + 1]).read_text(encoding="utf-8").splitlines()
        calls.append(pins)
        dependencies = [
            {"name": name, "version": version, "vulns": advisories.get(name.lower(), [])}
            for name, version in (pin.split("==") for pin in pins)
        ]
        found = any(dependency["vulns"] for dependency in dependencies)
        return ToolResult(True, json.dumps({"dependencies": dependencies, "fixes": []}), "", 1 if found else 0)

    monkeypatch.setattr(Tools.pip_audit, "run", run)
    return advisories, calls


@pytest.fixture
def safety(monkeypatch):
    """Make Safety available; set ``state["report"]`` to the JSON it prints."""
    state: dict = {"report": {"vulnerabilities": []}}
    monkeypatch.setattr(Tools.safety, "_path", "/usr/bin/safety")

    def run(args, **_kwargs):
        assert args[:2] == ["check", "-r"]
        return ToolResult(True, json.dumps(state["report"]), "", 64)

    monkeypatch.setattr(Tools.safety, "run", run)
    return state


def test_requirements_advisory_is_a_structured_finding_at_its_line(tmp_path, pip_audit):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    skill = _skill(tmp_path, {"requirements.txt": "# runtime\npyyaml==5.3\n"})

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert not result.passed and not result.is_incomplete
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.category == "DEPENDENCY"
    assert finding.check_name == "PYSEC-2020-176"
    assert finding.severity == Severity.HIGH
    assert finding.message == "pyyaml==5.3: PYSEC-2020-176 -> upgrade to 5.4"
    assert (finding.file_path, finding.line_number) == ("requirements.txt", 2)
    assert finding.suggestion == "Upgrade pyyaml to a fixed version (5.4) and update the pin."
    assert finding.metadata == {
        "package_name": "pyyaml",
        "resolved_version": "5.3",
        "declared_constraint": "pyyaml==5.3",
        "declaration": "line 2",
        "resolution_status": "resolved",
        "dependency_role": "runtime",
        "dependency_relationship": "direct",
        "advisory_id": "PYSEC-2020-176",
        "advisory_source": "pip-audit",
        "affected_range": None,
        "fixed_versions": ["5.4"],
        "aliases": ["CVE-2020-14343"],
    }
    assert result.summary.high_count == 1
    assert result.errors == ["[DEPENDENCY-HIGH] pyyaml==5.3: PYSEC-2020-176 -> upgrade to 5.4 in requirements.txt:2"]


def test_json_and_sarif_reports_carry_the_advisory_and_its_line(tmp_path, pip_audit):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    hashed = "pyyaml==5.3 \\\n    --hash=sha256:" + "a" * 64 + "\n"
    skill = _skill(tmp_path, {"requirements.txt": "idna==3.10 \\\n    --hash=sha256:" + "b" * 64 + "\n" + hashed})
    validator = DependencySecurityValidator(use_safety=False)
    result = validator.validate(skill)
    result.validator_name = validator.name

    row = json.loads(JSONReporter().render_all([result]))["results"][0]
    assert [(item["check_name"], item["line_number"]) for item in row["findings"]] == [("PYSEC-2020-176", 3)]
    assert row["summary"]["high_count"] == 1

    sarif = json.loads(SARIFReporter(include_timestamp=False, scan_root=skill).render_all([result]))
    (sarif_result,) = sarif["runs"][0]["results"]
    assert sarif_result["ruleId"] == "Dependency-Vulnerability-Audit/PYSEC-2020-176"
    assert sarif_result["level"] == "error"
    location = sarif_result["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"].endswith("demo-skill/requirements.txt")
    assert location["region"]["startLine"] == 3
    assert sarif_result["properties"]["metadata"]["declared_constraint"] == "pyyaml==5.3"


def test_package_names_use_pep_503_normalization(tmp_path, pip_audit):
    advisories, calls = pip_audit
    advisories["pillow-simd"] = [{"id": "PYSEC-1", "fix_versions": []}]
    skill = _skill(tmp_path, {"requirements.txt": "Pillow_SIMD==9.0\n"})

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert calls == [["pillow-simd==9.0"]]
    metadata = result.findings[0].metadata
    assert metadata["package_name"] == "pillow-simd"
    assert metadata["declared_constraint"] == "Pillow_SIMD==9.0"
    assert metadata["dependency_relationship"] == "direct"
    assert result.findings[0].suggestion == (
        "No fixed version of pillow-simd is listed; review the advisory before shipping this dependency."
    )


@pytest.mark.parametrize(
    ("filename", "role"),
    [
        ("requirements.txt", "runtime"),
        ("requirements-docs.txt", "docs"),
        ("requirements-dev.txt", "dev"),
        ("requirements_test.txt", "test"),
        ("requirements.tests.txt", "test"),
        ("requirements-build.txt", "build"),
        ("requirements-latest.txt", "runtime"),
    ],
)
def test_requirement_filename_sets_dependency_role(tmp_path, pip_audit, filename, role):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    skill = _skill(tmp_path, {filename: "pyyaml==5.3\n"})

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert [(finding.file_path, finding.metadata["dependency_role"]) for finding in result.findings] == [
        (filename, role)
    ]


def test_pyproject_advisory_names_the_declared_pin_and_group(tmp_path, pip_audit):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    advisories["idna"] = [{"id": "GHSA-demo-0001", "fix_versions": ["3.7"], "severity": "medium"}]
    pyproject = (
        '[project]\nname = "demo"\ndependencies = ["idna==3.10"]\n'
        '[project.optional-dependencies]\ndocs = ["requests==2.32.5", "PyYAML==5.3"]\ntest = ["idna==3.10"]\n'
    )
    skill = _skill(tmp_path, {"pyproject.toml": pyproject})

    result = DependencySecurityValidator(use_safety=False, fail_on_medium=True).validate(skill)

    findings = {finding.check_name: finding for finding in result.findings}
    assert set(findings) == {"PYSEC-2020-176", "GHSA-demo-0001"}
    yaml = findings["PYSEC-2020-176"]
    assert (yaml.file_path, yaml.line_number) == ("pyproject.toml", None)
    assert yaml.metadata["declared_constraint"] == "PyYAML==5.3"
    assert yaml.metadata["declaration"] == "optional group docs, declaration 2"
    assert yaml.metadata["dependency_role"] == "docs"
    assert yaml.metadata["dependency_relationship"] == "direct"
    # A pin repeated in an optional group keeps the runtime declaration.
    idna = findings["GHSA-demo-0001"]
    assert idna.severity == Severity.MEDIUM
    assert idna.metadata["declaration"] == "dependencies, declaration 1"
    assert idna.metadata["dependency_role"] == "runtime"
    # fail_on_medium still turns a medium pip-audit advisory into an error.
    assert any("GHSA-demo-0001" in error for error in result.errors)


def test_policy_overlay_can_change_one_advisory(tmp_path, pip_audit):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    skill = _skill(tmp_path, {"requirements.txt": "pyyaml==5.3\n"})
    result = DependencySecurityValidator(use_safety=False).validate(skill)
    assert not result.passed

    policy = ValidationPolicy(profile="reviewed", severity_overrides={"DEPENDENCY.PYSEC-2020-176": Severity.LOW})
    (overlaid,) = apply_policy([result], policy)

    assert overlaid.passed
    assert overlaid.findings[0].severity == Severity.LOW


def test_safety_findings_use_the_common_applicability_metadata(tmp_path, pip_audit, safety):
    skill = _skill(tmp_path, {"requirements.txt": "Pillow_SIMD==9.0\n"})
    safety["report"] = {
        "vulnerabilities": [
            {
                "package_name": "pillow-simd",
                "analyzed_version": "9.0",
                "vulnerability_id": "51457",
                "CVE": "CVE-2026-0001",
                "severity": None,
                "advisory": None,
                "vulnerable_spec": ["<9.1"],
                "fixed_versions": ["9.1"],
            }
        ]
    }

    result = DependencySecurityValidator().validate(skill)

    assert result.passed and not result.is_incomplete
    (finding,) = result.findings
    assert finding.category == "DEPENDENCY"
    assert finding.check_name == "SAFETY-51457"
    assert finding.severity == Severity.MEDIUM
    assert finding.message == "pillow-simd: SAFETY-51457"
    assert (finding.file_path, finding.line_number) == ("requirements.txt", 1)
    assert finding.metadata == {
        "package_name": "pillow-simd",
        "resolved_version": "9.0",
        "declared_constraint": "Pillow_SIMD==9.0",
        "declaration": "line 1",
        "resolution_status": "resolved",
        "dependency_role": "runtime",
        "dependency_relationship": "direct",
        "advisory_id": "SAFETY-51457",
        "advisory_source": "Safety",
        "affected_range": "<9.1",
        "fixed_versions": ["9.1"],
        "aliases": ["CVE-2026-0001"],
    }


@pytest.mark.parametrize(
    "report",
    [
        [{"package_name": "idna", "vulnerability_id": "1", "severity": "critical"}],
        {"vulnerabilities": [{"package_name": "idna", "vulnerability_id": "1", "severity": {"cvssv3": 9.8}}]},
        {"vulnerabilities": [{"package_name": "idna", "vulnerability_id": "1", "advisory": 7}]},
        {"vulnerabilities": [{"package_name": None, "vulnerability_id": None, "severity": None}]},
        {"vulnerabilities": "invalid"},
        "not a report",
    ],
    ids=["top-level-list", "object-severity", "non-text-advisory", "null-fields", "invalid-list", "not-a-report"],
)
def test_malformed_safety_output_never_crashes(tmp_path, pip_audit, safety, report):
    safety["report"] = report
    skill = _skill(tmp_path, {"requirements.txt": "idna==3.10\n"})

    result = DependencySecurityValidator().validate(skill)

    assert not result.is_incomplete
    for finding in result.findings:
        assert finding.category == "DEPENDENCY"
        assert finding.metadata["advisory_source"] == "Safety"
        assert finding.severity in {Severity.CRITICAL, Severity.MEDIUM}
    # Only a high or critical advisory fails the run.
    assert result.passed is not any(finding.severity == Severity.CRITICAL for finding in result.findings)


def test_null_pip_audit_severity_falls_back_instead_of_crashing(tmp_path, pip_audit):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [{**PYYAML_ADVISORY, "severity": None}]
    skill = _skill(tmp_path, {"requirements.txt": "pyyaml==5.3\n"})

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert not result.is_incomplete
    assert [(finding.check_name, finding.severity) for finding in result.findings] == [
        ("PYSEC-2020-176", Severity.HIGH)
    ]


def test_callers_without_declarations_keep_the_tagged_message():
    """The private plugin audit binds its own pins and converts this tagged form."""
    report = {"dependencies": [{"name": "foo", "version": "1.0", "vulns": [{"id": "GHSA-x", "fix_versions": []}]}]}
    result = ValidationResult()

    DependencySecurityValidator(use_safety=False)._process_pip_audit_result(
        ToolResult(True, json.dumps(report), "", 1), result, "requirements.txt"
    )

    assert result.findings == []
    assert result.errors == ["[CVE-HIGH] foo==1.0: GHSA-x"]


def test_warnings_name_the_physical_line_after_hash_continuations(tmp_path, pip_audit):
    text = "idna==3.10 \\\n    --hash=sha256:" + "a" * 64 + "\nrequests>=2\n"
    skill = _skill(tmp_path, {"requirements.txt": text})

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert result.is_incomplete
    assert any(warning.startswith("requirements.txt: line 3 requires an exact") for warning in result.warnings)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a==1\nb==2",
        "a==1 \\\n  --hash=sha256:00 \\\r\n  --hash=sha256:11\r\nb==2\r\n",
        "a==1 \\",
        "a==1 \\\n",
        "a==1 \\\\\nb==2\n",
        "a==1 \\\rb==2\n",
        "a==1 \\\r\r\nb==2",
        "\\\n\\\n\nc==3\x0bd==4\x85",
    ],
)
def test_continuation_join_matches_the_previous_text(text):
    lines = DependencySecurityValidator._requirement_lines(text)

    assert [line for _number, line in lines] == re.sub(r"\\\r?\n", " ", text).splitlines()
    physical = text.splitlines()
    for number, line in lines:
        assert line.startswith(physical[number - 1].removesuffix("\\"))
