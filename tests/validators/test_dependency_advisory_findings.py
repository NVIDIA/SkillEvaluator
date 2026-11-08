# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency advisories are structured findings bound to the pin that declared them."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
from rich.console import Console

from skillevaluator.reporting import JSONReporter, MarkdownReporter
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.sarif_reporter import SARIFReporter
from skillevaluator.utils.tool_runner import Severity, ToolResult, Tools
from skillevaluator.validators.base import ValidationResult
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.policy import ValidationPolicy, apply_policy

PYYAML_ADVISORY = {"id": "PYSEC-2020-176", "fix_versions": ["5.4"], "aliases": ["CVE-2020-14343"]}


def _skill(tmp_path: Path, files: dict[str, str], name: str = "demo-skill") -> Path:
    skill = tmp_path / name
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Demo\n---\n", encoding="utf-8")
    for filename, text in files.items():
        (skill / filename).write_text(text, encoding="utf-8")
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
        ("requirements-devel.txt", "dev"),
        ("requirements-devtools.txt", "dev"),
        ("requirements_unittest.txt", "test"),
        ("requirements-pytest.txt", "test"),
        ("requirements-lint.txt", "dev"),
        ("requirements-latest.txt", "runtime"),
        ("requirements-docker.txt", "runtime"),
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


@pytest.mark.parametrize(
    ("group", "role"),
    [
        ("devtools", "dev"),
        ("dev-tools", "dev"),
        ("unittest", "test"),
        ("testutils", "test"),
        ("pytest", "test"),
        ("Docs", "docs"),
        ("device", "runtime"),
        ("latest", "runtime"),
        ("docker", "runtime"),
    ],
)
def test_optional_group_name_sets_dependency_role(group, role):
    assert DependencySecurityValidator._role_from_name(group) == role


def test_requirements_files_are_audited_in_sorted_order(tmp_path, pip_audit, monkeypatch):
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    advisories["idna"] = [{"id": "PYSEC-2024-60", "fix_versions": ["3.7"]}]
    skill = _skill(tmp_path, {"requirements.txt": "pyyaml==5.3\n", "requirements-dev.txt": "idna==3.6\n"})
    # Directory order differs between filesystems; list the files in reverse.
    glob = Path.glob

    def reversed_glob(self, pattern, *args, **kwargs):
        return iter(sorted(glob(self, pattern, *args, **kwargs), reverse=True))

    monkeypatch.setattr(Path, "glob", reversed_glob)

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert [(finding.file_path, finding.check_name) for finding in result.findings] == [
        ("requirements-dev.txt", "PYSEC-2024-60"),
        ("requirements.txt", "PYSEC-2020-176"),
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


def test_policy_overlay_keeps_the_incomplete_reason(tmp_path, pip_audit):
    """Rebuilding a result from changed findings keeps warnings that no finding mirrors."""
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    skill = _skill(tmp_path, {"requirements.txt": "pyyaml==5.3\nrequests>=2\n"})
    result = DependencySecurityValidator(use_safety=False).validate(skill)
    reason = (
        "requirements.txt: line 2 requires an exact package version; "
        "URLs, local paths, includes and pip options are not audited"
    )
    assert result.warnings == [reason]

    policy = ValidationPolicy(profile="reviewed", severity_overrides={"DEPENDENCY.PYSEC-2020-176": Severity.LOW})
    (overlaid,) = apply_policy([result], policy)

    assert overlaid.is_incomplete and not overlaid.passed
    assert overlaid.errors == []
    assert overlaid.warnings == [
        "[DEPENDENCY-LOW] pyyaml==5.3: PYSEC-2020-176 -> upgrade to 5.4 in requirements.txt:1",
        reason,
    ]
    assert (overlaid.summary.errors, overlaid.summary.warnings) == (0, 2)
    # apply_policy runs more than once in a CLI run; a second pass changes nothing.
    apply_policy([overlaid], policy)
    assert overlaid.warnings[1:] == [reason]


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
    ids=["top-level-list", "cvss-not-an-object", "non-text-advisory", "null-fields", "invalid-list", "not-a-report"],
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


# The severity object Safety prints with an API key.
SAFETY_CVSS_OBJECT = {
    "source": "CVE-2024-3651",
    "cvssv2": None,
    "cvssv3": {
        "base_score": 9.8,
        "impact_score": 5.9,
        "base_severity": "CRITICAL",
        "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    },
}


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("high", Severity.HIGH),
        (" Critical ", Severity.CRITICAL),
        (None, Severity.MEDIUM),
        ("unknown", Severity.MEDIUM),
        (SAFETY_CVSS_OBJECT, Severity.CRITICAL),
        ({"cvssv3": {"base_severity": "NONE", "base_score": 9.1}}, Severity.CRITICAL),
        ({"cvssv3": {"base_score": 7.5}}, Severity.HIGH),
        ({"cvss_v3": {"base_severity": "high"}}, Severity.HIGH),
        ({"cvssv3": {"base_score": "9.8"}, "cvssv2": {"base_score": 7.5}}, Severity.HIGH),
        ({"cvssv2": {"base_score": 9.3, "base_severity": "HIGH"}}, Severity.CRITICAL),
        ({"cvss_v2": {"base_severity": "HIGH"}}, Severity.HIGH),
        ({"source": "CVE-2024-3651"}, Severity.MEDIUM),
        ({"cvssv3": {"base_score": True}}, Severity.MEDIUM),
        ({"cvssv3": {"base_score": float("nan")}}, Severity.MEDIUM),
        ({"cvssv3": {"base_score": 11}}, Severity.MEDIUM),
    ],
    ids=[
        "word-high",
        "word-padded",
        "null",
        "unknown-word",
        "object-critical",
        "object-v3-score",
        "object-v3-score-only",
        "object-v3-key-spelling",
        "object-v2-score-after-bad-v3",
        "object-v2-score",
        "object-v2-level-only",
        "object-without-score",
        "object-bool-score",
        "object-nan-score",
        "object-out-of-range",
    ],
)
def test_safety_severity_reads_level_words_and_cvss_objects(tmp_path, pip_audit, safety, severity, expected):
    """With an API key, Safety prints severity as a CVSS object; a critical one fails the run."""
    safety["report"] = {
        "vulnerabilities": [
            {
                "package_name": "idna",
                "analyzed_version": "3.6",
                "vulnerability_id": "51457",
                "advisory": "x",
                "severity": severity,
            }
        ]
    }
    skill = _skill(tmp_path, {"requirements.txt": "idna==3.6\n"})

    result = DependencySecurityValidator().validate(skill)

    assert not result.is_incomplete
    assert [(finding.check_name, finding.severity) for finding in result.findings] == [("SAFETY-51457", expected)]
    assert result.passed is (expected not in {Severity.CRITICAL, Severity.HIGH})


@pytest.mark.parametrize(
    ("advisory", "check_name"),
    [
        ({"vulnerability_id": "51457"}, "SAFETY-51457"),
        ({"vulnerability_id": 51457}, "SAFETY-51457"),
        ({"vulnerability_id": None, "id": 7}, "SAFETY-7"),
        ({"vulnerability_id": "PYUP-2024-1"}, "PYUP-2024-1"),
        ({"vulnerability_id": True}, "Unknown"),
    ],
    ids=["text", "number", "id-number", "named", "bool"],
)
def test_safety_advisory_ids_name_the_finding(tmp_path, pip_audit, safety, advisory, check_name):
    safety["report"] = {
        "vulnerabilities": [{"package_name": "idna", "analyzed_version": "3.6", "severity": "high", **advisory}]
    }
    skill = _skill(tmp_path, {"requirements.txt": "idna==3.6\n"})

    result = DependencySecurityValidator().validate(skill)

    (finding,) = result.findings
    assert (finding.check_name, finding.message) == (check_name, f"idna: {check_name}")
    assert finding.metadata["advisory_id"] == check_name


def test_folder_run_audits_every_skill_after_a_critical_advisory(tmp_path, pip_audit):
    """A critical advisory is not a configuration error, so a folder run still audits later skills."""
    advisories, calls = pip_audit
    advisories["aaa"] = [{"id": "GHSA-crit-0001", "fix_versions": ["2.0"], "severity": "critical"}]
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    root = tmp_path / "skills"
    root.mkdir()
    _skill(root, {"requirements.txt": "aaa==1.0\n"}, name="a-skill")
    _skill(root, {"requirements.txt": "pyyaml==5.3\n"}, name="b-skill")
    _skill(root, {"requirements.txt": "numpy>=1\n"}, name="c-skill")

    result = DependencySecurityValidator(use_safety=False).validate(root)

    assert calls == [["aaa==1.0"], ["pyyaml==5.3"]]
    assert [(finding.file_path, finding.check_name, finding.severity) for finding in result.findings] == [
        ("[a-skill] requirements.txt", "GHSA-crit-0001", Severity.CRITICAL),
        ("[b-skill] requirements.txt", "PYSEC-2020-176", Severity.HIGH),
    ]
    assert result.incomplete_scans == ["pip-audit"]
    # Accepting the first skill's advisory leaves the second one failing the run.
    policy = ValidationPolicy(profile="reviewed", severity_overrides={"DEPENDENCY.GHSA-crit-0001": Severity.LOW})
    (overlaid,) = apply_policy([result], policy)
    assert not overlaid.passed
    assert overlaid.summary.high_count == 1


def test_safety_skips_an_advisory_pip_audit_already_reported(tmp_path, pip_audit, safety):
    """One vulnerability is one finding: Safety's copy of a pip-audit advisory is dropped by ID or CVE."""
    advisories, _ = pip_audit
    advisories["pyyaml"] = [{"id": "PYSEC-2021-142", "fix_versions": ["5.4"], "aliases": ["CVE-2020-14343"]}]
    safety["report"] = {
        "vulnerabilities": [
            {
                "package_name": "PyYAML",
                "analyzed_version": "5.3.0",
                "vulnerability_id": "39611",
                "CVE": "CVE-2020-14343",
            },
            {"package_name": "pyyaml", "analyzed_version": "5.3", "vulnerability_id": "PYSEC-2021-142"},
            {"package_name": "pyyaml", "analyzed_version": "5.3", "vulnerability_id": "38100", "CVE": "CVE-2019-20477"},
            {"package_name": "pyyaml", "analyzed_version": "5.2", "vulnerability_id": "39612", "CVE": "CVE-2020-14343"},
        ]
    }
    skill = _skill(tmp_path, {"requirements.txt": "pyyaml==5.3\n"})

    result = DependencySecurityValidator().validate(skill)

    assert [(finding.check_name, finding.metadata["advisory_source"]) for finding in result.findings] == [
        ("PYSEC-2021-142", "pip-audit"),
        ("SAFETY-38100", "Safety"),
        ("SAFETY-39612", "Safety"),
    ]


def test_safety_advisory_without_a_matching_pin_is_unbound(tmp_path, pip_audit, safety):
    """Safety reads the raw file, so it can report a pin the audit skipped for its marker."""
    safety["report"] = {
        "vulnerabilities": [
            {"package_name": "legacy-lib", "analyzed_version": "1.0", "vulnerability_id": "70001", "severity": "high"}
        ]
    }
    skill = _skill(tmp_path, {"requirements-dev.txt": 'idna==3.10\nlegacy-lib==1.0; python_version < "3"\n'})

    result = DependencySecurityValidator().validate(skill)

    (finding,) = result.findings
    assert (finding.file_path, finding.line_number) == ("requirements-dev.txt", None)
    assert finding.metadata == {
        "package_name": "legacy-lib",
        "resolved_version": "1.0",
        "declared_constraint": None,
        "declaration": None,
        "resolution_status": "unbound",
        "dependency_role": "dev",
        "dependency_relationship": None,
        "advisory_id": "SAFETY-70001",
        "advisory_source": "Safety",
        "affected_range": None,
        "fixed_versions": [],
        "aliases": [],
    }
    assert not result.passed


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
    ("text", "numbers"),
    [
        ("", []),
        ("a==1\nb==2", [1, 2]),
        ("a==1 \\\n  --hash=sha256:00 \\\r\n  --hash=sha256:11\r\nb==2\r\n", [1, 4]),
        ("a==1 \\", [1]),
        ("a==1 \\\n", [1]),
        ("a==1 \\\\\nb==2\n", [1]),
        ("a==1 \\\rb==2\n", [1, 2]),
        ("a==1 \\\r\r\nb==2", [1, 2, 3]),
        ("\\\n\\\n\nc==3\x0bd==4\x85", [1, 4, 4]),
        ("# a\x0cnote\npyyaml==5.3\n", [1, 1, 2]),
        ("a==1\u2028b==2\nc==3\x1ed==4", [1, 1, 2, 2]),
    ],
    ids=[
        "empty",
        "no-final-newline",
        "hash-continuations-crlf",
        "trailing-backslash",
        "continuation-at-end",
        "escaped-backslash",
        "backslash-cr",
        "backslash-cr-crlf",
        "vertical-tab-and-nel",
        "form-feed",
        "line-separator-and-record-separator",
    ],
)
def test_continuation_join_matches_the_previous_text(text, numbers):
    """The joined text is unchanged; only CR, LF and CRLF end a numbered line."""
    lines = DependencySecurityValidator._requirement_lines(text)

    assert [line for _number, line in lines] == re.sub(r"\\\r?\n", " ", text).splitlines()
    assert [number for number, _line in lines] == numbers


def test_finding_line_follows_editor_line_breaks(tmp_path, pip_audit):
    """A form feed inside a comment does not shift the line that SARIF and editors show."""
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    skill = _skill(tmp_path, {"requirements.txt": "# a\x0cnote\npyyaml==5.3\n"})

    result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert [(finding.file_path, finding.line_number) for finding in result.findings] == [("requirements.txt", 2)]


def test_cli_report_keeps_the_incomplete_reason_next_to_an_advisory(tmp_path, pip_audit):
    """A confirmed advisory fails the check, and the CLI still says why the audit was incomplete."""
    advisories, _ = pip_audit
    advisories["pyyaml"] = [PYYAML_ADVISORY]
    skill = _skill(tmp_path, {"requirements.txt": "pyyaml==5.3\nrequests>=2\n"})
    validator = DependencySecurityValidator(use_safety=False)
    result = validator.validate(skill)
    result.validator_name = validator.name
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None)

    CLIReporter(console=console).render_result(result, console)

    output = buffer.getvalue()
    assert result.is_incomplete and result.status == "incomplete"
    assert "[FAIL] Validation failed" in output
    assert "[WARN] requirements.txt: line 2 requires an exact package version" in output
    assert output.count("PYSEC-2020-176 -> upgrade to 5.4") == 1
    assert output.index("Warnings:") < output.index("Issues:")


def test_markdown_report_keeps_the_incomplete_reason_next_to_an_advisory(tmp_path, pip_audit):
    """The PR-comment report still says why the audit was incomplete when an advisory is listed."""
    advisories, _ = pip_audit
    advisories["idna"] = [{"id": "OSV-2024-1", "fix_versions": ["3.11"], "severity": "medium"}]
    skill = _skill(tmp_path, {"requirements.txt": "idna==3.10\nrequests>=2\n"})
    validator = DependencySecurityValidator(use_safety=False)
    result = validator.validate(skill)
    result.validator_name = validator.name

    output = MarkdownReporter().render_all([result])

    assert result.status == "incomplete"
    assert "**Incomplete scanners:** pip-audit" in output
    assert "- ⚠️ requirements.txt: line 2 requires an exact package version" in output
    assert "idna==3.10: OSV-2024-1 -&gt; upgrade to 3.11 | <code>requirements.txt:1</code>" in output
    # The advisory is in the findings table, not repeated as a warning line.
    assert "- ⚠️ [DEPENDENCY-MEDIUM]" not in output


def test_markdown_report_keeps_execution_errors_as_the_incomplete_reason():
    """With an execution error and no finding, the error explains the incomplete scan, as before."""
    result = ValidationResult()
    result.validator_name = "Dependency Vulnerability Audit"
    result.add_error("pip-audit crashed")
    result.add_warning("requirements.txt: pip-audit did not complete")
    result.mark_scan_incomplete("pip-audit")

    output = MarkdownReporter().render_all([result])

    assert "- ❌ pip-audit crashed" in output
    assert "- ⚠️ requirements.txt: pip-audit did not complete" not in output
