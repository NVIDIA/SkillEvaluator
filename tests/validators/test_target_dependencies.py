# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.dependencies import DependencySecurityValidator


def audit(tmp_path, text, monkeypatch, *, exit_code=0, output=None, available=True):
    target = tmp_path / "pyproject.toml"
    target.write_text(text)
    calls = []
    monkeypatch.setattr(type(Tools.pip_audit), "is_available", property(lambda _self: available))

    def run(args, **kwargs):
        assert "--local" not in args
        assert "--no-deps" in args and "--disable-pip" in args
        requirements = Path(args[args.index("-r") + 1]).read_text().splitlines()
        calls.extend(requirements)
        dependencies = [
            {"name": pin.split("==")[0].lower(), "version": pin.split("==")[1], "vulns": []} for pin in requirements
        ]
        return ToolResult(
            True, output if output is not None else json.dumps({"dependencies": dependencies}), "", exit_code
        )

    monkeypatch.setattr(Tools.pip_audit, "run", run)
    return DependencySecurityValidator(use_safety=False)._audit_pyproject(target), calls


def test_audits_target_pins_and_optional_groups(tmp_path, monkeypatch):
    result, calls = audit(
        tmp_path,
        '[project]\ndependencies=["Requests==2.32.5"]\n[project.optional-dependencies]\nextra=["idna==3.10", "idna==3.10"]',
        monkeypatch,
    )
    assert result.status == "passed"
    assert calls == ["Requests==2.32.5", "idna==3.10"]
    assert result.metadata["dependency_coverage"]["audited_pins"] == calls
    assert result.metadata["dependency_coverage"]["transitive_dependencies"] == "not audited"


@pytest.mark.parametrize(
    "declaration", ["requests>=2", "requests==2.*", "requests @ https://example.com/pkg.whl", "--local", "bad\n--local"]
)
def test_unresolved_declarations_are_incomplete(tmp_path, monkeypatch, declaration):
    text = "[project]\ndependencies=" + json.dumps([declaration, "idna==3.10"])
    result, calls = audit(tmp_path, text, monkeypatch)
    assert result.status == "incomplete"
    assert calls == ["idna==3.10"]
    assert declaration not in " ".join(result.warnings)


@pytest.mark.parametrize(
    "text",
    [
        "[tool.poetry]\nname='example'",
        "[project]\ndynamic=['dependencies']",
        "[project]\ndependencies='invalid'",
        "[project]\noptional-dependencies=[]",
        "[project\n",
    ],
)
def test_invalid_or_dynamic_metadata_is_incomplete(tmp_path, monkeypatch, text):
    result, calls = audit(tmp_path, text, monkeypatch)
    assert result.status == "incomplete"
    assert not calls


def test_markers_are_explicitly_scoped(tmp_path, monkeypatch):
    result, calls = audit(tmp_path, "[project]\ndependencies=[\"idna==3.10; python_version < '2'\"]", monkeypatch)
    assert result.status == "passed"
    assert not calls
    assert "skipped" in " ".join(result.messages)


@pytest.mark.parametrize(
    "output,exit_code",
    [
        ("oops", 0),
        ('{"dependencies":[]}', 0),
        ('{"dependencies":[]}', 2),
        ('{"dependencies":[{"name":"idna","version":"3.10","vulns":[]}]}', 1),
    ],
)
def test_failed_or_incomplete_inventory_is_incomplete(tmp_path, monkeypatch, output, exit_code):
    result, _ = audit(
        tmp_path, '[project]\ndependencies=["idna==3.10"]', monkeypatch, output=output, exit_code=exit_code
    )
    assert result.status == "incomplete"
    assert result.metadata["dependency_coverage"]["requested_pins"] == ["idna==3.10"]
    assert result.metadata["dependency_coverage"]["audited_pins"] == []


def test_missing_scanner_is_incomplete(tmp_path, monkeypatch):
    result, calls = audit(tmp_path, '[project]\ndependencies=["idna==3.10"]', monkeypatch, available=False)
    assert result.status == "incomplete"
    assert not calls


@pytest.mark.parametrize("fields", [{"aliases": [{"cvss": None}]}, {"severity": []}, {"fix_versions": [None]}])
def test_invalid_advisory_fields_are_incomplete(tmp_path, monkeypatch, fields):
    output = json.dumps(
        {"dependencies": [{"name": "idna", "version": "3.10", "vulns": [{"id": "PYSEC-TEST", **fields}]}]}
    )
    result, _ = audit(tmp_path, '[project]\ndependencies=["idna==3.10"]', monkeypatch, output=output, exit_code=1)
    assert result.status == "incomplete"
    assert result.metadata["dependency_coverage"]["audited_pins"] == []


def test_vulnerable_target_preserves_findings(tmp_path, monkeypatch):
    output = json.dumps(
        {
            "dependencies": [
                {"name": "requests", "version": "2.19.1", "vulns": [{"id": "PYSEC-TEST", "fix_versions": ["2.32.5"]}]}
            ]
        }
    )
    result, _ = audit(tmp_path, '[project]\ndependencies=["requests==2.19.1"]', monkeypatch, output=output, exit_code=1)
    assert result.status == "failed"
    assert "PYSEC-TEST" in " ".join(result.errors)


def test_symlink_is_not_read(tmp_path):
    external = tmp_path / "outside.toml"
    external.write_text('[project]\ndependencies=["idna==3.10"]')
    link = tmp_path / "pyproject.toml"
    link.symlink_to(external)
    assert DependencySecurityValidator()._audit_pyproject(link).status == "incomplete"


def test_folder_target_coverage_survives_report_exports(tmp_path, monkeypatch):
    for name in ("first", "second"):
        skill = tmp_path / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Control\n---\n")
    validator = DependencySecurityValidator(use_safety=False)

    def audit_target(target):
        result, _ = audit(target, '[project]\ndependencies=["idna==3.10"]', monkeypatch)
        return result

    monkeypatch.setattr(validator, "_validate_single_skill", audit_target)
    result = validator.validate(tmp_path)
    coverage = result.metadata["dependency_coverage"]
    assert {item["target"] for item in coverage} == {"first", "second"}
    assert all(item["audited_pins"] == ["idna==3.10"] for item in coverage)
    json_data = json.loads(JSONReporter().render_all([result]))
    assert json_data["results"][0]["dependency_coverage"] == coverage
    assert HTMLReporter()._results_to_dict([result])[0]["dependency_coverage"] == coverage
