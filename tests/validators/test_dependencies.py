# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared-dependency auditing: the audited set comes from the target files."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.models.result import Severity
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.dependencies import (
    UNVERIFIED_CHECK_NAME,
    DependencySecurityValidator,
    parse_dependency_declaration,
    parse_requirements_text,
)

_SKILL_MD = "---\nname: dep-skill\ndescription: Dependency audit fixture\n---\n# Dep skill\n"


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "dep-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    return skill


class _RecordingPipAudit:
    """Fake ``Tools.pip_audit.run`` that snapshots the audited requirement file."""

    def __init__(self, stdout: str = '{"dependencies": []}', exit_code: int = 0) -> None:
        self.calls: list[dict] = []
        self.stdout = stdout
        self.exit_code = exit_code

    def __call__(self, args: list[str], *, cwd: Path | None = None, **_kwargs) -> ToolResult:
        audit_file = Path(args[args.index("-r") + 1])
        self.calls.append(
            {
                "args": list(args),
                "cwd": cwd,
                "audit_file": audit_file,
                "content": audit_file.read_text(encoding="utf-8"),
            }
        )
        return ToolResult(success=True, stdout=self.stdout, stderr="", exit_code=self.exit_code)


@pytest.fixture
def pip_audit_available():
    with (
        patch.object(Tools.pip_audit, "_path", "/usr/bin/pip-audit"),
        patch.object(Tools.pip_audit, "_configuration_error", None),
        patch.object(Tools.safety, "_path", None),
    ):
        yield


def test_pyproject_audits_declared_pins_not_the_running_environment(tmp_path: Path, pip_audit_available) -> None:
    """Regression: ``pip-audit --local`` audited SkillEvaluator's own environment."""
    skill = _skill(tmp_path)
    (skill / "pyproject.toml").write_text(
        "[project]\n"
        'name = "target"\n'
        'version = "0.1.0"\n'
        "dependencies = [\n"
        '  "PyYAML==5.3",\n'
        "  \"urllib3 == 1.26.0 ; python_version >= '3.8'\",\n"
        '  "requests>=2.0",\n'
        "]\n"
        "[project.optional-dependencies]\n"
        'docs = ["Jinja2[i18n]==2.10"]\n',
        encoding="utf-8",
    )
    fake = _RecordingPipAudit()

    with patch.object(Tools.pip_audit, "run", side_effect=fake):
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert "--local" not in call["args"]
    assert {"--no-deps", "--disable-pip"} <= set(call["args"])
    # Exactly the target's pins, normalized; nothing from the running environment.
    assert call["content"] == "jinja2==2.10\npyyaml==5.3\nurllib3==1.26.0\n"
    assert "click" not in call["content"]
    assert call["cwd"] != skill
    assert not call["audit_file"].exists(), "temporary audit file must be removed"
    assert result.passed


def test_unpinned_declarations_are_info_unverified(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "pyproject.toml").write_text(
        '[project]\nname = "target"\nversion = "0.1.0"\n'
        'dependencies = ["requests>=2.0", "idna", "flask==2.*", "pkg @ https://example.com/pkg.whl"]\n',
        encoding="utf-8",
    )

    with patch.object(Tools.pip_audit, "run") as run:
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    run.assert_not_called()
    unverified = [f for f in result.findings if f.check_name == UNVERIFIED_CHECK_NAME]
    assert [f.metadata["package_name"] for f in unverified] == ["requests", "idna", "flask", "pkg"]
    assert all(f.severity == Severity.INFO for f in unverified)
    assert all("cannot audit a floating version" in f.message for f in unverified)
    assert result.passed
    assert any("no exactly pinned dependencies" in m for m in result.messages)


def test_requirements_options_never_reach_pip_audit(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "requirements.txt").write_text(
        "--index-url https://packages.invalid/simple\n"
        "-e git+https://example.invalid/repo.git#egg=evil\n"
        "-r other.txt\n"
        "flask==0.12 \\\n"
        "    --hash=sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "werkzeug>=1.0  # floating\n",
        encoding="utf-8",
    )
    fake = _RecordingPipAudit()

    with patch.object(Tools.pip_audit, "run", side_effect=fake):
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert [call["content"] for call in fake.calls] == ["flask==0.12\n"]
    assert {"--no-deps", "--disable-pip"} <= set(fake.calls[0]["args"])
    assert any("skipped 3 option line(s)" in m for m in result.messages)
    unverified = [f for f in result.findings if f.check_name == UNVERIFIED_CHECK_NAME]
    assert [(f.metadata["package_name"], f.line_number) for f in unverified] == [("werkzeug", 6)]


def test_pinned_vulnerability_is_reported_once(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "pyproject.toml").write_text(
        '[project]\nname = "target"\nversion = "0.1.0"\ndependencies = ["pyyaml==5.3"]\n',
        encoding="utf-8",
    )
    stdout = json.dumps(
        {
            "dependencies": [
                {
                    "name": "pyyaml",
                    "version": "5.3",
                    "vulns": [
                        {"id": "PYSEC-2020-96", "fix_versions": ["5.3.1"]},
                        {"id": "PYSEC-2020-96", "fix_versions": ["5.3.1"]},
                    ],
                }
            ]
        }
    )

    advisory = {"id": "PYSEC-2020-96", "database_specific": {"severity": "HIGH"}}
    with (
        patch.object(Tools.pip_audit, "run", side_effect=_RecordingPipAudit(stdout=stdout, exit_code=1)),
        patch("skillevaluator.validators.dependency_ecosystems.fetch_osv_record", return_value=advisory),
    ):
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert not result.passed
    # A structured finding (policy, SARIF and BENCHMARK.md see it), with the advisory's severity.
    assert [e for e in result.errors if "PYSEC-2020-96" in e] == [
        "[DEPENDENCY-HIGH] pyyaml==5.3: PYSEC-2020-96 -> upgrade to 5.3.1 in pyproject.toml"
    ]
    assert any("pyproject.toml: Found 1 vulnerability(ies)" in m for m in result.messages)


def test_conflicting_pins_are_audited_in_separate_batches(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "pyproject.toml").write_text(
        '[project]\nname = "target"\nversion = "0.1.0"\n'
        "dependencies = [\"numpy==1.26.4; python_version < '3.13'\", \"numpy==2.1.0; python_version >= '3.13'\"]\n",
        encoding="utf-8",
    )
    fake = _RecordingPipAudit()

    with patch.object(Tools.pip_audit, "run", side_effect=fake):
        DependencySecurityValidator(use_safety=False).validate(skill)

    assert sorted(call["content"] for call in fake.calls) == ["numpy==1.26.4\n", "numpy==2.1.0\n"]


def test_pip_audit_failure_without_json_is_a_warning(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "requirements.txt").write_text("pyyaml==5.3\n", encoding="utf-8")

    def _fail(args, **_kwargs):
        return ToolResult(success=True, stdout="", stderr="ERROR: network unreachable\n", exit_code=1)

    with patch.object(Tools.pip_audit, "run", side_effect=_fail):
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert any("pip-audit failed: ERROR: network unreachable" in w for w in result.warnings)
    # Standalone skills keep the warning; plugin runs mark the Python audit INCOMPLETE
    # (see test_dependency_ecosystems.py).
    assert not result.is_incomplete


def test_linked_dependency_file_is_refused(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    outside = tmp_path / "outside-requirements.txt"
    outside.write_text("pyyaml==5.3\n", encoding="utf-8")
    try:
        (skill / "requirements.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with patch.object(Tools.pip_audit, "run") as run:
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    run.assert_not_called()
    assert any("refusing to read dependency file" in w for w in result.warnings)


def test_invalid_pyproject_is_reported_without_auditing(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "pyproject.toml").write_text("[project\n", encoding="utf-8")

    with patch.object(Tools.pip_audit, "run") as run:
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    run.assert_not_called()
    assert any("could not parse dependency declarations" in w for w in result.warnings)


def test_poetry_bare_versions_are_exact(tmp_path: Path, pip_audit_available) -> None:
    skill = _skill(tmp_path)
    (skill / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\npython = "^3.12"\nrequests = "2.31.0"\nhttpx = { version = "^0.27" }\n',
        encoding="utf-8",
    )
    fake = _RecordingPipAudit()

    with patch.object(Tools.pip_audit, "run", side_effect=fake):
        result = DependencySecurityValidator(use_safety=False).validate(skill)

    assert [call["content"] for call in fake.calls] == ["requests==2.31.0\n"]
    unverified = [f.metadata["package_name"] for f in result.findings if f.check_name == UNVERIFIED_CHECK_NAME]
    assert unverified == ["httpx"]


@pytest.mark.parametrize(
    ("raw", "name", "exact"),
    [
        ("requests==2.31.0", "requests", "2.31.0"),
        ("Foo_Bar[extra]==1.0.post1 ; sys_platform == 'linux'", "foo-bar", "1.0.post1"),
        ("pkg==1.0.*", "pkg", None),
        ("pkg===1.0", "pkg", None),
        ("pkg>=1,<2", "pkg", None),
        ("pkg==1.0,==1.0", "pkg", None),
        ("pkg", "pkg", None),
        ("pkg @ https://example.invalid/pkg-1.0.whl", "pkg", None),
        ("pkg==1.0\n--index-url https://packages.invalid", "pkg", None),
        ("./local/path", None, None),
    ],
)
def test_parse_dependency_declaration(raw: str, name: str | None, exact: str | None) -> None:
    declaration = parse_dependency_declaration(raw, line_number=None, role="runtime")
    assert declaration.name == name
    assert declaration.exact_version == exact


def test_parse_requirements_tracks_logical_line_numbers() -> None:
    declarations, skipped = parse_requirements_text(
        "# header\n\nalpha==1.0 \\\n  --hash=sha256:abc\nbeta>=2\n--extra-index-url https://x.invalid\n",
        role="requirements",
    )
    assert skipped == 1
    assert [(d.name, d.exact_version, d.line_number) for d in declarations] == [
        ("alpha", "1.0", 3),
        ("beta", None, 5),
    ]
