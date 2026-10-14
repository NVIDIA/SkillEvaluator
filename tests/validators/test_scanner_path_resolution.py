# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scanner invocations must not expose relative paths as option tokens.

A skill directory named like an option token (e.g. ``--exclude=*``) that
reaches a scanner argv as a bare relative positional is parsed by the
scanner as an option, silently excluding the skill from analysis. Most
scanners therefore receive ``skill_path.resolve()``. Gitleaks no-git scans
instead use paths relative to the requested scan root so allowlists cannot
match unrelated absolute ancestors.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.constants import SCAN_EXCLUDED_DIRS
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator

MALICIOUS_DIR_NAME = "--exclude=all"
GITLEAKS_ALLOWLIST_DIR_NAMES = sorted(
    {
        "test",
        "tests",
        "example",
        "examples",
        "fixture",
        "fixtures",
        "mock",
        "mocks",
        *SCAN_EXCLUDED_DIRS,
    }
)

CLEAN_RESULT = ToolResult(success=True, stdout=json.dumps({"results": []}), stderr="", exit_code=0)


@pytest.fixture
def malicious_skill_path(tmp_path, monkeypatch):
    """A relative path to a skill directory named like a scanner option."""
    skill_dir = tmp_path / MALICIOUS_DIR_NAME
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: evil\n---\n")
    (skill_dir / "payload.py").write_text("import os\n")
    monkeypatch.chdir(tmp_path)
    return Path(MALICIOUS_DIR_NAME)


def _assert_resolved(argv_path: str, relative_path: Path) -> None:
    assert argv_path == str(relative_path.resolve())
    assert Path(argv_path).is_absolute()
    assert not argv_path.startswith("-")


class TestBanditPathResolution:
    @patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
    @patch.object(Tools.bandit, "run")
    def test_bandit_argv_uses_resolved_path(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        CodeRiskValidator()._run_bandit(malicious_skill_path)

        args = mock_run.call_args.args[0]
        _assert_resolved(args[args.index("-r") + 1], malicious_skill_path)


class TestSemgrepPathResolution:
    @patch.object(Tools.semgrep, "_path", "/usr/bin/semgrep")
    @patch.object(Tools.semgrep, "run")
    def test_semgrep_argv_uses_resolved_path(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        CodeRiskValidator()._run_semgrep(malicious_skill_path)

        args = mock_run.call_args.args[0]
        _assert_resolved(args[-1], malicious_skill_path)


class TestGitleaksPathResolution:
    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run")
    def test_gitleaks_no_git_uses_dot_source_from_resolved_skill_dir(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        SecretsValidator()._validate_single_skill(malicious_skill_path)

        args = mock_run.call_args.args[0]
        assert args[args.index("--source") + 1] == "."
        assert "--no-git" in args
        assert "--gitleaks-ignore-path" not in args
        assert mock_run.call_args.kwargs["cwd"] == malicious_skill_path.resolve()

    @pytest.mark.parametrize("ancestor_name", GITLEAKS_ALLOWLIST_DIR_NAMES)
    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run")
    def test_gitleaks_no_git_does_not_expose_allowlisted_ancestors(
        self, mock_run, tmp_path, ancestor_name
    ):
        mock_run.return_value = CLEAN_RESULT
        skill_path = tmp_path / ancestor_name / "production-skill"
        skill_path.mkdir(parents=True)

        SecretsValidator()._validate_single_skill(skill_path)

        args = mock_run.call_args.args[0]
        assert args[args.index("--source") + 1] == "."
        assert mock_run.call_args.kwargs["cwd"] == skill_path.resolve()

    @pytest.mark.parametrize("ancestor_name", GITLEAKS_ALLOWLIST_DIR_NAMES)
    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run")
    def test_gitleaks_no_git_direct_file_uses_safe_relative_source(
        self, mock_run, tmp_path, ancestor_name
    ):
        mock_run.return_value = CLEAN_RESULT
        target = tmp_path / ancestor_name / "--exclude=all.mdc"
        target.parent.mkdir(parents=True)
        target.write_text("rule content\n")

        SecretsValidator()._validate_single_skill(target)

        args = mock_run.call_args.args[0]
        assert args[args.index("--source") + 1] == f"./{target.name}"
        assert "--gitleaks-ignore-path" not in args
        assert mock_run.call_args.kwargs["cwd"] == target.parent.resolve()

    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run")
    def test_gitleaks_git_history_keeps_resolved_source(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        SecretsValidator(scan_git_history=True)._validate_single_skill(malicious_skill_path)

        args = mock_run.call_args.args[0]
        _assert_resolved(args[args.index("--source") + 1], malicious_skill_path)
        assert "--no-git" not in args
        assert mock_run.call_args.kwargs.get("cwd") is None


class TestSkillspectorPathResolution:
    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run")
    def test_skillspector_argv_uses_resolved_path(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        SecurityValidator()._run_skillspector(malicious_skill_path)

        args = mock_run.call_args.args[0]
        _assert_resolved(args[args.index("scan") + 1], malicious_skill_path)
