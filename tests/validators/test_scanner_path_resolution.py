# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scanner argv must receive resolved absolute paths.

A skill directory named like an option token (e.g. ``--exclude=*``) that
reaches a scanner argv as a bare relative positional is parsed by the
scanner as an option, silently excluding the skill from analysis. Every
external scanner call site must therefore pass ``skill_path.resolve()``.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator

MALICIOUS_DIR_NAME = "--exclude=all"

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
    def test_gitleaks_argv_uses_resolved_path(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        SecretsValidator()._validate_single_skill(malicious_skill_path)

        args = mock_run.call_args.args[0]
        _assert_resolved(args[args.index("--source") + 1], malicious_skill_path)


class TestSkillspectorPathResolution:
    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run")
    def test_skillspector_argv_uses_resolved_path(self, mock_run, malicious_skill_path):
        mock_run.return_value = CLEAN_RESULT

        SecurityValidator()._run_skillspector(malicious_skill_path)

        args = mock_run.call_args.args[0]
        _assert_resolved(args[args.index("scan") + 1], malicious_skill_path)
