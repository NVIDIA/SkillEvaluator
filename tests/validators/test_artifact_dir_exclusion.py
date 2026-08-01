# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact dirs (evals/, results/, ...) must be skipped, not just filtered.

A skill that has been through Tier 3 accumulates hundreds of MB under
``evals/results/``. Tier 1 must never traverse those subtrees
(``iter_scannable_files`` pruning), never hand them to scanners that lack
exclude flags (skillspector scans a filtered copy), and never report
findings from them (gitleaks config allowlist, skillspector issue filter).
Regression source: a real skill dir with 798MB / 53k artifact files took
~25 minutes to validate and reported 270 artifact-only secret findings.
"""

import json
import os
import re
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.constants import SCAN_EXCLUDED_DIRS
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.base import iter_scannable_files
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator


@pytest.fixture
def skill_with_artifacts(tmp_path):
    """A skill dir carrying Tier 3 artifact subtrees."""
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\n---\n")
    (skill / "payload.py").write_text("import os\n")
    junk = skill / "evals" / "results" / "20260101_000000"
    junk.mkdir(parents=True)
    (junk / "leak.py").write_text("password = 'hunter2'\n")
    (skill / "results").mkdir()
    (skill / "results" / "old.py").write_text("x = 1\n")
    return skill


# =============================================================================
# iter_scannable_files MUST PRUNE, NOT POST-FILTER
# =============================================================================


class TestIterScannableFilesPrunes:
    def test_returns_only_live_files(self, skill_with_artifacts):
        found = iter_scannable_files(skill_with_artifacts, {".py"})
        assert [f.name for f in found] == ["payload.py"]

    def test_never_descends_into_excluded_dirs(self, skill_with_artifacts, monkeypatch):
        visited: list[str] = []
        real_scandir = os.scandir

        def spy(path=".", *args, **kwargs):
            visited.append(os.fspath(path))
            return real_scandir(path, *args, **kwargs)

        monkeypatch.setattr(os, "scandir", spy)
        iter_scannable_files(skill_with_artifacts, {".py"})

        excluded_visits = [p for p in visited if Path(p).name in SCAN_EXCLUDED_DIRS]
        assert excluded_visits == []

    def test_walks_the_tree_once_regardless_of_extension_count(self, skill_with_artifacts, monkeypatch):
        visited: list[str] = []
        real_scandir = os.scandir

        def spy(path=".", *args, **kwargs):
            visited.append(os.fspath(path))
            return real_scandir(path, *args, **kwargs)

        monkeypatch.setattr(os, "scandir", spy)
        iter_scannable_files(skill_with_artifacts, {".py", ".md", ".sh", ".js", ".txt"})

        root_visits = [p for p in visited if Path(p) == skill_with_artifacts]
        assert len(root_visits) == 1

    def test_file_root_still_matches_without_walk(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("x = 1\n")
        assert iter_scannable_files(f, {".py"}) == [f]
        assert iter_scannable_files(f, {".md"}) == []

    def test_endswith_semantics_preserved(self, tmp_path):
        skill = tmp_path / "s"
        skill.mkdir()
        hidden = skill / ".md"
        hidden.write_text("hidden\n")
        regular = skill / "readme.md"
        regular.write_text("hi\n")
        found = {f.name for f in iter_scannable_files(skill, {".md"}, excluded_files=())}
        assert found == {".md", "readme.md"}


# =============================================================================
# SKILLSPECTOR SCANS A FILTERED COPY (it has no exclude flag of its own)
# =============================================================================


def _clean_spector_result():
    return ToolResult(
        success=True,
        stdout=json.dumps(
            {
                "risk_assessment": {"score": 0, "severity": "LOW", "recommendation": "SAFE"},
                "issues": [],
                "metadata": {"llm_requested": False, "llm_available": False},
            }
        ),
        stderr="",
        exit_code=0,
    )


class TestSkillspectorFilteredCopy:
    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run")
    def test_scans_copy_without_artifact_dirs(self, mock_run, skill_with_artifacts):
        seen: dict = {}

        def capture(args, **kwargs):
            scanned = Path(args[args.index("scan") + 1])
            seen["path"] = scanned
            seen["has_manifest"] = (scanned / "SKILL.md").is_file()
            seen["has_payload"] = (scanned / "payload.py").is_file()
            seen["artifact_dirs"] = sorted(d.name for d in scanned.iterdir() if d.name in SCAN_EXCLUDED_DIRS)
            return _clean_spector_result()

        mock_run.side_effect = capture
        SecurityValidator()._run_skillspector(skill_with_artifacts)

        assert seen["path"] != skill_with_artifacts.resolve()
        assert seen["path"].name == "my-skill"
        assert seen["has_manifest"] and seen["has_payload"]
        assert seen["artifact_dirs"] == []
        assert not seen["path"].exists()  # temp copy cleaned up afterwards

    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run")
    def test_clean_skill_scanned_in_place(self, mock_run, tmp_path):
        skill = tmp_path / "clean-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: clean-skill\n---\n")
        mock_run.return_value = _clean_spector_result()

        SecurityValidator()._run_skillspector(skill)

        args = mock_run.call_args.args[0]
        assert args[args.index("scan") + 1] == str(skill.resolve())

    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run")
    def test_findings_map_back_to_original_paths(self, mock_run, skill_with_artifacts):
        def report_on_copy(args, **kwargs):
            scanned = args[args.index("scan") + 1]
            data = {
                "skill": {"name": "my-skill", "source": scanned},
                "risk_assessment": {"score": 80, "severity": "HIGH", "recommendation": "DO_NOT_INSTALL"},
                "issues": [
                    {
                        "id": "SS001",
                        "category": "execution",
                        "pattern": "eval-usage",
                        "severity": "HIGH",
                        "confidence": 1.0,
                        "location": {"file": f"{scanned}/payload.py", "start_line": 1},
                        "finding": "dangerous call",
                    }
                ],
            }
            return ToolResult(success=False, stdout=json.dumps(data), stderr="", exit_code=1)

        mock_run.side_effect = report_on_copy
        result = SecurityValidator()._run_skillspector(skill_with_artifacts)

        assert len(result.findings) == 1
        assert result.findings[0].file_path == str(skill_with_artifacts.resolve() / "payload.py")


# =============================================================================
# GITLEAKS CONFIG EXCLUDES ARTIFACT DIRS
# =============================================================================


class TestGitleaksArtifactExclusion:
    def test_config_allowlists_artifact_dir_paths(self):
        config_path = SecretsValidator()._create_gitleaks_config()
        try:
            config = tomllib.loads(config_path.read_text())
        finally:
            config_path.unlink(missing_ok=True)

        paths = config["allowlist"]["paths"]
        for name in SCAN_EXCLUDED_DIRS:
            pattern = f"(^|/){re.escape(name)}(/|$)"
            assert pattern in paths, f"gitleaks allowlist missing artifact dir {name!r}"


# =============================================================================
# SKILLSPECTOR ISSUES UNDER ARTIFACT DIRS ARE DROPPED
# =============================================================================


class TestSkillspectorArtifactIssueFilter:
    def test_issue_under_artifact_dir_is_filtered(self):
        issue = {"location": {"file": "evals/results/20260101/leak.py", "start_line": 3}}
        assert SecurityValidator._is_generated_artifact_issue(issue) is True

    def test_issue_under_absolute_artifact_path_is_filtered(self):
        issue = {"location": {"file": "/home/u/skill/evals/leak.py", "start_line": 3}}
        assert SecurityValidator._is_generated_artifact_issue(issue) is True

    def test_live_issue_is_kept(self):
        issue = {"location": {"file": "scripts/deploy.py", "start_line": 3}}
        assert SecurityValidator._is_generated_artifact_issue(issue) is False
