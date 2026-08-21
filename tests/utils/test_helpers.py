# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.utils module."""

import subprocess
from pathlib import Path
from unittest.mock import patch

from skillevaluator.utils import find_skills_in_directory, get_skill_name_from_path
from skillevaluator.utils.helpers import (
    _ssh_to_https,
    resolve_git_remote_url,
)


class TestFindSkillsInDirectory:
    """Tests for find_skills_in_directory function."""

    def test_find_skill_from_skill_md_file_names(self, tmp_path: Path):
        """Both accepted skill manifest casings resolve to their parent skill dir."""
        for filename, dirname in (("SKILL.md", "upper-skill-md"), ("skill.md", "lower-skill-md")):
            skill_dir = tmp_path / dirname
            skill_dir.mkdir()
            skill_md = skill_dir / filename
            skill_md.write_text("---\nname: my-skill\n---\n")

            result = find_skills_in_directory(skill_md)
            assert len(result) == 1, filename
            assert result[0] == skill_dir, filename

    def test_find_no_skills_from_non_skill_file(self, tmp_path: Path):
        """Test that non-SKILL.md files return empty list."""
        other_file = tmp_path / "README.md"
        other_file.write_text("# README")

        result = find_skills_in_directory(other_file)
        assert len(result) == 0

    def test_find_skills_in_directory_recursive(self, tmp_path: Path):
        """Test finding multiple skills recursively."""
        # Create nested skill structure
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill1 = skills_dir / "skill-one"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("---\nname: skill-one\n---\n")

        skill2 = skills_dir / "skill-two"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("---\nname: skill-two\n---\n")

        result = find_skills_in_directory(skills_dir)
        assert len(result) == 2
        assert skill1 in result
        assert skill2 in result

    def test_find_skills_mixed_case(self, tmp_path: Path):
        """Test finding skills with both SKILL.md and skill.md."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill1 = skills_dir / "skill-upper"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("---\nname: skill-upper\n---\n")

        skill2 = skills_dir / "skill-lower"
        skill2.mkdir()
        (skill2 / "skill.md").write_text("---\nname: skill-lower\n---\n")

        result = find_skills_in_directory(skills_dir)
        assert len(result) == 2

    def test_find_no_skills_in_empty_directory(self, tmp_path: Path):
        """Test that empty directory returns empty list."""
        result = find_skills_in_directory(tmp_path)
        assert len(result) == 0


class TestGetSkillNameFromPath:
    """Tests for get_skill_name_from_path function."""

    def test_get_name_from_supported_path_shapes(self, tmp_path: Path):
        """Directories, manifest files, and nested dirs all use the skill directory name."""
        skill_dir = tmp_path / "my-awesome-skill"
        skill_dir.mkdir()

        file_skill_dir = tmp_path / "my-skill"
        file_skill_dir.mkdir()
        skill_md = file_skill_dir / "SKILL.md"
        skill_md.write_text("content")

        nested_dir = tmp_path / "team-skills" / "my-team" / "deep-skill"
        nested_dir.mkdir(parents=True)

        cases = [
            (skill_dir, "my-awesome-skill"),
            (skill_md, "my-skill"),
            (nested_dir, "deep-skill"),
        ]
        for path, expected in cases:
            assert get_skill_name_from_path(path) == expected, path


class TestSshToHttps:
    """Tests for _ssh_to_https credential stripping and URL conversion."""

    def test_url_conversion(self):
        cases = [
            # HTTPS with CI token credentials
            (
                "https://github-actions:example-token@github.com/example/project.git",
                "https://github.com/example/project",
            ),
            # HTTPS with generic user:password credentials
            (
                "https://user:password@github.com/group/repo.git",
                "https://github.com/group/repo",
            ),
            # HTTPS with token-only credential (oauth2)
            (
                "https://oauth2:some-token-value@github.com/org/project",
                "https://github.com/org/project",
            ),
            # Plain HTTPS without credentials (unchanged)
            (
                "https://github.com/example/project.git",
                "https://github.com/example/project",
            ),
            (
                "https://github.com/example/project",
                "https://github.com/example/project",
            ),
            # SSH URL (existing behaviour, no credentials to strip)
            (
                "ssh://git@github.com/example/project.git",
                "https://github.com/example/project",
            ),
            # git@ shorthand
            (
                "git@github.com:example/project.git",
                "https://github.com/example/project",
            ),
            # Unsupported scheme returns None
            ("ftp://host/path", None),
        ]
        for remote_url, expected in cases:
            assert _ssh_to_https(remote_url) == expected, remote_url


class TestMakeTimestampedBasename:
    """Tests for ``make_timestamped_basename`` filename construction."""


_CI_ENV_KEYS = ("GITHUB_REF_NAME", "GITHUB_SHA", "GITHUB_REPOSITORY")


def _clean_ci_env(**overrides: str) -> dict[str, str]:
    """Build an os.environ patch that clears CI vars except the given overrides."""
    import os

    env = {k: os.environ[k] for k in os.environ if k not in _CI_ENV_KEYS}
    env.update(overrides)
    return env


def _init_git_repo(path: Path, remote_url: str) -> str:
    """Create a repository with one commit and return its HEAD revision."""
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    (path / "README.md").write_text("test repository\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "Initial commit"], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote_url], check=True)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


class TestResolveGitRemoteUrl:
    """Tests for the Git remote URL resolver used in HTML report links."""

    def test_returns_none_for_non_git_path(self, tmp_path: Path):
        """A path outside any git repo (and no source_url) returns ``None``."""
        assert resolve_git_remote_url(tmp_path) is None

    @patch("skillevaluator.utils.helpers.subprocess.check_output")
    def test_builds_https_tree_url_from_ssh_remote(self, mock_check_output, tmp_path: Path):
        """GitHub SSH origin + local branch + relative path => tree URL."""
        repo_root = tmp_path / "repo"
        sub_dir = repo_root / "skills"
        sub_dir.mkdir(parents=True)

        mock_check_output.side_effect = [
            f"{repo_root}\n",
            "git@github.com:example/project.git\n",
            "main\n",
        ]
        with patch.dict("os.environ", _clean_ci_env(), clear=True):
            url = resolve_git_remote_url(sub_dir)

        assert url == "https://github.com/example/project/tree/main/skills"

    @patch("skillevaluator.utils.helpers.subprocess.check_output")
    def test_ci_branch_overrides_detached_head(self, mock_check_output, tmp_path: Path):
        """GITHUB_REF_NAME takes precedence when the checkout is detached."""
        repo_root = tmp_path / "repo"
        sub_dir = repo_root / "src"
        sub_dir.mkdir(parents=True)
        mock_check_output.side_effect = [
            f"{repo_root}\n",
            "git@github.com:g/r.git\n",
        ]
        env = _clean_ci_env(
            GITHUB_REF_NAME="feature/x",
            GITHUB_SHA="deadbeef",
            GITHUB_REPOSITORY="g/r",
        )
        with patch.dict("os.environ", env, clear=True):
            url = resolve_git_remote_url(sub_dir)
        # Branch wins over the SHA because the scan is same-repo.
        assert url == "https://github.com/g/r/tree/feature/x/src"

    def test_pull_request_merge_ref_uses_local_head_without_valid_github_sha(self, tmp_path: Path):
        """A missing or malformed workflow SHA cannot replace the checked-out revision."""
        repo_root = tmp_path / "repo"
        head = _init_git_repo(repo_root, "git@github.com:g/r.git")
        sub_dir = repo_root / "src"
        sub_dir.mkdir()

        for github_sha in (None, "not-a-commit"):
            env = _clean_ci_env(GITHUB_REF_NAME="58/merge", GITHUB_REPOSITORY="g/r")
            if github_sha is not None:
                env["GITHUB_SHA"] = github_sha
            with patch.dict("os.environ", env, clear=True):
                url = resolve_git_remote_url(sub_dir)

            assert url == f"https://github.com/g/r/tree/{head}/src"

    def test_pull_request_merge_ref_uses_secondary_checkout_head(self, tmp_path: Path):
        """Source links use the target checkout even when workflow metadata describes another repository."""
        repo_root = tmp_path / "secondary"
        head = _init_git_repo(repo_root, "https://github.com/example/secondary.git")
        skill_dir = repo_root / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        workflow_sha = "a" * 40
        assert head != workflow_sha
        env = _clean_ci_env(
            GITHUB_REF_NAME="58/merge",
            GITHUB_SHA=workflow_sha,
            GITHUB_REPOSITORY="NVIDIA/SkillEvaluator",
        )

        with patch.dict("os.environ", env, clear=True):
            url = resolve_git_remote_url(skill_dir)

        assert url == f"https://github.com/example/secondary/tree/{head}/skills/demo"
