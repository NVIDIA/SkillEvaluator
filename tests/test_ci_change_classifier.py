# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.classify_ci_changes import (
    changed_paths,
    is_docs_only,
    is_tier3_platform_related,
    main,
    parse_name_status_z,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "ci-test@example.com")
    _git(tmp_path, "config", "user.name", "CI Test")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.mdx").write_text("docs source\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("root source\n", encoding="utf-8")
    return tmp_path, _commit(tmp_path, "initial")


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        ([b"docs/index.mdx"], True),
        ([b"fern/fern.config.json"], False),
        ([b"fern/package.json"], False),
        ([b"fern/package-lock.json"], False),
        ([b"fern/new-config.yml"], False),
        ([b"docs/index.mdx", b"fern/package-lock.json"], False),
        ([b"fern/docs.yml", b"docs/assets/logo.svg"], True),
        ([b"docs"], False),
        ([b"fern"], False),
        ([b"docs-old/index.mdx"], False),
        ([b"fernicious/docs.yml"], False),
        ([b"README.md"], True),
        ([b"CHANGELOG.md"], True),
        ([b"CHANGELOG.md", b"docs/index.mdx", b"fern/docs.yml"], True),
        ([b"CHANGELOG.md", b"src/skillevaluator/cli.py"], False),
        ([b"changelog.md"], False),
        ([b"CONTRIBUTING.md"], False),
        ([b"skills/skillevaluator/README.md"], False),
        ([b"tests/CHANGELOG.md"], False),
        ([b"skills/skillevaluator/SKILL.md"], False),
        ([b"src/skillevaluator/tier3/reference_skills/demo/SKILL.md"], False),
        ([b"tests/golden/benchmark_tier1.md"], False),
        ([b"docs/index.mdx", b"src/skillevaluator/cli.py"], False),
        ([], False),
    ],
)
def test_is_docs_only(paths: list[bytes], expected: bool) -> None:
    assert is_docs_only(paths) is expected


@pytest.mark.parametrize(
    "paths",
    [
        # The source that holds the macOS and Windows sandbox code.
        [b"src/skillevaluator/tier3/harbor/local_sandbox.py"],
        [b"src/skillevaluator/tier3/reference_skills/demo/SKILL.md"],
        [b"src/skillevaluator/tier3_environments.py"],
        [b"src/skillevaluator/tier3_new_module.py"],
        [b"src/skillevaluator/utils/secure_fs.py"],
        # A directory replaced by a file or a link shows up under its bare name.
        [b"src/skillevaluator/tier3"],
        [b"src/skillevaluator/utils"],
        # What the two jobs run, and what those tests load.
        [b"tests/test_harbor_local_mode.py"],
        [b"tests/validators/test_rubric_eval.py"],
        [b"tests/embedding/test_catalog.py"],
        [b"tests/conftest.py"],
        [b"tests/embedding/conftest.py"],
        [b"tests/fixtures/local_mode_contract.json"],
        [b"tests/fixtures"],
        [b"tests/changelog_entries.py"],
        # Anything that is not plain source, tests or documentation.
        [b"pyproject.toml"],
        [b"uv.lock"],
        [b".github/workflows/ci.yml"],
        [b"scripts/classify_ci_changes.py"],
        [b"ci/gitlab/all-tiers.yml"],
        [b"skills/skillevaluator/SKILL.md"],
        [b"src/other_package/module.py"],
        [b"src/skillevaluator"],
        [b"tests"],
        # One related path is enough.
        [b"src/skillevaluator/reporting/html.py", b"src/skillevaluator/tier3/runner.py"],
        [b"CHANGELOG.md", b"docs/index.mdx", b"uv.lock"],
        # Nothing to go on.
        [],
    ],
)
def test_tier3_platform_jobs_run_for_related_and_unknown_paths(paths: list[bytes]) -> None:
    assert is_tier3_platform_related(paths) is True


@pytest.mark.parametrize(
    "paths",
    [
        [b"src/skillevaluator/reporting/html.py"],
        [b"src/skillevaluator/cli.py"],
        [b"src/skillevaluator/validators/security.py"],
        # ``utils`` further down the tree is not the shared utilities package.
        [b"src/skillevaluator/deduplication/utils/chunker.py"],
        [b"src/skillevaluator/tier2/collector.py"],
        [b"src/skillevaluator/reporting/templates/report.html"],
        # Tests the two jobs do not run, and their data.
        [b"tests/test_cli.py"],
        [b"tests/tier3/test_budget_reservation.py"],
        [b"tests/golden/benchmark_tier1.md"],
        # Documentation never counts.
        [b"docs/index.mdx"],
        [b"CHANGELOG.md", b"README.md", b"fern/docs.yml"],
        [b"CHANGELOG.md", b"src/skillevaluator/reporting/html.py", b"tests/reporting/test_html.py"],
    ],
)
def test_tier3_platform_jobs_are_skipped_for_unrelated_source_and_tests(paths: list[bytes]) -> None:
    assert is_tier3_platform_related(paths) is False


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"M\0docs/index.mdx\0", [b"docs/index.mdx"]),
        (b"D\0docs/old.mdx\0", [b"docs/old.mdx"]),
        (
            b"R100\0docs/old.mdx\0src/new.py\0",
            [b"docs/old.mdx", b"src/new.py"],
        ),
        (
            b"C075\0docs/source.mdx\0fern/copied.yml\0",
            [b"docs/source.mdx", b"fern/copied.yml"],
        ),
        (
            b"M\0docs/file with spaces.mdx\0M\0docs/line\nbreak.mdx\0",
            [b"docs/file with spaces.mdx", b"docs/line\nbreak.mdx"],
        ),
        (b"M\0docs/non-utf8-\xff.mdx\0", [b"docs/non-utf8-\xff.mdx"]),
        (b"", []),
    ],
)
def test_parse_name_status_z_returns_every_changed_path(payload: bytes, expected: list[bytes]) -> None:
    assert parse_name_status_z(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        b"R100\0docs/old.mdx\0",
        b"M\0",
        b"Z\0docs/index.mdx\0",
        b"\0docs/index.mdx\0",
        b"M\0docs/index.mdx",
    ],
)
def test_parse_name_status_z_rejects_malformed_records(payload: bytes) -> None:
    with pytest.raises(ValueError, match="Git status record"):
        parse_name_status_z(payload)


def _classify(
    repo: Path,
    base: str,
    head: str,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    return main(["--repo", str(repo), "--base", base, "--head", head])


def test_main_classifies_a_real_docs_only_diff(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, base = git_repo
    (repo / "docs" / "index.mdx").write_text("updated\n", encoding="utf-8")
    head = _commit(repo, "docs")
    output = tmp_path / "github-output"

    assert _classify(repo, base, head, output, monkeypatch) == 0

    assert capsys.readouterr().out == "docs_only=true\ntier3_platform=false\n"
    assert output.read_text(encoding="utf-8") == "docs_only=true\ntier3_platform=false\n"


def test_main_classifies_a_real_mixed_diff_as_full_ci(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, base = git_repo
    (repo / "docs" / "index.mdx").write_text("updated\n", encoding="utf-8")
    (repo / "source.py").write_text("print('changed')\n", encoding="utf-8")
    head = _commit(repo, "mixed")
    output = tmp_path / "github-output"

    assert _classify(repo, base, head, output, monkeypatch) == 0

    assert capsys.readouterr().out == "docs_only=false\ntier3_platform=true\n"
    assert output.read_text(encoding="utf-8") == "docs_only=false\ntier3_platform=true\n"


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("src/skillevaluator/reporting/html.py", "docs_only=false\ntier3_platform=false\n"),
        ("src/skillevaluator/tier3/runner.py", "docs_only=false\ntier3_platform=true\n"),
    ],
)
def test_main_classifies_a_real_source_diff_for_the_tier3_platform_jobs(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changed: str,
    expected: str,
) -> None:
    repo, base = git_repo
    (repo / changed).parent.mkdir(parents=True)
    (repo / changed).write_text("print('changed')\n", encoding="utf-8")
    head = _commit(repo, "source")
    output = tmp_path / "github-output"

    assert _classify(repo, base, head, output, monkeypatch) == 0

    assert capsys.readouterr().out == expected
    assert output.read_text(encoding="utf-8") == expected


def test_main_treats_a_deleted_docs_file_as_docs_only(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = git_repo
    (repo / "docs" / "index.mdx").unlink()
    head = _commit(repo, "delete docs")

    assert _classify(repo, base, head, tmp_path / "github-output", monkeypatch) == 0
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == "docs_only=true\ntier3_platform=false\n"


def test_main_checks_both_sides_of_a_rename(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = git_repo
    (repo / "src").mkdir()
    _git(repo, "mv", "docs/index.mdx", "src/index.py")
    head = _commit(repo, "rename out of docs")

    assert _classify(repo, base, head, tmp_path / "github-output", monkeypatch) == 0
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == "docs_only=false\ntier3_platform=true\n"


def test_main_treats_a_rename_within_docs_as_docs_only(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = git_repo
    _git(repo, "mv", "docs/index.mdx", "docs/renamed.mdx")
    head = _commit(repo, "rename within docs")

    assert _classify(repo, base, head, tmp_path / "github-output", monkeypatch) == 0
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == "docs_only=true\ntier3_platform=false\n"


def test_main_checks_the_source_of_a_rename_into_docs(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = git_repo
    _git(repo, "mv", "pyproject.toml", "docs/pyproject.mdx")
    head = _commit(repo, "rename into docs")

    assert _classify(repo, base, head, tmp_path / "github-output", monkeypatch) == 0
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == "docs_only=false\ntier3_platform=true\n"


def test_changed_paths_detects_an_unmodified_copy_source_outside_docs(
    git_repo: tuple[Path, str],
) -> None:
    repo, base = git_repo
    (repo / "docs" / "copied-root.mdx").write_bytes((repo / "pyproject.toml").read_bytes())
    head = _commit(repo, "copy root file into docs")

    paths = changed_paths(repo, base, head)

    assert b"pyproject.toml" in paths
    assert b"docs/copied-root.mdx" in paths
    assert is_docs_only(paths) is False


def test_main_uses_the_merge_base_when_the_base_branch_advances(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, original_base = git_repo
    _git(repo, "checkout", "--quiet", "-b", "feature")
    (repo / "docs" / "index.mdx").write_text("feature docs\n", encoding="utf-8")
    feature_head = _commit(repo, "feature docs")

    _git(repo, "checkout", "--quiet", "--detach", original_base)
    (repo / "source.py").write_text("base advanced\n", encoding="utf-8")
    advanced_base = _commit(repo, "advance base")

    assert _classify(repo, advanced_base, feature_head, tmp_path / "github-output", monkeypatch) == 0
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == "docs_only=true\ntier3_platform=false\n"


@pytest.mark.parametrize("revision", ["0" * 40, "not-a-sha"])
def test_main_fails_closed_for_invalid_revisions(
    git_repo: tuple[Path, str],
    revision: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, head = git_repo
    output = tmp_path / "github-output"

    assert _classify(repo, revision, head, output, monkeypatch) == 0

    captured = capsys.readouterr()
    assert captured.out == "docs_only=false\ntier3_platform=true\n"
    assert "falling back to full CI" in captured.err
    assert output.read_text(encoding="utf-8") == "docs_only=false\ntier3_platform=true\n"


def test_main_fails_closed_for_an_empty_diff(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, head = git_repo
    output = tmp_path / "github-output"
    output.write_text("existing=value\n", encoding="utf-8")

    assert _classify(repo, head, head, output, monkeypatch) == 0

    captured = capsys.readouterr()
    assert captured.out == "docs_only=false\ntier3_platform=true\n"
    assert "no changed paths" in captured.err
    assert output.read_text(encoding="utf-8") == "existing=value\ndocs_only=false\ntier3_platform=true\n"


def test_main_fails_closed_outside_a_git_repository(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, head = git_repo
    output = tmp_path / "github-output"

    assert _classify(tmp_path / "missing-repo", head, head, output, monkeypatch) == 0

    captured = capsys.readouterr()
    assert captured.out == "docs_only=false\ntier3_platform=true\n"
    assert "falling back to full CI" in captured.err
    assert output.read_text(encoding="utf-8") == "docs_only=false\ntier3_platform=true\n"


def test_cli_classifies_a_real_fern_only_diff(git_repo: tuple[Path, str], tmp_path: Path) -> None:
    repo, base = git_repo
    (repo / "fern").mkdir()
    (repo / "fern" / "docs.yml").write_text("navigation: []\n", encoding="utf-8")
    head = _commit(repo, "fern docs")
    output = tmp_path / "github-output"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "classify_ci_changes.py"),
            "--repo",
            str(repo),
            "--base",
            base,
            "--head",
            head,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"GITHUB_OUTPUT": str(output)},
    )

    assert result.stdout == "docs_only=true\ntier3_platform=false\n"
    assert result.stderr == ""
    assert output.read_text(encoding="utf-8") == "docs_only=true\ntier3_platform=false\n"


def test_output_write_failure_is_not_silently_downgraded(
    git_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = git_repo
    (repo / "docs" / "index.mdx").write_text("updated\n", encoding="utf-8")
    head = _commit(repo, "docs")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path))

    with pytest.raises(OSError):
        main(["--repo", str(repo), "--base", base, "--head", head])
