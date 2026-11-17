# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.classify_ci_changes import DOC_FILES, DOC_PREFIXES
from scripts.select_docs_tests import main, select_docs_tests

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    "source",
    [
        'TEXT = (ROOT / "CHANGELOG.md").read_text()\n',
        'TEXT = (ROOT / "README.md").read_text()\n',
        'PAGE = ROOT / "docs" / "reports.mdx"\n',
        "PAGE = ROOT / 'docs' / 'reports.mdx'\n",
        'PAGE = ROOT / "docs/reports.mdx"\n',
        'NAV = ROOT / "fern" / "docs.yml"\n',
        'NAV = ROOT / "fern/docs.yml"\n',
    ],
)
def test_selects_a_test_file_that_names_a_documentation_file(tmp_path: Path, source: str) -> None:
    _write(tmp_path, "tests/test_reads_docs.py", source)
    _write(tmp_path, "tests/test_unrelated.py", 'VALUE = json.loads("{}")\n')

    assert select_docs_tests(tmp_path) == ["tests/test_reads_docs.py"]


@pytest.mark.parametrize(
    "source",
    [
        'NAME = "mydocs/page.mdx"\n',
        'NAME = "docstring"\n',
        "import json\n",
        'NAME = "CHANGELOG"\n',
    ],
)
def test_leaves_out_a_test_file_that_names_no_documentation_file(tmp_path: Path, source: str) -> None:
    _write(tmp_path, "tests/test_unrelated.py", source)

    assert select_docs_tests(tmp_path) == []


def test_every_path_the_classifier_calls_documentation_selects_its_tests(tmp_path: Path) -> None:
    """A path added to the classifier is covered here without a second edit."""
    expected = []
    for index, path in enumerate(sorted(DOC_FILES)):
        expected.append(f"tests/test_file_{index}.py")
        _write(tmp_path, expected[-1], f'TEXT = (ROOT / "{path.decode()}").read_text()\n')
    for index, prefix in enumerate(DOC_PREFIXES):
        expected.append(f"tests/test_prefix_{index}.py")
        _write(tmp_path, expected[-1], f'PAGE = ROOT / "{prefix.decode()}page.mdx"\n')
    _write(tmp_path, "tests/test_unrelated.py", "VALUE = 1\n")

    assert select_docs_tests(tmp_path) == sorted(expected)


def test_selects_a_test_file_that_reads_documentation_through_a_helper(tmp_path: Path) -> None:
    _write(tmp_path, "tests/changelog_entries.py", 'CHANGELOG = ROOT / "CHANGELOG.md"\n')
    _write(tmp_path, "tests/middle.py", "import changelog_entries\n")
    _write(tmp_path, "tests/test_plain_import.py", "from changelog_entries import CHANGELOG\n")
    _write(tmp_path, "tests/reporting/test_package_import.py", "from tests.changelog_entries import CHANGELOG\n")
    _write(tmp_path, "tests/reporting/test_relative_import.py", "from .. import changelog_entries\n")
    _write(tmp_path, "tests/test_through_two_helpers.py", "import middle\n")
    _write(tmp_path, "tests/test_unrelated.py", "import json\n")

    assert select_docs_tests(tmp_path) == [
        "tests/reporting/test_package_import.py",
        "tests/reporting/test_relative_import.py",
        "tests/test_plain_import.py",
        "tests/test_through_two_helpers.py",
    ]


def test_lists_only_collected_test_files(tmp_path: Path) -> None:
    reads_docs = 'TEXT = (ROOT / "README.md").read_text()\n'
    _write(tmp_path, "tests/test_prefixed.py", reads_docs)
    _write(tmp_path, "tests/tier3/suffixed_test.py", reads_docs)
    _write(tmp_path, "tests/helper.py", reads_docs)
    _write(tmp_path, "tests/conftest.py", reads_docs)
    _write(tmp_path, "src/test_outside_tests.py", reads_docs)

    assert select_docs_tests(tmp_path) == ["tests/test_prefixed.py", "tests/tier3/suffixed_test.py"]


@pytest.mark.parametrize("helper_module", ["test_doc_reader", "doc_reader_test"])
def test_follows_a_documentation_reader_in_another_collected_test_module(
    tmp_path: Path, helper_module: str
) -> None:
    _write(
        tmp_path,
        f"tests/{helper_module}.py",
        'def read_page():\n    return (ROOT / "README.md").read_text()\n\ndef test_smoke():\n    assert True\n',
    )
    _write(
        tmp_path,
        "tests/test_consumer.py",
        f'from {helper_module} import read_page\n\ndef test_readme_contract():\n    assert "required" in read_page()\n',
    )

    assert select_docs_tests(tmp_path) == sorted([f"tests/{helper_module}.py", "tests/test_consumer.py"])


def test_follows_a_documentation_reader_in_a_package_initializer(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/doc_helpers/__init__.py",
        'def read_page():\n    return (ROOT / "README.md").read_text()\n',
    )
    _write(tmp_path, "tests/middle.py", "from doc_helpers import read_page\n")
    _write(
        tmp_path,
        "tests/test_consumer.py",
        'from middle import read_page\n\ndef test_readme_contract():\n    assert "required" in read_page()\n',
    )

    assert select_docs_tests(tmp_path) == ["tests/test_consumer.py"]


def test_selects_a_test_file_it_cannot_parse(tmp_path: Path) -> None:
    """Its imports are unknown, so it cannot be ruled out; pytest then reports the syntax error."""
    _write(tmp_path, "tests/test_broken.py", "def broken(:\n")

    assert select_docs_tests(tmp_path) == ["tests/test_broken.py"]


def test_main_prints_one_selected_file_per_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, "tests/test_b.py", 'TEXT = "CHANGELOG.md"\n')
    _write(tmp_path, "tests/test_a.py", 'TEXT = "README.md"\n')
    _write(tmp_path, "tests/test_unrelated.py", "VALUE = 1\n")

    assert main(["--root", str(tmp_path)]) == 0

    assert capsys.readouterr().out == "tests/test_a.py\ntests/test_b.py\n"


def test_main_refuses_an_empty_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """pytest with no paths runs the whole suite, so an empty answer must not look like success."""
    _write(tmp_path, "tests/test_unrelated.py", "VALUE = 1\n")

    assert main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no documentation tests found" in captured.err


def test_the_repository_selection_covers_tests_known_to_read_documentation() -> None:
    selected = set(select_docs_tests(REPO_ROOT))
    every_test_file = {
        path.relative_to(REPO_ROOT).as_posix()
        for pattern in ("test_*.py", "*_test.py")
        for path in (REPO_ROOT / "tests").rglob(pattern)
    }

    assert {
        # Documentation pages.
        "tests/test_metrics_reference_docs.py",
        "tests/test_benchmark_rollout_docs.py",
        # The readme and the changelog.
        "tests/test_oss_packaging.py",
        # The changelog, in a subdirectory.
        "tests/reporting/test_tier3_incomplete_arms.py",
        # The Fern navigation file.
        "tests/test_ci_workflows.py",
    } <= selected
    assert selected <= every_test_file
    assert len(selected) < len(every_test_file) / 2, "the docs lane should stay a small part of the suite"
