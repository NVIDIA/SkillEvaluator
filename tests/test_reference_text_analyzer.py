# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the packaged text-analyzer reference script."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ANALYZE_TEXT = (
    Path(__file__).resolve().parents[1]
    / "src/skillevaluator/tier3/reference_skills/text-analyzer/scripts/analyze_text.py"
)


def _load_analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reference_script_analyze_text", ANALYZE_TEXT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top_words_skip_punctuation_only_tokens() -> None:
    stats = _load_analyzer().analyze("Wait... what? ... ok ... - fine - yes! \u2014 & \u2013")

    assert [entry["word"] for entry in stats["top_words"]] == ["wait", "what", "ok", "fine", "yes"]
    assert stats["word_count"] == 12


def test_top_words_keep_numbers_and_inner_punctuation() -> None:
    stats = _load_analyzer().analyze("v2 v2 don't don't 3.14 well-known")

    assert stats["top_words"] == [
        {"word": "v2", "count": 2},
        {"word": "don't", "count": 2},
        {"word": "3.14", "count": 1},
        {"word": "well-known", "count": 1},
    ]


def test_starter_eval_text_statistics_are_unchanged() -> None:
    stats = _load_analyzer().analyze(
        "The quick brown fox jumps over the lazy dog. The dog barked at the fox. The fox ran away quickly."
    )

    assert stats["word_count"] == 20
    assert stats["character_count"] == 97
    assert stats["sentence_count"] == 3
    assert stats["top_words"][:3] == [
        {"word": "the", "count": 5},
        {"word": "fox", "count": 3},
        {"word": "dog", "count": 2},
    ]


@pytest.mark.parametrize(
    ("text", "sentence_count"),
    [
        ("Open input.txt and report the totals.", 1),
        ("Pi is 3.14. Done!", 2),
        ("Ship v1.2.3 today.", 1),
        ("See https://example.com/index.html for details.", 1),
        ("Write to user@example.com.", 1),
        ('He said "go." Then left.', 2),
        ("She said \u201cGo now.\u201d Then left.", 2),
        ("She said \u2018Go now.\u2019 Then left.", 2),
        ("He wrote \u00abfin.\u00bb Then left.", 2),
        ("(see above.) Next", 2),
        ("It was founded in 1993.[1] The company grew.", 2),
        ("Done.[1][2] Next.", 2),
        ("This is **done.** Next one.", 2),
        ("_She left._ Then he came.", 2),
        ("Wait... what?", 2),
        ("Really?! Yes.", 2),
        ("hello world", 1),
        ("", 0),
        ("  \n  ", 0),
        ("First.\nSecond.", 2),
        (
            "The quick brown fox jumps over the lazy dog. The dog barked at the fox. The fox ran away quickly.",
            3,
        ),
    ],
)
def test_sentence_count_ends_only_before_whitespace_or_end(text: str, sentence_count: int) -> None:
    assert _load_analyzer().analyze(text)["sentence_count"] == sentence_count


def test_sentence_count_is_linear_on_long_terminator_runs() -> None:
    analyzer = _load_analyzer()
    started = time.perf_counter()
    result = analyzer.analyze("." * 50_000 + "x")
    elapsed = time.perf_counter() - started

    assert result["sentence_count"] == 1
    assert elapsed < 2.0


def _run_analyzer(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ANALYZE_TEXT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_file_argument_reports_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    result = _run_analyzer("--file", str(path))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: file not found: {path}\n"


def test_file_argument_rejects_a_directory(tmp_path: Path) -> None:
    result = _run_analyzer("--file", str(tmp_path))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: not a file: {tmp_path}\n"
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="permission bits do not apply on Windows")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="permission bits do not apply when running as root"
)
def test_file_argument_rejects_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "secret.txt"
    path.write_text("hello", encoding="utf-8")
    path.chmod(0)
    try:
        result = _run_analyzer("--file", str(path))
    finally:
        path.chmod(0o644)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: cannot read file: {path} (Permission denied)\n"
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="permission bits do not apply on Windows")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="permission bits do not apply when running as root"
)
def test_file_argument_rejects_a_file_under_an_unreadable_directory(tmp_path: Path) -> None:
    directory = tmp_path / "locked"
    directory.mkdir()
    path = directory / "secret.txt"
    path.write_text("hello", encoding="utf-8")
    directory.chmod(0o000)
    try:
        result = _run_analyzer("--file", str(path))
    finally:
        directory.chmod(0o755)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: cannot read file: {path} (Permission denied)\n"
    assert "Traceback" not in result.stderr
