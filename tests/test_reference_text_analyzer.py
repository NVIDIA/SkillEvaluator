# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the packaged text-analyzer reference script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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
