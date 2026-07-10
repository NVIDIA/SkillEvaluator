# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 command wrappers preserve useful failure diagnostics."""

from __future__ import annotations

from pathlib import Path

from skillevaluator.tier2 import commands


def test_similarity_check_wraps_unexpected_validator_exceptions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class BrokenSimilarityValidator:
        def __init__(self, **_kwargs):
            pass

        def validate(self, _content_path: Path):
            raise RuntimeError("catalog exploded")

    monkeypatch.setattr(commands, "SimilarityValidator", BrokenSimilarityValidator)

    (result,) = commands.run_similarity_check(tmp_path, catalog=tmp_path / "catalog.json")

    assert result.validator_name == "Similarity Check"
    assert result.validator_description == "Tier 2 check"
    assert result.passed is False
    assert result.errors == ["Similarity Check failed: catalog exploded"]
