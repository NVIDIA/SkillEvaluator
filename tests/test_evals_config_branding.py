# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from skillevaluator.tier3.evals_config import load_evals_config


def _write_config(skill_path: Path, mode: str) -> None:
    evals_dir = skill_path / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "config.yml").write_text(
        f"schema_version: 1\ngrading:\n  mode: {mode}\n",
        encoding="utf-8",
    )


def test_public_grading_mode_normalizes_for_the_existing_engine(tmp_path: Path) -> None:
    _write_config(tmp_path, "default_plus_custom")

    config, _ = load_evals_config(tmp_path)

    assert config["grading"]["mode"] == "default_plus_custom"


def test_legacy_grading_mode_remains_readable(tmp_path: Path) -> None:
    _write_config(tmp_path, "default_plus_custom")

    config, _ = load_evals_config(tmp_path)

    assert config["grading"]["mode"] == "default_plus_custom"
