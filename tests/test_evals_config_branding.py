# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config


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


@pytest.mark.parametrize(
    ("legacy_mode", "normalized_mode"),
    [
        ("aces_default", "default"),
        ("aces_plus_custom", "default_plus_custom"),
    ],
)
def test_legacy_grading_mode_remains_readable(tmp_path: Path, legacy_mode: str, normalized_mode: str) -> None:
    """Retired grading-mode spellings stay accepted and normalize to current names."""
    _write_config(tmp_path, legacy_mode)

    config, _ = load_evals_config(tmp_path)

    assert config["grading"]["mode"] == normalized_mode


def test_unknown_grading_mode_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "creative_scoring")

    with pytest.raises(EvalsConfigError, match=r"grading\.mode"):
        load_evals_config(tmp_path)


def test_agent_model_is_trimmed_during_config_loading(tmp_path: Path) -> None:
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "config.yml").write_text(
        'schema_version: 1\nharbor:\n  agents:\n    opencode:\n      model: "  vendor/custom-model  "\n',
        encoding="utf-8",
    )

    config, _ = load_evals_config(tmp_path)

    assert config["harbor"]["agents"]["opencode"]["model"] == "vendor/custom-model"


def test_claude_alias_config_key_is_persisted_canonically(tmp_path: Path) -> None:
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  agents:\n    claude:\n      model: anthropic/claude-sonnet\n",
        encoding="utf-8",
    )

    config, _ = load_evals_config(tmp_path)

    assert config["harbor"]["agents"] == {"claude-code": {"model": "anthropic/claude-sonnet"}}


def test_claude_alias_and_canonical_config_keys_are_rejected(tmp_path: Path) -> None:
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  agents:\n    claude:\n      model: first\n    claude-code:\n      model: second\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match=r"claude.*claude-code.*same agent"):
        load_evals_config(tmp_path)
