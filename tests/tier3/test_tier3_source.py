# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.tier3.evals_spec import validate_tier3_source


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "source-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: source-skill\ndescription: test\n---\n")
    return skill


def test_source_preflight_reports_missing_source(tmp_path: Path) -> None:
    source, checks = validate_tier3_source(_skill(tmp_path))

    assert source == "missing"
    assert any(check.status == "error" for check in checks)


def test_source_preflight_accepts_yaml_dataset(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.yaml").write_text(
        "skill_name: source-skill\n"
        "evals:\n"
        "  - id: source-001\n"
        "    prompt: do the thing\n"
        "    expected_output: the thing is done\n"
    )

    source, checks = validate_tier3_source(skill)

    assert source == "evals_json"
    assert not [check for check in checks if check.status in {"missing", "error"}]


def test_source_preflight_rejects_empty_dataset(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(json.dumps({"skill_name": "source-skill", "evals": []}))

    source, checks = validate_tier3_source(skill)

    assert source == "evals_json"
    assert any(check.status == "error" and "empty" in check.message.lower() for check in checks)


def test_configured_native_source_fails_when_directory_is_missing(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text("schema_version: 1\nharbor:\n  task_source: native_harbor\n")

    source, checks = validate_tier3_source(skill)

    assert source == "native_harbor"
    assert any(check.status == "error" and check.path == "evals/harbor/" for check in checks)
