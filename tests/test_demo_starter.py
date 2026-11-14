# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.demo_starter import FIXTURES, check_quality

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", ["docs/starter-demo.mdx", "examples/starter/README.md"])
def test_released_commands_do_not_reuse_the_editable_checkout(page):
    text = (ROOT / page).read_text()
    commands = [line for line in text.splitlines() if line.startswith("uv run ") and "--with " in line]
    assert commands
    for command in commands:
        assert "--isolated --no-project" in command
        assert "https://github.com/NVIDIA/SkillEvaluator/releases/download/" in command
        assert ".whl" in command


def test_starter_runs_real_cli_and_preserves_sources(tmp_path):
    output = tmp_path / "starter with spaces"
    source = ROOT / "src/skillevaluator/tier3/reference_skills/text-analyzer/SKILL.md"
    original = source.read_bytes()
    command = [sys.executable, str(ROOT / "scripts/demo_starter.py"), "--output", str(output)]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: starter output" in result.stdout
    assert source.read_bytes() == original
    expected = json.loads((FIXTURES / "expected.json").read_text())
    assert json.loads((output / "analysis.json").read_text()) == expected["analysis"]
    for label in ("quality", "regressed"):
        report = json.loads((output / label / "skillevaluator-quality.json").read_text())
        check_quality(report, expected[label])
        assert (output / f"{label}.txt").is_file()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["live_agent_evaluation"] == "not run"
    assert receipt["skillevaluator_version"]
    dataset = json.loads((output / "text-analyzer/evals/evals.json").read_text())
    assert dataset["skill_name"] == "text-analyzer"
    assert "27 characters" in dataset["evals"][0]["expected_output"]
    assert (output / "text-analyzer/evals/files/input.txt").read_text() == (FIXTURES / "input.txt").read_text()

    before = (output / "receipt.json").read_bytes()
    repeat = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert repeat.returncode == 1
    assert "new --output directory" in repeat.stderr
    assert (output / "receipt.json").read_bytes() == before


@pytest.fixture
def quality_report():
    return {
        "overall_status": "passed",
        "overall_passed": True,
        "quality_summary": [{"overall_score": 86.2, "grade": "B"}],
        "publication_status": "incomplete",
        "publication": {"eligible": False, "reasons": ["Live evaluation did not run"]},
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "overall_status", "failed"),
        (None, "overall_passed", False),
        ("quality_summary", "overall_score", 100),
        ("quality_summary", "grade", "A"),
        (None, "publication_status", "pass"),
        ("publication", "eligible", True),
        ("publication", "reasons", []),
    ],
)
def test_output_checks_reject_changed_or_misleading_results(quality_report, section, key, value):
    expected = json.loads((FIXTURES / "expected.json").read_text())["quality"]
    check_quality(quality_report, expected)
    changed = copy.deepcopy(quality_report)
    target = changed if section is None else changed[section]
    if section == "quality_summary":
        target = target[0]
    target[key] = value
    with pytest.raises(ValueError):
        check_quality(changed, expected)
