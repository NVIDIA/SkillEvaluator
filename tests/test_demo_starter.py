# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import demo_starter
from scripts.demo_starter import FIXTURES, check_quality

ROOT = Path(__file__).resolve().parents[1]
# What the release wheel pinned in the starter guide reports for the starter and its
# shortened copy. The checkout's evaluator can score them differently until a release
# includes its scoring changes.
PINNED_RELEASE_SCORES = {"quality": 84.5, "regressed": 71.0}


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
    expected = demo_starter.load_expected()
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


def test_installed_release_is_checked_against_the_pinned_release_scores(tmp_path, monkeypatch):
    """The documented run checks the pinned release's scores, not the checkout's."""
    # Demo files outside the evaluator's source tree, as when the release wheel runs them.
    fixtures = tmp_path / "checkout" / "examples" / "starter"
    shutil.copytree(FIXTURES, fixtures)
    monkeypatch.setattr(demo_starter, "FIXTURES", fixtures)
    monkeypatch.setattr(demo_starter, "version", lambda _: "0.9.0")
    expected = json.loads((fixtures / "expected.json").read_text(encoding="utf-8"))

    def release_cli(command, log, expected_exit):
        if "quality-check" not in command:
            log.write_text(json.dumps(expected["analysis"]), encoding="utf-8")
            return
        reports = Path(command[command.index("-o") + 1])
        reports.mkdir()
        report = {
            "overall_status": expected[log.stem]["status"],
            "overall_passed": expected_exit == 0,
            "quality_summary": [
                {"overall_score": PINNED_RELEASE_SCORES[log.stem], "grade": expected[log.stem]["grade"]}
            ],
            "publication_status": expected[log.stem]["publication"],
            "publication": {"eligible": False, "reasons": ["Live evaluation did not run"]},
        }
        (reports / "skillevaluator-quality.json").write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(demo_starter, "run", release_cli)

    receipt = demo_starter.demo(tmp_path / "starter")

    assert receipt["live_agent_evaluation"] == "not run"


@pytest.mark.parametrize(
    ("page", "template"),
    [
        ("docs/starter-demo.mdx", "score {score:g}, grade {grade}"),
        ("examples/starter/README.md", "scores {score:g}/100 ({grade})"),
    ],
)
def test_starter_docs_show_the_pinned_release_scores(page, template):
    text = (ROOT / page).read_text(encoding="utf-8")
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
    for label, score in PINNED_RELEASE_SCORES.items():
        assert template.format(score=score, grade=expected[label]["grade"]) in text


@pytest.fixture
def quality_report():
    return {
        "overall_status": "passed",
        "overall_passed": True,
        "quality_summary": [{"overall_score": 84.5, "grade": "B"}],
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


@pytest.mark.parametrize(
    ("evaluator_version", "quality_score", "regressed_score"),
    [("0.9.0", 84.5, 71.0), ("1.0.0", 88.8, 72.8), ("1.0.1", 88.8, 72.8)],
)
def test_installed_starter_uses_reviewed_version_scores(monkeypatch, evaluator_version, quality_score, regressed_score):
    # Installed releases have no checkout source origin. Keep the old pinned
    # guide's contract while qualifying the new wheel against its real outputs.
    monkeypatch.setattr(demo_starter, "find_spec", lambda _: None)
    monkeypatch.setattr(demo_starter, "version", lambda _: evaluator_version)
    expected = demo_starter.load_expected()
    assert expected["quality"]["score"] == quality_score
    assert expected["regressed"]["score"] == regressed_score
    assert "evaluator_versions" not in expected
    assert "checkout" not in expected
