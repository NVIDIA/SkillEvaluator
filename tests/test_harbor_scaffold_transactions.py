# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3 import commands
from skillevaluator.tier3.harbor.adapter import _write_dataset_toml
from skillevaluator.tier3.toml_utils import toml_quote


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "sample-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: sample-skill\ndescription: Test fixture.\n---\n# Sample\n",
        encoding="utf-8",
    )
    return skill


def test_harbor_scaffold_verifier_timeout_covers_all_structured_judge_attempts(tmp_path: Path) -> None:
    skill = _skill(tmp_path)

    assert (
        commands.init_harbor_task(
            skill,
            force=False,
            case_id="case-001",
            mode="default",
            language="python",
            with_config=True,
        )
        == 0
    )
    task = tomllib.loads((skill / "evals" / "harbor" / "case-001" / "task.toml").read_text(encoding="utf-8"))

    assert task["verifier"]["timeout_sec"] >= (3 * 2 * 90) + 60


@pytest.mark.parametrize("case_id", ["../outside", r"..\outside", "/tmp/outside", r"C:\outside"])
def test_scaffold_rejects_escaping_case_id_before_mutation(tmp_path: Path, case_id: str) -> None:
    skill = _skill(tmp_path)

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill), "--case-id", case_id])

    assert result.exit_code != 0
    assert not (skill / "evals").exists()


@pytest.mark.parametrize("case_id", ["dataset.toml", "README.md", "metric.py", "results"])
def test_scaffold_rejects_harbor_root_reserved_name_before_mutation(tmp_path: Path, case_id: str) -> None:
    skill = _skill(tmp_path)

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill), "--case-id", case_id])

    assert result.exit_code != 0
    assert not (skill / "evals").exists()


def test_scaffold_rejects_symlinked_evals_root_without_touching_target(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (skill / "evals").symlink_to(outside, target_is_directory=True)

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill)])

    assert result.exit_code != 0
    assert (skill / "evals").is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_custom_grader_rejects_symlinked_evals_root_without_touching_target(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (skill / "evals").symlink_to(outside, target_is_directory=True)

    result = CliRunner().invoke(cli, ["init-custom-grader", str(skill)])

    assert result.exit_code != 0
    assert (skill / "evals").is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outside / "grader.py").exists()


@pytest.mark.parametrize(
    ("invalid_name", "invalid_content"),
    [
        pytest.param("evals.json", "{not-json", id="malformed-eval-dataset"),
        pytest.param("config.yml", "harbor: [not-valid\n", id="malformed-eval-config"),
    ],
)
def test_scaffold_rejects_malformed_existing_authored_state_without_with_config(
    tmp_path: Path,
    invalid_name: str,
    invalid_content: str,
) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    if invalid_name != "evals.json":
        (evals / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "sample-skill",
                    "evals": [{"id": "case-001", "prompt": "Q", "expected_output": "A"}],
                }
            ),
            encoding="utf-8",
        )
    invalid = evals / invalid_name
    invalid.write_text(invalid_content, encoding="utf-8")
    before = {path.name: path.read_bytes() for path in evals.iterdir()}

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill)])

    assert result.exit_code != 0
    assert not (evals / "harbor").exists()
    assert {path.name: path.read_bytes() for path in evals.iterdir()} == before


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"id": "case-001", "expected_output": "A"}, id="missing-prompt"),
        pytest.param({"id": "case-001", "prompt": 7, "expected_output": "A"}, id="numeric-prompt"),
        pytest.param({"id": "case-001", "prompt": "Q"}, id="missing-expected-output"),
        pytest.param({"id": "case-001", "prompt": "Q", "expected_output": []}, id="non-string-output"),
        pytest.param(
            {"id": "case-001", "prompt": "Q", "expected_output": "A", "assertions": "not-a-list"},
            id="non-list-assertions",
        ),
        pytest.param(
            {"id": "case-001", "prompt": "Q", "expected_output": "A", "assertions": [7]},
            id="non-string-assertion",
        ),
        pytest.param(
            {"id": "case-001", "prompt": "Q", "question": 7, "expected_output": "A"},
            id="invalid-effective-question-alias",
        ),
        pytest.param(
            {"id": "case-001", "prompt": "Q", "expected_output": "A", "ground_truth": []},
            id="invalid-effective-ground-truth-alias",
        ),
        pytest.param(
            {
                "id": "case-001",
                "prompt": "Q",
                "expected_output": "A",
                "assertions": ["Expected behavior"],
                "expected_behavior": "not-a-list",
            },
            id="invalid-effective-behavior-alias",
        ),
    ],
)
def test_scaffold_rejects_contract_invalid_existing_dataset_before_publish(
    tmp_path: Path,
    entry: dict[str, object],
) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    dataset = evals / "evals.json"
    dataset.write_text(json.dumps({"skill_name": "sample-skill", "evals": [entry]}), encoding="utf-8")
    before = dataset.read_bytes()

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill)])

    assert result.exit_code != 0
    assert dataset.read_bytes() == before
    assert not (evals / "harbor").exists()


def test_force_late_config_failure_preserves_complete_old_scaffold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill(tmp_path)
    assert (
        commands.init_harbor_task(
            skill, force=False, case_id="case-001", mode="custom_only", language="python", with_config=True
        )
        == 0
    )
    case_dir = skill / "evals" / "harbor" / "case-001"
    sentinel = case_dir / "old-sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    old_dataset = (skill / "evals" / "harbor" / "dataset.toml").read_bytes()
    old_config = (skill / "evals" / "config.yml").read_bytes()

    original_write_text = Path.write_text

    def fail_config(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path.name == "config.yml":
            raise OSError("injected late config failure")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_config)

    with pytest.raises(OSError, match="injected late config failure"):
        commands.init_harbor_task(
            skill, force=True, case_id="case-001", mode="default", language="shell", with_config=True
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (skill / "evals" / "harbor" / "dataset.toml").read_bytes() == old_dataset
    assert (skill / "evals" / "config.yml").read_bytes() == old_config


def test_custom_grader_force_late_config_failure_preserves_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill(tmp_path)
    assert (
        commands.init_custom_grader(
            skill,
            mode="custom_only",
            language="python",
            force=False,
            no_config=False,
        )
        == 0
    )
    evals = skill / "evals"
    old_grader = (evals / "grader.py").read_bytes()
    old_dataset = (evals / "evals.json").read_bytes()
    old_config = (evals / "config.yml").read_bytes()

    original_write_text = Path.write_text

    def fail_config(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path.name == "config.yml":
            raise OSError("injected late config failure")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_config)

    with pytest.raises(OSError, match="injected late config failure"):
        commands.init_custom_grader(
            skill,
            mode="default_plus_custom",
            language="shell",
            force=True,
            no_config=False,
        )

    assert (evals / "grader.py").read_bytes() == old_grader
    assert not (evals / "grader.sh").exists()
    assert (evals / "evals.json").read_bytes() == old_dataset
    assert (evals / "config.yml").read_bytes() == old_config


def test_scaffold_without_with_config_keeps_opt_in_default(tmp_path: Path) -> None:
    skill = _skill(tmp_path)

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill)])

    assert result.exit_code == 0, result.output
    assert not (skill / "evals" / "config.yml").exists()


def test_scaffold_preserves_existing_dataset_tasks(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    for case_id in ("case-001", "case-002"):
        result = CliRunner().invoke(cli, ["init-harbor-task", str(skill), "--case-id", case_id])
        assert result.exit_code == 0, result.output

    dataset = tomllib.loads((skill / "evals" / "harbor" / "dataset.toml").read_text(encoding="utf-8"))
    assert [task["name"] for task in dataset["tasks"]] == ["nvidia/case-001", "nvidia/case-002"]
    payload = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert payload["evals"][0]["id"] == "case-001"


def test_force_success_removes_stale_case_files_transactionally(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    assert (
        commands.init_harbor_task(
            skill, force=False, case_id="case-001", mode="custom_only", language="python", with_config=False
        )
        == 0
    )
    sentinel = skill / "evals" / "harbor" / "case-001" / "stale.txt"
    sentinel.write_text("remove", encoding="utf-8")

    assert (
        commands.init_harbor_task(
            skill, force=True, case_id="case-001", mode="custom_only", language="shell", with_config=False
        )
        == 0
    )

    assert not sentinel.exists()
    assert (skill / "evals" / "harbor" / "case-001" / "tests" / "grader.sh").is_file()


def test_existing_yaml_dataset_is_not_shadowed(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    existing = evals / "evals.yaml"
    existing.write_text(
        "skill_name: sample-skill\nevals:\n  - id: case-001\n    prompt: Q\n    expected_output: A\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill)])

    assert result.exit_code == 0, result.output
    assert existing.is_file()
    assert not (evals / "evals.json").exists()


def test_with_config_updates_existing_config_yaml_without_shadow_file(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    config = evals / "config.yaml"
    config.write_text(
        "schema_version: 1\nharbor:\n  n_attempts: 3\ngrading:\n  mode: default\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill), "--with-config"])

    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert parsed["harbor"]["task_source"] == "native_harbor"
    assert parsed["harbor"]["n_attempts"] == 3
    assert not (evals / "config.yml").exists()


def test_casefold_colliding_existing_case_is_rejected_without_mutation(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    assert (
        commands.init_harbor_task(
            skill, force=False, case_id="Case-001", mode="custom_only", language="python", with_config=False
        )
        == 0
    )
    sentinel = skill / "evals" / "harbor" / "Case-001" / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill), "--case-id", "case-001"])

    assert result.exit_code != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_native_dataset_writer_quotes_hostile_directory_names(tmp_path: Path) -> None:
    hostile = 'case"\n[[tasks]]\nname="injected'

    _write_dataset_toml(tmp_path, [hostile])

    parsed = tomllib.loads((tmp_path / "dataset.toml").read_text(encoding="utf-8"))
    assert parsed["tasks"] == [{"name": f"nvidia/{hostile}"}]


@pytest.mark.parametrize("value", ['quote"slash\\line\n', "emoji-😀", "\x7f"])
def test_toml_quote_round_trips_valid_strings(value: str) -> None:
    assert tomllib.loads(f"value = {toml_quote(value)}\n")["value"] == value


def test_toml_quote_rejects_lone_surrogates() -> None:
    with pytest.raises(ValueError, match="surrogate"):
        toml_quote("bad-\ud800")
