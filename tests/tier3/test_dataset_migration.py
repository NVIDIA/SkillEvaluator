# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless migration must preserve runtime grading and authored metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.dataset_utils import load_dataset_entries


def _write(path: Path, data: object) -> Path:
    if path.suffix in {".yaml", ".yml"}:
        content = yaml.safe_dump(data, allow_unicode=True)
    elif path.suffix == ".jsonl":
        content = "\n".join(json.dumps(entry) for entry in data)
    else:
        content = json.dumps(data, ensure_ascii=False)
    path.write_text(content + "\n", encoding="utf-8")
    return path


def _case(**fields: object) -> dict:
    return {"id": 7, "question": "Résumé", "ground_truth": "done", "expected_skill": "demo", **fields}


def test_cli_previews_without_writing_and_writes_only_explicit_new_output(tmp_path: Path) -> None:
    source = _write(tmp_path / "legacy.json", [_case(expected_skill=None, should_trigger=False)])
    original = source.read_bytes()
    runner = CliRunner()
    result = runner.invoke(cli, ["tier3", "migrate-dataset", str(source), "--skill-name", "demo"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["evals"][0]["expected_skill"] is None
    assert list(tmp_path.iterdir()) == [source]
    destination = tmp_path / "converted.yaml"
    result = runner.invoke(
        cli, ["tier3", "migrate-dataset", str(source), "--skill-name", "demo", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(destination.read_text())["evals"][0]["should_trigger"] is False
    assert source.read_bytes() == original
    result = runner.invoke(
        cli, ["tier3", "migrate-dataset", str(source), "--output", str(destination), "--skill-name", "demo"]
    )
    assert result.exit_code != 0
    assert "exist" in result.output.lower()
    assert source.read_bytes() == original


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".yaml", ".yml"])
@pytest.mark.parametrize("output_suffix", [".json", ".yaml", ".yml"])
def test_migration_preserves_every_authored_field_and_ids(tmp_path: Path, suffix: str, output_suffix: str) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    entries = [
        _case(
            expected_script="run.py",
            expected_behavior=["reads skill"],
            acceptable_alternates=["other"],
            files=["data/a.txt"],
            extension={"nested": [None, False, 3]},
            generation={"model": "original"},
            checks={"rubric": ["correct"]},
        ),
        _case(id="negative", expected_skill=None, should_trigger=False, files=[], expected_script=None),
    ]
    source = _write(tmp_path / f"source{suffix}", entries)
    original = source.read_bytes()
    destination = tmp_path / f"converted{output_suffix}"
    result = migrate_dataset(source, skill_name="demo", destination=destination)
    assert result.dataset == {
        "skill_name": "demo",
        "evals": [
            {
                "id": 7,
                "prompt": "Résumé",
                "expected_output": "done",
                "expected_skill": "demo",
                "expected_script": "run.py",
                "assertions": ["reads skill"],
                "acceptable_skills": ["other"],
                "files": ["data/a.txt"],
                "extension": {"nested": [None, False, 3]},
                "generation": {"model": "original"},
                "checks": {"rubric": ["correct"]},
            },
            {
                "id": "negative",
                "prompt": "Résumé",
                "expected_output": "done",
                "expected_skill": None,
                "should_trigger": False,
                "files": [],
                "expected_script": None,
            },
        ],
    }
    assert source.read_bytes() == original
    assert migrate_dataset(destination, skill_name=None).dataset == result.dataset


def test_preferred_wrapper_metadata_survives_idempotent_conversion(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    data = {
        "skill_name": "demo",
        "evals": [{"id": "one", "prompt": "work", "expected_output": "done"}],
        "generation": {"provider": "original"},
        "checks": ["exact"],
        "extension": False,
    }
    source = _write(tmp_path / "preferred.json", data)
    assert migrate_dataset(source, skill_name=None).dataset == data


@pytest.mark.parametrize(
    "data,skill_name,error",
    [
        ([_case(prompt="different")], "demo", "conflicting.*question.*prompt"),
        ([_case(acceptable_alternates=["one"], acceptable_skills=["two"])], "demo", "conflicting.*acceptable"),
        ([_case(), 42], "demo", "entry 1.*object"),
        ({"cases": "invalid"}, "demo", "cases.*array"),
        ([], "demo", "empty"),
        ([_case(id="../escape")], "demo", "invalid case id"),
        ([_case(id=7), _case(id="7")], "demo", "colliding case id"),
        ([_case(expected_skill="one"), _case(id="two", expected_skill="two")], None, "ambiguous.*skill"),
        ([_case(expected_skill=None)], None, "skill-name"),
        ([_case()], " ", "skill-name"),
        ({"skill_name": "one", "evals": [_case()]}, "two", "conflicting.*skill"),
        ([_case(expected_skill=42)], "demo", "expected_skill"),
        ([_case(question=None)], "demo", "prompt"),
    ],
)
def test_invalid_or_ambiguous_input_is_rejected_before_writing(
    tmp_path: Path, data: object, skill_name: str | None, error: str
) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    source = _write(tmp_path / "source.json", data)
    destination = tmp_path / "new.json"
    original = source.read_bytes()
    with pytest.raises(ValueError, match=error):
        migrate_dataset(source, skill_name=skill_name, destination=destination)
    assert not destination.exists()
    assert source.read_bytes() == original


def test_equal_aliases_are_collapsed_and_identity_can_be_inferred(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    source = _write(tmp_path / "source.json", [_case(prompt="Résumé", acceptable_alternates=[], acceptable_skills=[])])
    assert migrate_dataset(source, skill_name=None).dataset == {
        "skill_name": "demo",
        "evals": [
            {"id": 7, "prompt": "Résumé", "expected_output": "done", "expected_skill": "demo", "acceptable_skills": []}
        ],
    }


@pytest.mark.parametrize("destination_kind", ["source", "existing", "symlink", "dangling", "jsonl"])
def test_destinations_never_overwrite_and_jsonl_output_is_not_misrepresented(
    tmp_path: Path, destination_kind: str
) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    source = _write(tmp_path / "source.json", [_case()])
    destination = tmp_path / "new.json"
    if destination_kind == "source":
        destination = source
    elif destination_kind == "existing":
        destination.write_text("keep")
    elif destination_kind in {"symlink", "dangling"}:
        destination.symlink_to(source if destination_kind == "symlink" else tmp_path / "missing")
    else:
        destination = tmp_path / "new.jsonl"
    original = source.read_bytes()
    with pytest.raises(ValueError, match=r"source|exist|JSONL"):
        migrate_dataset(source, skill_name=None, destination=destination)
    assert source.read_bytes() == original
    if destination_kind == "existing":
        assert destination.read_text() == "keep"


def test_converted_routing_has_identical_real_grading_outcomes(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset
    from skillevaluator.tier3.eval_core.checks import (
        resolve_acceptable_skills,
        resolve_should_trigger,
        score_skill_execution,
    )

    source = _write(
        tmp_path / "source.json",
        [
            _case(),
            _case(id="negative", expected_skill=None, should_trigger=False),
            _case(id="false", should_trigger=False, acceptable_alternates=["other"]),
        ],
    )
    destination = tmp_path / "new.json"
    migrate_dataset(source, skill_name="demo", destination=destination)
    tool_calls = [
        {"action": "read_file", "action_input": {"path": "/skills/demo/SKILL.md"}, "observation": "skill instructions"}
    ]
    before, after = load_dataset_entries(source), load_dataset_entries(destination)
    for old, new in zip(before, after, strict=True):
        for calls in ([], tool_calls):
            scores = [
                score_skill_execution(
                    calls,
                    entry.get("expected_skill") or "",
                    entry.get("expected_script"),
                    resolve_should_trigger(entry),
                    evaluated_skill="demo",
                    acceptable_skills=resolve_acceptable_skills(entry),
                )
                for entry in (old, new)
            ]
            assert scores[0] == scores[1]
    assert score_skill_execution(tool_calls, "demo", evaluated_skill="demo")["score"] == 1.0
    assert score_skill_execution([], "demo", evaluated_skill="demo")["score"] == 0.0
    assert score_skill_execution(tool_calls, "", should_trigger=False, evaluated_skill="demo")["score"] == 0.0
    assert resolve_should_trigger(after[0]) is True
    assert resolve_should_trigger(after[1]) is False
    assert score_skill_execution([], "", should_trigger=False, evaluated_skill="demo")["score"] == 1.0


def test_unlabeled_legacy_routing_is_rejected_without_guessing(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset
    from skillevaluator.tier3.eval_core.checks import resolve_should_trigger

    entry = _case()
    del entry["expected_skill"]
    source = _write(tmp_path / "source.json", [entry])
    assert resolve_should_trigger(load_dataset_entries(source)[0]) is None
    with pytest.raises(ValueError, match=r"unlabeled routing.*expected_skill.*should_trigger"):
        migrate_dataset(source, skill_name="demo", destination=tmp_path / "new.json")
    assert not (tmp_path / "new.json").exists()


def test_explicit_false_routing_without_expected_skill_is_preserved(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset
    from skillevaluator.tier3.eval_core.checks import resolve_should_trigger

    entry = _case(should_trigger=False)
    del entry["expected_skill"]
    source = _write(tmp_path / "source.json", [entry])
    destination = tmp_path / "new.json"
    migrate_dataset(source, skill_name="demo", destination=destination)
    new = load_dataset_entries(destination)[0]
    assert new["expected_skill"] is None
    assert resolve_should_trigger(new) is False


@pytest.mark.parametrize(
    "suffix,content",
    [
        (".json", '[{"id":"one","question":"first","question":"lost","ground_truth":"done","expected_skill":"demo"}]'),
        (".yaml", "- id: one\n  question: first\n  question: lost\n  ground_truth: done\n  expected_skill: demo\n"),
    ],
)
def test_duplicate_authored_keys_are_rejected_instead_of_lost(tmp_path: Path, suffix: str, content: str) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    source = tmp_path / f"source{suffix}"
    source.write_text(content)
    with pytest.raises(ValueError, match=r"duplicate.*question"):
        migrate_dataset(source, skill_name=None)


def test_bundled_examples_use_preferred_objects_with_fifteen_cases() -> None:
    root = Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/reference_skills"
    datasets = sorted(root.glob("*/evals/evals.json"))
    assert len(datasets) == 5
    count = 0
    for source in datasets:
        data = json.loads(source.read_text())
        assert isinstance(data, dict), source
        assert data["skill_name"] == source.parents[1].name
        count += len(data["evals"])
        assert all("prompt" in entry and "expected_output" in entry for entry in data["evals"])
    assert count == 15


def test_converted_files_selectors_stage_identical_harbor_fixtures(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset
    from skillevaluator.tier3.harbor.adapter import _stage_task_inputs

    skill = tmp_path / "demo"
    evals = skill / "evals"
    files = evals / "files"
    files.mkdir(parents=True)
    (files / "wanted.txt").write_text("wanted")
    (files / "unrelated.txt").write_text("unrelated")
    source = _write(evals / "legacy.json", [_case(files=["files/wanted.txt"]), _case(id="empty", files=[])])
    destination = evals / "converted.json"
    migrate_dataset(source, skill_name=None, destination=destination)
    for index, (old, new) in enumerate(
        zip(load_dataset_entries(source), load_dataset_entries(destination), strict=True)
    ):
        for label, entry in (("old", old), ("new", new)):
            environment = tmp_path / f"{label}-{index}"
            environment.mkdir()
            _stage_task_inputs(
                environment, input_files_dir=files, entry=entry, source_skill_path=skill, evals_dir=evals
            )
            staged = sorted(
                path.relative_to(environment).as_posix() for path in environment.rglob("*") if path.is_file()
            )
            assert staged == (["input/wanted.txt"] if index == 0 else [])


def test_nested_alias_type_conflicts_are_not_collapsed(tmp_path: Path) -> None:
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    source = _write(tmp_path / "source.json", [_case(expected_behavior=[False], assertions=[0])])
    with pytest.raises(ValueError, match=r"conflicting.*expected_behavior.*assertions"):
        migrate_dataset(source, skill_name=None)
