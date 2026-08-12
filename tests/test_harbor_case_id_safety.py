# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security and compatibility tests for generated Harbor case identifiers."""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.case_ids import safe_child
from skillevaluator.tier3.evals_spec import validate_skillevaluators
from skillevaluator.tier3.harbor.adapter import _write_task_toml, generate_harbor_tasks, stage_native_harbor_tasks
from skillevaluator.tier3.toml_utils import toml_quote


def _write_skill(tmp_path: Path, case_ids: list[object]) -> Path:
    skill_path = tmp_path / "sample-skill"
    evals_dir = skill_path / "evals"
    evals_dir.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text(
        "---\nname: sample-skill\ndescription: Test fixture.\n---\n\n# Sample skill\n",
        encoding="utf-8",
    )
    payload = {
        "skill_name": "sample-skill",
        "evals": [
            {
                "id": case_id,
                "prompt": f"Run case {index}",
                "expected_output": "The case completes.",
            }
            for index, case_id in enumerate(case_ids)
        ],
    }
    (evals_dir / "evals.json").write_text(json.dumps(payload), encoding="utf-8")
    return skill_path


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("", id="empty"),
        pytest.param(".", id="current-directory"),
        pytest.param("..", id="parent-directory"),
        pytest.param("../outside", id="posix-traversal"),
        pytest.param("/tmp/absolute", id="posix-absolute"),
        pytest.param("nested/case", id="posix-separator"),
        pytest.param(r"..\outside", id="windows-traversal"),
        pytest.param(r"C:\temp\case", id="windows-drive-absolute"),
        pytest.param(r"\\server\share\case", id="windows-unc-absolute"),
        pytest.param(r"nested\case", id="windows-separator"),
        pytest.param("CON", id="windows-device"),
        pytest.param("nul.txt", id="windows-device-with-extension"),
        pytest.param("case with spaces", id="lossy-space"),
        pytest.param("case\nnext", id="control-newline"),
        pytest.param('case"next', id="quote"),
        pytest.param(".hidden", id="leading-dot"),
        pytest.param("case.", id="trailing-dot"),
        pytest.param("x" * 129, id="overlong"),
        pytest.param(None, id="null"),
        pytest.param(True, id="boolean"),
        pytest.param(1.5, id="float"),
    ],
)
def test_invalid_case_id_is_rejected_before_any_output_mutation(tmp_path: Path, case_id: object) -> None:
    skill_path = _write_skill(tmp_path, [case_id])
    output_parent = tmp_path / "output-parent"
    output_dir = output_parent / "generated"
    output_dir.mkdir(parents=True)
    inside_sentinel = output_dir / "existing-output.txt"
    outside_sentinel = output_parent / "outside-sentinel.txt"
    inside_sentinel.write_text("inside", encoding="utf-8")
    outside_sentinel.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="case id"):
        generate_harbor_tasks(skill_path, output_dir)

    assert inside_sentinel.read_text(encoding="utf-8") == "inside"
    assert outside_sentinel.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize(
    "case_ids",
    [
        pytest.param(["duplicate", "duplicate"], id="exact-duplicate"),
        pytest.param(["Case-001", "case-001"], id="case-insensitive-filesystem-collision"),
        pytest.param([1, "1"], id="numeric-string-collision"),
        pytest.param(["case/a", "case-a"], id="slash-slug-collision"),
        pytest.param(["case a", "case-a"], id="space-slug-collision"),
    ],
)
def test_colliding_case_ids_are_rejected_before_any_output_mutation(
    tmp_path: Path,
    case_ids: list[object],
) -> None:
    skill_path = _write_skill(tmp_path, case_ids)
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    sentinel = output_dir / "existing-output.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="case id"):
        generate_harbor_tasks(skill_path, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_numeric_agentskills_id_is_canonicalized_for_harbor(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, [1])
    output_dir = tmp_path / "generated"

    task_paths = generate_harbor_tasks(skill_path, output_dir)

    assert task_paths == [output_dir / "1"]
    task_config = tomllib.loads((task_paths[0] / "task.toml").read_text(encoding="utf-8"))
    assert task_config["metadata"]["entry_id"] == "1"
    entry = json.loads((task_paths[0] / "tests" / "entry.json").read_text(encoding="utf-8"))
    assert entry["id"] == "1"


def test_consecutive_dots_inside_case_id_are_safe(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["case..part"])
    output_dir = tmp_path / "generated"

    task_paths = generate_harbor_tasks(skill_path, output_dir)

    assert task_paths == [output_dir / "case..part"]
    assert (task_paths[0] / "task.toml").is_file()


def test_validate_evals_reports_unsafe_case_ids(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["safe-case", ".."])

    results = validate_skillevaluators(skill_path)

    assert any(result.status == "error" and "case id" in result.message for result in results)


def test_tier3_validate_cli_rejects_unsafe_case_id(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["safe-case", ".."])

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill_path), "--strict", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(item["status"] == "error" and "case id" in item["message"] for item in payload)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        pytest.param(
            "evals.json",
            json.dumps(
                {
                    "skill_name": "sample-skill",
                    "evals": [{"id": "..", "prompt": "Q", "expected_output": "A"}],
                }
            ),
            id="wrapped-json",
        ),
        pytest.param(
            "evals.jsonl",
            json.dumps({"id": "..", "question": "Q"}) + "\n",
            id="legacy-jsonl",
        ),
        pytest.param(
            "evals.yaml",
            "skill_name: sample-skill\nevals:\n  - id: '..'\n    prompt: Q\n    expected_output: A\n",
            id="wrapped-yaml",
        ),
        pytest.param(
            "evals.yml",
            "skill_name: sample-skill\nevals:\n  - id: '..'\n    prompt: Q\n    expected_output: A\n",
            id="wrapped-yml",
        ),
    ],
)
def test_all_dataset_formats_reject_unsafe_ids_before_creating_output(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    skill_path = _write_skill(tmp_path, ["placeholder"])
    evals_dir = skill_path / "evals"
    (evals_dir / "evals.json").unlink()
    (evals_dir / filename).write_text(content, encoding="utf-8")
    output_dir = tmp_path / "not-created"

    with pytest.raises(ValueError, match="case id"):
        generate_harbor_tasks(skill_path, output_dir)

    assert not output_dir.exists()


def test_valid_case_id_boundaries_round_trip_through_real_harbor_model(tmp_path: Path) -> None:
    from harbor.models.task.task import Task

    case_ids: list[object] = [0, "a", "A_b-c.1", "x" * 128]
    expected_ids = [str(case_id) for case_id in case_ids]
    skill_path = _write_skill(tmp_path, case_ids)
    output_dir = tmp_path / "generated"

    task_paths = generate_harbor_tasks(skill_path, output_dir)

    assert [path.name for path in task_paths] == expected_ids
    for expected_id, task_path in zip(expected_ids, task_paths, strict=True):
        task = Task(task_path)
        assert task.config.metadata["entry_id"] == expected_id
        entry = json.loads((task_path / "tests" / "entry.json").read_text(encoding="utf-8"))
        assert entry["id"] == expected_id

    dataset = tomllib.loads((output_dir / "dataset.toml").read_text(encoding="utf-8"))
    assert [task["name"] for task in dataset["tasks"]] == [f"nvidia/{case_id}" for case_id in sorted(expected_ids)]


def test_existing_task_symlink_is_rejected_without_touching_its_target(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["case-001"])
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    sentinel = outside_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    task_link = output_dir / "case-001"
    task_link.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        generate_harbor_tasks(skill_path, output_dir)

    assert task_link.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_symlinked_output_root_is_rejected_without_touching_its_target(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["case-001"])
    outside_dir = tmp_path / "outside"
    task_dir = outside_dir / "case-001"
    task_dir.mkdir(parents=True)
    sentinel = task_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    output_dir = tmp_path / "generated"
    output_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match=r"symlink|reparse point"):
        generate_harbor_tasks(skill_path, output_dir)

    assert output_dir.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_safe_child_rejects_symlinked_output_ancestor_without_touching_target(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    sentinel = outside_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match=r"symlink|reparse|junction"):
        safe_child(linked_parent / "generated", "case-001")

    assert not (outside_dir / "generated").exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(os.name != "posix" or not Path("/tmp").is_symlink(), reason="no trusted platform root alias")
def test_safe_child_accepts_root_owned_platform_temp_alias() -> None:
    base = Path("/tmp") / "skillevaluator-safe-child-not-created"

    assert safe_child(base, "case-001") == base / "case-001"


def test_symlinked_output_ancestor_leaves_target_untouched(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["case-001"])
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    sentinel = outside_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises((ValueError, OSError), match=r"symlink|reparse|non-directory"):
        generate_harbor_tasks(skill_path, linked_parent / "generated")

    assert sorted(path.name for path in outside_dir.iterdir()) == ["sentinel.txt"]
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_all_dynamic_task_toml_strings_and_keys_are_quoted_for_real_harbor(tmp_path: Path) -> None:
    from harbor.models.task.task import Task

    hostile = 'hostile"\nkeywords = ["injected"]'
    hostile_key = 'custom"\ninjected-key'
    skill_path = _write_skill(tmp_path, ["case-001"])
    payload_path = skill_path / "evals" / "evals.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["evals"][0]["expected_skill"] = hostile
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    environment = skill_path / "evals" / "environment"
    environment.mkdir()
    (environment / "mcp_servers.toml").write_text(
        "[[mcp_servers]]\n"
        f"name = {toml_quote(hostile)}\n"
        f"command = {toml_quote(hostile)}\n"
        f"args = [{toml_quote(hostile)}]\n"
        'transport = "stdio"\n'
        f"{toml_quote(hostile_key)} = {toml_quote(hostile)}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "generated"

    task_paths = generate_harbor_tasks(
        skill_path,
        output_dir,
        runtime_env={"SAFE_ENV": hostile},
        pre_agent_setup=[hostile],
        agent_workdir="/workspace/safe",
    )
    _write_task_toml(
        task_paths[0],
        {"id": "case-001", "expected_skill": hostile},
        has_skill=True,
        mcp_servers=[
            {
                "name": hostile,
                "command": hostile,
                "args": [hostile],
                "transport": "stdio",
                hostile_key: hostile,
            }
        ],
        docker_image=hostile,
        runtime_env={"SAFE_ENV": hostile},
        pre_agent_setup=[hostile],
        agent_workdir=hostile,
    )

    parsed = tomllib.loads((task_paths[0] / "task.toml").read_text(encoding="utf-8"))
    assert parsed["task"]["description"] == f"Skill evaluation task for {hostile}"
    assert parsed["metadata"]["skill"] == hostile
    assert parsed["environment"]["docker_image"] == hostile
    assert parsed["environment"]["workdir"] == hostile
    assert parsed["environment"]["env"]["SAFE_ENV"] == hostile
    assert parsed["environment"]["mcp_servers"] == [
        {
            "name": hostile,
            "command": hostile,
            "args": [hostile],
            "transport": "stdio",
            hostile_key: hostile,
        }
    ]
    task = Task(task_paths[0])
    assert task.config.task.description == f"Skill evaluation task for {hostile}"
    assert task.config.metadata["skill"] == hostile
    assert task.config.environment.docker_image == hostile
    assert task.config.environment.mcp_servers[0].name == hostile
    assert task.config.environment.mcp_servers[0].command == hostile
    assert task.config.environment.mcp_servers[0].args == [hostile]


def test_invalid_late_entry_leaves_existing_generated_output_unchanged(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["case-001", "case-002"])
    payload_path = skill_path / "evals" / "evals.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["evals"][1]["prompt"] = 7
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    sentinel = output_dir / "existing-output.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=r"str|prompt|operand"):
        generate_harbor_tasks(skill_path, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == ["existing-output.txt"]
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_native_default_without_eval_metadata_preserves_existing_output(tmp_path: Path) -> None:
    skill_path = tmp_path / "sample-skill"
    native_task = skill_path / "evals" / "harbor" / "case-001"
    native_task.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text(
        "---\nname: sample-skill\ndescription: Test fixture.\n---\n# Sample\n",
        encoding="utf-8",
    )
    (native_task / "instruction.md").write_text("Run the case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "native-output"
    output_dir.mkdir()
    sentinel = output_dir / "existing-output.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"require evals/evals.json metadata"):
        stage_native_harbor_tasks(skill_path, output_dir, grading_mode="default")

    assert sorted(path.name for path in output_dir.iterdir()) == ["existing-output.txt"]
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_existing_real_task_directory_is_safely_replaced(tmp_path: Path) -> None:
    from harbor.models.task.task import Task

    skill_path = _write_skill(tmp_path, ["case-001"])
    output_dir = tmp_path / "generated"
    existing_task = output_dir / "case-001"
    existing_task.mkdir(parents=True)
    stale = existing_task / "stale.txt"
    stale.write_text("remove", encoding="utf-8")

    task_paths = generate_harbor_tasks(skill_path, output_dir)

    assert task_paths == [existing_task]
    assert not stale.exists()
    assert Task(existing_task).config.metadata["entry_id"] == "case-001"


def test_existing_task_replacement_uses_transactional_publish_without_hardened_rmtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_path = _write_skill(tmp_path, ["case-001"])
    output_dir = tmp_path / "generated"
    existing_task = output_dir / "case-001"
    existing_task.mkdir(parents=True)
    sentinel = existing_task / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(shutil.rmtree, "avoids_symlink_attacks", False)

    task_paths = generate_harbor_tasks(skill_path, output_dir)

    assert task_paths == [existing_task]
    assert not sentinel.exists()


def test_native_mode_validates_eval_ids_before_replacing_output(tmp_path: Path) -> None:
    skill_path = _write_skill(tmp_path, ["safe-case", ".."])
    (skill_path / "evals" / "harbor").mkdir()
    output_dir = tmp_path / "native-output"
    output_dir.mkdir()
    sentinel = output_dir / "existing-output.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="case id"):
        stage_native_harbor_tasks(skill_path, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_native_mode_canonicalizes_numeric_eval_id_and_stages_task(tmp_path: Path) -> None:
    from harbor.models.task.task import Task

    skill_path = _write_skill(tmp_path, [1])
    native_task = skill_path / "evals" / "harbor" / "1"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Run the numeric case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/native-1"\n\n[metadata]\nentry_id = 1\n\n[environment]\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "native-output"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("remove", encoding="utf-8")

    task_paths = stage_native_harbor_tasks(skill_path, output_dir)

    assert task_paths == [output_dir / "1"]
    assert not (output_dir / "stale.txt").exists()
    entry = json.loads((task_paths[0] / "tests" / "entry.json").read_text(encoding="utf-8"))
    assert entry["id"] == "1"
    assert Task(task_paths[0]).config.metadata["entry_id"] == 1
