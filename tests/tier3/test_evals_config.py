# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from skillevaluator.tier3.dataset_utils import load_dataset_entries_with_format
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.evals_spec import validate_skillevaluators as validate_skill_evals


def test_load_evals_config_valid_harbor_policy(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(
        """\
schema_version: 1

harbor:
  task_source: native_harbor
  custom_dockerfile_mode: preserve
  n_attempts: 3
  pass_threshold: 0.60
  n_concurrent: 4
  max_agents: 2
  timeout_multiplier: 2.0
  agent_workdir: /app
  resources:
    cpus: 4
    memory_mb: 8192
    storage_mb: 4096
  agents:
    claude-code:
      model: aws/anthropic/bedrock-claude-opus-4-6

skill_workspace:
  mode: group
  include:
    - ../helper-skill

grading:
  mode: default_plus_custom
""",
        encoding="utf-8",
    )

    config, path = load_evals_config(skill)

    assert path == skill / "evals" / "config.yml"
    assert config["harbor"]["task_source"] == "native_harbor"
    assert config["harbor"]["custom_dockerfile_mode"] == "preserve"
    assert config["harbor"]["n_attempts"] == 3
    assert config["harbor"]["pass_threshold"] == 0.60
    assert config["harbor"]["agent_workdir"] == "/app"
    assert config["harbor"]["resources"] == {
        "cpus": 4,
        "memory_mb": 8192,
        "storage_mb": 4096,
    }
    assert config["harbor"]["agents"]["claude-code"]["model"].endswith("bedrock-claude-opus-4-6")
    assert config["skill_workspace"]["mode"] == "group"
    assert config["skill_workspace"]["include"] == ["../helper-skill"]
    assert config["grading"]["mode"] == "default_plus_custom"


def test_load_evals_config_missing_is_empty(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)

    config, path = load_evals_config(skill)

    assert config == {}
    assert path is None


def test_load_evals_config_rejects_unknown_keys(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(
        """\
schema_version: 1
harbor:
  surprise: true
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match="unknown harbor key"):
        load_evals_config(skill)


def test_load_evals_config_validates_resource_shapes(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(
        """\
schema_version: 1
harbor:
  resources:
    memory_mb: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match=r"harbor\.resources\.memory_mb must be >= 1"):
        load_evals_config(skill)


def test_load_evals_config_validates_ranges(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(
        """\
schema_version: 1
harbor:
  pass_threshold: 1.5
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match="pass_threshold"):
        load_evals_config(skill)


def test_validate_skill_evals_does_not_warn_expected_script_for_guide_only_skill(tmp_path):
    skill = tmp_path / "guide-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        '[{"id":"case-1","question":"Explain the guide."}]',
        encoding="utf-8",
    )

    messages = [r.message for r in validate_skill_evals(skill)]

    assert not any("expected_script" in msg for msg in messages)


def test_validate_skill_evals_warns_expected_script_when_scripts_exist(tmp_path):
    skill = tmp_path / "script-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "scripts").mkdir()
    (skill / "scripts" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        '[{"id":"case-1","question":"Run it."}]',
        encoding="utf-8",
    )

    messages = [r.message for r in validate_skill_evals(skill)]

    assert any("expected_script is missing" in msg for msg in messages)


def test_agentskills_evals_json_is_accepted_without_deprecation_warning(tmp_path):
    skill = tmp_path / "agent-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        """\
{
  "skill_name": "agent-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "Use the skill.",
      "expected_output": "The skill returns a useful answer.",
      "files": ["evals/files/input.txt"],
      "assertions": ["The answer references the input."]
    }
  ]
}
""",
        encoding="utf-8",
    )

    entries, dataset_format = load_dataset_entries_with_format(skill / "evals" / "evals.json")
    results = validate_skill_evals(skill)
    messages = [r.message for r in results]

    assert dataset_format == "agentskills"
    assert entries[0]["question"] == "Use the skill."
    assert entries[0]["ground_truth"] == "The skill returns a useful answer."
    assert entries[0]["expected_behavior"] == ["The answer references the input."]
    assert entries[0]["expected_skill"] == "agent-skill"
    assert not any("Deprecated eval dataset format" in msg for msg in messages)
    assert not any(r.status == "error" for r in results)


def test_agentskills_evals_json_requires_skill_name(tmp_path):
    skill = tmp_path / "agent-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        """\
{
  "evals": [
    {"id": 1, "prompt": "Use the skill.", "expected_output": "The skill answers."}
  ]
}
""",
        encoding="utf-8",
    )

    messages = [r.message for r in validate_skill_evals(skill)]

    assert any("skill_name" in msg for msg in messages)


def test_agentskills_evals_json_rejects_non_object_items(tmp_path):
    skill = tmp_path / "agent-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        """\
{
  "skill_name": "agent-skill",
  "evals": [
    {"id": 1, "prompt": "Use the skill.", "expected_output": "The skill answers."},
    "bad"
  ]
}
""",
        encoding="utf-8",
    )

    messages = [r.message for r in validate_skill_evals(skill)]

    assert any("evals[1]" in msg and "object" in msg for msg in messages)


def test_agentskills_evals_json_reports_authored_required_field_names(tmp_path):
    skill = tmp_path / "agent-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        """\
{
  "skill_name": "agent-skill",
  "evals": [
    {"id": 1, "prompt": "Use the skill."}
  ]
}
""",
        encoding="utf-8",
    )

    messages = [r.message for r in validate_skill_evals(skill)]

    assert any("expected_output" in msg for msg in messages)
    assert not any("ground_truth" in msg for msg in messages)


def test_legacy_evals_json_is_accepted_with_deprecation_warning(tmp_path):
    skill = tmp_path / "legacy-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        '[{"id":"case-1","question":"Use the skill."}]',
        encoding="utf-8",
    )

    results = validate_skill_evals(skill)

    assert not any(r.status == "error" for r in results)
    assert any("Deprecated eval dataset format" in r.message for r in results)
