# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skillevaluator.tier3.dataset_utils import load_dataset_entries_with_format
from skillevaluator.tier3.evals_config import (
    EvalsConfigError,
    load_evals_config,
    parse_environment_kwarg_overrides,
    validate_environment_kwargs,
)
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


def test_load_evals_config_rejects_environment_kwargs_because_they_are_operator_only(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  environment_kwargs:\n    region: us-west-2\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match=r"unknown harbor key.*environment_kwargs"):
        load_evals_config(skill)


@pytest.mark.parametrize(
    ("entry", "secret"),
    [
        ("api_key=do-not-render-this", "do-not-render-this"),
        ('headers={"Authorization":"do-not-render-this"}', "do-not-render-this"),
        ('headers={"X-API-Key":"do-not-render-this"}', "do-not-render-this"),
        ('metadata={"clientSecret":"do-not-render-this"}', "do-not-render-this"),
        ('nested={"sudoPassword":"do-not-render-this"}', "do-not-render-this"),
        ("endpoint=https://bearer-token@example.invalid/path", "bearer-token"),
        ("endpoint=https://user%3Apassword@example.invalid/path", "user%3Apassword"),
        ("endpoint=ssh://git@example.invalid/repo", "git@example.invalid"),
        ("proxy=https://user:do-not-render-this@example.test", "do-not-render-this"),
        ("missing-equals-do-not-render-this", "do-not-render-this"),
    ],
)
def test_cli_environment_kwargs_reject_secrets_without_echoing_values(entry, secret):
    with pytest.raises(ValueError) as caught:
        parse_environment_kwarg_overrides((entry,))

    assert secret not in str(caught.value)


def test_cli_environment_kwargs_allow_non_secret_key_names() -> None:
    assert parse_environment_kwarg_overrides(('key_name="evaluation-key"',)) == {"key_name": "evaluation-key"}


def test_cli_environment_kwargs_allow_kubernetes_secret_object_reference() -> None:
    assert parse_environment_kwarg_overrides(
        ('image_pull_secret="registry-credentials"',),
        env_mode="ack",
    ) == {"image_pull_secret": "registry-credentials"}


@pytest.mark.parametrize(
    ("env_mode", "reference"),
    [
        ("modal", "registry-credentials"),
        ("daytona", "registry-credentials"),
        ("ack", "username:password"),
        ("ack", "UPPER_CASE"),
        ("ack", "contains spaces"),
    ],
)
def test_cli_environment_kwargs_reject_image_pull_secret_outside_ack_or_invalid_kubernetes_names(
    env_mode: str,
    reference: str,
) -> None:
    with pytest.raises(ValueError, match="Invalid --environment-kwarg"):
        parse_environment_kwarg_overrides(
            (f"image_pull_secret={json.dumps(reference)}",),
            env_mode=env_mode,
        )


def test_environment_kwargs_reject_cycles_without_recursing() -> None:
    cyclic: dict[str, object] = {}
    cyclic["nested"] = cyclic

    with pytest.raises(ValueError, match="cyclic"):
        validate_environment_kwargs({"options": cyclic})


def test_environment_kwargs_reject_excessive_direct_api_nesting_without_recursing() -> None:
    nested: object = "leaf"
    for _ in range(40):
        nested = [nested]

    with pytest.raises(ValueError, match="nest at most"):
        validate_environment_kwargs({"options": nested})


def test_cli_environment_kwargs_reject_extreme_json_nesting_without_traceback() -> None:
    deeply_nested = "[" * 1200 + "0" + "]" * 1200

    with pytest.raises(ValueError, match="nest"):
        parse_environment_kwarg_overrides((f"options={deeply_nested}",))


@pytest.mark.parametrize(
    ("env_mode", "entry", "expected"),
    [
        (
            "daytona",
            'secrets={"TARGET_API_KEY":"organization-secret-name"}',
            {"secrets": {"TARGET_API_KEY": "organization-secret-name"}},
        ),
        (
            "modal",
            'secrets=["runtime-secret","telemetry-secret"]',
            {"secrets": ["runtime-secret", "telemetry-secret"]},
        ),
        (
            "modal",
            'registry_secret="private-registry-login"',
            {"registry_secret": "private-registry-login"},
        ),
        (
            "skypilot",
            'secrets=["cluster-secret"]',
            {"secrets": ["cluster-secret"]},
        ),
        (
            "cwsandbox",
            'secrets=[{"store":"team-store","name":"runtime-key","field":"value","env_var":"API_KEY"}]',
            {"secrets": [{"store": "team-store", "name": "runtime-key", "field": "value", "env_var": "API_KEY"}]},
        ),
        (
            "wandb",
            'secrets=[{"name":"runtime-key","env_var":"API_KEY"}]',
            {"secrets": [{"name": "runtime-key", "env_var": "API_KEY"}]},
        ),
    ],
)
def test_cli_environment_kwargs_allow_provider_secret_references(env_mode, entry, expected) -> None:
    assert parse_environment_kwarg_overrides((entry,), env_mode=env_mode) == expected


@pytest.mark.parametrize(
    ("env_mode", "entry"),
    [
        ("e2b", 'secrets=["not-supported"]'),
        ("daytona", 'secrets=["wrong-shape"]'),
        ("modal", 'secrets={"wrong":"shape"}'),
        ("cwsandbox", 'secrets=[{"value":"plaintext-not-a-reference"}]'),
    ],
)
def test_cli_environment_kwargs_reject_secret_reference_fields_outside_exact_harbor_shapes(env_mode, entry) -> None:
    with pytest.raises(ValueError, match="Invalid --environment-kwarg"):
        parse_environment_kwarg_overrides((entry,), env_mode=env_mode)


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
