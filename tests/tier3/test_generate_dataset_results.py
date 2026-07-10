# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil

from skillevaluator.tier3 import generate_dataset
from skillevaluator.tier3.generate_dataset import (
    _discover_trajectories,
    _run_agent_collect_trajectories,
    _to_agentskills_dataset,
)


def test_discover_trajectories_uses_env_results_root(tmp_path, monkeypatch):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    results_root = tmp_path / "external-results"
    trial = (
        results_root
        / "my-skill"
        / "latest"
        / "claude-code"
        / "with-skill"
        / "trials"
        / "case-001"
    )
    trial.mkdir(parents=True)
    trajectory = {"steps": [{"type": "assistant", "content": "done"}]}
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(results_root))

    assert _discover_trajectories(skill) == {"case-001": trajectory}


def test_discover_trajectories_results_dir_overrides_env(tmp_path, monkeypatch):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    env_root = tmp_path / "env-results"
    cli_root = tmp_path / "cli-results"
    trial = (
        cli_root
        / "my-skill"
        / "latest"
        / "claude-code"
        / "with-skill"
        / "trials"
        / "case-001"
    )
    trial.mkdir(parents=True)
    trajectory = {"steps": [{"type": "assistant", "content": "done"}]}
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(env_root))

    assert _discover_trajectories(skill, results_dir=cli_root) == {"case-001": trajectory}


def test_to_agentskills_dataset_preserves_aces_metadata():
    dataset = _to_agentskills_dataset(
        "my-skill",
        [
            {
                "id": "case-001",
                "question": "Use my-skill.",
                "ground_truth": "The agent uses the skill.",
                "expected_behavior": ["The agent reads SKILL.md."],
                "expected_skill": "my-skill",
                "expected_script": "main.py",
            }
        ],
    )

    assert dataset["skill_name"] == "my-skill"
    assert dataset["evals"][0]["prompt"] == "Use my-skill."
    assert dataset["evals"][0]["expected_output"] == "The agent uses the skill."
    assert dataset["evals"][0]["assertions"] == ["The agent reads SKILL.md."]
    assert dataset["evals"][0]["expected_skill"] == "my-skill"
    assert dataset["evals"][0]["expected_script"] == "main.py"


def test_dry_run_refine_does_not_write_dataset_when_no_trajectory(tmp_path, monkeypatch):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            str(skill),
            "--no-llm",
            "--dry-run",
            "--refine",
        ],
    )

    generate_dataset.main()

    assert not (skill / "evals" / "evals.json").exists()


def test_agent_collect_stages_agentskills_dataset(tmp_path, monkeypatch):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    cases = [
        {
            "id": "case-001",
            "question": "Use my-skill.",
            "ground_truth": "The agent uses the skill.",
            "expected_behavior": ["The agent reads SKILL.md."],
            "expected_skill": "my-skill",
            "expected_script": "main.py",
        }
    ]

    monkeypatch.setattr(shutil, "which", lambda _name, *_args, **_kwargs: "/usr/bin/tool")
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.runner.run_harbor_eval",
        lambda **_kwargs: {"agents": {}},
    )
    monkeypatch.setattr(
        "skillevaluator.tier3.generate_dataset._discover_trajectories",
        lambda *_args, **_kwargs: {},
    )

    _run_agent_collect_trajectories(skill, cases)

    data = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert data["skill_name"] == "my-skill"
    assert data["evals"][0]["prompt"] == "Use my-skill."
    assert data["evals"][0]["expected_output"] == "The agent uses the skill."
