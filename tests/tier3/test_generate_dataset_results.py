# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import stat
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

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
    skill_results = results_root / "my-skill"
    run_id = "20260709_120000"
    run_dir = skill_results / run_id
    trial = run_dir / "claude-code" / "with-skill" / "trials" / "case-001"
    trial.mkdir(parents=True)
    trajectory = {"steps": [{"type": "assistant", "content": "done"}]}
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (skill_results / "latest").symlink_to(run_id)

    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(results_root))

    assert _discover_trajectories(skill) == {"case-001": trajectory}


def test_discover_trajectories_results_dir_overrides_env(tmp_path, monkeypatch):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    env_root = tmp_path / "env-results"
    cli_root = tmp_path / "cli-results"
    skill_results = cli_root / "my-skill"
    run_id = "20260709_120000"
    run_dir = skill_results / run_id
    trial = run_dir / "claude-code" / "with-skill" / "trials" / "case-001"
    trial.mkdir(parents=True)
    trajectory = {"steps": [{"type": "assistant", "content": "done"}]}
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (skill_results / "latest").symlink_to(run_id)

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
    result = generate_dataset.main(
        [
            str(skill),
            "--no-llm",
            "--dry-run",
            "--refine",
        ]
    )

    assert not (skill / "evals" / "evals.json").exists()
    assert result.status == "preview"
    assert result.cases_count == 1
    assert result.dataset is not None


def test_main_invalid_skill_raises_domain_error_without_printing(tmp_path, capsys):
    from skillevaluator.evaluation import DatasetGenerationError

    with pytest.raises(DatasetGenerationError, match=rf"{tmp_path} does not contain a SKILL\.md"):
        generate_dataset.main([str(tmp_path)])

    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(("extra_args", "expected_cases"), [([], 1), (["--full"], 4)])
def test_main_reports_created_dataset_with_written_payload(tmp_path, extra_args, expected_cases):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )

    result = generate_dataset.main([str(skill), "--no-llm", *extra_args])

    assert result.status == "created"
    assert result.path == skill / "evals" / "evals.json"
    assert result.cases_count == expected_cases
    assert result.dataset == json.loads(result.path.read_text(encoding="utf-8"))


def test_force_write_failure_preserves_existing_dataset(tmp_path, monkeypatch):
    from skillevaluator.evaluation import DatasetGenerationError

    skill = tmp_path / "my-skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )
    output = evals / "evals.json"
    original = b'{"skill_name":"original","evals":[]}\n'
    output.write_bytes(original)

    def _partial_dump(_dataset, stream, **_kwargs):
        stream.write('{"partial":')
        raise OSError("disk full")

    monkeypatch.setattr(generate_dataset.json, "dump", _partial_dump)

    with pytest.raises(DatasetGenerationError, match="Could not write dataset"):
        generate_dataset.main([str(skill), "--no-llm", "--force"])

    assert output.read_bytes() == original
    assert list(evals.iterdir()) == [output]


def test_force_write_preserves_existing_dataset_permissions(tmp_path):
    skill = tmp_path / "my-skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )
    output = evals / "evals.json"
    output.write_text('{"skill_name":"original","evals":[]}\n', encoding="utf-8")
    output.chmod(0o640)

    generate_dataset.main([str(skill), "--no-llm", "--force"])

    assert stat.S_IMODE(output.stat().st_mode) == 0o640


def test_main_reports_existing_dataset_as_unchanged(tmp_path):
    skill = tmp_path / "my-skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )
    output = evals / "evals.json"
    output.write_text('{"skill_name": "my-skill", "evals": []}', encoding="utf-8")

    result = generate_dataset.main([str(skill), "--no-llm"])

    assert result.status == "unchanged"
    assert result.path == output
    assert result.dataset is None
    assert output.read_text(encoding="utf-8") == '{"skill_name": "my-skill", "evals": []}'


def test_command_uses_explicit_argv_without_mutating_process_state(tmp_path, monkeypatch):
    from skillevaluator.tier3 import commands

    observed: list[list[str]] = []
    sentinel = object()
    original_argv = sys.argv[:]
    monkeypatch.setattr(generate_dataset, "main", lambda argv=None: observed.append(list(argv or ())) or sentinel)

    result = commands.create_dataset(tmp_path, no_llm=True, dry_run=True)

    assert result is sentinel
    assert observed == [[str(tmp_path.resolve()), "--no-llm", "--dry-run"]]
    assert sys.argv == original_argv


def test_concurrent_programmatic_generation_has_no_cross_talk(tmp_path):
    from skillevaluator.evaluation import DatasetOptions, EvaluationService

    skills = []
    for name in ("alpha-skill", "beta-skill"):
        skill = tmp_path / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Does useful work\n---\n",
            encoding="utf-8",
        )
        skills.append(skill)

    def _generate(skill):
        return EvaluationService().create_dataset(DatasetOptions(skill_path=skill, no_llm=True))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_generate, skills))

    assert [result.status for result in results] == ["created", "created"]
    assert [result.path.parent.parent for result in results] == skills
    assert [result.dataset["skill_name"] for result in results] == ["alpha-skill", "beta-skill"]


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


def _parse(tmp_path, frontmatter: str):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n# body\n", encoding="utf-8")
    return generate_dataset._parse_skill(skill)


def test_parse_skill_folds_block_scalar_description(tmp_path):
    """``description: >-`` must fold, not be captured as the literal indicator."""
    parsed = _parse(
        tmp_path,
        "name: my-skill\ndescription: >-\n  Writing standards and a checklist.\n  Use when revising docs.",
    )
    assert parsed["description"] == "Writing standards and a checklist. Use when revising docs."
    assert parsed["name"] == "my-skill"


def test_parse_skill_keeps_literal_block_scalar_newlines(tmp_path):
    parsed = _parse(tmp_path, "name: my-skill\ndescription: |-\n  first line\n  second line")
    assert parsed["description"] == "first line\nsecond line"


def test_parse_skill_does_not_truncate_multiline_quoted_description(tmp_path):
    """A quoted scalar spanning lines was silently cut at the first line."""
    parsed = _parse(tmp_path, 'name: my-skill\ndescription: "first part\n  second part"')
    assert parsed["description"] == "first part second part"


def test_parse_skill_falls_back_to_defaults_on_malformed_frontmatter(tmp_path):
    """Unparseable YAML must degrade to the directory name and an empty description."""
    parsed = _parse(tmp_path, "name: [unclosed\ndescription: broken")
    assert parsed["name"] == "my-skill"
    assert parsed["description"] == ""
