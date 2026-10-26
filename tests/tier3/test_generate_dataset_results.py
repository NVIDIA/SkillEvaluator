# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skillevaluator.tier3 import generate_dataset
from skillevaluator.tier3.generate_dataset import (
    _discover_trajectories,
    _generate_full,
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

    assert _discover_trajectories(skill)["case-001"][0]["execution_status"] == "incomplete"


def test_discover_trajectories_opencode_txt_fallback(tmp_path):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    results_root = tmp_path / "results"
    skill_results = results_root / "my-skill"
    run_id = "20260709_120000"
    run_dir = skill_results / run_id
    trial = run_dir / "opencode" / "with-skill" / "trials" / "case-001"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"finished_at": "2026-10-24T00:00:00Z"}))
    (trial / "opencode.txt").write_text(
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"c1",'
        '"state":{"status":"completed","input":{"command":"echo hi"},"output":"hi"}}}\n',
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (skill_results / "latest").symlink_to(run_id)

    trajectories = _discover_trajectories(skill, results_dir=results_root)
    assert "case-001" in trajectories
    assert "echo hi" in trajectories["case-001"][0]["summary"]


def test_discover_trajectories_opencode_maps_harbor_folder_via_result_metadata(tmp_path):
    skill = tmp_path / "demo"
    skill.mkdir()
    results_root = tmp_path / "results"
    run_id = "20260709_120000"
    run_dir = results_root / "demo" / run_id
    trial_folder = "case-001__LRZctSP"
    trial = run_dir / "opencode" / "with-skill" / "trials" / trial_folder
    trial.mkdir(parents=True)
    trajectory = {"steps": [{"tool_calls": [{"tool": "Read"}]}]}
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    trial.joinpath("result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_folder,
                "config": {"task": {"path": "tasks/case-001"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (results_root / "demo" / "latest").symlink_to(run_id)

    found = _discover_trajectories(skill, results_dir=results_root)
    assert list(found) == ["case-001"]
    assert found["case-001"][0]["execution_status"] == "incomplete"


def test_discover_trajectories_maps_harbor_trial_folder_to_case_id(tmp_path, monkeypatch):
    """Harbor persists trials as ``{case_id}__{suffix}``; refine looks up by case id."""
    skill = tmp_path / "demo"
    skill.mkdir()
    results_root = tmp_path / "results"
    run_id = "20260709_120000"
    run_dir = results_root / "demo" / run_id
    trial = run_dir / "claude-code" / "with-skill" / "trials" / "demo-001__Lmi47iy"
    trial.mkdir(parents=True)
    trajectory = {"steps": [{"tool_calls": [{"tool": "Read"}]}]}
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    trial.joinpath("result.json").write_text(
        json.dumps(
            {
                "trial_name": "demo-001__Lmi47iy",
                "config": {"task": {"path": "tasks/demo-001"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (results_root / "demo" / "latest").symlink_to(run_id)

    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(results_root))

    found = _discover_trajectories(skill)
    assert "demo-001" in found
    assert "demo-001__Lmi47iy" not in found
    assert found["demo-001"][0]["execution_status"] == "incomplete"


def _write_results_trial(
    tmp_path: Path,
    *,
    skill_name: str,
    trial_folder: str,
    trajectory: dict[str, object],
    reward: dict[str, object] | None = None,
) -> Path:
    skill = tmp_path / skill_name
    skill.mkdir(exist_ok=True)
    results_root = tmp_path / "results"
    run_id = "20260709_120000"
    run_dir = results_root / skill_name / run_id
    trial = run_dir / "claude-code" / "with-skill" / "trials" / trial_folder
    trial.mkdir(parents=True)
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    if reward is not None:
        trial.joinpath("reward.json").write_text(json.dumps(reward), encoding="utf-8")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (results_root / skill_name / "latest").symlink_to(run_id)
    return skill


def test_discover_trajectories_prefers_reward_entry_id_over_folder_prefix(tmp_path, monkeypatch):
    """Stop-on-pass folders must not collapse to the job prefix when entry_id is present."""
    trajectory = {"steps": [{"tool_calls": [{"tool": "Read"}]}]}
    skill = _write_results_trial(
        tmp_path,
        skill_name="demo",
        trial_folder="harbor-job__demo-001__Lmi47iy",
        trajectory=trajectory,
        reward={"entry_id": "demo-001", "overall": 1.0},
    )
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    found = _discover_trajectories(skill)
    assert list(found) == ["demo-001"]
    assert found["demo-001"][0]["execution_status"] == "incomplete"


def test_discover_trajectories_keeps_distinct_case_ids_with_double_underscore(tmp_path, monkeypatch):
    """Case ids containing ``__`` must not collapse to a shared prefix without metadata."""
    trajectory_a = {"steps": [{"message": "a"}]}
    trajectory_b = {"steps": [{"message": "b"}]}
    skill = tmp_path / "demo"
    skill.mkdir()
    results_root = tmp_path / "results"
    run_id = "20260709_120000"
    run_dir = results_root / "demo" / run_id
    trials_dir = run_dir / "claude-code" / "with-skill" / "trials"
    for folder, traj, entry_id in (
        ("case__one", trajectory_a, "case__one"),
        ("case__two", trajectory_b, "case__two"),
    ):
        trial = trials_dir / folder
        trial.mkdir(parents=True)
        trial.joinpath("trajectory.json").write_text(json.dumps(traj), encoding="utf-8")
        trial.joinpath("reward.json").write_text(json.dumps({"entry_id": entry_id}), encoding="utf-8")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (results_root / "demo" / "latest").symlink_to(run_id)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(results_root))

    found = _discover_trajectories(skill)
    assert set(found) == {"case__one", "case__two"}
    assert found["case__one"][0]["execution_status"] == "incomplete"
    assert found["case__two"][0]["execution_status"] == "incomplete"


def test_discover_trajectories_ambiguous_folder_without_reward_uses_full_name(tmp_path, monkeypatch):
    """Without reward metadata, ambiguous ``__`` folders keep the full directory name."""
    trajectory = {"steps": [{"tool_calls": []}]}
    skill = _write_results_trial(
        tmp_path,
        skill_name="demo",
        trial_folder="case__one",
        trajectory=trajectory,
    )
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    found = _discover_trajectories(skill)
    assert list(found) == ["case__one"]


def test_discover_trajectories_preserves_versioned_case_id_without_reward(tmp_path, monkeypatch):
    """IDs like ``case__v2`` must not be truncated when only the folder name is present."""
    trajectory = {"steps": [{"tool_calls": []}]}
    skill = _write_results_trial(
        tmp_path,
        skill_name="demo",
        trial_folder="case__v2",
        trajectory=trajectory,
    )
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    found = _discover_trajectories(skill)
    assert list(found) == ["case__v2"]


def test_discover_trajectories_resolves_shortuuid_folder_from_result_json(tmp_path, monkeypatch):
    """All-letter Harbor tails resolve via result.json task metadata, not suffix guessing."""
    trajectory = {"steps": [{"tool_calls": []}]}
    skill = tmp_path / "demo"
    skill.mkdir()
    results_root = tmp_path / "results"
    run_id = "20260709_120000"
    run_dir = results_root / "demo" / run_id
    trial_folder = "case-001__LRZctSP"
    trial = run_dir / "claude-code" / "with-skill" / "trials" / trial_folder
    trial.mkdir(parents=True)
    trial.joinpath("trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    trial.joinpath("result.json").write_text(
        json.dumps(
            {
                "trial_name": trial_folder,
                "config": {"task": {"path": "tasks/case-001"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (results_root / "demo" / "latest").symlink_to(run_id)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(results_root))

    found = _discover_trajectories(skill)
    assert list(found) == ["case-001"]


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

    assert _discover_trajectories(skill, results_dir=cli_root)["case-001"][0]["source"] == str(trial)


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


@pytest.mark.parametrize(("extra_args", "expected_cases"), [([], 1), (["--full"], 3)])
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


def test_main_full_with_author_negative_writes_four_cases(tmp_path):
    skill = tmp_path / "my-skill"
    evals = skill / "evals"
    evals.mkdir(parents=True)
    (evals / "EVAL.md").write_text("## Negative Cases\n- What is the capital of Peru?\n", encoding="utf-8")
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Does useful work\n---\n",
        encoding="utf-8",
    )

    result = generate_dataset.main([str(skill), "--no-llm", "--full"])

    assert result.cases_count == 4
    assert any(case["id"].endswith("-neg-001") for case in result.dataset["evals"])


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

    def evaluate_draft(**_kwargs):
        staged = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
        assert staged["evals"][0]["authoring_status"] == "draft_requires_oracle_review"
        return {"agents": {}}

    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.runner.run_harbor_eval",
        evaluate_draft,
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
    assert "authoring_status" not in cases[0]


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


def test_no_llm_negative_case_does_not_name_the_skill():
    """Author-provided negatives must stay off-skill and must not name the skill."""
    skill = {
        "name": "pdf-extractor",
        "description": "Extracts tables from PDF files",
        "scripts": [],
        "eval_prompt": "## Negative Cases\n- What is the capital of Peru?",
    }
    cases = _generate_full(skill)
    negative = next(c for c in cases if c["id"] == "pdf-extractor-neg-001")
    assert negative["expected_skill"] is None
    assert "pdf-extractor" not in negative["question"]
    assert "pdf-extractor" not in negative["ground_truth"]
    for behavior in negative["expected_behavior"]:
        assert "pdf-extractor" not in behavior
    assert "directly answers the unrelated question" in negative["expected_behavior"][0]
    domain = {"pdf", "extractor", "extracts", "tables"}
    question_tokens = set(re.findall(r"[a-z0-9]+", negative["question"].lower()))
    assert not domain & question_tokens


def test_no_llm_negative_case_omits_planning_skills_without_author_negative():
    """Planning skills omit the negative bucket unless eval guidance supplies one."""
    for skill in (
        {
            "name": "errand-planner",
            "description": "Organizes weekend errands efficiently in a new city",
            "scripts": [],
            "eval_prompt": "",
        },
        {
            "name": "day-planner",
            "description": "Plans grocery runs and appointments across a busy week",
            "scripts": [],
            "eval_prompt": "",
        },
    ):
        cases = _generate_full(skill)
        assert all(not c["id"].endswith("-neg-001") for c in cases)
        assert len(cases) == 3


def test_no_llm_negative_case_uses_author_provided_negative_section():
    skill = {
        "name": "errand-planner",
        "description": "Organizes weekend errands efficiently in a new city",
        "scripts": [],
        "eval_prompt": "## Negative Cases\n- What is the capital of Peru?",
    }
    cases = _generate_full(skill)
    negative = next(c for c in cases if c["id"] == "errand-planner-neg-001")
    assert negative["expected_skill"] is None
    assert negative["question"] == "What is the capital of Peru?"


def test_no_llm_negative_case_omits_without_author_negative():
    """Template mode omits the negative bucket unless eval guidance supplies one."""
    for skill in (
        {
            "name": "media-transcoder",
            "description": "Changes sound recordings between lossless formats while retaining tags",
            "scripts": [],
            "eval_prompt": "",
        },
        {
            "name": "music-reencoder",
            "description": "Changes songs between codecs while keeping tags",
            "scripts": [],
            "eval_prompt": "",
        },
    ):
        cases = _generate_full(skill)
        assert all(not c["id"].endswith("-neg-001") for c in cases)
        assert len(cases) == 3


def test_parse_skill_includes_tools_dir_scripts(tmp_path):
    skill = tmp_path / "tools-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: tools-skill\ndescription: Spec-compliant executables live in tools/.\n---\n# x\n",
        encoding="utf-8",
    )
    tools = skill / "tools"
    tools.mkdir()
    (tools / "run.py").write_text("print('hello')\n", encoding="utf-8")
    parsed = generate_dataset._parse_skill(skill)
    assert parsed["scripts"] == ["run.py"]


_BLOCK_HARBOR_AND_RESOLVE_CASE = """
import importlib.abc
import json
import sys
from pathlib import Path


class _BlockHarbor(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "harbor" or name.startswith("harbor."):
            raise ImportError(name)
        return None


sys.meta_path.insert(0, _BlockHarbor())
from skillevaluator.tier3 import generate_dataset

case_id = generate_dataset._case_id_from_trial_dir(Path(sys.argv[1]))
print(json.dumps({"case_id": case_id, "harbor_loaded": "harbor" in sys.modules}))
"""


def test_refine_resolves_result_case_ids_without_the_tier3_extra(tmp_path: Path) -> None:
    import subprocess

    trial = tmp_path / "case-007__abc1234"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"task_id": {"path": str(tmp_path / "tasks" / "case-007")}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-c", _BLOCK_HARBOR_AND_RESOLVE_CASE, str(trial)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"case_id": "case-007", "harbor_loaded": False}


_CASE = {
    "id": "demo-001",
    "question": "Return zero.",
    "ground_truth": "0",
    "expected_behavior": ["Returns numeric zero."],
}


def test_refine_preserves_existing_dataset_bytes_and_contract(tmp_path, monkeypatch):
    import hashlib

    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n")
    original = {
        "skill_name": "demo",
        "owner": "author",
        "evals": [
            {
                "id": "authored-case",
                "prompt": "Return just JSON with false and zero.",
                "expected_output": '{"enabled": false, "count": 0}',
                "assertions": ["The entire response is one JSON object."],
                "split": "holdout",
                "outcome_contract": {"type": "json"},
            }
        ],
    }
    path = skill / "evals/evals.json"
    frozen = json.dumps(original).encode()
    path.write_bytes(frozen)
    observations = {"authored-case": [{"execution_status": "completed", "summary": "Agent answered wrong prose."}]}
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_a, **_kw: observations)
    monkeypatch.setattr(generate_dataset, "_generate_simple", lambda *_a: pytest.fail("must not regenerate"))
    result = generate_dataset.main([str(skill), "--refine", "--no-llm"])
    assert path.read_bytes() == frozen
    assert result.path == skill / "evals/refinement.json"
    saved = json.loads(result.path.read_text())
    assert saved["frozen_dataset"]["sha256"] == hashlib.sha256(frozen).hexdigest()
    assert saved["authoring_status"] == "observations_and_unvalidated_proposals"
    assert {
        k: v for k, v in saved["evals"][0].items() if k not in {"trajectory_observations", "expected_skill"}
    } == original["evals"][0]
    assert saved["evals"][0]["trajectory_observations"] == observations["authored-case"]


def test_refine_does_not_launch_when_explicit_results_have_no_trajectories(tmp_path, monkeypatch):
    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n")
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        generate_dataset, "_run_agent_collect_trajectories", lambda *_a, **_kw: pytest.fail("must not run")
    )
    generate_dataset.main([str(skill), "--refine", "--no-llm", "--from-results", str(tmp_path)])


def test_discover_real_trial_identity_arms_and_incomplete_status(tmp_path):
    run = tmp_path / "run"
    for arm, suffix, status in [
        ("with-skill", "dH3tNDP", "completed"),
        ("with-skill", "FwQkMp8", "failed"),
        ("without-skill", "EaGaoh9", "completed"),
        ("without-skill", "Sf4bcsQ", "incomplete"),
    ]:
        trial = run / "opencode" / arm / "trials" / f"case-with__separator__{suffix}"
        trial.mkdir(parents=True)
        result = {
            "task_name": "nvidia/skillevaluator-case-with__separator",
            "task_id": {"path": "/work/tasks/case-with__separator"},
            "finished_at": "2026-10-24T00:00:00Z" if status != "incomplete" else None,
            "exception_info": {"exception_type": "AgentTimeoutError"} if status == "failed" else None,
        }
        (trial / "result.json").write_text(json.dumps(result))
        (trial / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "source": "agent",
                            "tool_calls": [
                                {"function_name": "read", "arguments": {"filePath": "/workspace/input.json"}}
                            ],
                        },
                        {"source": "agent", "message": "This observed answer is not an independently checked oracle."},
                    ]
                }
            )
        )
    observations = _discover_trajectories(tmp_path, from_results=str(run))
    assert set(observations) == {"case-with__separator"}
    rows = observations["case-with__separator"]
    assert len(rows) == 4
    assert {row["arm"] for row in rows} == {"with-skill", "without-skill"}
    assert {row["agent"] for row in rows} == {"opencode"}
    assert sorted(row["execution_status"] for row in rows) == ["completed", "completed", "failed", "incomplete"]
    for row in rows:
        if row["execution_status"] == "completed":
            assert "/workspace/input.json" in row["summary"]
        else:
            assert "summary" not in row


def test_context_captures_full_skill_references_root_scripts_and_bounds(tmp_path):
    import hashlib

    (tmp_path / "references").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "assets/specs").mkdir(parents=True)
    (tmp_path / "SKILL.md").write_text(
        "---\nname: demo\ndescription: >\n  Check actual\n  outcomes.\n---\n"
        + "x" * 4100
        + "\nSee references/vault.md, check_headers.py and assets/specs/metric.json.\nauth.py is interactive."
    )
    reference = b"Only KVv2 is supported.\n"
    (tmp_path / "references/vault.md").write_bytes(reference)
    (tmp_path / "check_headers.py").write_text("compliant = failures == 0 and errors == 0\n")
    (tmp_path / "scripts/auth.py").write_text("print('interactive')\n")
    (tmp_path / "assets/specs/metric.json").write_text('{"metrics": [{"name": "exact-match"}]}\n')
    (tmp_path / "assets/specs/unlinked.json").write_text('{"not_context": true}\n')
    (tmp_path / "references/large.md").write_text("x" * (generate_dataset._CONTEXT_FILE_BYTES + 1))
    context = generate_dataset._parse_skill(tmp_path)
    assert context["description"] == "Check actual outcomes."
    assert len(context["content"]) > 4000
    assert "check_headers.py" in context["scripts"]
    assert "auth.py" in context["interactive_scripts"]
    assert "Only KVv2" in context["supporting_content"]
    assert "compliant =" in context["supporting_content"]
    assert '"name": "exact-match"' in context["supporting_content"]
    assert "not_context" not in context["supporting_content"]
    assert any(row["path"] == "assets/specs/metric.json" for row in context["source_files"])
    assert (
        next(row for row in context["source_files"] if row["path"] == "references/vault.md")["sha256"]
        == hashlib.sha256(reference).hexdigest()
    )
    assert context["omitted_sources"] == [{"path": "references/large.md", "reason": "file size limit exceeded"}]


def test_context_refuses_linked_reference(tmp_path):
    from skillevaluator.utils.secure_fs import SecurePathError

    skill = tmp_path / "skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n")
    (tmp_path / "outside.md").write_text("Must not be sent to the generator.")
    (skill / "references/policy.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(SecurePathError):
        generate_dataset._parse_skill(skill)


@pytest.mark.parametrize(
    "invalid", [[], [{**_CASE, "ground_truth": ""}], [_CASE, _CASE], [{**_CASE, "expected_behavior": [False]}]]
)
def test_invalid_generation_is_not_saved(tmp_path, monkeypatch, invalid):
    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n")
    monkeypatch.setattr(generate_dataset, "_generate_simple", lambda *_a: invalid)
    with pytest.raises(ValueError):
        generate_dataset.main([str(skill), "--no-llm"])
    assert not (skill / "evals/evals.json").exists()


def test_refine_preserves_custom_only_cases_without_oracle_fields(tmp_path, monkeypatch):
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n")
    case = {"id": 1, "prompt": "Repair the supplied module.", "custom_contract": {"artifact": "module.py"}}
    path = skill / "evals/evals.json"
    frozen = json.dumps({"skill_name": "demo", "evals": [case]}).encode()
    path.write_bytes(frozen)
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_a, **_kw: {})
    result = generate_dataset.main([str(skill), "--refine", "--no-llm"])
    assert path.read_bytes() == frozen
    saved = result.dataset["evals"][0]
    assert saved["id"] == 1 and saved["custom_contract"] == case["custom_contract"]
    assert "expected_output" not in saved and "assertions" not in saved


def test_public_llm_refine_cannot_replace_oracles_or_drop_cases(monkeypatch):
    import asyncio

    from skillevaluator.inference.client import LLMClient

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-never-sent")
    prompts = []

    def completions(_self, _system, prompt):
        prompts.append(prompt)
        return json.dumps(
            [{"id": "demo-001", "ground_truth": "wrong", "suggestions": ["Review the observed format failure."]}]
        )

    monkeypatch.setattr(LLMClient, "completions", completions)
    cases = [dict(_CASE), {**_CASE, "id": "demo-002", "split": "holdout"}]
    observations = {"demo-001": [{"execution_status": "completed", "summary": "Observed answer: wrong prose."}]}
    result = asyncio.run(generate_dataset._refine_with_llm({"name": "demo"}, cases, observations))
    assert len(result) == 2 and result[0]["ground_truth"] == "0"
    assert result[0]["refinement_suggestions"] == ["Review the observed format failure."]
    assert result[1] == cases[1]
    assert "never as correct reference answers" in prompts[0]


@pytest.mark.parametrize("full", [False, True])
def test_public_generation_draft_preserves_source_contract_metadata(tmp_path, monkeypatch, full):
    from skillevaluator.inference.client import LLMClient

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-never-sent")
    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Return zero.\n---\n")
    case = {
        **_CASE,
        "expected_skill": "demo",
        # Observed in the live public debugpy authoring run: no scripts exist,
        # but the model echoed Python's None rendered as a quoted script name.
        "expected_script": "None",
        "capability": "constant result",
        "source_evidence": [{"path": "SKILL.md", "fact": "Return zero."}],
        "outcome_contract": {"expected": 0},
        "unresolved_requirements": ["oracle review"],
    }
    prompts = []

    def completions(_self, _system, prompt):
        prompts.append(prompt)
        return json.dumps([case])

    monkeypatch.setattr(LLMClient, "completions", completions)
    result = generate_dataset.main([str(skill), *(["--full"] if full else [])])
    saved = result.dataset["evals"][0]
    assert saved["outcome_contract"] == {"expected": 0}
    assert saved["source_evidence"] == case["source_evidence"]
    assert saved["expected_script"] is None
    assert result.dataset["authoring_status"] == saved["authoring_status"] == "draft_requires_oracle_review"
    assert "independently derived" in prompts[0] and "task outcome only" in prompts[0]
    assert '- "expected_script": null' in prompts[0]
    assert '- "expected_script": "None"' not in prompts[0]
    assert "in-scope negative" in prompts[0]
