# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval datasets must not grade the no-skill baseline on using the skill it does not have."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3 import generate_dataset
from skillevaluator.tier3.dataset_utils import (
    detect_dataset_format,
    load_dataset_entries_with_format,
    normalize_dataset_entries,
)
from skillevaluator.tier3.evals_spec import (
    EVALS_SPEC,
    _baseline_fairness_warnings,
    validate_skillevaluators,
    validate_tier3_source,
)

_REPO = Path(__file__).resolve().parents[2]


def _entry(**fields) -> dict:
    return {"id": "case-1", "expected_skill": "release-review", **fields}


@pytest.mark.parametrize(
    "prompt",
    [
        "Use the skill at skills/release-review to check manifest.yaml",
        "Read release-review/SKILL.md and then check manifest.yaml",
    ],
)
def test_prompt_pointing_at_skill_files_warns(prompt):
    (warning,) = _baseline_fairness_warnings(_entry(question=prompt), "agentskills")
    assert warning.startswith("prompt points at the skill's files")


@pytest.mark.parametrize(
    "assertion",
    [
        "Read release-review/SKILL.md before acting",
        "The agent read SKILL.md before taking action",
        "The agent read the release-review SKILL.md before executing commands",
        "The agent used the release-review skill",
        "Identified release-review as the right skill",
        "The agent invoked release-review before checking the manifest",
        "The agent activated `release-review`.",
    ],
)
def test_assertion_grading_skill_use_warns(assertion):
    entry = _entry(question="Check manifest.yaml for blockers", expected_behavior=[assertion])
    (warning,) = _baseline_fairness_warnings(entry, "agentskills")
    assert warning.startswith("assertions grades skill use")


def test_expected_output_grading_skill_use_warns():
    entry = _entry(question="Check manifest.yaml", ground_truth="The agent used release-review and found 2 blockers")
    (warning,) = _baseline_fairness_warnings(entry, "legacy")
    assert warning.startswith("ground_truth grades skill use")


def test_one_warning_per_entry_for_graded_text():
    entry = _entry(
        question="Check manifest.yaml",
        ground_truth="The agent used the release-review skill",
        expected_behavior=["The agent read SKILL.md", "The agent used release-review."],
    )
    (warning,) = _baseline_fairness_warnings(entry, "agentskills")
    assert warning.startswith("expected_output grades skill use")


@pytest.mark.parametrize(
    "entry",
    [
        _entry(
            question="Use release-review to check manifest.yaml",
            ground_truth="Two release blockers: missing SBOM and an unsigned image",
            expected_behavior=["Lists both blockers", "Names the file each blocker comes from"],
        ),
        _entry(expected_skill=None, question="What is semantic versioning?", expected_behavior=["Explains it"]),
    ],
)
def test_outcome_entries_do_not_warn(entry):
    assert _baseline_fairness_warnings(entry, "agentskills") == []


@pytest.mark.parametrize(
    ("entry", "dataset_format"),
    [
        # A skill named after the CLI it wraps: the baseline can run the same CLI.
        (
            {
                "expected_skill": "glab",
                "question": "List my open merge requests in group foo",
                "ground_truth": "The open MRs, listed using glab mr list",
                "expected_behavior": ["The agent ran glab mr list for group foo"],
            },
            "legacy",
        ),
        (
            {
                "expected_skill": "kubectl",
                "question": "Which pods are crash-looping in namespace web?",
                "ground_truth": "The crash-looping pods, found by using kubectl get pods",
                "expected_behavior": ["The agent used kubectl get pods -n web"],
            },
            "legacy",
        ),
        # A skill named after a file format: "the input PDF" is the task's file.
        (
            {
                "expected_skill": "pdf",
                "question": "What is the invoice total in invoice.pdf?",
                "expected_behavior": ["Reads the input PDF and returns the invoice total"],
            },
            "agentskills",
        ),
        # "the <format>" names the task's file, not the skill.
        (
            {
                "expected_skill": "pdf",
                "question": "What is the invoice total in invoice.pdf?",
                "expected_behavior": ["Loads the PDF and returns the invoice total"],
            },
            "agentskills",
        ),
        (
            {
                "expected_skill": "csv",
                "question": "Total the revenue in sales.csv by region",
                "expected_behavior": ["The agent loaded the CSV and grouped revenue by region"],
            },
            "agentskills",
        ),
        # A skill whose task output, or input, is a SKILL.md.
        (
            {
                "expected_skill": "skill-creator",
                "question": "Write a SKILL.md for my csv-cleaner script",
                "ground_truth": "A SKILL.md with valid frontmatter",
                "expected_behavior": ["Writes skills/csv-cleaner/SKILL.md with a name and description"],
            },
            "legacy",
        ),
        (
            {
                "expected_skill": "skill-linter",
                "question": "Read SKILL.md in ./my-skill and fix its frontmatter",
                "expected_behavior": ["Opens SKILL.md in ./my-skill", "The fixed SKILL.md has valid frontmatter"],
            },
            "legacy",
        ),
    ],
    ids=[
        "cli-named",
        "cli-named-with-args",
        "format-named",
        "format-named-the-pdf",
        "format-named-the-csv",
        "authoring",
        "reviewing",
    ],
)
def test_task_text_that_shares_the_skill_name_or_file_does_not_warn(entry, dataset_format):
    assert _baseline_fairness_warnings({"id": "case-1", **entry}, dataset_format) == []


@pytest.mark.parametrize(
    ("skill", "assertion"),
    [("glab", "The agent used glab to list MRs"), ("git", "The agent used git to commit the change")],
)
def test_used_name_then_connective_warns_even_for_cli_named_skill(skill, assertion):
    # Only CLI arguments or a noun after the name exempt it, so "used <name> to ..."
    # still warns for a CLI-named skill.
    entry = {"id": "case-1", "expected_skill": skill, "question": "q", "expected_behavior": [assertion]}
    (warning,) = _baseline_fairness_warnings(entry, "agentskills")
    assert warning.startswith("assertions grades skill use")


@pytest.mark.parametrize(
    ("field", "entry"),
    [
        ("prompt", {"question": "Open SKILL.md and fix its frontmatter"}),
        (
            "assertions",
            {
                "question": "Fix the frontmatter of the SKILL.md in ./my-skill",
                "expected_behavior": ["The agent read the SKILL.md and reported the missing description"],
            },
        ),
    ],
    ids=["prompt", "assertion"],
)
def test_bare_read_skill_md_warns_even_for_skill_md_editing_skill(field, entry):
    # A SKILL.md is exempt only when the task writes it or the text names its
    # location, so a bare "read/open SKILL.md" still means the skill's own instructions.
    (warning,) = _baseline_fairness_warnings({"id": "case-1", "expected_skill": "skill-linter", **entry}, "agentskills")
    assert warning.startswith(f"{field} ")
    assert "SKILL.md" in warning


def test_spec_example_dataset_is_fair_to_the_baseline(tmp_path: Path):
    example = next(spec for spec in EVALS_SPEC if spec.rel_path == "evals/evals.json").example
    path = tmp_path / "evals.json"
    path.write_text(example, encoding="utf-8")
    entries, dataset_format = load_dataset_entries_with_format(path)

    assert entries
    for entry in entries:
        assert _baseline_fairness_warnings(entry, dataset_format) == []


# Pages whose dataset examples users and agents copy.
_DATASET_DOC_PAGES = ["docs/eval-datasets.mdx", "skills/skillevaluator/references/tier3-datasets.md"]


@pytest.mark.parametrize("page", _DATASET_DOC_PAGES)
def test_documented_dataset_examples_are_fair_to_the_baseline(page):
    checked = 0
    text = (_REPO / page).read_text(encoding="utf-8")
    for block in re.findall(r"```json[^\n]*\n(.*?)```", text, re.DOTALL):
        try:
            data = json.loads(block)
            entries = normalize_dataset_entries(data)
        except (json.JSONDecodeError, ValueError):
            continue
        for entry in entries:
            if entry.get("id") and entry.get("question"):
                checked += 1
                assert _baseline_fairness_warnings(entry, detect_dataset_format(data)) == [], (page, entry["id"])
    assert checked, f"no dataset examples found in {page}"


def test_documented_eval_md_behaviors_are_fair_to_the_baseline():
    # EVAL.md `## Behaviors` become expected_behavior entries, which the
    # baseline is graded on, so they must not ask for SKILL.md or skill scripts.
    text = (_REPO / "docs" / "eval-datasets.mdx").read_text(encoding="utf-8")
    behaviors = []
    for block in re.findall(r"```markdown[^\n]*\n(.*?)```", text, re.DOTALL):
        if section := re.search(r"^## Behaviors\n(.*?)(?:^## |\Z)", block, re.DOTALL | re.MULTILINE):
            behaviors += re.findall(r"^- (.+)$", section.group(1), re.MULTILINE)

    assert behaviors, "no EVAL.md behaviors found in docs/eval-datasets.mdx"
    for behavior in behaviors:
        assert not re.search(r"\bSKILL\.md\b|\b[\w-]+\.(?:py|sh)\b", behavior), behavior


def test_spec_guidance_names_the_command_that_reports_fairness_warnings():
    notes = next(spec for spec in EVALS_SPEC if spec.rel_path == "evals/evals.json").validation_notes
    (note,) = [note for note in notes if "no-skill baseline" in note]
    command = re.search(r"`skillevaluator ([\w -]+)`", note)

    assert command is not None, note
    result = CliRunner().invoke(cli, [*command.group(1).split(), "--help"])
    assert result.exit_code == 0, result.output
    assert "--strict" in result.output


def _write_release_review_skill(tmp_path: Path, assertions: list[str]) -> Path:
    skill = tmp_path / "release-review"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: release-review\ndescription: Review releases\n---\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "release-review",
                "evals": [
                    {
                        "id": "release-review-001",
                        "prompt": "Review manifest.yaml for release blockers",
                        "expected_output": "Two release blockers: missing SBOM and an unsigned image",
                        "assertions": assertions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return skill


def test_validate_evals_reports_fairness_warnings(tmp_path: Path):
    skill = _write_release_review_skill(tmp_path, ["Read release-review/SKILL.md"])
    dataset = skill / "evals" / "evals.json"
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload["evals"][0]["prompt"] = "Use the skill at skills/release-review to review manifest.yaml"
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    warnings = [result for result in validate_skillevaluators(skill) if result.status == "warning"]

    assert {result.path for result in warnings} == {"entry release-review-001"}
    assert any("points at the skill's files" in result.message for result in warnings)
    assert any("grades skill use" in result.message for result in warnings)


@pytest.mark.parametrize(("strict", "exit_code"), [(True, 1), (False, 0)], ids=["strict", "default"])
def test_tier3_validate_cli_fails_on_fairness_warning_only_with_strict(tmp_path: Path, strict: bool, exit_code: int):
    skill = _write_release_review_skill(tmp_path, ["The agent read SKILL.md before reviewing", "Lists both blockers"])
    args = ["tier3", "validate", str(skill), "--json", *(["--strict"] if strict else [])]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == exit_code, result.output
    payload = json.loads(result.output)
    fairness = [item for item in payload if item["path"] == "entry release-review-001"]
    assert [item["status"] for item in fairness] == ["warning"]
    assert fairness[0]["message"].startswith("assertions grades skill use ('read SKILL.md')")
    assert not [item for item in payload if item["status"] == "error"]


def test_tier3_validate_strict_passes_an_outcome_dataset(tmp_path: Path):
    skill = _write_release_review_skill(tmp_path, ["Lists both blockers", "Names the file each blocker comes from"])

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])

    assert result.exit_code == 0, result.output


def test_fairness_warnings_do_not_block_the_tier3_source_check(tmp_path: Path):
    # Runs, autopilot and `validate` gate on errors from this check, never on warnings.
    skill = _write_release_review_skill(tmp_path, ["The agent used the release-review skill"])

    source, checks = validate_tier3_source(skill)

    assert source == "evals_json"
    assert [check.status for check in checks if check.path == "entry release-review-001"] == ["warning"]
    assert not [check for check in checks if check.status in {"error", "missing"}]


def _skill(**overrides) -> dict:
    return {
        "name": "release-review",
        "description": "Reviews release manifests for policy problems.",
        "scripts": ["scripts/review.py"],
        "content": "",
        "eval_prompt": "",
        **overrides,
    }


@pytest.mark.parametrize(
    "eval_prompt",
    ["", "## Negative Cases\n- What is the capital of Peru?"],
    ids=["no-guidance", "with-negative"],
)
@pytest.mark.parametrize("generate", [generate_dataset._generate_simple, generate_dataset._generate_full])
def test_template_generator_cases_are_fair_to_the_baseline(generate, eval_prompt):
    cases = generate(_skill(eval_prompt=eval_prompt))

    assert cases
    for case in cases:
        assert _baseline_fairness_warnings(case, "legacy") == [], case
        assert "SKILL.md" not in json.dumps(case)
        assert "review.py" not in json.dumps([case["ground_truth"], case["expected_behavior"]])


def test_template_positive_cases_grade_the_task_outcome():
    cases = generate_dataset._generate_full(_skill())

    assert [case["id"] for case in cases] == ["release-review-001", "release-review-002", "release-review-003"]
    for case in cases:
        assert case["expected_skill"] == "release-review"
        assert case["expected_script"] == "scripts/review.py"
        assert "reviews release manifests for policy problems" in case["ground_truth"]
        assert case["expected_behavior"][0] == (
            "The agent produced concrete results for the request "
            "(reviews release manifests for policy problems), not only a plan"
        )
        assert case["expected_behavior"][-1] == generate_dataset.SECURITY_BEHAVIOR


def test_template_cases_keep_developer_behaviors():
    skill = _skill(eval_prompt="## Behaviors\n- The agent lists every blocker with its manifest line")

    (case,) = generate_dataset._generate_simple(skill)

    assert "The agent lists every blocker with its manifest line" in case["expected_behavior"]
    assert case["expected_behavior"][-1] == generate_dataset.SECURITY_BEHAVIOR


def test_template_negative_case_is_unchanged():
    without_guidance = generate_dataset._generate_full(_skill())
    with_guidance = generate_dataset._generate_full(
        _skill(eval_prompt="## Negative Cases\n- What is the capital of Peru?")
    )

    assert not [case for case in without_guidance if case["id"].endswith("-neg-001")]
    assert with_guidance[-1] == {
        "id": "release-review-neg-001",
        "question": "What is the capital of Peru?",
        "expected_skill": None,
        "expected_script": None,
        "ground_truth": "The agent answered an unrelated question without loading or applying this skill",
        "expected_behavior": [
            "The agent responded without reading or applying this skill",
            "The agent did not invoke this skill's tools or scripts",
            generate_dataset.SECURITY_BEHAVIOR,
        ],
    }


def test_autopilot_template_case_does_not_grade_skill_use(tmp_path: Path):
    skill = tmp_path / "release-review"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: release-review\ndescription: Reviews release manifests for policy problems.\n---\n# Body\n",
        encoding="utf-8",
    )
    (skill / "scripts" / "review.py").write_text("print('ok')\n", encoding="utf-8")

    output = generate_dataset.generate_one_case(skill, use_llm=False)

    (case,) = json.loads(output.read_text(encoding="utf-8"))["evals"]
    graded = json.dumps([case["expected_output"], case["assertions"]])
    assert not re.search(r"SKILL\.md|review\.py|used the|identified", graded), graded
    assert case["expected_script"] == "review.py"
    assert not [result for result in validate_skillevaluators(skill) if result.path == f"entry {case['id']}"], (
        "generated starter case should pass the fairness check"
    )


@pytest.fixture
def llm_prompts(monkeypatch) -> list[str]:
    """Stub the public provider and capture every dataset prompt it is sent."""
    from skillevaluator import provider_config
    from skillevaluator.inference.client import LLMClient

    prompts: list[str] = []

    def _completions(_self, _system_prompt: str, user_prompt: str, **_kwargs) -> str:
        prompts.append(user_prompt)
        return "[]"

    monkeypatch.setattr(
        provider_config, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai", model="test-model")
    )
    monkeypatch.setattr(LLMClient, "completions", _completions)
    return prompts


@pytest.mark.parametrize("full", [False, True], ids=["simple", "full"])
def test_llm_generation_prompt_examples_are_fair_to_the_baseline(llm_prompts, full):
    asyncio.run(generate_dataset._generate_with_llm(_skill(), full=full))

    (prompt,) = llm_prompts
    good_examples = re.findall(r'GOOD: "([^"]+)"', prompt)
    assert good_examples
    for example in good_examples:
        entry = _entry(ground_truth=example, expected_behavior=[example])
        assert _baseline_fairness_warnings(entry, "legacy") == [], example
        assert not re.search(r"release-review|SKILL\.md|review\.py", example), example
    assert "skill_execution" in prompt
    assert "observable checks on the task outcome" in prompt


def test_llm_refine_prompt_does_not_ask_for_skill_use_behaviors(llm_prompts):
    skill = _skill(scripts=["review.py"])
    cases = generate_dataset._generate_simple(skill)

    asyncio.run(generate_dataset._refine_with_llm(skill, cases, {cases[0]["id"]: _SKILL_USE_TRAJECTORY}))

    (prompt,) = llm_prompts
    instructions = prompt.split("REFINEMENT INSTRUCTIONS", 1)[1]
    assert "skill_execution" in instructions
    assert not re.search(r'"The agent executed\s+[^"]*\.py', instructions)
    assert not re.search(r"\b\w+\.py\b", instructions)
    for example in re.findall(r'"(The agent[^"]*)"', instructions):
        assert _baseline_fairness_warnings(_entry(expected_behavior=[example]), "legacy") == [], example


_SKILL_USE_TRAJECTORY = {
    "steps": [
        {
            "source": "agent",
            "tool_calls": [{"function_name": "read", "arguments": {"path": "/skills/release-review/SKILL.md"}}],
        },
        {
            "source": "agent",
            "tool_calls": [
                {"function_name": "read", "arguments": {"path": "/skills/release-review/references/policy.md"}}
            ],
        },
        {
            "source": "agent",
            "tool_calls": [
                {"function_name": "bash", "arguments": {"command": "python scripts/review.py --manifest release.yaml"}}
            ],
        },
        {"source": "agent", "tool_calls": [{"function_name": "read", "arguments": {"path": "/work/release.yaml"}}]},
        {
            "source": "agent",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "grep -n image release.yaml"}}],
        },
        {
            "source": "agent",
            "message": "The manifest has two blockers: a missing changelog entry and an unsigned image.",
        },
    ]
}


@pytest.mark.parametrize("with_skill_context", [True, False], ids=["skill", "case-only"])
def test_template_refine_drops_skill_use_and_keeps_task_work(with_skill_context):
    skill = _skill(scripts=["review.py"])
    cases = generate_dataset._generate_simple(skill)

    (refined,) = generate_dataset._refine_from_trajectory_template(
        cases, {cases[0]["id"]: _SKILL_USE_TRAJECTORY}, skill=skill if with_skill_context else None
    )

    behaviors = refined["expected_behavior"]
    assert behaviors[:2] == ["The agent read release.yaml", "The agent executed: grep -n image release.yaml"]
    assert not [item for item in behaviors if re.search(r"SKILL\.md|review\.py|policy\.md", item)]
    assert refined["ground_truth"].startswith("The manifest has two blockers")
    assert _baseline_fairness_warnings(refined, "legacy") == []


@pytest.mark.parametrize(
    ("text", "uses"),
    [
        ("/root/.claude/skills/release-review/SKILL.md", True),
        ("/workspace/skills/release-review/references/policy.md", True),
        ("cat ~/.agents/skills/release-review/notes.txt", True),
        ("python3 ./review.py --manifest m.yaml", True),
        ("/workspace/skills/release-review-extra/notes.txt", False),
        ("python3 preview.py", False),
        ("cat review.python.txt", False),
        ("grep -n image release.yaml", False),
    ],
)
def test_uses_skill_files_matches_whole_names_only(text, uses):
    assert generate_dataset._uses_skill_files(text, "release-review", ["scripts/review.py", None]) is uses
