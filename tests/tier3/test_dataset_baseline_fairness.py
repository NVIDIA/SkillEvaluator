# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval datasets must not grade the no-skill baseline on using the skill it does not have."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3 import generate_dataset
from skillevaluator.tier3.dataset_utils import load_dataset_entries_with_format, normalize_dataset_entries
from skillevaluator.tier3.evals_spec import (
    EVALS_SPEC,
    _baseline_fairness_warnings,
    validate_skillevaluators,
    validate_tier3_source,
)

_REPO = Path(__file__).resolve().parents[2]


def _entry(**fields) -> dict:
    """A legacy entry (question, ground_truth, expected_behavior)."""
    return {"id": "case-1", "expected_skill": "release-review", **fields}


def _case(skill_name: str = "release-review", **fields) -> dict:
    """An agentskills.io entry (prompt, expected_output, assertions), normalized as the loader does."""
    (entry,) = normalize_dataset_entries({"skill_name": skill_name, "evals": [{"id": "case-1", **fields}]})
    return entry


@pytest.mark.parametrize(
    "prompt",
    [
        "Use the skill at skills/release-review to check manifest.yaml",
        "Read release-review/SKILL.md and then check manifest.yaml",
    ],
)
def test_prompt_pointing_at_skill_files_warns(prompt):
    (warning,) = _baseline_fairness_warnings(_case(prompt=prompt))
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
        "The agent followed the SKILL.md workflow step by step",
        "Follows SKILL.md step by step",
    ],
)
def test_assertion_grading_skill_use_warns(assertion):
    entry = _case(prompt="Check manifest.yaml for blockers", assertions=[assertion])
    (warning,) = _baseline_fairness_warnings(entry)
    assert warning.startswith("assertions grades skill use")


def test_expected_output_grading_skill_use_warns():
    entry = _entry(question="Check manifest.yaml", ground_truth="The agent used release-review and found 2 blockers")
    (warning,) = _baseline_fairness_warnings(entry)
    assert warning.startswith("ground_truth grades skill use")


def test_one_warning_per_entry_for_graded_text():
    entry = _case(
        prompt="Check manifest.yaml",
        expected_output="The agent used the release-review skill",
        assertions=["The agent read SKILL.md", "The agent used release-review."],
    )
    (warning,) = _baseline_fairness_warnings(entry)
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
    assert _baseline_fairness_warnings(entry) == []


@pytest.mark.parametrize(
    "entry",
    [
        # A skill named after the CLI it wraps: the baseline can run the same CLI.
        {
            "expected_skill": "glab",
            "question": "List my open merge requests in group foo",
            "ground_truth": "The open MRs, listed using glab mr list",
            "expected_behavior": ["The agent ran glab mr list for group foo"],
        },
        {
            "expected_skill": "kubectl",
            "question": "Which pods are crash-looping in namespace web?",
            "ground_truth": "The crash-looping pods, found by using kubectl get pods",
            "expected_behavior": ["The agent used kubectl get pods -n web"],
        },
        # A skill named after a file format: "the input PDF" is the task's file.
        {
            "expected_skill": "pdf",
            "question": "What is the invoice total in invoice.pdf?",
            "expected_behavior": ["Reads the input PDF and returns the invoice total"],
        },
        # "the <format>" names the task's file, not the skill.
        {
            "expected_skill": "pdf",
            "question": "What is the invoice total in invoice.pdf?",
            "expected_behavior": ["Loads the PDF and returns the invoice total"],
        },
        {
            "expected_skill": "csv",
            "question": "Total the revenue in sales.csv by region",
            "expected_behavior": ["The agent loaded the CSV and grouped revenue by region"],
        },
        # A skill whose task output, or input, is a SKILL.md.
        {
            "expected_skill": "skill-creator",
            "question": "Write a SKILL.md for my csv-cleaner script",
            "ground_truth": "A SKILL.md with valid frontmatter",
            "expected_behavior": ["Writes skills/csv-cleaner/SKILL.md with a name and description"],
        },
        {
            "expected_skill": "skill-linter",
            "question": "Read SKILL.md in ./my-skill and fix its frontmatter",
            "expected_behavior": ["Opens SKILL.md in ./my-skill", "The fixed SKILL.md has valid frontmatter"],
        },
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
def test_task_text_that_shares_the_skill_name_or_file_does_not_warn(entry):
    assert _baseline_fairness_warnings({"id": "case-1", **entry}) == []


@pytest.mark.parametrize(
    ("skill", "assertion"),
    [("glab", "The agent used glab to list MRs"), ("git", "The agent used git to commit the change")],
)
def test_used_name_then_connective_warns_even_for_cli_named_skill(skill, assertion):
    # Only CLI arguments or a noun after the name exempt it, so "used <name> to ..."
    # still warns for a CLI-named skill.
    entry = _case(skill, prompt="q", assertions=[assertion])
    (warning,) = _baseline_fairness_warnings(entry)
    assert warning.startswith("assertions grades skill use")


@pytest.mark.parametrize(
    ("field", "entry"),
    [
        ("prompt", {"prompt": "Open SKILL.md and fix its frontmatter"}),
        (
            "assertions",
            {
                "prompt": "Fix the frontmatter of the SKILL.md in ./my-skill",
                "assertions": ["The agent read the SKILL.md and reported the missing description"],
            },
        ),
    ],
    ids=["prompt", "assertion"],
)
def test_bare_read_skill_md_warns_even_for_skill_md_editing_skill(field, entry):
    # A SKILL.md is exempt only when the task writes it or the text names its
    # location, so a bare "read/open SKILL.md" still means the skill's own instructions.
    (warning,) = _baseline_fairness_warnings(_case("skill-linter", **entry))
    assert warning.startswith(f"{field} ")
    assert "SKILL.md" in warning


def test_spec_example_dataset_is_fair_to_the_baseline(tmp_path: Path):
    example = next(spec for spec in EVALS_SPEC if spec.rel_path == "evals/evals.json").example
    path = tmp_path / "evals.json"
    path.write_text(example, encoding="utf-8")
    entries, _dataset_format = load_dataset_entries_with_format(path)

    assert entries
    for entry in entries:
        assert _baseline_fairness_warnings(entry) == []


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
                assert _baseline_fairness_warnings(entry) == [], (page, entry["id"])
    assert checked, f"no dataset examples found in {page}"


def test_public_starter_dataset_passes_strict_validate(tmp_path: Path):
    # scripts/demo_starter.py copies this dataset next to the text-analyzer skill, and
    # docs/starter-demo.mdx runs the paired evaluation on it.
    reference = _REPO / "src" / "skillevaluator" / "tier3" / "reference_skills" / "text-analyzer"
    skill = tmp_path / "text-analyzer"
    (skill / "evals" / "files").mkdir(parents=True)
    (skill / "SKILL.md").write_bytes((reference / "SKILL.md").read_bytes())
    shutil.copytree(reference / "scripts", skill / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    starter = _REPO / "examples" / "starter"
    (skill / "evals" / "evals.json").write_bytes((starter / "evals.json").read_bytes())
    (skill / "evals" / "files" / "input.txt").write_bytes((starter / "input.txt").read_bytes())

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])

    assert result.exit_code == 0, result.output


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


def test_all_tier_ci_tier3_dataset_check_is_strict_about_fairness_warnings(tmp_path: Path, monkeypatch):
    # All-tier CI runs its advisory Tier 3 step only on a dataset that passes strict validation.
    # A staged verifier bundle in another test can leave its own secret_redaction module behind;
    # without it, the CI scripts import ci.secret_redaction.
    monkeypatch.delitem(sys.modules, "secret_redaction", raising=False)
    from ci.run_tiers import ConfigurationError, dataset_preflight

    unfair = _write_release_review_skill(tmp_path / "unfair", ["The agent used the release-review skill"])
    fair = _write_release_review_skill(tmp_path / "fair", ["Lists both blockers"])

    with pytest.raises(ConfigurationError, match="strict validation"):
        dataset_preflight(unfair)
    assert "evals/evals.json" in dataset_preflight(fair)


def test_accept_traces_reports_fairness_warnings_on_promoted_cases(tmp_path: Path):
    # Cases promoted from reviewed production traces are validated with the same check.
    from skillevaluator.tier3.trace_import import accept_traces, import_traces

    skill = tmp_path / "release-review"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: release-review\ndescription: Review releases\n---\n", encoding="utf-8")
    export = tmp_path / "export.jsonl"
    export.write_text(
        json.dumps({"session_id": "session-one", "prompt": "Review manifest.yaml for release blockers"}),
        encoding="utf-8",
    )
    review = tmp_path / "review"
    document = import_traces(skill, export, review)
    document["candidates"][0]["review"].update(
        decision="accept",
        expected_output="Two release blockers: missing SBOM and an unsigned image",
        assertions=["The agent read release-review/SKILL.md"],
    )
    (review / "candidates.json").write_text(json.dumps(document), encoding="utf-8")

    result = accept_traces(skill, review)

    (case_id,) = result["added"]
    assert [check.status for check in result["validation"] if check.path == f"entry {case_id}"] == ["warning"]


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
        assert _baseline_fairness_warnings(case, skill_scripts=["review.py"]) == [], case
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


@pytest.mark.parametrize("grounding", ["skill", "intent"])
@pytest.mark.parametrize("generate", [generate_dataset._generate_simple, generate_dataset._generate_full])
def test_template_cases_keep_the_grounding_order_and_stay_fair(generate, grounding):
    # --grounding intent puts the authored EVAL.md behaviors first; the outcome check replaces skill use.
    authored = ["The agent lists every blocker with its manifest line", "The agent names the policy each one breaks"]
    skill = _skill(eval_prompt="## Behaviors\n" + "".join(f"- {item}\n" for item in authored), grounding=grounding)
    outcome = (
        "The agent produced concrete results for the request "
        "(reviews release manifests for policy problems), not only a plan"
    )
    leading = [*authored, outcome] if grounding == "intent" else [outcome, *authored]

    cases = generate(skill)

    assert cases
    for case in cases:
        assert case["expected_behavior"] == [
            *leading,
            generate_dataset._FINAL_RESPONSE_BEHAVIOR,
            generate_dataset.SECURITY_BEHAVIOR,
        ]
        assert _baseline_fairness_warnings(case, skill_scripts=["review.py"]) == []


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
        assert _baseline_fairness_warnings(entry) == [], example
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
        assert _baseline_fairness_warnings(_entry(expected_behavior=[example])) == [], example


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

    # create-eval-dataset --refine passes no authored case ids for cases it generated.
    (refined,) = generate_dataset._refine_from_trajectory_template(
        cases,
        {cases[0]["id"]: _SKILL_USE_TRAJECTORY},
        skill=skill if with_skill_context else None,
        authored_case_ids=set(),
    )

    behaviors = refined["expected_behavior"]
    assert behaviors[:2] == ["The agent read release.yaml", "The agent executed: grep -n image release.yaml"]
    assert not [item for item in behaviors if re.search(r"SKILL\.md|review\.py|policy\.md", item)]
    assert refined["ground_truth"].startswith("The manifest has two blockers")
    assert _baseline_fairness_warnings(refined, skill_scripts=["review.py"]) == []


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


# --- Skill scripts, negation and wording the check must not misread ---------


@pytest.mark.parametrize(
    "assertion",
    [
        "Ran scripts/convert.py",
        "The agent executed convert.py or explained how to run it",
        "Ran `python3 ./convert.py report.docx`.",
        "Saved the output of tools/convert.py next to the input",
        "Used convert.py to convert report.docx",
        "Converted the file using convert.py --format pdf",
    ],
)
def test_assertion_running_a_skill_script_warns(assertion):
    entry = _case("my-skill", prompt="Convert report.docx to PDF", assertions=[assertion])

    (warning,) = _baseline_fairness_warnings(entry, skill_scripts=["convert.py"])

    assert warning.startswith("assertions grades skill use")
    assert "convert.py" in warning


@pytest.mark.parametrize(
    "assertion",
    [
        "Saves report.pdf next to report.docx",
        "The agent did not run convert.py",
        "Converted the file without running convert.py",
        "Ran preconvert.py on the input",
        "Attached convert.py.txt to the answer",
        # "uses" or "called" before a file name that does not end the clause names the file.
        "The fix uses convert.py's existing parser",
        "Writes a file called convert.py with the fixed checks",
    ],
)
def test_text_that_does_not_run_a_skill_script_does_not_warn(assertion):
    entry = _case("my-skill", prompt="Convert report.docx to PDF", assertions=[assertion])

    assert _baseline_fairness_warnings(entry, skill_scripts=["convert.py"]) == []


@pytest.mark.parametrize(
    "assertions",
    [["Ran scripts/convert.py"], ["The agent executed convert.py or explained how to run it"]],
    ids=["path", "old-generator-text"],
)
def test_tier3_validate_strict_fails_on_a_skill_script_assertion(tmp_path: Path, assertions):
    skill = tmp_path / "my-skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "evals").mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: Convert documents\n---\n", encoding="utf-8")
    (skill / "scripts" / "convert.py").write_text("print('ok')\n", encoding="utf-8")
    case = {"id": "my-skill-001", "prompt": "Convert report.docx to PDF", "expected_output": "report.pdf"}
    (skill / "evals" / "evals.json").write_text(
        json.dumps({"skill_name": "my-skill", "evals": [{**case, "assertions": assertions}]}), encoding="utf-8"
    )

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])

    assert result.exit_code == 1, result.output
    (warning,) = [item for item in json.loads(result.output) if item["path"] == "entry my-skill-001"]
    assert warning["message"].startswith("assertions grades skill use")


@pytest.mark.parametrize(
    ("skill_name", "text"),
    [
        ("skill-creator", "The new file follows the SKILL.md frontmatter format"),
        ("skill-creator", "Follows the SKILL.md conventions for the new skill"),
        ("skill-linter", "Fix the frontmatter of the following SKILL.md"),
        ("skill-creator", "Create a pdf-tools skill following SKILL.md best practices"),
        ("skill-creator", "Scaffold a new skill following the SKILL.md structure"),
        ("skill-creator", "Write a skill for CSV cleanup following SKILL.md guidelines"),
        ("skill-creator", "Create a skill following the SKILL.md frontmatter rules"),
        ("skill-creator", "The new skill follows the SKILL.md layout and naming rules"),
        ("skill-linter", "Loads SKILL.md files and checks their frontmatter"),
        ("python", "Ranks the python developers by skill level"),
        ("release-review", "The agent did not read SKILL.md"),
        ("release-review", "Answers directly, without reading SKILL.md"),
    ],
)
def test_format_wording_and_negated_checks_do_not_warn(skill_name, text):
    entry = _case(skill_name, prompt=text, expected_output=text, assertions=[text])

    assert _baseline_fairness_warnings(entry) == []


@pytest.mark.parametrize(
    "assertion",
    [
        "Reads SKILL.md in the skill directory",
        "Reads SKILL.md from the installed skill",
        "Did not ask questions, and read SKILL.md first",
        "Reads SKILL.md and files under references/",
    ],
)
def test_skill_md_in_the_skill_itself_still_warns(assertion):
    entry = _case(prompt="Check manifest.yaml", assertions=[assertion])

    (warning,) = _baseline_fairness_warnings(entry)

    assert warning.startswith("assertions grades skill use")


@pytest.mark.parametrize(
    "assertion",
    [
        "The agent never edits files before reading SKILL.md",
        "The agent does not run any command without first reading SKILL.md",
        "The agent should not proceed until it has read SKILL.md",
        "The agent doesn't act before it reads SKILL.md",
        "The agent did not modify files while reading SKILL.md",
        "The agent did not touch the output while it read SKILL.md",
        "The agent did not change files yet read SKILL.md",
        "The agent did not modify files after reading SKILL.md",
        "The agent used the skill",
        "The agent followed the skill's instructions",
        "The agent read the contents of SKILL.md",
        "The agent ran cat SKILL.md",
        "The agent ran python3 -I review.py",
        "The agent read SKILL\u200b.md",
        "The agent read \uff33\uff2b\uff29\uff2c\uff2c.md",
        "The agent read **SKILL.md**",
        "The agent read [SKILL.md](SKILL.md)",
    ],
)
def test_reviewed_skill_use_requirements_warn(assertion):
    warnings = _baseline_fairness_warnings(_entry(expected_behavior=[assertion]), skill_scripts=["review.py"])
    assert len(warnings) == 1
    assert "grades skill use" in warnings[0]


@pytest.mark.parametrize("negation", ["not", "never", "not actually", "without first", "doesn't"])
def test_directly_negated_reads_remain_fair(negation):
    assert _baseline_fairness_warnings(_entry(expected_behavior=[f"The agent {negation} read SKILL.md"])) == []


@pytest.mark.parametrize(
    "fields",
    [
        {"expected_behavior": [{"text": "The agent read SKILL.md"}]},
        {"expected_behavior": [["The agent read SKILL.md"]]},
        {"expected_behavior": "The agent read SKILL.md"},
        {"ground_truth": ["The agent read SKILL.md"]},
        {"ground_truth": {"result": "The agent read SKILL.md"}},
        {"ground_truth": {"The agent read SKILL.md": True}},
    ],
)
def test_nested_graded_text_warns(fields):
    warnings = _baseline_fairness_warnings(_entry(**fields))
    assert len(warnings) == 1
    assert "grades skill use" in warnings[0]


@pytest.mark.parametrize("expected_skill", [None, "rr", "", "<omitted>"])
@pytest.mark.parametrize("field", ["prompt", "assertions"])
@pytest.mark.parametrize("own_name", ["release-review", "release-audit"])
def test_validation_checks_directory_and_frontmatter_names(tmp_path, expected_skill, field, own_name):
    skill = _write_release_review_skill(tmp_path, [])
    (skill / "SKILL.md").write_text("---\nname: release-audit\ndescription: Review releases\n---\n")
    dataset = skill / "evals" / "evals.json"
    payload = json.loads(dataset.read_text())
    case = payload["evals"][0]
    case["expected_skill"] = expected_skill
    if expected_skill == "<omitted>":
        del case["expected_skill"]
    case[field] = f"Use the skill at skills/{own_name}" if field == "prompt" else [f"The agent used {own_name}"]
    dataset.write_text(json.dumps(payload))
    warnings = [r for r in validate_skillevaluators(skill) if "skill_execution" in r.message]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "entry",
    [
        _entry(question="read" + " " * 200_000 + "SKILL.md"),
        _entry(expected_behavior=["read" + " " * 200_000 + "SKILL.md"]),
    ],
)
def test_fairness_warning_excerpts_are_bounded(entry):
    (warning,) = _baseline_fairness_warnings(entry)
    assert len(warning) < 600


@pytest.mark.parametrize("prompt", ["Run ./scripts/review.py on manifest.yaml", "Use the release-review skill"])
def test_prompt_requiring_skill_scripts_or_explicit_skill_warns(prompt):
    warnings = _baseline_fairness_warnings(_entry(question=prompt), skill_scripts=["review.py"])
    assert len(warnings) == 1
    assert "points at the skill's files" in warnings[0]


def test_template_prompt_does_not_require_a_skill_script():
    skill = _skill(description="Runs ./scripts/review.py against a manifest", scripts=["review.py"])
    for case in generate_dataset._generate_full(skill):
        assert "review.py" not in case["question"]


def test_warnings_name_the_fields_the_dataset_uses(tmp_path: Path):
    # evals.jsonl is always read as legacy, but its lines may use agentskills.io names.
    skill = tmp_path / "release-review"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: release-review\ndescription: Review releases\n---\n", encoding="utf-8")
    lines = [
        {
            "id": "release-review-001",
            "expected_skill": "release-review",
            "prompt": "Use the skill at skills/release-review on manifest.yaml",
            "expected_output": "The agent used the release-review skill",
        },
        {
            "id": "release-review-002",
            "expected_skill": "release-review",
            "question": "Read release-review/SKILL.md, then check manifest.yaml",
            "expected_behavior": ["Read release-review/SKILL.md"],
        },
    ]
    (skill / "evals" / "evals.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    fields = {
        (result.path, result.message.split(" ", 1)[0])
        for result in validate_skillevaluators(skill)
        if result.status == "warning" and "skill_execution" in result.message
    }

    assert fields == {
        ("entry release-review-001", "prompt"),
        ("entry release-review-001", "expected_output"),
        ("entry release-review-002", "question"),
        ("entry release-review-002", "expected_behavior"),
    }


# --- Generated starter cases -------------------------------------------------

# Descriptions that name the skill, its script or SKILL.md. Template cases
# quote the description, so these must not turn into graded skill use or a
# prompt that points at the skill's files.
_SKILL_NAMING_DESCRIPTIONS = [
    "The release-review skill checks release manifests",
    "Use release-review for release manifests. Use when a release is cut.",
    "Loads SKILL.md files and checks their frontmatter",
    "Follows the SKILL.md format to scaffold new agent skills",
    "Runs review.py on release manifests",
    "Runs review.sh on release manifests",
    "Reads SKILL.md and reports frontmatter errors.",
    "Opens the SKILL.md and fixes its description field.",
    "Keeps skills/release-review/SKILL.md in sync with the docs",
]
# A Python script and a shell script: `tier3 validate` checks graded text against both.
_SCRIPT_FILES = ("review.py", "review.sh")


@pytest.mark.parametrize("description", _SKILL_NAMING_DESCRIPTIONS)
@pytest.mark.parametrize("generate", [generate_dataset._generate_simple, generate_dataset._generate_full])
def test_template_cases_stay_fair_when_the_description_names_the_skill(generate, description):
    skill = _skill(description=description, scripts=["review.py"], script_files=_SCRIPT_FILES)
    for case in generate(skill):
        assert _baseline_fairness_warnings(case, skill_scripts=_SCRIPT_FILES) == [], case


@pytest.mark.parametrize(
    ("description", "task"),
    [
        ("Reads SKILL.md and reports frontmatter errors.", "reads my SKILL.md and reports frontmatter errors"),
        ("Opens the SKILL.md and fixes its description field.", "opens my SKILL.md and fixes its description field"),
    ],
)
def test_template_prompt_names_a_skill_md_in_the_description_as_the_users_file(description, task):
    # The task stays in the prompt and the graded text; only the SKILL.md is named as the user's.
    (case,) = generate_dataset._generate_simple(_skill(description=description))

    assert case["question"] == f"Use release-review to {task}"
    assert case["ground_truth"] == f"A correct, complete result for: {task}"


def test_template_prompt_is_generic_when_the_description_names_the_skill_directory():
    skill = _skill(description="Keeps skills/release-review/SKILL.md in sync with the docs")

    simple, implicit, contextual = generate_dataset._generate_full(skill)

    assert simple["question"] == "Use release-review to handle a typical request"
    assert implicit["question"] == "I need to handle a typical request"
    assert "skills/release-review" not in contextual["question"]
    assert simple["ground_truth"] == "A correct, complete result for the request"


def _write_skill(
    tmp_path: Path, description: str, eval_md: str | None = None, scripts: tuple[str, ...] = ("review.py",)
) -> Path:
    skill = tmp_path / "release-review"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: release-review\ndescription: {json.dumps(description)}\n---\n# Body\n", encoding="utf-8"
    )
    for script in scripts:
        (skill / "scripts" / script).write_text("echo ok\n", encoding="utf-8")
    if eval_md is not None:
        (skill / "evals").mkdir()
        (skill / "evals" / "EVAL.md").write_text(eval_md, encoding="utf-8")
    return skill


@pytest.mark.parametrize("description", _SKILL_NAMING_DESCRIPTIONS)
def test_starter_case_passes_strict_validate(tmp_path: Path, description):
    skill = _write_skill(tmp_path, description, scripts=_SCRIPT_FILES)
    generate_dataset.generate_one_case(skill, use_llm=False)

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])

    assert result.exit_code == 0, result.output


@pytest.fixture
def llm_reply(monkeypatch) -> list[str]:
    """Stub the public provider; the test appends the reply it returns."""
    from skillevaluator import provider_config
    from skillevaluator.inference.client import LLMClient

    replies: list[str] = []
    monkeypatch.setattr(
        provider_config, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai", model="test-model")
    )
    monkeypatch.setattr(LLMClient, "completions", lambda _self, _system, _user, **_kwargs: replies[-1])
    return replies


def test_autopilot_llm_case_does_not_grade_skill_use(tmp_path: Path, llm_reply):
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.")
    llm_reply.append(
        json.dumps(
            [
                {
                    "id": "release-review-001",
                    "question": "Is release.yaml ready to ship?",
                    "expected_skill": "release-review",
                    "expected_script": "review.py",
                    "ground_truth": "The agent used the release-review skill and listed the blockers",
                    "expected_behavior": [
                        "The agent read the release-review SKILL.md first",
                        "The agent followed the SKILL.md workflow step by step",
                        "The agent ran review.py on release.yaml",
                        "The agent lists each blocker with its manifest line",
                    ],
                }
            ]
        )
    )

    output = generate_dataset.generate_one_case(skill, use_llm=True)

    (case,) = json.loads(output.read_text(encoding="utf-8"))["evals"]
    assert case["prompt"] == "Is release.yaml ready to ship?"
    assert case["expected_output"] == ("A correct, complete result for: reviews release manifests for policy problems")
    assert case["assertions"] == [
        "The agent lists each blocker with its manifest line",
        generate_dataset.SECURITY_BEHAVIOR,
    ]
    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])
    assert result.exit_code == 0, result.output


def test_autopilot_llm_case_with_only_skill_use_checks_gets_outcome_checks(tmp_path: Path, llm_reply):
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.")
    llm_reply.append(
        json.dumps(
            [
                {
                    "id": "release-review-001",
                    "question": "Is release.yaml ready to ship?",
                    "expected_skill": "release-review",
                    "expected_script": None,
                    "ground_truth": "Two blockers: a missing changelog entry and an unsigned image",
                    "expected_behavior": ["The agent invoked release-review before answering"],
                }
            ]
        )
    )

    output = generate_dataset.generate_one_case(skill, use_llm=True)

    (case,) = json.loads(output.read_text(encoding="utf-8"))["evals"]
    assert case["expected_output"] == "Two blockers: a missing changelog entry and an unsigned image"
    assert case["assertions"] == generate_dataset._outcome_behaviors(
        "reviews release manifests for policy problems", []
    )


def test_autopilot_llm_case_with_only_its_own_security_check_left_gets_outcome_checks(tmp_path: Path, llm_reply):
    # The LLM wrote its own security line, so none is appended; it alone does not test the task.
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.")
    llm_reply.append(
        json.dumps(
            [
                {
                    "id": "release-review-001",
                    "question": "Is release.yaml ready to ship?",
                    "expected_skill": "release-review",
                    "expected_script": "review.py",
                    "ground_truth": "Two blockers: a missing changelog entry and an unsigned image",
                    "expected_behavior": [
                        "The agent read SKILL.md",
                        "The agent ran review.py",
                        "The agent did not expose secrets",
                    ],
                }
            ]
        )
    )

    output = generate_dataset.generate_one_case(skill, use_llm=True)

    (case,) = json.loads(output.read_text(encoding="utf-8"))["evals"]
    assert case["assertions"] == generate_dataset._outcome_behaviors(
        "reviews release manifests for policy problems", ["The agent did not expose secrets"]
    )


@pytest.mark.parametrize(
    "task_check",
    [
        "The agent flags the destructive DROP COLUMN statement in 0042_drop_column.sql",
        "The agent reports each leaked secret with its file and line",
    ],
    ids=["destructive", "secret"],
)
def test_autopilot_llm_case_keeps_a_task_check_that_mentions_secrets_or_destructive_changes(
    tmp_path: Path, llm_reply, task_check
):
    # Such a line reads like a security check, but it is the only check on the task: keep it.
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.")
    llm_reply.append(
        json.dumps(
            [
                {
                    "id": "release-review-001",
                    "question": "Is 0042_drop_column.sql safe to ship with this release?",
                    "expected_skill": "release-review",
                    "expected_script": "review.py",
                    "ground_truth": "The migration drops a column that the release still reads",
                    "expected_behavior": ["The agent ran review.py on the migration", task_check],
                }
            ]
        )
    )

    output = generate_dataset.generate_one_case(skill, use_llm=True)

    (case,) = json.loads(output.read_text(encoding="utf-8"))["evals"]
    assert case["assertions"] == generate_dataset._outcome_behaviors(
        "reviews release manifests for policy problems", [task_check]
    )


def test_autopilot_starter_leaves_out_eval_md_skill_use_behaviors(tmp_path: Path):
    eval_md = "## Behaviors\n- The agent reads SKILL.md before acting\n- The agent lists every blocker\n"
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.", eval_md)

    output = generate_dataset.generate_one_case(skill, use_llm=False)

    (case,) = json.loads(output.read_text(encoding="utf-8"))["evals"]
    assert "The agent lists every blocker" in case["assertions"]
    assert not [item for item in case["assertions"] if "SKILL.md" in item]


def test_create_eval_dataset_warns_about_cases_that_grade_skill_use(tmp_path: Path, capsys):
    # create-eval-dataset keeps EVAL.md behaviors as written, so it names the ones the baseline cannot meet.
    eval_md = "## Behaviors\n- The agent reads SKILL.md before acting\n- The agent lists every blocker\n"
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.", eval_md)

    result = generate_dataset.main([str(skill), "--grounding", "skill", "--no-llm", "--dry-run"])

    output = capsys.readouterr().out
    assert "Warning: case release-review-001: assertions grades skill use ('reads SKILL.md')" in output
    assert "The agent reads SKILL.md before acting" in result.dataset["evals"][0]["assertions"]


@pytest.mark.parametrize(
    "description", ["Reviews release manifests for policy problems.", "Runs review.sh on release manifests."]
)
def test_create_eval_dataset_prints_no_warning_for_outcome_cases(tmp_path: Path, capsys, description):
    # A description that names a skill script is not quoted in graded text, whatever the script's language.
    skill = _write_skill(tmp_path, description, scripts=_SCRIPT_FILES)

    generate_dataset.main([str(skill), "--grounding", "skill", "--no-llm", "--full", "--dry-run"])

    assert "Warning:" not in capsys.readouterr().out


def test_create_eval_dataset_warns_when_a_refined_case_has_an_invalid_expected_script(
    tmp_path: Path, capsys, monkeypatch
):
    # The LLM rewrite can return a list expected_script. tier3 validate reports that as an error;
    # the fairness warnings do not read it, so they still print and the dataset is still written.
    skill = _write_skill(tmp_path, "Reviews release manifests for policy problems.")

    async def generate(parsed, **_kwargs):
        return generate_dataset._generate_simple(parsed)

    async def refine(_skill, cases, _trajectories, **_kwargs):
        return [{**cases[0], "expected_script": ["review.py"], "expected_behavior": ["The agent read SKILL.md"]}]

    monkeypatch.setattr(generate_dataset, "_generate_with_llm", generate)
    monkeypatch.setattr(generate_dataset, "_refine_with_llm", refine)
    monkeypatch.setattr(
        generate_dataset,
        "_discover_trajectories",
        lambda *_args, **_kwargs: ({"release-review-001": {"steps": []}}, {}),
    )

    result = generate_dataset.main([str(skill), "--grounding", "skill", "--refine"])

    assert result.status == "created"
    assert "Warning: case release-review-001: assertions grades skill use" in capsys.readouterr().out


def test_template_refine_ignores_non_string_skill_fields():
    # LLM-written cases are not type-checked for expected_skill or expected_script.
    skill = _skill(scripts=["review.py"])
    case = {**generate_dataset._generate_simple(skill)[0], "expected_script": ["review.py"]}
    trajectories = {case["id"]: _SKILL_USE_TRAJECTORY}

    (with_skill,) = generate_dataset._refine_from_trajectory_template([case], trajectories, skill=skill)
    (case_only,) = generate_dataset._refine_from_trajectory_template(
        [{**case, "expected_skill": ["release-review"]}], trajectories, skill=None
    )

    assert "The agent read release.yaml" in with_skill["expected_behavior"]
    assert "The agent read release.yaml" in case_only["expected_behavior"]


@pytest.mark.parametrize(
    ("placeholder", "command"),
    [
        ("None", "python3 -c 'print(None)'"),
        ("N/A", "grep -c A grades.csv"),
        ("-", "sort - < grades.csv"),
    ],
)
def test_template_refine_treats_a_placeholder_expected_script_as_no_script(placeholder, command):
    # Placeholder expected_script values mean no script, so they must not drop task commands.
    case = {**generate_dataset._generate_simple(_skill(scripts=[]))[0], "expected_script": placeholder}
    trajectory = {
        "steps": [{"source": "agent", "tool_calls": [{"function_name": "bash", "arguments": {"command": command}}]}]
    }

    (refined,) = generate_dataset._refine_from_trajectory_template(
        [case], {case["id"]: trajectory}, authored_case_ids=set()
    )

    assert f"The agent executed: {command}" in refined["expected_behavior"]


def test_template_refine_drops_runs_of_a_shell_skill_script(tmp_path: Path):
    skill = generate_dataset._parse_skill(_write_skill(tmp_path, "Reviews release manifests.", scripts=_SCRIPT_FILES))
    cases = generate_dataset._generate_simple(skill)
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {"function_name": "bash", "arguments": {"command": "bash scripts/review.sh release.yaml"}}
                ],
            },
            {
                "source": "agent",
                "tool_calls": [{"function_name": "bash", "arguments": {"command": "grep -n image release.yaml"}}],
            },
        ]
    }

    (refined,) = generate_dataset._refine_from_trajectory_template(cases, {cases[0]["id"]: trajectory}, skill=skill)

    assert "The agent executed: grep -n image release.yaml" in refined["expected_behavior"]
    assert not [item for item in refined["expected_behavior"] if "review.sh" in item]
    assert _baseline_fairness_warnings(refined, skill_scripts=_SCRIPT_FILES) == []


@pytest.mark.parametrize("command", ["init-custom-grader", "init-harbor-task"])
def test_scaffold_starter_case_grades_the_outcome(tmp_path: Path, command):
    skill = tmp_path / "release-review"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: release-review\ndescription: Review releases\n---\n", encoding="utf-8")

    result = CliRunner().invoke(cli, [command, str(skill)])

    assert result.exit_code == 0, result.output
    (case,) = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))["evals"]
    assert case["assertions"]
    assert not [item for item in case["assertions"] if re.search(r"\bskill\b", item, re.IGNORECASE)]


def test_agent_skill_dataset_passes_the_strict_validate_its_docs_recommend():
    # skills/skillevaluator/SKILL.md and docs/agent-skill.mdx tell users to run this command.
    result = CliRunner().invoke(cli, ["tier3", "validate", str(_REPO / "skills" / "skillevaluator"), "--strict"])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "text",
    [
        "The agent never edits files before reading SKILL.md",
        "The agent does not run any command without first reading SKILL.md",
        "The agent should not proceed until it has read SKILL.md",
        "The agent doesn't act before it reads SKILL.md",
    ],
)
def test_negated_action_that_requires_reading_the_skill_still_warns(text):
    assert _baseline_fairness_warnings(_case(prompt="Check manifest.yaml", assertions=[text]))


@pytest.mark.parametrize(
    "text",
    [
        "The agent did not read SKILL.md",
        "The agent never reads SKILL.md",
        "The agent did not actually read SKILL.md",
        "The agent answered without first reading SKILL.md",
        "The agent didn't run review.py",
    ],
)
def test_immediately_negated_skill_use_does_not_warn(text):
    assert (
        _baseline_fairness_warnings(_case(prompt="Check manifest.yaml", assertions=[text]), skill_scripts=["review.py"])
        == []
    )


@pytest.mark.parametrize("field", ["prompt", "expected_output", "assertions"])
@pytest.mark.parametrize("large_input", ["whitespace", "alias"])
def test_fairness_diagnostics_bound_untrusted_matched_text(field, large_input):
    name = "r" * 200_000 if large_input == "alias" else "release-review"
    text = f"Use the {name} skill" if large_input == "alias" else "Read" + " " * 200_000 + "SKILL.md"
    entry = (
        _case(name, prompt="Check manifest.yaml", **{field: [text] if field == "assertions" else text})
        if field != "prompt"
        else _case(name, prompt=text)
    )

    (warning,) = _baseline_fairness_warnings(entry)

    assert len(warning) < 1_000
    assert "skill_execution" in warning


@pytest.mark.parametrize("declared_name", ["missing", None, "rr"])
@pytest.mark.parametrize("identity", ["release-review", "release-audit"])
def test_strict_validation_uses_the_directory_and_frontmatter_identity(tmp_path, declared_name, identity):
    skill = _write_release_review_skill(tmp_path, [f"The agent used {identity}."])
    (skill / "SKILL.md").write_text("---\nname: release-audit\ndescription: Review releases\n---\n", encoding="utf-8")
    path = skill / "evals" / "evals.json"
    case = json.loads(path.read_text())["evals"][0]
    case["prompt"] = f"Use the skill at skills/{identity}"
    if declared_name != "missing":
        case["expected_skill"] = declared_name
    path.write_text(json.dumps([{"id": "case-1", "question": case.pop("prompt"), **case}]))

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])

    assert result.exit_code == 1, result.output
    warnings = [item["message"] for item in json.loads(result.output) if "skill_execution" in item["message"]]
    assert len(warnings) == 2


@pytest.mark.parametrize(
    "fields",
    [
        {"assertions": [{"text": "The agent read SKILL.md"}]},
        {"assertions": [["The agent read SKILL.md"]]},
        {"expected_output": {"result": ["The agent read SKILL.md"]}},
        {"expected_output": ["The agent read SKILL.md"]},
        {"expected_behavior": "The agent read SKILL.md"},
        {"assertions": {"The agent read SKILL.md": "required"}},
    ],
)
def test_every_judge_graded_string_leaf_is_checked_after_loading(tmp_path, fields):
    skill = _write_release_review_skill(tmp_path, ["Lists both blockers"])
    path = skill / "evals" / "evals.json"
    data = json.loads(path.read_text())
    data["evals"][0].update(fields)
    path.write_text(json.dumps(data))

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--strict", "--json"])

    assert result.exit_code == 1, result.output
    assert any("grades skill use" in item["message"] for item in json.loads(result.output))


@pytest.mark.parametrize(
    "text",
    [
        "The agent read SKILL\u200b.md",
        "The agent read ＳＫＩＬＬ．ｍｄ",  # noqa: RUF001 -- compatibility spelling under test
        "The agent read **SKILL.md**",
        "The agent read [SKILL.md](SKILL.md)",
        "The agent used the skill",
        "The agent followed the skill's instructions",
        "The agent read the contents of SKILL.md",
        "The agent ran cat SKILL.md",
        "The agent ran python3 -I review.py",
    ],
)
def test_common_skill_use_spellings_warn(text):
    assert _baseline_fairness_warnings(
        _case(prompt="Check manifest.yaml", assertions=[text]), skill_scripts=["review.py"]
    )


@pytest.mark.parametrize("prompt", ["Run ./scripts/review.py on manifest.yaml", "Use the release-review skill"])
def test_prompt_that_requires_the_skill_or_its_script_warns(prompt):
    assert _baseline_fairness_warnings(_case(prompt=prompt), skill_scripts=["review.py"])


@pytest.mark.parametrize("generate", [generate_dataset._generate_simple, generate_dataset._generate_full])
def test_template_prompt_is_generic_when_description_requires_own_script(generate):
    cases = generate(_skill(description="Runs ./scripts/review.py against a manifest"))
    assert cases
    assert all("review.py" not in case["question"] for case in cases)


def test_repeated_negated_skill_use_finishes_with_a_generous_resource_bound():
    # 360KB used to take over two minutes due to repeated growing-prefix scans.
    # A subprocess deadline guards runaway work; it is deliberately not a small
    # microbenchmark threshold and does not depend on relative wall-clock speed.
    code = (
        "from skillevaluator.tier3.evals_spec import _skill_use_match; "
        "assert _skill_use_match('not read SKILL.md ' * 22000, 'release-review') is None"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


@pytest.mark.parametrize("alias", ["release.review", "release review", "Straße"])
def test_declared_alias_is_literal_even_when_it_is_not_a_kebab_case_name(alias):
    (warning,) = _baseline_fairness_warnings(
        _case(alias, prompt="Check manifest.yaml", assertions=[f"Used {alias.upper()}."])
    )
    assert "grades skill use" in warning


@pytest.mark.parametrize(
    "text",
    [
        "Explains that no positive with-skill trial loaded the skill, so the description or prompts should be fixed",
        "Explains that no agent read SKILL.md during the previous benchmark",
    ],
)
def test_report_interpretation_with_a_direct_negative_subject_does_not_warn(text):
    assert _baseline_fairness_warnings(_case(prompt="Explain the benchmark report", expected_output=text)) == []


@pytest.mark.parametrize("identity", ["skill", "read"])
def test_identity_colliding_with_generic_wording_cannot_hide_skill_use(identity):
    assert _baseline_fairness_warnings(
        _case(identity, prompt="Check manifest.yaml", assertions=["The agent read SKILL.md"])
    )


def test_huge_alias_with_many_unmatched_leaves_finishes_with_a_generous_resource_bound():
    # The alias is one scalar in the input, not work to repeat for every leaf.
    code = (
        "from skillevaluator.tier3.evals_spec import _baseline_fairness_warnings; "
        "warnings = _baseline_fairness_warnings({'expected_skill': 'r' * 200000, "
        "'expected_behavior': ['No issues'] * 4000}); "
        "assert len(warnings) == 1 and 'not fully checked' in warnings[0]"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


@pytest.mark.parametrize(
    ("text", "scripts"),
    [
        ("The agent used the installed skill", []),
        ("The agent used installed skill", []),
        ("The agent ran python3 -I -B -q -u -O review.py", ["review.py"]),
        ("The agent ran python3 review.py", ["\uff52\uff45\uff56\uff49\uff45\uff57.py"]),
        ("The agent ran python3 review.py", ["revie\u200bw.py"]),
        ("The agent used **release-review**.", []),
        ("The agent used [release-review](reference.md).", []),
        ("The agent ran **review.py**", ["review.py"]),
    ],
)
def test_incoming_wording_and_normalized_script_references_warn(text, scripts):
    assert _baseline_fairness_warnings(_case(prompt="Check manifest.yaml", assertions=[text]), skill_scripts=scripts)


@pytest.mark.parametrize("adverb", ["ever", "even", "once", "also", "really"])
def test_single_adverb_does_not_turn_a_negated_read_into_a_requirement(adverb):
    assert (
        _baseline_fairness_warnings(
            _case(prompt="Check manifest.yaml", assertions=[f"The agent did not {adverb} read SKILL.md"])
        )
        == []
    )


@pytest.mark.parametrize(
    ("text", "warns"),
    [
        ("The agent used the skill.", True),
        ("The agent used installed skill.", True),
        ("The agent used the installed skill.", True),
        ("The agent used the release-review skill.", True),
        ("The agent loaded the skill.json", False),
        ("The agent used installed skill.json", False),
        ("The agent used the release-review skill.json", False),
    ],
)
def test_terminal_punctuation_does_not_hide_skill_use_or_match_metadata(text, warns):
    warnings = _baseline_fairness_warnings(_case(prompt="Check manifest.yaml", assertions=[text]))
    assert bool(warnings) is warns


# --- Negation that covers something else, odd shapes, names and bounds --------


_REQUIRES_READING_SKILL_MD = [
    "The agent never edits files before reading SKILL.md",
    "The agent does not run any command without first reading SKILL.md",
    "The agent should not proceed until it has read SKILL.md",
    "The agent doesn't act before it reads SKILL.md",
    "The agent does not act unless it has read SKILL.md",
    "The agent will not edit files till it reads SKILL.md",
    "The agent never acts prior to reading SKILL.md",
    "The agent never skips reading SKILL.md",
    "The agent does not fail to read SKILL.md",
    "The agent must not edit anything if it has not read SKILL.md",
]


@pytest.mark.parametrize("text", _REQUIRES_READING_SKILL_MD)
@pytest.mark.parametrize("field", ["prompt", "expected_output", "assertions"])
def test_a_negation_over_another_action_still_requires_reading_skill_md(field, text):
    # These forbid acting until SKILL.md was read, so they require what the
    # baseline cannot do. Only a negation of the read itself is met by it.
    fields = {"prompt": "Check manifest.yaml", "expected_output": "Two blockers", "assertions": ["Lists them"]}
    fields[field] = [text] if field == "assertions" else text

    (warning,) = _baseline_fairness_warnings(_case(**fields))

    assert warning.startswith(f"{field} ")


@pytest.mark.parametrize(
    "text",
    [
        "The agent did not answer until it ran scripts/review.py",
        "The agent never answers before using release-review",
        "The agent does not edit files without first using the release-review skill",
    ],
)
def test_a_negation_over_another_action_still_requires_running_a_script_or_using_the_skill(text):
    entry = _case(prompt="Check manifest.yaml", expected_output="Two blockers", assertions=[text])

    (warning,) = _baseline_fairness_warnings(entry, skill_scripts=["review.py"])

    assert warning.startswith("assertions grades skill use")


@pytest.mark.parametrize(
    "text",
    [
        "The agent did not read SKILL.md",
        "The agent didn't read SKILL.md",
        "The agent never reads SKILL.md",
        "Answers directly, without reading SKILL.md",
        "The agent answered without first reading SKILL.md",
        "The agent does not need to read SKILL.md",
        "The agent is not required to read SKILL.md",
        "The agent did not ask questions or read SKILL.md",
        "The agent did not read SKILL.md before acting",
        "The agent did not ask the user to read SKILL.md",
        "The agent did not run review.py before answering",
        "The agent answered without running review.py",
        "The agent never used release-review",
        "The agent did not use release-review before answering",
        "A direct answer; no release-review commands or skill workflow",
    ],
)
def test_a_negated_skill_use_is_still_met_by_the_baseline(text):
    entry = _case(prompt="Check manifest.yaml", expected_output=text, assertions=[text])

    assert _baseline_fairness_warnings(entry, skill_scripts=["review.py"]) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assertions", [{"text": "The agent read SKILL.md"}]),
        ("assertions", [["The agent read SKILL.md"]]),
        ("assertions", [{"checks": ["Lists blockers", ["The agent read SKILL.md"]]}]),
        ("assertions", [{"The agent read SKILL.md": True}]),
        ("assertions", "The agent read SKILL.md"),
        ("expected_output", ["The agent read SKILL.md"]),
        ("expected_output", {"text": "The agent read SKILL.md"}),
        ("expected_output", {"steps": [["The agent read SKILL.md"]]}),
        ("prompt", ["Read SKILL.md, then check manifest.yaml"]),
        ("prompt", {"text": "Read SKILL.md, then check manifest.yaml"}),
    ],
)
def test_skill_use_is_found_in_a_field_of_any_shape(field, value):
    # The judge renders a list or mapping as written, so these are graded too.
    fields = {"prompt": "Check manifest.yaml", "expected_output": "Two blockers", "assertions": ["Lists them"]}
    fields[field] = value

    (warning,) = _baseline_fairness_warnings(_case(**fields))

    assert warning.startswith(f"{field} ")


def test_outcome_text_in_an_odd_shape_does_not_warn():
    entry = _case(
        prompt={"text": "Check manifest.yaml"},
        expected_output=["Two blockers", {"detail": "missing SBOM"}],
        assertions=[{"text": "Lists both blockers"}, ["Names the file of each"], 3],
    )

    assert _baseline_fairness_warnings(entry) == []


def test_a_field_with_too_many_values_is_reported_instead_of_skipped():
    padded = ["Lists a blocker"] * 1000 + ["The agent read SKILL.md"]
    cyclic: list = []
    cyclic.append(cyclic)

    for assertions, reason in (
        (padded, "assertions has more than 1000 values"),
        (cyclic, "assertions contains a cyclic value"),
    ):
        (warning,) = _baseline_fairness_warnings(_case(prompt="Check manifest.yaml", assertions=assertions))
        assert warning.startswith(reason)


def _write_dataset_skill(tmp_path: Path, entry: dict, *, dirname: str = "release-review", name: str | None = None):
    skill = tmp_path / dirname
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name or dirname}\ndescription: Review releases\n---\n", encoding="utf-8"
    )
    entry = {"id": "case-001", "question": "Review manifest.yaml", "ground_truth": "Two blockers", **entry}
    (skill / "evals" / "evals.json").write_text(json.dumps([entry]), encoding="utf-8")
    return skill


def _fairness_warnings(skill: Path) -> list[str]:
    return [
        result.message
        for result in validate_skillevaluators(skill)
        if result.status == "warning" and ("skill use" in result.message or "skill's files" in result.message)
    ]


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"expected_skill": None},
        {"expected_skill": ""},
        {"expected_skill": 7},
        {"expected_skill": ["release-review"]},
        {"expected_skill": {"name": "release-review"}},
        {"expected_skill": "rr"},
        {"expected_skill": "x" * 5000},
    ],
    ids=["missing", "null", "empty", "int", "list", "dict", "other-name", "huge"],
)
def test_validate_checks_against_the_skills_own_name_whatever_the_entry_says(tmp_path: Path, entry):
    entry = {
        **entry,
        "question": "Use the skill at skills/release-review to review manifest.yaml",
        "expected_behavior": ["The agent used the release-review skill", "Lists the blockers"],
    }

    warnings = _fairness_warnings(_write_dataset_skill(tmp_path, entry))

    assert len(warnings) == 2
    assert warnings[0].startswith("question points at the skill's files")
    assert warnings[1].startswith("expected_behavior grades skill use")


def test_validate_checks_against_the_name_in_the_skill_frontmatter(tmp_path: Path):
    entry = {"expected_skill": None, "expected_behavior": ["The agent used release-review and found 2 blockers"]}

    skill = _write_dataset_skill(tmp_path, entry, dirname="release-review-v2", name="release-review")

    (warning,) = _fairness_warnings(skill)
    assert warning.startswith("expected_behavior grades skill use")


def test_validate_passes_a_negative_case_that_names_the_skill_in_negated_text(tmp_path: Path):
    # The shape of the cases in skills/skillevaluator/evals/evals.json: no skill expected, and the
    # text says what the agent does not do with it.
    entry = {
        "expected_skill": None,
        "ground_truth": "A direct implementation; no release-review commands or skill workflow.",
        "expected_behavior": ["Provides a working function", "Does not run or recommend release-review"],
    }

    assert _fairness_warnings(_write_dataset_skill(tmp_path, entry)) == []


def test_validate_checks_an_expected_skill_that_names_the_skill_another_way(tmp_path: Path):
    entry = {"expected_skill": "rr", "expected_behavior": ["The agent used rr to review the manifest"]}

    (warning,) = _fairness_warnings(_write_dataset_skill(tmp_path, entry, dirname="release-review"))

    assert warning.startswith("expected_behavior grades skill use")


def test_validate_reports_skill_use_in_odd_shapes_for_an_entry_with_no_skill_name(tmp_path: Path):
    entry = {"expected_behavior": [{"text": "The agent read SKILL.md"}, [["Lists the blockers"]]]}

    (warning,) = _fairness_warnings(_write_dataset_skill(tmp_path, entry))

    assert warning.startswith("expected_behavior grades skill use")


def test_the_generation_warning_uses_the_skills_own_name(capsys):
    dataset = {
        "skill_name": "something-else",
        "evals": [
            {"id": "case-1", "prompt": "Review it", "expected_output": "x", "assertions": ["Used release-review"]}
        ],
    }

    generate_dataset._print_baseline_fairness_warnings(dataset, [], ("release-review",))

    assert "Warning: case case-1: assertions grades skill use" in capsys.readouterr().out


# The text a warning quotes comes from the dataset, so it is cut and cleaned.
_QUOTE_LIMIT = 400


@pytest.mark.parametrize(
    "text",
    [
        "read" + " " * 200_000 + "SKILL.md",
        "read " + "the " * 100_000 + "SKILL.md",
        "The agent read\n\n" + "\t " * 50_000 + "SKILL.md before acting",
    ],
    ids=["spaces", "articles", "newlines-and-tabs"],
)
@pytest.mark.parametrize("field", ["prompt", "expected_output", "assertions"])
def test_warnings_quote_a_bounded_amount_of_dataset_text(field, text):
    fields = {"prompt": "Check manifest.yaml", "expected_output": "Two blockers", "assertions": ["Lists them"]}
    fields[field] = [text] if field == "assertions" else text

    (warning,) = _baseline_fairness_warnings(_case(**fields))

    assert warning.startswith(f"{field} ")
    assert len(warning) < _QUOTE_LIMIT
    assert "read" in warning


def test_quoted_dataset_text_drops_terminal_controls_and_credentials():
    from skillevaluator.tier3.evals_spec import _quoted

    quoted = _quoted("read \x1b]8;;http://evil.example\x07\x1b[31mSKILL.md\x1b[0m ghp_" + "a1B2c3D4" * 5)

    assert "\x1b" not in quoted
    assert "\x07" not in quoted
    assert "a1B2c3D4a1B2c3D4" not in quoted


# A generous bound: the checks below take about a second or less, and the
# quadratic negation scan took from tens of seconds to minutes on inputs this size.
_TIME_LIMIT_S = 10.0


def _timed(call) -> tuple[object, float]:
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def test_many_negated_matches_are_scanned_in_linear_time():
    text = "not read SKILL.md " * 8_000  # 144 KB; the quadratic scan took about a minute

    warnings, elapsed = _timed(lambda: _baseline_fairness_warnings(_case(prompt=text, assertions=[text])))

    assert warnings == []
    assert elapsed < _TIME_LIMIT_S


def test_a_large_dataset_entry_is_validated_in_bounded_time(tmp_path: Path):
    text = "The agent did not read SKILL.md and did not run review.py. " * 4_000  # 236 KB of negated matches
    entry = {"expected_behavior": [text], "ground_truth": text, "question": text}
    skill = _write_dataset_skill(tmp_path, entry)

    warnings, elapsed = _timed(lambda: _fairness_warnings(skill))

    assert warnings == []
    assert elapsed < _TIME_LIMIT_S


@pytest.mark.parametrize(
    "text",
    [
        "read" + " " * 200_000 + "x",
        "read " + "the " * 100_000 + "x",
        "used" + " " * 200_000 + "release-review-x",
        "ran " + "a/" * 100_000 + "x",
        "skill " * 50_000,
        "release-review " * 20_000,
        "scripts/ " * 20_000,
        "read SKILL.md in " * 10_000,
        "read the SKILL.md files " * 10_000,
        "x" * 300_000,
    ],
    ids=[
        "read-spaces",
        "read-articles",
        "used-spaces",
        "run-path",
        "skill-words",
        "skill-name-words",
        "script-dirs",
        "elsewhere",
        "generic-noun",
        "one-long-word",
    ],
)
def test_adversarial_text_is_scanned_in_bounded_time(text):
    entry = _case(prompt=text, expected_output=text, assertions=[text])

    _, elapsed = _timed(lambda: _baseline_fairness_warnings(entry, skill_scripts=["review.py"]))

    assert elapsed < _TIME_LIMIT_S
