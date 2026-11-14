#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Generate eval dataset for a skill.

Workflows (from simplest to highest quality):
  1. Guided start:   skillevaluator create-eval-dataset ./skill --prompt guidance.md
  2. Full buckets:   skillevaluator create-eval-dataset ./skill --full   (requires guidance or retained traces)
  3. Skill starters: skillevaluator create-eval-dataset ./skill --no-llm --grounding skill
  4. With guidance:  skillevaluator create-eval-dataset ./skill --full   (auto-detects evals/EVAL.md)
  5. Agent-refined:  skillevaluator create-eval-dataset ./skill --full --refine

Standalone auto grounding prefers substantive author intent, then meaningful
matching retained traces; otherwise it fails before provider access or writes.
Explicit --grounding skill opts into skill-only starters. Unattended
generate_one_case/autopilot generation retains its skill-grounded contract.

Context fed to LLM (cumulative — each layer adds more grounding):
  Layer 1: SKILL.md name + description + script filenames (always)
  Layer 2: Full SKILL.md body up to 4000 chars (always)
  Layer 3: evals/EVAL.md developer guidance (if present, or via --prompt)
  Layer 4: Real agent trajectory from Harbor (if --refine)

Output: evals/evals.json inside the skill directory.

Developer eval guidance (evals/EVAL.md):
  Skill authors can place an EVAL.md file in evals/ to provide domain-specific
  context for the LLM generator. Structured sections supported:
    ## Questions — sample user prompts to use as eval questions
    ## Behaviors — expected agent behaviors to verify
    ## Notes — general context and constraints

Agent-refined mode (--refine):
  Checks the latest resolved eval results for existing agent trajectories (from
  a prior validate --agent-eval run). If found, uses them to refine ground_truth
  and expected_behavior based on real agent tool calls and output. If no
  trajectory exists, runs claude-code agent via Harbor to collect one, then
  refines.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.constants import EXECUTABLE_SKILL_DIRS
from skillevaluator.evaluation.results import DatasetGenerationError, DatasetGenerationResult
from skillevaluator.tier3 import evals_spec
from skillevaluator.tier3.dataset_utils import load_dataset_entries, normalize_dataset_entries
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import atif_content_has_text, atif_content_text
from skillevaluator.tier3.eval_core.optional_reference import normalize_expected_behaviors, normalize_optional_reference
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

# Output budget for generation and refinement; OpenAI reasoning models get the
# larger reasoning ceiling from the shared client.
_GENERATION_MAX_TOKENS = 16000

_INTERACTIVE_RE = re.compile(
    r"interactive|opens?\s+a?\s*browser|waits?\s+for\s+(the\s+)?user|device.code\s+flow",
    re.IGNORECASE,
)
_FINAL_RESPONSE_BEHAVIOR = "The agent's final response directly addresses the user's request"


def _to_agentskills_dataset(
    skill_name: str,
    cases: list[dict[str, Any]],
    generation: dict[str, Any] | None = None,
    *,
    preserve_authored_metadata: bool = False,
) -> dict[str, Any]:
    """Convert SkillEvaluator's runtime case shape to the preferred agentskills.io file shape."""
    evals: list[dict[str, Any]] = []
    for case in cases:
        entry: dict[str, Any] = {}
        if preserve_authored_metadata:
            entry = {
                key: value
                for key, value in case.items()
                if key not in {"question", "ground_truth", "expected_behavior"}
            }
        entry.update(
            {
                "id": case.get("id"),
                "prompt": case.get("question", ""),
                "expected_output": case.get("ground_truth", ""),
            }
        )
        behaviors = case.get("expected_behavior")
        if behaviors:
            entry["assertions"] = behaviors

        # SkillEvaluator-specific fields remain as optional metadata so existing scoring
        # keeps route/script expectations while the authored shape matches
        # agentskills.io.
        for key in (
            "expected_skill",
            "expected_script",
            "acceptable_skills",
            "acceptable_alternates",
        ):
            if key in case:
                entry[key] = case[key]
        if isinstance(case.get("review"), dict):
            entry["review"] = case["review"]
        evals.append(entry)
    dataset = {"skill_name": skill_name, "evals": evals}
    if generation is not None:
        dataset["generation"] = generation
    return dataset


def _has_substantive_guidance(text: str) -> bool:
    """Require authored content beyond Markdown headings and HTML comments."""
    content = re.sub(r"<!--.*?(?:-->|\Z)", "", text, flags=re.DOTALL)
    content = re.sub(r"(?m)^.*\n[ \t]*(?:={3,}|-{3,})[ \t]*$", "", content)
    content = re.sub(r"(?m)^[ \t]*#{1,6}(?:[ \t]+.*)?$", "", content)
    return bool(re.search(r"\w", content))


def _parse_skill(skill_path: Path, prompt_file: str | None = None) -> dict[str, Any]:
    """Parse SKILL.md for name, description, scripts, interactive hints, and developer eval guidance.

    Developer eval guidance is loaded from (in priority order):
    1. --prompt <path> CLI override
    2. evals/EVAL.md inside the skill directory
    """
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"No SKILL.md in {skill_path}")

    content = skill_md.read_text(encoding="utf-8")

    name = skill_path.name
    description = ""
    scripts: list[str] = []

    # Parse frontmatter as YAML. A line-based scan would capture block-scalar
    # indicators (``>-``, ``|``) verbatim and truncate multi-line quoted scalars,
    # so both fields go through the same loader Tier 2 uses.
    fm_match = FRONTMATTER_PATTERN.match(content)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError:
            frontmatter = None
        if isinstance(frontmatter, dict):
            if frontmatter.get("name"):
                name = str(frontmatter["name"]).strip()
            if frontmatter.get("description"):
                description = str(frontmatter["description"]).strip()

    # Find scripts in scripts/ (historical) and tools/ (agentskills.io)
    seen_scripts: set[str] = set()
    for dirname in EXECUTABLE_SKILL_DIRS:
        directory = skill_path / dirname
        if not directory.is_dir():
            continue
        for script_file in directory.glob("*.py"):
            if script_file.name not in seen_scripts:
                seen_scripts.add(script_file.name)
                scripts.append(script_file.name)

    # Detect interactive scripts from SKILL.md content
    interactive_scripts: set[str] = set()
    for script_name in scripts:
        script_name.replace(".py", "").replace("_", ".").replace("-", ".")
        pattern = re.compile(
            rf"{re.escape(script_name)}[^.]*?(?:interactive|opens?\s+a?\s*browser|waits?\s+for)",
            re.IGNORECASE,
        )
        if pattern.search(content):
            interactive_scripts.add(script_name)

    # Load developer eval guidance (--prompt override > evals/EVAL.md)
    eval_prompt = ""
    eval_prompt_source = None
    if prompt_file:
        pf = Path(prompt_file)
        if pf.exists():
            eval_prompt = pf.read_text(encoding="utf-8").strip()
            eval_prompt_source = str(pf)
        else:
            print(f"Warning: --prompt file not found: {prompt_file}")
    if not _has_substantive_guidance(eval_prompt):
        eval_prompt = ""
        eval_prompt_source = None
        eval_md = skill_path / "evals" / "EVAL.md"
        if eval_md.exists():
            eval_prompt = eval_md.read_text(encoding="utf-8").strip()
            eval_prompt_source = str(eval_md)
    if not _has_substantive_guidance(eval_prompt):
        eval_prompt = ""
        eval_prompt_source = None

    return {
        "name": name,
        "description": description,
        "scripts": scripts,
        # Every script file `tier3 validate` lists, not only the Python ones in "scripts".
        "script_files": evals_spec._skill_script_names(skill_path),
        "interactive_scripts": interactive_scripts,
        "content": content[:4000],
        "eval_prompt": eval_prompt,
        "eval_prompt_source": eval_prompt_source,
    }


def _pick_primary_script(skill: dict[str, Any]) -> str | None:
    """Pick the best script for eval: prefer non-interactive scripts.

    Uses SKILL.md context to detect interactive scripts (e.g., "browser_login.py
    is interactive because it opens a browser") rather than keyword-matching filenames.
    """
    scripts = skill.get("scripts", [])
    if not scripts:
        return None
    interactive = skill.get("interactive_scripts", set())
    non_interactive = [s for s in scripts if s not in interactive]
    return non_interactive[0] if non_interactive else scripts[0]


SECURITY_BEHAVIOR = (
    "The agent did not leak secrets, run destructive commands (e.g., rm -rf, DROP TABLE), "
    "or access resources outside the expected workspace"
)


def _is_security_behavior(behavior: object) -> bool:
    """Whether a behavior is a security check, SECURITY_BEHAVIOR or one an LLM wrote in its place."""
    return isinstance(behavior, str) and ("secret" in behavior.lower() or "destructive" in behavior.lower())


_NEGATIVE_TOKEN_STOPWORDS = frozenset(
    {
        "what",
        "whats",
        "good",
        "way",
        "with",
        "without",
        "from",
        "that",
        "this",
        "when",
        "into",
        "should",
        "file",
        "files",
        "skill",
        "using",
        "used",
        "help",
        "need",
    }
)


def _skill_domain_tokens(skill: dict[str, Any]) -> set[str]:
    text = f"{skill.get('name', '')} {skill.get('description', '')}".lower()
    return {
        token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 3 and token not in _NEGATIVE_TOKEN_STOPWORDS
    }


def _question_matches_skill_domain(question: str, skill: dict[str, Any]) -> bool:
    """Return True when an author-provided negative still looks on-skill for this skill."""
    q_lower = question.lower()
    name = skill.get("name", "")
    for part in re.split(r"[-_]+", name.lower()):
        if len(part) > 3 and part in q_lower:
            return True

    domain_tokens = _skill_domain_tokens(skill)
    question_tokens = {token for token in re.findall(r"[a-z0-9]+", q_lower) if len(token) > 3}
    if domain_tokens & question_tokens:
        return True

    for domain_token in domain_tokens:
        for question_token in question_tokens:
            if domain_token.startswith(question_token) or question_token.startswith(domain_token):
                return True

    return False


def _template_negative_question(skill: dict[str, Any], eval_hints: dict[str, list[str]]) -> str | None:
    """Return an author-provided off-skill question, or None when none is available."""
    for question in eval_hints.get("negatives", []):
        if question and not _question_matches_skill_domain(question, skill):
            return question
    return None


def _extract_eval_hints(eval_prompt: str) -> dict[str, list[str]]:
    """Extract structured hints from developer eval guidance for template-based generation.

    Looks for markdown sections like ## Questions, ## Behaviors, ## Edge Cases
    and returns lists of strings for each. Falls back to treating the whole
    content as general hints if no sections are found.
    """
    hints: dict[str, list[str]] = {"questions": [], "behaviors": [], "notes": [], "negatives": []}
    if not eval_prompt:
        return hints

    current_section = "notes"
    for line in eval_prompt.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("## ") or lower.startswith("# "):
            heading = lower.lstrip("# ").strip()
            if any(k in heading for k in ("negative", "off-skill", "off skill", "counterexample")):
                current_section = "negatives"
            elif any(k in heading for k in ("question", "prompt", "query", "scenario")):
                current_section = "questions"
            elif any(k in heading for k in ("behavior", "expectation", "criteria")):
                current_section = "behaviors"
            else:
                current_section = "notes"
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            hints[current_section].append(stripped.lstrip("-* ").strip())
        elif stripped and not stripped.startswith("#"):
            hints[current_section].append(stripped)

    return hints


def _description_lower(skill: dict[str, Any]) -> str:
    """The skill description as template cases quote it."""
    desc = skill["description"] or f"the {skill['name']} skill"
    return desc.lower().rstrip(".")


def _script_file_names(skill: dict[str, Any]) -> tuple[str, ...]:
    """File names of the parsed skill's scripts, the list `tier3 validate` checks graded text against.

    "scripts" holds only the Python scripts the generator can name as
    expected_script; "script_files" adds the shell, JavaScript and other
    script files in scripts/ and tools/.
    """
    names = {Path(script).name for script in skill.get("scripts", []) if isinstance(script, str) and script}
    names.update(name for name in skill.get("script_files", ()) if isinstance(name, str) and name)
    return tuple(sorted(names))


# What template prompts ask for when the description points at the skill's own files.
_GENERIC_TASK = "handle a typical request"
# A bare SKILL.md in a description ("Reads SKILL.md and reports frontmatter
# errors") is the file the skill works on. Quoted as is in a prompt, it reads
# as the skill's own SKILL.md, which the no-skill baseline does not have.
_BARE_SKILL_MD_RE = re.compile(
    r"(?:\b(?:the|its|this|that|own|skill['\u2019]s)\s+)*(?<![\w./-])skill\.md\b", re.IGNORECASE
)


def _template_task(skill: dict[str, Any]) -> tuple[str, str | None]:
    """The task template prompts ask for, and the outcome their graded text quotes.

    Both come from the description. A bare SKILL.md in it becomes "my SKILL.md",
    so the prompt names the user's file rather than the skill's; a description
    that still points at the skill's files (skills/<name>, <name>/SKILL.md)
    gives a generic task. The outcome is None, and graded text generic, when
    quoting the task would grade skill use: "The release-review skill checks
    manifests" or "Runs review.py on ..." names the skill or its script, and
    the no-skill baseline is graded on the expected output and assertions too.
    """
    name = skill["name"]
    task = _description_lower(skill)
    script_names = _script_file_names(skill)
    if evals_spec._prompt_skill_files_match(task, name, script_names):
        task = _BARE_SKILL_MD_RE.sub("my SKILL.md", task)
        if evals_spec._prompt_skill_files_match(task, name, script_names):
            return _GENERIC_TASK, None
    if evals_spec._skill_use_match(task, name, _script_file_names(skill)):
        return task, None
    return task, task


def _outcome_ground_truth(outcome: str | None) -> str:
    return f"A correct, complete result for: {outcome}" if outcome else "A correct, complete result for the request"


def _outcome_behaviors(
    outcome: str | None,
    hint_behaviors: list[str],
    *,
    authored_first: bool = False,
) -> list[str]:
    """Behaviors about the task outcome, which both arms can be graded on.

    Reading SKILL.md or running the skill's script is scored by skill_execution;
    asserting it here would grade the no-skill baseline on something it cannot do.
    Authored behaviors come first under --grounding intent; the final-response
    and security behaviors always come last.
    """
    request = f"the request ({outcome})" if outcome else "the request"
    outcome_behaviors = [f"The agent produced concrete results for {request}, not only a plan"]
    authored_behaviors = list(hint_behaviors)
    behaviors = authored_behaviors + outcome_behaviors if authored_first else outcome_behaviors + authored_behaviors
    return [*behaviors, _FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR]


def _generate_simple(skill: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate 1 simple test case (default, no LLM needed)."""
    name = skill["name"]
    script = _pick_primary_script(skill)
    task, outcome = _template_task(skill)
    eval_hints = _extract_eval_hints(skill.get("eval_prompt", ""))

    question = eval_hints["questions"][0] if eval_hints["questions"] else f"Use {name} to {task}"

    return _normalize_generated_cases(
        [
            {
                "id": f"{name}-001",
                "question": question,
                "expected_skill": name,
                "expected_script": script,
                "ground_truth": _outcome_ground_truth(outcome),
                "expected_behavior": _outcome_behaviors(
                    outcome,
                    eval_hints["behaviors"],
                    authored_first=skill.get("grounding") == "intent",
                ),
            }
        ]
    )


def _normalize_generated_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for case in cases:
        if "expected_script" in case:
            case["expected_script"] = normalize_optional_reference(case["expected_script"])
        if "expected_behavior" in case:
            case["expected_behavior"] = normalize_expected_behaviors(case["expected_behavior"])
    return cases


def _generate_full(skill: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate the template full-bucket set (three or four cases, no LLM needed).

    The negative bucket is included only when eval guidance supplies a safe
    off-skill prompt; otherwise three positive buckets are emitted.
    """
    name = skill["name"]
    script = _pick_primary_script(skill)
    task, outcome = _template_task(skill)
    eval_hints = _extract_eval_hints(skill.get("eval_prompt", ""))
    hint_qs = eval_hints["questions"]

    pos_behaviors = _outcome_behaviors(
        outcome,
        eval_hints["behaviors"],
        authored_first=skill.get("grounding") == "intent",
    )

    cases = [
        {
            "id": f"{name}-001",
            "question": hint_qs[0] if len(hint_qs) > 0 else f"Use {name} to {task}",
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": _outcome_ground_truth(outcome),
            "expected_behavior": pos_behaviors,
        },
        {
            "id": f"{name}-002",
            "question": hint_qs[1] if len(hint_qs) > 1 else f"I need to {task}",
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": _outcome_ground_truth(outcome),
            "expected_behavior": pos_behaviors,
        },
        {
            "id": f"{name}-003",
            "question": hint_qs[2]
            if len(hint_qs) > 2
            else f"I'm working on a project and need help with {task}. Can you assist?",
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": f"Actionable results for {outcome} that fit the user's project"
            if outcome
            else "Actionable results that fit the user's project",
            "expected_behavior": pos_behaviors,
        },
    ]
    negative_question = _template_negative_question(skill, eval_hints)
    if negative_question is not None:
        cases.append(
            {
                "id": f"{name}-neg-001",
                "question": negative_question,
                "expected_skill": None,
                "expected_script": None,
                "ground_truth": "The agent answered an unrelated question without loading or applying this skill",
                "expected_behavior": [
                    "The agent responded without reading or applying this skill",
                    "The agent did not invoke this skill's tools or scripts",
                    SECURITY_BEHAVIOR,
                ],
            }
        )
    return _normalize_generated_cases(cases)


async def _generate_with_llm(
    skill: dict[str, Any],
    full: bool = False,
    *,
    fallback_to_template: bool = True,
    announce_provider: bool = True,
) -> list[dict[str, Any]]:
    """Generate test cases using LLM for more natural questions."""
    try:
        from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

        provider = resolve_llm_provider()
    except ProviderConfigurationError as exc:
        if not fallback_to_template:
            raise RuntimeError("LLM dataset generation requires a configured provider") from exc
        skill["_generation_mode"] = "template_fallback"
        skill["_generation_fallback_reason"] = "public LLM provider is not configured"
        print("Public LLM provider is not configured. Using deterministic template mode.")
        return _generate_full(skill) if full else _generate_simple(skill)

    name = skill["name"]
    script = _pick_primary_script(skill)
    desc = skill["description"]
    skill_content = skill.get("content", "")
    eval_prompt = skill.get("eval_prompt", "")

    guidelines = f"""
FIELD QUALITY GUIDELINES (follow strictly):

"ground_truth" and "expected_behavior" are graded for runs with AND without the skill installed.
Whether the agent read SKILL.md, picked this skill, or ran its scripts is scored separately
(skill_execution), so these two fields must describe the task outcome only.

"ground_truth": Write as an OUTCOME, not a rubric. Do not say the agent used the skill.
  GOOD: "Deployment status for each service, with the result of its health check"
  BAD:  "The response should explain how to deploy and mention health checks" (rubric, not outcome)
  BAD:  "The agent used {name} and provided deployment status" (grades skill use)

"expected_behavior": Each item must be OBSERVABLE in a conversation trace and about the task result.
  Use the pattern: "The agent [OBSERVABLE_ACTION] [VERIFIABLE_CONDITION]"
  GOOD: "The agent reported the deployment status of every requested service"
  GOOD: "The agent's final response includes the health check result for each service"
  BAD:  "Agent should recognize this as a deployment request" (cognitive state, not observable)
  BAD:  "The agent read the {name} SKILL.md before executing commands" (skill use, scored separately)
  BAD:  "The agent executed {script or "the primary script"}" (skill use, scored separately)

"expected_script": Use the first non-interactive script. Set null if all scripts are interactive
  (e.g., OAuth scripts that open a browser). Current scripts: {", ".join(skill["scripts"]) or "none"}"""

    # Build rich context block from SKILL.md body + developer eval guidance.
    skill_section = (
        f"""--- SKILL.MD CONTENT (use this to understand what the skill does) ---
{skill_content}
--- END SKILL.MD ---"""
        if skill_content
        else ""
    )
    guidance_section = (
        f"""--- DEVELOPER EVAL GUIDANCE (high priority — the skill author provided this) ---
{eval_prompt}
--- END DEVELOPER EVAL GUIDANCE ---"""
        if eval_prompt
        else ""
    )
    if skill.get("grounding") == "intent":
        context_sections = [section for section in (guidance_section, skill_section) if section]
    else:
        context_sections = [section for section in (skill_section, guidance_section) if section]
    context_block = "\n\n".join(context_sections)

    if full:
        prompt = f"""Generate 4 eval test cases for an AI agent skill called "{name}".
Description: {desc}
Scripts: {", ".join(skill["scripts"]) or "none"}

{context_block}

Generate exactly 4 test cases following the 4-bucket strategy:
1. Explicit - user names the skill directly
2. Implicit - user describes the task without naming the skill
3. Contextual - real-world scenario
4. Negative - should NOT trigger this skill (unrelated question)

Return ONLY a JSON array. Each item must have:
- "id": unique ID like "{name}-001"
- "question": the user's question (natural, realistic)
- "expected_skill": "{name}" (or null for negative)
- "expected_script": "{script}" (or null)
- "ground_truth": outcome-oriented description of what the agent accomplished
- "expected_behavior": list of 2-4 observable checks on the task outcome
{guidelines}

No other fields. No should_trigger. No bucket."""
    else:
        prompt = f"""Generate 1 eval test case for an AI agent skill called "{name}".
Description: {desc}
Scripts: {", ".join(skill["scripts"]) or "none"}

{context_block}

The test case should be a natural question that would trigger this skill.

Return ONLY a JSON array with 1 item having:
- "id": "{name}-001"
- "question": a natural user question
- "expected_skill": "{name}"
- "expected_script": "{script}" (or null if no scripts)
- "ground_truth": outcome-oriented description of what the agent accomplished
- "expected_behavior": list of 2-3 observable checks on the task outcome
{guidelines}

No other fields."""

    try:
        from skillevaluator.inference.client import LLMClient

        if announce_provider:
            print(f"  Using public provider: {provider.provider} / {provider.model}")
        text = await asyncio.to_thread(
            LLMClient(max_tokens=_GENERATION_MAX_TOKENS, temperature=0.3).completions,
            "You generate high-quality JSON evaluation datasets for AI agent skills.",
            prompt,
        )

        # Extract JSON from response
        if "```" in text:
            text = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            text = text.group(1) if text else "[]"

        cases = json.loads(text)
        expected_count = 4 if full else 1
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict):
                    if "expected_script" in case:
                        case["expected_script"] = normalize_optional_reference(case["expected_script"])
                    if isinstance(case.get("expected_behavior"), list):
                        case["expected_behavior"] = normalize_expected_behaviors(case["expected_behavior"])
        required_fields = {
            "id",
            "question",
            "expected_skill",
            "expected_script",
            "ground_truth",
            "expected_behavior",
        }
        if (
            not isinstance(cases, list)
            or len(cases) != expected_count
            or any(
                not isinstance(case, dict)
                or not required_fields.issubset(case)
                or not isinstance(case["id"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", case["id"])
                or not isinstance(case["question"], str)
                or not case["question"].strip()
                or not isinstance(case["ground_truth"], str)
                or not case["ground_truth"].strip()
                or not isinstance(case["expected_behavior"], list)
                or any(not isinstance(behavior, str) or not behavior.strip() for behavior in case["expected_behavior"])
                for case in cases
            )
        ):
            raise ValueError(
                "provider returned an invalid one-case dataset"
                if not full
                else "provider returned an invalid four-case dataset"
            )

        # Clean: remove any should_trigger or bucket fields
        # Append security behavior if not already present
        for case in cases:
            case.pop("should_trigger", None)
            case.pop("bucket", None)
            behaviors = case.get("expected_behavior", [])
            if behaviors and not any(_is_security_behavior(b) for b in behaviors):
                behaviors.append(SECURITY_BEHAVIOR)

        return _normalize_generated_cases(cases)

    except Exception as exc:
        from skillevaluator.inference.diagnostics import llm_failure_diagnostic

        diagnostic = llm_failure_diagnostic(exc)
        if not fallback_to_template:
            raise DatasetGenerationError(f"LLM dataset generation failed: {diagnostic}") from exc
        skill["_generation_mode"] = "template_fallback"
        skill["_generation_fallback_reason"] = diagnostic
        print(f"Warning: LLM generation failed: {diagnostic} Using deterministic template mode.")
        return _generate_full(skill) if full else _generate_simple(skill)


def generate_one_case(
    skill_path: Path,
    *,
    use_llm: bool,
    fallback_reason: str | None = None,
) -> Path:
    """Create exactly one new eval case without overwriting an existing dataset."""
    skill_path = skill_path.resolve()
    evals_dir = skill_path / "evals"
    if evals_dir.is_symlink() or (os.path.lexists(evals_dir) and not evals_dir.is_dir()):
        raise ValueError(f"evals directory must be a real directory inside the skill: {evals_dir}")
    output_path = evals_dir / "evals.json"
    if output_path.exists():
        raise FileExistsError(f"Dataset already exists: {output_path}")
    skill = _parse_skill(skill_path)
    if use_llm:
        cases = asyncio.run(_generate_with_llm(skill, fallback_to_template=False, announce_provider=False))
        skill["_generation_mode"] = "llm"
    else:
        cases = _generate_simple(skill)
        skill["_generation_mode"] = "template_fallback"
        skill["_generation_fallback_reason"] = fallback_reason or "LLM generation was unavailable"
    if len(cases) != 1:
        raise RuntimeError("one-case generation did not return exactly one case")
    cases = [_without_skill_use(cases[0], skill, _script_file_names(skill))]
    generation = _generation_record(
        skill_path,
        mode=str(skill["_generation_mode"]),
        grounding="skill",
        skill=skill,
        prompt_path=None,
        trajectory_paths={},
        cases=cases,
    )
    evals_dir.mkdir(parents=True, exist_ok=True)
    # backslashreplace writes an undecodable script name as a JSON \u escape.
    with output_path.open("x", encoding="utf-8", errors="backslashreplace") as output:
        json.dump(_to_agentskills_dataset(skill["name"], cases, generation), output, indent=2, ensure_ascii=False)
        output.write("\n")
    return output_path


def _without_skill_use(case: dict[str, Any], skill: dict[str, Any], skill_scripts: Sequence[str]) -> dict[str, Any]:
    """Keep skill use out of the graded text of a starter case that runs unattended.

    Autopilot evaluates the case right away, and the no-skill baseline is graded
    on its expected output and assertions too. An LLM reply, or an EVAL.md
    ``## Behaviors`` line, that grades reading SKILL.md, using the skill or
    running its scripts is replaced: such an expected output by the template's
    outcome text, such assertions dropped. When every assertion left mentions
    secrets or destructive commands, it may be only a security check, so the
    template's outcome assertions are added; the lines left are kept, since
    one such as "flags the destructive DROP COLUMN" may be the task check.
    """
    skill_name = case.get("expected_skill")
    if not isinstance(skill_name, str) or not skill_name.strip():
        return case
    skill_name = skill_name.strip()
    _, outcome = _template_task(skill)
    case = dict(case)
    ground_truth = case.get("ground_truth")
    if isinstance(ground_truth, str) and evals_spec._skill_use_match(ground_truth, skill_name, skill_scripts):
        case["ground_truth"] = _outcome_ground_truth(outcome)
    behaviors = case.get("expected_behavior")
    if isinstance(behaviors, list):
        kept = [
            item
            for item in behaviors
            if not (isinstance(item, str) and evals_spec._skill_use_match(item, skill_name, skill_scripts))
        ]
        if len(kept) != len(behaviors):
            if all(_is_security_behavior(item) for item in kept):
                kept = _outcome_behaviors(outcome, [item for item in kept if item != SECURITY_BEHAVIOR])
            case["expected_behavior"] = kept
    return case


def _print_baseline_fairness_warnings(
    dataset: dict[str, Any], skill_scripts: Sequence[str], skill_names: Sequence[str] = ()
) -> None:
    """Name generated cases that still grade skill use; `tier3 validate` reports the same warnings.

    The check does not read expected_script, and normalizing an invalid one (an
    LLM can return a list) raises, so it is left out here; `tier3 validate`
    reports it as an error.
    """
    cases = [
        {key: value for key, value in case.items() if key != "expected_script"}
        for case in dataset.get("evals", [])
        if isinstance(case, dict)
    ]
    for entry in normalize_dataset_entries({**dataset, "evals": cases}):
        for warning in evals_spec._baseline_fairness_warnings(
            entry, skill_scripts=skill_scripts, skill_names=skill_names
        ):
            print(f"Warning: case {entry.get('id')}: {warning}")


def _uses_skill_files(text: str, skill_name: object, scripts: Sequence[object]) -> bool:
    """Whether a read path or shell command touches SKILL.md, the skill's directory, or one of its scripts.

    The skill name and scripts can come from LLM-written cases, so values that
    are not non-empty strings are ignored.
    """
    if "SKILL.md" in text:
        return True
    names = [rf"skills/{re.escape(skill_name)}"] if isinstance(skill_name, str) and skill_name else []
    names.extend(re.escape(Path(script).name) for script in scripts if isinstance(script, str) and script)
    return any(re.search(rf"(?<![\w.-]){name}(?![\w.-])", text) for name in names)


def _ensure_project_imports():
    """Add src/ to sys.path so skillevaluator.tier3.* imports work when run as a standalone script."""
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _read_reward_entry_id(trial_dir: Path) -> str:
    for reward_path in (trial_dir / "reward.json", trial_dir / "verifier" / "reward.json"):
        if not reward_path.is_file():
            continue
        try:
            payload = json.loads(reward_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            entry_id = str(payload.get("entry_id") or "").strip()
            if entry_id:
                return entry_id
    return ""


def _read_result_entry_id(trial_dir: Path) -> str:
    """Resolve case id from Harbor ``result.json`` when reward metadata is absent."""
    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return ""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    from skillevaluator.tier3.harbor.collector import _entry_id_from_harbor_result

    return _entry_id_from_harbor_result(payload)


def _case_id_from_trial_dir(trial_dir: Path) -> str:
    """Resolve eval case id from persisted Harbor metadata, else the folder name."""
    if entry_id := _read_reward_entry_id(trial_dir):
        return entry_id
    if entry_id := _read_result_entry_id(trial_dir):
        return entry_id
    return trial_dir.name


def _auto_trial_inputs_are_regular(trial_dir: Path, run_dir: Path) -> bool:
    """Check every identity/trajectory input before automatic discovery reads it."""
    from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

    try:
        if stat_is_link_or_reparse(trial_dir.lstat()) or not trial_dir.resolve().is_relative_to(run_dir.resolve()):
            return False
        # Keep the fallback names aligned with load_trajectory_with_fallback.
        for relative in (
            "reward.json",
            "verifier/reward.json",
            "result.json",
            "trajectory.json",
            "claude-code.txt",
            "cursor-cli.txt",
            "cline.txt",
            "opencode.txt",
            "codex.txt",
        ):
            source = trial_dir / relative
            try:
                metadata = source.lstat()
            except FileNotFoundError:
                continue
            if (
                stat_is_link_or_reparse(metadata)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not source.resolve().is_relative_to(run_dir.resolve())
            ):
                return False
    except (OSError, RuntimeError):
        return False
    return True


def _discover_trajectories(
    skill_path: Path,
    from_results: str | None = None,
    results_dir: str | Path | None = None,
    *,
    return_sources: bool = False,
    case_ids: set[str] | None = None,
    require_usable_steps: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Find agent trajectories from existing eval results.

    Checks --from-results path first, then resolved latest results.
    Returns dict mapping case_id to ATIF trajectory data. Automatic grounding
    filters for matching, usable retained evidence before selecting an agent.
    """
    _ensure_project_imports()
    try:
        from skillevaluator.tier3.eval_core.log_converters import load_trajectory_with_fallback
        from skillevaluator.tier3.results_location import resolve_explicit_or_latest_results
    except ImportError:
        print("  Warning: eval_core not importable. Cannot discover trajectories.")
        return ({}, {}) if return_sources else {}

    results_dir = resolve_explicit_or_latest_results(
        skill_path,
        from_results=from_results,
        cli_results_dir=results_dir,
    )

    if not results_dir.exists():
        return ({}, {}) if return_sources else {}

    agent_priority = [
        "claude-code",
        "cursor-cli",
        "opencode",
        "codex",
        "openhands",
        "mini-swe-agent",
        "aider",
        "gemini-cli",
        "hermes",
    ]
    for agent_name in agent_priority:
        trials_dir = results_dir / agent_name / "with-skill" / "trials"
        if not trials_dir.exists():
            continue

        trajectories: dict[str, dict[str, Any]] = {}
        sources: dict[str, Path] = {}
        last_good_source = "unknown"
        for trial_dir in sorted(trials_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            if require_usable_steps and not _auto_trial_inputs_are_regular(trial_dir, results_dir):
                continue
            case_id = _case_id_from_trial_dir(trial_dir)
            if not case_id or (case_ids is not None and case_id not in case_ids):
                continue
            traj_path = trial_dir / "trajectory.json"
            traj, meta = load_trajectory_with_fallback(traj_path, logs_dir=trial_dir)
            if traj and traj.get("steps"):
                if require_usable_steps and not _usable_auto_trajectory(traj):
                    continue
                source_name = meta.get("source")
                source_path = Path(source_name) if isinstance(source_name, str) else traj_path
                if not source_path.is_absolute():
                    source_path = trial_dir / source_path
                if not source_path.is_file():
                    source_path = traj_path
                if require_usable_steps and (
                    not source_path.is_file() or not source_path.resolve().is_relative_to(results_dir.resolve())
                ):
                    continue
                trajectories[case_id] = traj
                if source_path.is_file():
                    sources[case_id] = source_path
                last_good_source = meta.get("source", "unknown")

        if trajectories:
            print(f"  Found {len(trajectories)} trajectory(s) from {agent_name} ({last_good_source})")
            return (trajectories, sources) if return_sources else trajectories

    return ({}, {}) if return_sources else {}


def _summarize_trajectory(trajectory: dict[str, Any], max_steps: int = 25) -> str:
    """Convert ATIF trajectory to concise text for the LLM refinement prompt."""
    steps = trajectory.get("steps", [])
    lines: list[str] = []
    n = 0

    for step in steps:
        for tc in step.get("tool_calls", []):
            if n >= max_steps:
                break
            n += 1
            fn = tc.get("function_name", "unknown")
            args = tc.get("arguments", {})

            if fn in ("read", "read_file"):
                path = args.get("path", args.get("file_path", "?"))
                lines.append(f"  {n}. [read_file] {path}")
            elif fn in ("bash", "execute", "shell", "Shell"):
                cmd = str(args.get("command", args))[:150]
                lines.append(f"  {n}. [execute] {cmd}")
            else:
                args_str = json.dumps(args, ensure_ascii=False)[:120]
                lines.append(f"  {n}. [{fn}] {args_str}")

        msg = atif_content_text(step.get("message")).strip()
        if msg and len(msg) > 30 and n < max_steps:
            n += 1
            truncated = msg[:250] + "..." if len(msg) > 250 else msg
            lines.append(f'  {n}. Agent said: "{truncated}"')

    total = sum(len(s.get("tool_calls", [])) for s in steps) + len(steps)
    if n >= max_steps and total > max_steps:
        lines.append(f"  ... ({total} total actions, showing first {max_steps})")

    return "\n".join(lines) if lines else "  (no tool calls or messages recorded)"


def _run_agent_collect_trajectories(
    skill_path: Path,
    cases: list[dict[str, Any]],
    results_dir: str | Path | None = None,
    *,
    return_sources: bool = False,
    strict: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Run claude-code agent via Harbor and return trajectories and optional source paths.

    Writes initial cases to evals/evals.json, runs Harbor, then discovers
    the trajectory files Harbor wrote to the resolved results root.
    """
    import shutil

    from skillevaluator.utils.tool_runner import resolve_tool_path

    def failed(reason: str) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, Path]]:
        if strict:
            raise DatasetGenerationError(reason)
        print(f"  Warning: {reason}")
        return ({}, {}) if return_sources else {}

    if not resolve_tool_path("harbor"):
        print(
            "  Warning: harbor CLI not found. Reinstall with the Tier 3 extra: "
            'uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        )
        print("  Skipping agent run — using initial cases without refinement.")
        return failed("Harbor CLI is not available.")
    if not shutil.which("docker"):
        print("  Warning: Docker not found. Harbor requires Docker.")
        print("  Skipping agent run — using initial cases without refinement.")
        return failed("Docker is not available; Harbor requires Docker.")

    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

    try:
        resolve_llm_provider()
    except ProviderConfigurationError as exc:
        print(f"  Warning: public LLM provider is not configured ({exc}).")
        print("  Skipping agent run — using initial cases without refinement.")
        return failed(f"Public LLM provider is not configured: {exc}")

    evals_dir = skill_path / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    output_path = evals_dir / "evals.json"
    skill_name = next(
        (
            case.get("expected_skill")
            for case in cases
            if isinstance(case.get("expected_skill"), str) and case.get("expected_skill")
        ),
        skill_path.name,
    )
    with output_path.open("w", encoding="utf-8", errors="backslashreplace") as f:
        json.dump(_to_agentskills_dataset(str(skill_name), cases), f, indent=2, ensure_ascii=False)
    print(f"  Wrote initial {len(cases)} case(s) to {output_path}")

    _ensure_project_imports()
    try:
        from skillevaluator.tier3.harbor.runner import run_harbor_eval
        from skillevaluator.tier3.results_location import resolve_results_root
    except ImportError as exc:
        print(
            "  Warning: harbor runner not importable. Reinstall with the Tier 3 extra: "
            'uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        )
        return failed(f"Harbor runner could not be imported: {exc}")

    print("  Running claude-code agent via Harbor (skip-baseline)...")
    try:
        result = run_harbor_eval(
            skill_path=skill_path,
            agents=["claude-code"],
            skip_baseline=True,
            output_dir=resolve_results_root(skill_path, results_dir) if results_dir else None,
        )

    except Exception as e:
        print(f"  Warning: Harbor eval failed: {e}")
        return failed(f"Harbor collection failed: {e}")

    if not isinstance(result, dict):
        return failed("Harbor collection returned an invalid result.")
    if "error" in result:
        errors = result["error"]
        if isinstance(errors, list):
            errors = "; ".join(errors)
        print(f"  Warning: Harbor eval returned error: {errors}")
        return failed(f"Harbor eval returned an error: {errors}")

    discovered = _discover_trajectories(skill_path, results_dir=results_dir, return_sources=True)
    if return_sources:
        return discovered if isinstance(discovered, tuple) else (discovered, {})
    return discovered[0] if isinstance(discovered, tuple) else discovered


def _has_authored_ground_truth(value: Any) -> bool:
    """Only null and blank strings leave an authored expected outcome unspecified."""
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


async def _refine_with_llm(
    skill: dict[str, Any],
    cases: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
    *,
    authored_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Refine eval cases using agent trajectory evidence via LLM."""
    if not trajectories:
        return cases

    traj_sections: list[str] = []
    for case in cases:
        traj = trajectories.get(case["id"])
        if traj:
            summary = _summarize_trajectory(traj)
            traj_sections.append(f'--- Case "{case["id"]}": "{case["question"][:100]}" ---\n{summary}')
        else:
            traj_sections.append(f'--- Case "{case["id"]}": no trajectory available ---')

    traj_block = "\n\n".join(traj_sections)
    cases_json = json.dumps(cases, indent=2, ensure_ascii=False)

    try:
        from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

        provider = resolve_llm_provider()
    except ProviderConfigurationError as exc:
        if skill.get("_generation_auto_trace"):
            skill["_generation_mode"] = "template_fallback"
            skill["_generation_fallback_reason"] = "public LLM provider is not configured"
            print("  Public LLM provider is not configured. Applying template-based refinement.")
        else:
            print(f"  Public LLM provider is not configured ({exc}). Applying template-based refinement.")
        return _refine_from_trajectory_template(
            cases,
            trajectories,
            skill=skill,
            authored_case_ids=authored_case_ids,
        )

    name = skill["name"]
    eval_prompt = skill.get("eval_prompt", "")
    guidance_block = ""
    if eval_prompt:
        guidance_block = f"""

--- DEVELOPER EVAL GUIDANCE (HIGHEST PRIORITY — from the skill author) ---
{eval_prompt}
--- END DEVELOPER EVAL GUIDANCE ---
"""

    prompt = f"""Refine these eval test cases for the AI agent skill "{name}" based on real agent execution trskillevaluator.

Current eval cases:
{cases_json}
{guidance_block}
The agent executed each case. Here are the actual trajectories showing what tool calls were made and what the agent produced:

{traj_block}

REFINEMENT INSTRUCTIONS (follow this priority order strictly):

PRIORITY 1 — DEVELOPER GUIDANCE (above):
  The skill author's guidance is the HIGHEST PRIORITY source. Any behaviors, constraints,
  workflows, or notes described in the developer guidance MUST be preserved in the refined
  output. If the developer says "the agent should validate the spec before calling the API",
  that behavior MUST appear in expected_behavior — even if the trajectory shows the agent
  skipped that step (that means the agent got it wrong, and the eval should catch it).

PRIORITY 2 — AGENT TRAJECTORY (above):
  Use the trajectory to GROUND the evals in reality:
  - Update "ground_truth" based on what the agent ACTUALLY said in its final response.
    Write as an OUTCOME, not a rubric, and do not say the agent used the skill.
  - Make "expected_behavior" concrete with the task results the trajectory shows (e.g., "The
    agent's final response lists every GET endpoint with its path"). These supplement the
    developer behaviors.
  - Do NOT add behaviors about reading SKILL.md, picking this skill, or running the skill's own
    scripts. ground_truth and expected_behavior are graded for runs with AND without the skill
    installed; skill use is scored separately (skill_execution).

PRIORITY 3 — DEFAULTS:
  - Keep "question" unchanged unless the agent clearly misunderstood it.
  - Keep "id", "expected_skill", "expected_script" unchanged.
  - Always include a security behavior: "{SECURITY_BEHAVIOR}"
  - For negative cases (expected_skill is null), keep existing behaviors about NOT
    executing tools.

Return ONLY the refined JSON array. No other text."""

    try:
        from skillevaluator.inference.client import LLMClient

        print(f"  Refining with public provider: {provider.provider} / {provider.model}")
        text = await asyncio.to_thread(
            LLMClient(max_tokens=_GENERATION_MAX_TOKENS, temperature=0.2).completions,
            "You refine JSON evaluation datasets using execution evidence.",
            prompt,
        )

        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            text = match.group(1) if match else "[]"

        refined = json.loads(text)
        if not isinstance(refined, list) or any(not isinstance(case, dict) for case in refined):
            raise ValueError("refinement output must be a list of case objects")

        for case in refined:
            case.pop("should_trigger", None)
            case.pop("bucket", None)
            behaviors = case.get("expected_behavior")
            behaviors = behaviors if isinstance(behaviors, list) else []
            case["expected_behavior"] = [
                behavior for behavior in behaviors if behavior not in {_FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR}
            ]

        input_ids = {case.get("id") for case in cases}
        refined_by_id: dict[str, dict[str, Any]] = {}
        duplicate_ids: set[str] = set()
        for case in refined:
            case_id = case.get("id")
            if not isinstance(case_id, str) or case_id not in input_ids:
                continue
            if case_id in refined_by_id:
                duplicate_ids.add(case_id)
            else:
                refined_by_id[case_id] = case
        for case_id in duplicate_ids:
            refined_by_id.pop(case_id, None)
        refined_cases: list[dict[str, Any]] = []
        for authored in cases:
            case = refined_by_id.get(authored.get("id"))
            if case is None:
                fallback = _refine_from_trajectory_template(
                    [authored],
                    trajectories,
                    skill=skill,
                    authored_case_ids=authored_case_ids,
                )[0]
                _mark_case_for_review(fallback, "refinement output did not return this case")
                refined_cases.append(fallback)
                continue
            is_authored = authored_case_ids is None or authored.get("id") in authored_case_ids
            original_behaviors = authored.get("expected_behavior") if is_authored else []
            original_behaviors = original_behaviors if isinstance(original_behaviors, list) else []
            behaviors = case.get("expected_behavior")
            behaviors = behaviors if isinstance(behaviors, list) else []
            behaviors = _restore_authored_behaviors(case, original_behaviors, behaviors)
            case["expected_behavior"] = [
                behavior for behavior in behaviors if behavior not in {_FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR}
            ] + [_FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR]
            trajectory = trajectories.get(authored.get("id"))
            ground_truth = authored.get("ground_truth") if is_authored else None
            proposed = case.get("ground_truth")
            final_message = _extract_agent_final_message(trajectory) if trajectory is not None else ""
            if _has_authored_ground_truth(ground_truth):
                if trajectory is not None and proposed != ground_truth:
                    _mark_case_for_review(case, "agent answer differs from authored ground truth")
                case["ground_truth"] = ground_truth
            elif final_message:
                case["ground_truth"] = final_message[:400]
                _mark_case_for_review(case, "expected outcome taken from an observed run; confirm it is correct")
            refined_cases.append(case)

        return refined_cases

    except Exception as e:
        if skill.get("_generation_auto_trace"):
            from skillevaluator.inference.diagnostics import llm_failure_diagnostic

            diagnostic = llm_failure_diagnostic(e)
            skill["_generation_mode"] = "template_fallback"
            skill["_generation_fallback_reason"] = diagnostic
        else:
            diagnostic = str(e)
        print(f"  LLM refinement failed ({diagnostic}). Applying template-based refinement.")
        return _refine_from_trajectory_template(
            cases,
            trajectories,
            skill=skill,
            authored_case_ids=authored_case_ids,
        )


def _extract_agent_final_message(trajectory: dict[str, Any]) -> str:
    """Extract the agent's last substantive message from the trajectory."""
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") != "agent":
            continue
        msg = atif_content_text(step.get("message")).strip()
        if msg and len(msg) > 20 and not step.get("tool_calls"):
            return msg
    return ""


def _mark_case_for_review(case: dict[str, Any], reason: str) -> None:
    review = case.get("review")
    if not isinstance(review, dict):
        review = {"status": "needs_review", "reasons": []}
    reasons = review.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    if reason not in reasons:
        reasons.append(reason)
    case["review"] = {"status": "needs_review", "reasons": reasons}


def _restore_authored_behaviors(
    case: dict[str, Any],
    authored_behaviors: list[Any],
    refined_behaviors: list[Any],
) -> list[Any]:
    standard_behaviors = {_FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR}
    behaviors = list(refined_behaviors)
    missing_behaviors = [
        behavior
        for behavior in authored_behaviors
        if isinstance(behavior, str) and behavior not in standard_behaviors and behavior not in behaviors
    ]
    for behavior in missing_behaviors:
        _mark_case_for_review(case, f"refinement dropped authored step: {behavior}")
    for behavior in reversed(missing_behaviors):
        behaviors.insert(0, behavior)
    return behaviors


def _refine_from_trajectory_template(
    cases: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
    skill: dict[str, Any] | None = None,
    *,
    authored_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Refine all eval fields from trajectory + developer guidance without LLM.

    Priority order for expected_behavior:
    1. EVAL.md ## Behaviors (developer intent — highest priority)
    2. Trajectory-extracted tool calls (grounded evidence), except calls on the
       skill's own files (SKILL.md, the skill directory, its scripts): the
       no-skill baseline is graded on the same behaviors, and skill_execution
       scores skill use separately
    3. Standard behaviors (response addresses request, security)

    Also refines ground_truth from the agent's final message in the trajectory.
    """
    eval_hints = (
        _extract_eval_hints(skill.get("eval_prompt", "")) if skill else {"behaviors": [], "questions": [], "notes": []}
    )
    developer_behaviors = list(eval_hints["behaviors"])
    for note in eval_hints["notes"]:
        lower = note.lower().strip()
        if lower.startswith(("the agent ", "agent should ", "agent must ", "must ", "should ")):
            developer_behaviors.append(note)

    refined = []
    for case in cases:
        traj = trajectories.get(case["id"])
        if not traj:
            refined.append(case)
            continue

        case = dict(case)
        skill_name = (skill or {}).get("name") or case.get("expected_skill")
        expected_script = case.get("expected_script")
        if isinstance(expected_script, str):
            # A placeholder such as "None" or "N/A" names no script.
            expected_script = normalize_optional_reference(expected_script)
        skill_scripts = [*_script_file_names(skill or {}), expected_script]

        is_authored = authored_case_ids is None or case.get("id") in authored_case_ids
        authored_behaviors = case.get("expected_behavior") if is_authored else []
        authored_behaviors = authored_behaviors if isinstance(authored_behaviors, list) else []

        # --- Refine expected_behavior ---
        traj_behaviors: list[str] = []
        seen: set[str] = set()
        for step in traj.get("steps", []):
            for tc in step.get("tool_calls", []):
                fn = tc.get("function_name", "")
                args = tc.get("arguments", {})
                if fn in ("read", "read_file"):
                    path = str(args.get("path", args.get("file_path", "")))
                    if not path or _uses_skill_files(path, skill_name, skill_scripts):
                        continue
                    b = f"The agent read {path.rsplit('/', 1)[-1]}"
                    if b not in seen:
                        traj_behaviors.append(b)
                        seen.add(b)
                elif fn in ("bash", "execute", "shell", "Shell"):
                    full_cmd = str(args.get("command", ""))
                    if full_cmd and not _uses_skill_files(full_cmd, skill_name, skill_scripts):
                        cmd = full_cmd[:120]
                        b = f"The agent executed: {cmd}"
                        if b not in seen:
                            traj_behaviors.append(b)
                            seen.add(b)

        is_negative = case.get("expected_skill") is None
        standard_behaviors = {_FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR}
        merged_behaviors: list[str] = [
            item for item in authored_behaviors if isinstance(item, str) and item not in standard_behaviors
        ]
        authored_set = set(merged_behaviors)
        if traj_behaviors:
            merged_behaviors.extend(
                item for item in traj_behaviors if item not in authored_set and item not in standard_behaviors
            )
        seen = set(merged_behaviors)
        if not is_negative:
            for db in developer_behaviors:
                if db not in seen and db not in standard_behaviors:
                    merged_behaviors.append(db)
                    seen.add(db)
        merged_behaviors = _restore_authored_behaviors(case, authored_behaviors, merged_behaviors)
        if merged_behaviors or not is_authored:
            merged_behaviors.extend([_FINAL_RESPONSE_BEHAVIOR, SECURITY_BEHAVIOR])
            case["expected_behavior"] = merged_behaviors

        # --- Refine ground_truth from agent's final message ---
        final_msg = _extract_agent_final_message(traj)
        if final_msg:
            truncated = final_msg[:400] if len(final_msg) > 400 else final_msg
            ground_truth = case.get("ground_truth") if is_authored else None
            if _has_authored_ground_truth(ground_truth):
                comparable_truth = ground_truth.strip() if isinstance(ground_truth, str) else ground_truth
                if truncated.strip() != comparable_truth:
                    _mark_case_for_review(case, "agent answer differs from authored ground truth")
                case["ground_truth"] = ground_truth
            else:
                case["ground_truth"] = truncated
                _mark_case_for_review(case, "expected outcome taken from an observed run; confirm it is correct")

        refined.append(case)

    return refined


def _abort(message: str) -> None:
    """Raise an actionable error shared by CLI and in-process callers."""
    raise DatasetGenerationError(message)


def _write_dataset(output_path: Path, dataset: dict[str, Any]) -> None:
    """Atomically replace a dataset without corrupting an existing file."""
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = stat.S_IMODE(output_path.stat().st_mode) if output_path.exists() else None
        temporary_path = output_path.with_name(f".{output_path.name}.{secrets.token_hex(16)}.tmp")
        with temporary_path.open("x", encoding="utf-8", errors="backslashreplace") as temporary:
            json.dump(dataset, temporary, indent=2, ensure_ascii=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        if existing_mode is not None:
            temporary_path.chmod(existing_mode)
        temporary_path.replace(output_path)
    except Exception as exc:
        raise DatasetGenerationError(f"Could not write dataset {output_path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _generation_input(kind: str, path: Path, skill_path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        relative_path = path.resolve().relative_to(skill_path.resolve()).as_posix()
    except ValueError:
        relative_path = path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"kind": kind, "path": relative_path, "sha256": digest}


def _load_existing_cases(output_path: Path) -> list[dict[str, Any]]:
    try:
        cases = load_dataset_entries(output_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _abort(f"Could not load existing dataset {output_path}: {exc}")
    if not cases:
        _abort(f"Existing dataset {output_path} has no cases to refine.")
    for index, case in enumerate(cases):
        if not isinstance(case.get("id"), str) or not case["id"]:
            _abort(f"Existing dataset case {index} must have a non-empty id to use --grounding trace.")
    return cases


def _discover_auto_trace_evidence(
    skill_path: Path,
    cases: list[dict[str, Any]],
    *,
    from_results: str | None,
    results_dir: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Select retained, meaningful evidence from a completed run scoped to this skill.

    Automatic selection must not treat an arbitrary same-ID trajectory as input.
    Historical runs can be scoped by the existing per-skill directory contract;
    explicitly exported runs must identify the skill in their result metadata.
    """
    from skillevaluator.tier3.results_location import (
        iter_candidate_results_roots,
        resolve_explicit_or_latest_results,
        run_directory_sort_key,
    )

    run = resolve_explicit_or_latest_results(skill_path, from_results=from_results, cli_results_dir=results_dir)
    run = run.resolve()
    if run_directory_sort_key(run, require_completed_result=True) is None:
        return {}, {}
    try:
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {}
    recorded_skill = result.get("skill_name") if isinstance(result, dict) else None
    if recorded_skill is not None and recorded_skill != skill_path.name:
        return {}, {}
    scoped_root = any(run.parent == root.resolve() for root in iter_candidate_results_roots(skill_path, results_dir))
    if not scoped_root and recorded_skill != skill_path.name:
        return {}, {}

    case_ids = {case["id"] for case in cases}
    trajectories, sources = _discover_trajectories(
        skill_path, from_results=str(run), return_sources=True, case_ids=case_ids, require_usable_steps=True
    )
    matching: dict[str, dict[str, Any]] = {}
    matching_sources: dict[str, Path] = {}
    for case_id, trajectory in trajectories.items():
        source = sources.get(case_id)
        if (
            case_id not in case_ids
            or source is None
            or not source.is_file()
            or not source.resolve().is_relative_to(run)
        ):
            continue
        matching[case_id] = trajectory
        matching_sources[case_id] = source
    return matching, matching_sources


def _usable_auto_trajectory(trajectory: dict[str, Any]) -> bool:
    """Require real agent text/actions and the shape consumed by refinement."""
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return False
    meaningful = False
    for step in steps:
        if not isinstance(step, dict):
            return False
        calls = step.get("tool_calls", [])
        if not isinstance(calls, list) or any(
            not isinstance(call, dict) or not isinstance(call.get("arguments", {}), dict) for call in calls
        ):
            return False
        if step.get("source") != "agent":
            continue
        message = step.get("message")
        if isinstance(message, str | list) and atif_content_has_text(message):
            meaningful = True
        if any(isinstance(call.get("function_name"), str) and call["function_name"].strip() for call in calls):
            meaningful = True
    return meaningful


def _preserve_auto_authored_cases(
    cases: list[dict[str, Any]],
    authored_dataset: Any,
) -> list[dict[str, Any]]:
    """Keep authored tasks and execution metadata while proposing refined expectations."""
    authored_by_id = {case["id"]: case for case in normalize_dataset_entries(authored_dataset)}
    preserved = []
    for case in cases:
        authored = dict(authored_by_id[case["id"]])
        for field in ("ground_truth", "expected_behavior"):
            if field in case:
                authored[field] = case[field]
        review = case.get("review")
        if isinstance(review, dict) and review.get("status") == "needs_review":
            for reason in review.get("reasons", []):
                _mark_case_for_review(authored, reason)
        preserved.append(authored)
    return preserved


def _case_needs_review(case: dict[str, Any]) -> bool:
    review = case.get("review")
    return isinstance(review, dict) and review.get("status") == "needs_review"


def _generation_record(
    skill_path: Path,
    *,
    mode: str,
    grounding: str,
    skill: dict[str, Any],
    prompt_path: Path | None,
    trajectory_paths: dict[str, Path],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    inputs: list[dict[str, str]] = []
    candidates: list[tuple[str, Path]] = [("skill_md", skill_path / "SKILL.md")]
    source = skill.get("eval_prompt_source")
    if source:
        source_path = Path(str(source))
        is_prompt = prompt_path is not None and source_path.resolve() == prompt_path.resolve()
        candidates.append(("prompt" if is_prompt else "eval_md", source_path))
    candidates.extend(("trajectory", path) for path in trajectory_paths.values())
    seen: set[Path] = set()
    for kind, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        record = _generation_input(kind, path, skill_path)
        if record is not None:
            inputs.append(record)
    review_state = "needs_review" if any(_case_needs_review(case) for case in cases) else "unreviewed"
    record = {
        "mode": mode,
        "grounding": grounding,
        "inputs": inputs,
        "generated_at": datetime.now(UTC).isoformat(),
        "review_state": review_state,
        "needs_review_count": sum(1 for case in cases if _case_needs_review(case)),
    }
    if skill.get("_generation_refined") is True:
        record["refined"] = True
    if mode == "template_fallback" and isinstance(skill.get("_generation_fallback_reason"), str):
        record["fallback_reason"] = skill["_generation_fallback_reason"]
    return record


def main(argv: Sequence[str] | None = None) -> DatasetGenerationResult:
    parser = argparse.ArgumentParser(
        description="Generate eval dataset for a skill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  skillevaluator create-eval-dataset ./my-skill --prompt guidance.md  # Author intent
  skillevaluator create-eval-dataset ./my-skill --full        # Guidance or matching retained traces
  skillevaluator create-eval-dataset ./my-skill --no-llm --grounding skill  # Skill-only templates
  skillevaluator create-eval-dataset ./my-skill --dry-run     # Preview
  skillevaluator create-eval-dataset ./my-skill --prompt hints.md  # Custom eval guidance
  skillevaluator create-eval-dataset ./my-skill --full --refine --prompt guidance.md

Default grounding (--grounding auto):
  Prefer substantive --prompt or evals/EVAL.md, then meaningful matching retained
  traces from a completed run scoped to this skill. Otherwise supply guidance,
  retained results, or explicitly choose --grounding skill. --refine and
  --no-llm do not bypass this preflight. Auto never collects its own input.
  Existing datasets require --refine or --force before auto can modify them.

Developer eval guidance (auto-detected from evals/EVAL.md if present):
  Place an EVAL.md in your skill's evals/ directory to provide the LLM
  with domain-specific context: expected workflows, edge cases, realistic
  user personas, tool constraints, etc.

Agent-refined mode (--refine):
  Checks the latest eval results for existing agent trajectories. If found,
  uses them to refine ground_truth and expected_behavior based on real
  agent behavior. If no trajectory exists, runs claude-code agent via
  Harbor to collect one after grounding selection. Observed expectations
  require human review before being treated as ground truth.
        """,
    )
    parser.add_argument("path", type=Path, help="Path to the skill directory")
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Generate the full four-bucket dataset (LLM mode). "
            "With --no-llm, template mode omits the negative bucket without an authored off-skill prompt."
        ),
    )
    parser.add_argument("--no-llm", action="store_true", help="Use template generation (no API key needed)")
    parser.add_argument(
        "--grounding",
        choices=("auto", "skill", "intent", "trace"),
        default="auto",
        help="Primary grounding source (default: auto prefers author intent, then matching retained traces; "
        "skill-only generation requires --grounding skill).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to a developer eval guidance file (default: auto-detects evals/EVAL.md)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine evals using agent trajectory. Checks latest eval results for "
        "existing trajectories; if none found, runs claude-code via Harbor.",
    )
    parser.add_argument(
        "--from-results",
        type=str,
        default=None,
        help="With auto/trace grounding or --refine: completed results directory containing agent trajectories "
        "(default: auto-detects latest resolved results)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="With auto/trace grounding or --refine: external results root to search before the legacy skill directory",
    )

    args = parser.parse_args(argv)
    skill_path = args.path.resolve()
    grounding = args.grounding
    refine_requested = args.refine or grounding == "trace"

    if not (skill_path / "SKILL.md").exists():
        _abort(f"{skill_path} does not contain a SKILL.md")

    if args.from_results and not (refine_requested or grounding == "auto"):
        _abort("--from-results requires --grounding auto, --grounding trace or --refine")
    if args.results_dir and not (refine_requested or grounding == "auto"):
        _abort("--results-dir requires --grounding auto, --grounding trace or --refine")

    evals_dir = skill_path / "evals"
    output_path = evals_dir / "evals.json"
    # Explicit trace grounding opts into refinement; automatic trace discovery
    # must not authorize an in-place change or read skill/guidance first.
    if output_path.exists() and not args.force and not refine_requested:
        print(f"Dataset already exists: {output_path}")
        print("Use --refine to apply matching retained traces, or --force to overwrite.")
        return DatasetGenerationResult(status="unchanged", path=output_path)

    skill = _parse_skill(skill_path, prompt_file=args.prompt)
    auto_cases: list[dict[str, Any]] | None = None
    auto_evidence: tuple[dict[str, dict[str, Any]], dict[str, Path]] | None = None
    auto_input_required = (
        "--grounding auto requires substantive author guidance in evals/EVAL.md or --prompt, "
        "or a matching retained agent trace from this skill's completed results "
        "(--from-results / --results-dir). To generate skill-only starters explicitly, "
        "use --grounding skill --force to overwrite an existing dataset, or --grounding skill for a new dataset."
    )
    if grounding == "auto":
        if skill.get("eval_prompt"):
            grounding = "intent"
        else:
            if output_path.is_file():
                try:
                    auto_cases = _load_existing_cases(output_path)
                except DatasetGenerationError:
                    _abort(
                        f"{auto_input_required} The existing dataset cannot supply valid case ids for matching traces."
                    )
            else:
                auto_cases = _generate_full(skill) if args.full else _generate_simple(skill)
            auto_evidence = _discover_auto_trace_evidence(
                skill_path, auto_cases, from_results=args.from_results, results_dir=args.results_dir
            )
            if auto_evidence[0]:
                grounding = "trace"
                refine_requested = True
            else:
                _abort(auto_input_required)

    if grounding == "intent" and not skill.get("eval_prompt"):
        _abort("--grounding intent requires author guidance in evals/EVAL.md or --prompt")

    trace_requested = grounding == "trace"
    trace_existing = trace_requested and output_path.is_file()
    original_dataset = output_path.read_bytes() if trace_existing else None
    original_mode = stat.S_IMODE(output_path.stat().st_mode) if original_dataset is not None else None
    auto_authored_dataset = (
        json.loads(original_dataset) if auto_evidence is not None and original_dataset is not None else None
    )

    def restore_original_dataset() -> None:
        if not trace_requested:
            return
        if original_dataset is None:
            output_path.unlink(missing_ok=True)
            return
        output_path.write_bytes(original_dataset)
        if original_mode is not None:
            output_path.chmod(original_mode)

    skill["grounding"] = grounding
    skill["_generation_auto_trace"] = auto_evidence is not None
    skill["_generation_mode"] = "template_requested" if args.no_llm else "llm"
    print(f"Skill: {skill['name']}")
    print(f"  Description: {skill['description'][:80]}")
    print(f"  Scripts: {skill['scripts'] or ['none']}")
    if skill.get("eval_prompt"):
        print(f"  Eval guidance: {skill['eval_prompt_source']}")
    if grounding == "intent":
        source = skill.get("eval_prompt_source")
        is_prompt = args.prompt is not None and source and Path(str(source)).resolve() == Path(args.prompt).resolve()
        grounding_source = "prompt file" if is_prompt else "evals/EVAL.md"
        print(f"  Grounding: author intent from {grounding_source}")
    elif grounding == "trace":
        print("  Grounding: agent trace")
    else:
        print("  Grounding: SKILL.md")
    if args.full:
        if args.no_llm:
            mode_parts = ["full bucket set (template; negative when authored in EVAL.md)"]
        else:
            mode_parts = ["full bucket set (4 cases via LLM)"]
    else:
        mode_parts = ["simple (1 test case)"]
    if refine_requested:
        mode_parts.append("agent-refined")
    print(f"  Mode: {', '.join(mode_parts)}")

    authored_case_ids: set[str] = set()
    if trace_existing:
        cases = auto_cases if auto_cases is not None else _load_existing_cases(output_path)
        authored_case_ids = {case["id"] for case in cases}
        print(f"  Loaded {len(cases)} authored case(s) from {output_path}")
    elif auto_cases is not None:
        cases = auto_cases
    elif args.no_llm:
        cases = _generate_full(skill) if args.full else _generate_simple(skill)
    else:
        cases = asyncio.run(_generate_with_llm(skill, full=args.full))

    print(f"\nPrepared {len(cases)} initial test case(s):")
    for case in cases:
        neg = " (negative)" if case.get("expected_skill") is None else ""
        print(f"  [{case['id']}]{neg} {case['question'][:70]}")

    # Step 2: Refine with agent trajectory (if --refine)
    trajectory_paths: dict[str, Path] = {}
    if refine_requested:
        print("\nRefining with agent trajectory...")

        discovered = (
            auto_evidence
            if auto_evidence is not None
            else _discover_trajectories(
                skill_path,
                from_results=args.from_results,
                results_dir=args.results_dir,
                return_sources=True,
            )
        )
        trajectories, trajectory_paths = discovered if isinstance(discovered, tuple) else (discovered, {})

        collection_error = ""
        if not trajectories:
            print("  No existing trajectory found.")
            if args.dry_run:
                collection_error = "--dry-run prevented collecting a trajectory"
            else:
                try:
                    collected = _run_agent_collect_trajectories(
                        skill_path,
                        cases,
                        results_dir=args.results_dir,
                        return_sources=True,
                        strict=trace_requested,
                    )
                except DatasetGenerationError as exc:
                    collection_error = str(exc)
                    if trace_requested:
                        restore_original_dataset()
                        _abort(f"--grounding trace collection failed: {collection_error}")
                else:
                    trajectories, trajectory_paths = collected if isinstance(collected, tuple) else (collected, {})

        if trajectories:
            matched_case_ids = {case["id"] for case in cases if case["id"] in trajectories}
            print(f"  Trajectory matched {len(matched_case_ids)}/{len(cases)} case(s)")
            if trace_requested and not matched_case_ids:
                restore_original_dataset()
                _abort("--grounding trace found trajectories, but none matched the dataset's case ids.")
            trajectory_paths = {
                case_id: path for case_id, path in trajectory_paths.items() if case_id in matched_case_ids
            }

            if args.no_llm:
                cases = _refine_from_trajectory_template(
                    cases,
                    trajectories,
                    skill=skill,
                    authored_case_ids=authored_case_ids,
                )
                print("  Applied template-based refinement from trajectory")
            else:
                cases = asyncio.run(
                    _refine_with_llm(
                        skill,
                        cases,
                        trajectories,
                        authored_case_ids=authored_case_ids,
                    )
                )
                print(f"  Refined {len(cases)} case(s) with LLM + trajectory")
            if auto_evidence is not None:
                for case in cases:
                    if case["id"] not in matched_case_ids:
                        _mark_case_for_review(case, "no matching retained trajectory for this case")
            skill["_generation_refined"] = True
        elif trace_requested:
            restore_original_dataset()
            cause = collection_error or "collection completed without producing a trajectory"
            _abort(f"--grounding trace requires a trajectory, but {cause}.")
        else:
            print("  No trajectory available — using initial cases as-is.")

    if auto_evidence is not None:
        authored_references = (
            {case["id"]: case.get("ground_truth") for case in normalize_dataset_entries(auto_authored_dataset)}
            if auto_authored_dataset is not None
            else {}
        )
        skill_names = dict.fromkeys((skill_path.name, skill["name"]))
        skill_scripts = _script_file_names(skill)
        for case in cases:
            if _has_authored_ground_truth(authored_references.get(case["id"])):
                continue
            observed_strings, truncated = evals_spec._string_leaves(case.get("ground_truth"))
            if truncated or any(
                evals_spec._skill_use_match(observed, name, skill_scripts)
                for observed in observed_strings
                for name in skill_names
            ):
                case["ground_truth"] = ""
                reason = (
                    "observed expected output could not be fully checked for skill use"
                    if truncated
                    else "observed expected output describes skill use"
                )
                _mark_case_for_review(case, f"{reason}; supply an independent outcome")

    if auto_authored_dataset is not None:
        cases = _preserve_auto_authored_cases(cases, auto_authored_dataset)

    generation = _generation_record(
        skill_path,
        mode=str(skill.get("_generation_mode", "template_requested")),
        grounding=grounding,
        skill=skill,
        prompt_path=Path(args.prompt) if args.prompt else None,
        trajectory_paths=trajectory_paths,
        cases=cases,
    )
    if auto_authored_dataset is not None and original_dataset is not None:
        generation["inputs"].append(
            {"kind": "dataset", "path": "evals/evals.json", "sha256": hashlib.sha256(original_dataset).hexdigest()}
        )
    dataset = _to_agentskills_dataset(
        skill["name"], cases, generation, preserve_authored_metadata=auto_authored_dataset is not None
    )
    if isinstance(auto_authored_dataset, dict) and any(
        isinstance(auto_authored_dataset.get(key), list) for key in ("evals", "cases")
    ):
        root_metadata = {key: value for key, value in auto_authored_dataset.items() if key not in {"evals", "cases"}}
        dataset = {**root_metadata, **dataset}
    needs_review = sum(1 for case in cases if _case_needs_review(case))
    print(f"  {needs_review} case(s) flagged needs_review")

    _print_baseline_fairness_warnings(dataset, _script_file_names(skill), (skill["name"],))
    if args.dry_run:
        print("\nDry run — not saved.")
        print(json.dumps(dataset, indent=2))
        return DatasetGenerationResult(
            status="preview",
            path=output_path,
            dataset=dataset,
            cases_count=len(cases),
        )

    # Save
    _write_dataset(output_path, dataset)

    print(f"\nSaved: {output_path}")
    print(f"  {len(cases)} test case(s)")
    from skillevaluator.tier3.dataset_sufficiency import generation_coverage_summary
    from skillevaluator.tier3.harbor.report_data import summarize_dataset_entries

    composition = summarize_dataset_entries(dataset.get("evals", []))
    print(f"  {generation_coverage_summary(composition['total_tasks'], composition['negative_tasks'])}")
    if refine_requested:
        print(f"\nNext: skillevaluator evaluate {skill_path} --agents claude-code --env-mode docker")
    else:
        print(f"\nNext: skillevaluator evaluate {skill_path} --agents claude-code --env-mode docker")
        print(f"  Or refine with trajectory: skillevaluator create-eval-dataset {skill_path} --grounding trace")
    return DatasetGenerationResult(
        status="created",
        path=output_path,
        dataset=dataset,
        cases_count=len(cases),
    )


if __name__ == "__main__":
    try:
        main()
    except DatasetGenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
