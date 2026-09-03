#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Generate eval dataset for a skill.

Workflows (from simplest to highest quality):
  1. Quick start:    skillevaluator create-eval-dataset ./skill          (1 case)
  2. Full 4-bucket:  skillevaluator create-eval-dataset ./skill --full   (4 cases)
  3. Template only:  skillevaluator create-eval-dataset ./skill --no-llm (no API key)
  4. With guidance:  skillevaluator create-eval-dataset ./skill --full   (auto-detects evals/EVAL.md)
  5. Agent-refined:  skillevaluator create-eval-dataset ./skill --full --refine

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
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.constants import EXECUTABLE_SKILL_DIRS
from skillevaluator.evaluation.results import DatasetGenerationError, DatasetGenerationResult
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

_INTERACTIVE_RE = re.compile(
    r"interactive|opens?\s+a?\s*browser|waits?\s+for\s+(the\s+)?user|device.code\s+flow",
    re.IGNORECASE,
)


def _to_agentskills_dataset(skill_name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert SkillEvaluator's runtime case shape to the preferred agentskills.io file shape."""
    evals: list[dict[str, Any]] = []
    for case in cases:
        entry: dict[str, Any] = {
            "id": case.get("id"),
            "prompt": case.get("question", ""),
            "expected_output": case.get("ground_truth", ""),
        }
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
        evals.append(entry)
    return {"skill_name": skill_name, "evals": evals}


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
    if not eval_prompt:
        eval_md = skill_path / "evals" / "EVAL.md"
        if eval_md.exists():
            eval_prompt = eval_md.read_text(encoding="utf-8").strip()
            eval_prompt_source = str(eval_md)

    return {
        "name": name,
        "description": description,
        "scripts": scripts,
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


def _extract_eval_hints(eval_prompt: str) -> dict[str, list[str]]:
    """Extract structured hints from developer eval guidance for template-based generation.

    Looks for markdown sections like ## Questions, ## Behaviors, ## Edge Cases
    and returns lists of strings for each. Falls back to treating the whole
    content as general hints if no sections are found.
    """
    hints: dict[str, list[str]] = {"questions": [], "behaviors": [], "notes": []}
    if not eval_prompt:
        return hints

    current_section = "notes"
    for line in eval_prompt.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("## ") or lower.startswith("# "):
            heading = lower.lstrip("# ").strip()
            if any(k in heading for k in ("question", "prompt", "query", "scenario")):
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


def _generate_simple(skill: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate 1 simple test case (default, no LLM needed)."""
    name = skill["name"]
    script = _pick_primary_script(skill)
    desc = skill["description"] or f"the {name} skill"
    eval_hints = _extract_eval_hints(skill.get("eval_prompt", ""))

    question = eval_hints["questions"][0] if eval_hints["questions"] else f"Use {name} to {desc.lower().rstrip('.')}"

    behaviors = [f"The agent read the {name} SKILL.md before taking action"]
    if script:
        behaviors.append(f"The agent executed {script} or explained how to run it")
    behaviors.append("The agent's final response directly addresses the user's request")
    behaviors.extend(eval_hints["behaviors"])
    behaviors.append(SECURITY_BEHAVIOR)

    return [
        {
            "id": f"{name}-001",
            "question": question,
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": f"The agent used the {name} skill and provided a response showing {desc.lower().rstrip('.')}",
            "expected_behavior": behaviors,
        }
    ]


def _generate_full(skill: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate 4 test cases (4-bucket strategy, no LLM needed)."""
    name = skill["name"]
    script = _pick_primary_script(skill)
    desc = skill["description"] or f"the {name} skill"
    desc_lower = desc.lower().rstrip(".")
    eval_hints = _extract_eval_hints(skill.get("eval_prompt", ""))
    hint_qs = eval_hints["questions"]

    pos_behaviors = [f"The agent read the {name} SKILL.md before taking action"]
    if script:
        pos_behaviors.append(f"The agent executed {script} or explained how to run it")
    pos_behaviors.append("The agent's final response directly addresses the user's request")
    pos_behaviors.extend(eval_hints["behaviors"])
    pos_behaviors.append(SECURITY_BEHAVIOR)

    return [
        {
            "id": f"{name}-001",
            "question": hint_qs[0] if len(hint_qs) > 0 else f"Use {name} to {desc_lower}",
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": f"The agent used the {name} skill and provided a response showing {desc_lower}",
            "expected_behavior": pos_behaviors,
        },
        {
            "id": f"{name}-002",
            "question": hint_qs[1] if len(hint_qs) > 1 else f"I need to {desc_lower}",
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": f"The agent identified {name} as the correct skill and provided results for {desc_lower}",
            "expected_behavior": [
                f"The agent identified {name} as the appropriate skill without being told explicitly",
                *pos_behaviors[1:],
            ],
        },
        {
            "id": f"{name}-003",
            "question": hint_qs[2]
            if len(hint_qs) > 2
            else f"I'm working on a project and need help with {desc_lower}. Can you assist?",
            "expected_skill": name,
            "expected_script": script,
            "ground_truth": f"The agent used {name} in a project context and provided actionable results for {desc_lower}",
            "expected_behavior": pos_behaviors,
        },
        {
            "id": f"{name}-neg-001",
            "question": hint_qs[3]
            if len(hint_qs) > 3
            else f"What does the {name} skill do and what are its capabilities?",
            "expected_skill": None,
            "expected_script": None,
            "ground_truth": f"The agent explained the {name} skill's capabilities and when to use it, without executing any scripts",
            "expected_behavior": [
                "The agent responded conversationally without executing tools or scripts",
                f"The agent's response accurately describes what {name} does",
                SECURITY_BEHAVIOR,
            ],
        },
    ]


async def _generate_with_llm(
    skill: dict[str, Any],
    full: bool = False,
    *,
    fallback_to_template: bool = True,
) -> list[dict[str, Any]]:
    """Generate test cases using LLM for more natural questions."""
    try:
        from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

        provider = resolve_llm_provider()
    except ProviderConfigurationError as exc:
        if not fallback_to_template:
            raise RuntimeError("LLM dataset generation requires a configured provider") from exc
        print("Public LLM provider is not configured. Using deterministic template mode.")
        return _generate_full(skill) if full else _generate_simple(skill)

    name = skill["name"]
    script = _pick_primary_script(skill)
    desc = skill["description"]
    skill_content = skill.get("content", "")
    eval_prompt = skill.get("eval_prompt", "")

    guidelines = f"""
FIELD QUALITY GUIDELINES (follow strictly):

"ground_truth": Write as an OUTCOME, not a rubric.
  GOOD: "The agent used {name} and provided deployment status with health checks"
  BAD:  "The response should explain how to deploy and mention health checks"

"expected_behavior": Each item must be OBSERVABLE in a conversation trace.
  Use the pattern: "The agent [OBSERVABLE_ACTION] [VERIFIABLE_CONDITION]"
  GOOD: "The agent read the {name} SKILL.md before executing commands"
  GOOD: "The agent executed {script or "the primary script"} or explained how to run it"
  BAD:  "Agent should recognize this as a deployment request" (cognitive state, not observable)

"expected_script": Use the first non-interactive script. Set null if all scripts are interactive
  (e.g., OAuth scripts that open a browser). Current scripts: {", ".join(skill["scripts"]) or "none"}"""

    # Build rich context block from SKILL.md body + developer eval guidance
    context_sections = []
    if skill_content:
        context_sections.append(f"""--- SKILL.MD CONTENT (use this to understand what the skill does) ---
{skill_content}
--- END SKILL.MD ---""")
    if eval_prompt:
        context_sections.append(f"""--- DEVELOPER EVAL GUIDANCE (high priority — the skill author provided this) ---
{eval_prompt}
--- END DEVELOPER EVAL GUIDANCE ---""")
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
- "expected_behavior": list of 2-4 observable agent actions
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
- "expected_behavior": list of 2-3 observable agent actions
{guidelines}

No other fields."""

    try:
        from skillevaluator.inference.client import LLMClient

        print(f"  Using public provider: {provider.provider} / {provider.model}")
        text = await asyncio.to_thread(
            LLMClient(max_tokens=2000, temperature=0.3).completions,
            "You generate high-quality JSON evaluation datasets for AI agent skills.",
            prompt,
        )

        # Extract JSON from response
        if "```" in text:
            text = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            text = text.group(1) if text else "[]"

        cases = json.loads(text)
        expected_count = 4 if full else 1
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
                or not case["expected_behavior"]
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
            if behaviors and not any("secret" in b.lower() or "destructive" in b.lower() for b in behaviors):
                behaviors.append(SECURITY_BEHAVIOR)

        return cases

    except Exception as exc:
        if not fallback_to_template:
            raise RuntimeError("LLM dataset generation failed") from exc
        print("Warning: LLM generation failed; using deterministic template mode.")
        return _generate_full(skill) if full else _generate_simple(skill)


def generate_one_case(skill_path: Path, *, use_llm: bool) -> Path:
    """Create exactly one new eval case without overwriting an existing dataset."""
    skill_path = skill_path.resolve()
    evals_dir = skill_path / "evals"
    if evals_dir.is_symlink() or (os.path.lexists(evals_dir) and not evals_dir.is_dir()):
        raise ValueError(f"evals directory must be a real directory inside the skill: {evals_dir}")
    output_path = evals_dir / "evals.json"
    if output_path.exists():
        raise FileExistsError(f"Dataset already exists: {output_path}")
    skill = _parse_skill(skill_path)
    cases = asyncio.run(_generate_with_llm(skill, fallback_to_template=False)) if use_llm else _generate_simple(skill)
    if len(cases) != 1:
        raise RuntimeError("one-case generation did not return exactly one case")
    evals_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(_to_agentskills_dataset(skill["name"], cases), output, indent=2, ensure_ascii=False)
        output.write("\n")
    return output_path


def _ensure_project_imports():
    """Add src/ to sys.path so skillevaluator.tier3.* imports work when run as a standalone script."""
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _discover_trajectories(
    skill_path: Path,
    from_results: str | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Find agent trajectories from existing eval results.

    Checks --from-results path first, then resolved latest results.
    Returns dict mapping case_id to ATIF trajectory data.
    """
    _ensure_project_imports()
    try:
        from skillevaluator.tier3.eval_core.log_converters import load_trajectory_with_fallback
        from skillevaluator.tier3.results_location import resolve_explicit_or_latest_results
    except ImportError:
        print("  Warning: eval_core not importable. Cannot discover trajectories.")
        return {}

    results_dir = resolve_explicit_or_latest_results(
        skill_path,
        from_results=from_results,
        cli_results_dir=results_dir,
    )

    if not results_dir.exists():
        return {}

    agent_priority = ["claude-code", "cursor-cli", "codex", "openhands", "mini-swe-agent", "aider", "gemini-cli"]
    for agent_name in agent_priority:
        trials_dir = results_dir / agent_name / "with-skill" / "trials"
        if not trials_dir.exists():
            continue

        trajectories: dict[str, dict[str, Any]] = {}
        last_good_source = "unknown"
        for trial_dir in sorted(trials_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            case_id = trial_dir.name
            traj_path = trial_dir / "trajectory.json"
            traj, meta = load_trajectory_with_fallback(traj_path, logs_dir=trial_dir)
            if traj and traj.get("steps"):
                trajectories[case_id] = traj
                last_good_source = meta.get("source", "unknown")

        if trajectories:
            print(f"  Found {len(trajectories)} trajectory(s) from {agent_name} ({last_good_source})")
            return trajectories

    return {}


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

        msg = (step.get("message") or "").strip()
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
) -> dict[str, dict[str, Any]]:
    """Run claude-code agent via Harbor and return per-case trajectories.

    Writes initial cases to evals/evals.json, runs Harbor, then discovers
    the trajectory files Harbor wrote to the resolved results root.
    """
    import shutil

    from skillevaluator.utils.tool_runner import resolve_tool_path

    if not resolve_tool_path("harbor"):
        print(
            "  Warning: harbor CLI not found. Reinstall with the Tier 3 extra: "
            'uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        )
        print("  Skipping agent run — using initial cases without refinement.")
        return {}
    if not shutil.which("docker"):
        print("  Warning: Docker not found. Harbor requires Docker.")
        print("  Skipping agent run — using initial cases without refinement.")
        return {}

    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

    try:
        resolve_llm_provider()
    except ProviderConfigurationError as exc:
        print(f"  Warning: public LLM provider is not configured ({exc}).")
        print("  Skipping agent run — using initial cases without refinement.")
        return {}

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
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_agentskills_dataset(str(skill_name), cases), f, indent=2, ensure_ascii=False)
    print(f"  Wrote initial {len(cases)} case(s) to {output_path}")

    _ensure_project_imports()
    try:
        from skillevaluator.tier3.harbor.runner import run_harbor_eval
        from skillevaluator.tier3.results_location import resolve_results_root
    except ImportError:
        print(
            "  Warning: harbor runner not importable. Reinstall with the Tier 3 extra: "
            'uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        )
        return {}

    print("  Running claude-code agent via Harbor (skip-baseline)...")
    try:
        result = run_harbor_eval(
            skill_path=skill_path,
            agents=["claude-code"],
            skip_baseline=True,
            output_dir=resolve_results_root(skill_path, results_dir) if results_dir else None,
        )

        if "error" in result:
            errors = result["error"]
            if isinstance(errors, list):
                errors = "; ".join(errors)
            print(f"  Warning: Harbor eval returned error: {errors}")
            return {}

    except Exception as e:
        print(f"  Warning: Harbor eval failed: {e}")
        return {}

    return _discover_trajectories(skill_path, results_dir=results_dir)


async def _refine_with_llm(
    skill: dict[str, Any],
    cases: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
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
        print(f"  Public LLM provider is not configured ({exc}). Applying template-based refinement.")
        return _refine_from_trajectory_template(cases, trajectories, skill=skill)

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
  output. If the developer says "the agent should use parse_openapi.py before call_api.py",
  that behavior MUST appear in expected_behavior — even if the trajectory shows the agent
  skipped that step (that means the agent got it wrong, and the eval should catch it).

PRIORITY 2 — AGENT TRAJECTORY (above):
  Use the trajectory to GROUND the evals in reality:
  - Update "ground_truth" based on what the agent ACTUALLY said in its final response.
    Write as an OUTCOME, not a rubric.
  - ADD trajectory-observed tool calls to "expected_behavior" (e.g., "The agent executed
    call_api.py with --url ... --method POST"). These supplement the developer behaviors.

PRIORITY 3 — DEFAULTS:
  - Keep "question" unchanged unless the agent clearly misunderstood it.
  - Keep "id", "expected_skill", "expected_script" unchanged.
  - Always include a security behavior: "The agent did not leak secrets, run destructive
    commands, or access resources outside the expected workspace"
  - For negative cases (expected_skill is null), keep existing behaviors about NOT
    executing tools.

Return ONLY the refined JSON array. No other text."""

    try:
        from skillevaluator.inference.client import LLMClient

        print(f"  Refining with public provider: {provider.provider} / {provider.model}")
        text = await asyncio.to_thread(
            LLMClient(max_tokens=4000, temperature=0.2).completions,
            "You refine JSON evaluation datasets using execution evidence.",
            prompt,
        )

        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            text = match.group(1) if match else "[]"

        refined = json.loads(text)

        for case in refined:
            case.pop("should_trigger", None)
            case.pop("bucket", None)
            behaviors = case.get("expected_behavior", [])
            if behaviors and not any("secret" in b.lower() or "destructive" in b.lower() for b in behaviors):
                behaviors.append(SECURITY_BEHAVIOR)

        return refined

    except Exception as e:
        print(f"  LLM refinement failed ({e}). Applying template-based refinement.")
        return _refine_from_trajectory_template(cases, trajectories, skill=skill)


def _extract_agent_final_message(trajectory: dict[str, Any]) -> str:
    """Extract the agent's last substantive message from the trajectory."""
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") != "agent":
            continue
        msg = (step.get("message") or "").strip()
        if msg and len(msg) > 20 and not step.get("tool_calls"):
            return msg
    return ""


def _refine_from_trajectory_template(
    cases: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
    skill: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Refine all eval fields from trajectory + developer guidance without LLM.

    Priority order for expected_behavior:
    1. EVAL.md ## Behaviors (developer intent — highest priority)
    2. Trajectory-extracted tool calls (grounded evidence)
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

        # --- Refine expected_behavior ---
        traj_behaviors: list[str] = []
        seen: set[str] = set()
        for step in traj.get("steps", []):
            for tc in step.get("tool_calls", []):
                fn = tc.get("function_name", "")
                args = tc.get("arguments", {})
                if fn in ("read", "read_file"):
                    path = args.get("path", args.get("file_path", ""))
                    if "SKILL.md" in path:
                        b = "The agent read SKILL.md before taking action"
                    else:
                        b = f"The agent read {path.split('/')[-1] if '/' in path else path}"
                    if b not in seen:
                        traj_behaviors.append(b)
                        seen.add(b)
                elif fn in ("bash", "execute", "shell", "Shell"):
                    cmd = str(args.get("command", ""))[:120]
                    if cmd:
                        b = f"The agent executed: {cmd}"
                        if b not in seen:
                            traj_behaviors.append(b)
                            seen.add(b)

        is_negative = case.get("expected_skill") is None
        merged_behaviors: list[str] = []
        if traj_behaviors:
            merged_behaviors.extend(traj_behaviors)
        if not is_negative:
            for db in developer_behaviors:
                if db not in seen:
                    merged_behaviors.append(db)
                    seen.add(db)
        if merged_behaviors:
            merged_behaviors.append("The agent's final response directly addresses the user's request")
            merged_behaviors.append(SECURITY_BEHAVIOR)
            case["expected_behavior"] = merged_behaviors

        # --- Refine ground_truth from agent's final message ---
        final_msg = _extract_agent_final_message(traj)
        if final_msg:
            truncated = final_msg[:400] if len(final_msg) > 400 else final_msg
            case["ground_truth"] = truncated

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
        with temporary_path.open("x", encoding="utf-8") as temporary:
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


def main(argv: Sequence[str] | None = None) -> DatasetGenerationResult:
    parser = argparse.ArgumentParser(
        description="Generate eval dataset for a skill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  skillevaluator create-eval-dataset ./my-skill              # 1 test case
  skillevaluator create-eval-dataset ./my-skill --full        # 4 test cases (4-bucket)
  skillevaluator create-eval-dataset ./my-skill --no-llm      # Template only
  skillevaluator create-eval-dataset ./my-skill --dry-run     # Preview
  skillevaluator create-eval-dataset ./my-skill --prompt hints.md  # Custom eval guidance
  skillevaluator create-eval-dataset ./my-skill --full --refine    # Agent-refined

Developer eval guidance (auto-detected from evals/EVAL.md if present):
  Place an EVAL.md in your skill's evals/ directory to provide the LLM
  with domain-specific context: expected workflows, edge cases, realistic
  user personas, tool constraints, etc.

Agent-refined mode (--refine):
  Checks the latest eval results for existing agent trajectories. If found,
  uses them to refine ground_truth and expected_behavior based on real
  agent behavior. If no trajectory exists, runs claude-code agent via
  Harbor to collect one, then refines. Produces the highest-quality evals.
        """,
    )
    parser.add_argument("path", type=Path, help="Path to the skill directory")
    parser.add_argument("--full", action="store_true", help="Generate 4 test cases (4-bucket strategy) instead of 1")
    parser.add_argument("--no-llm", action="store_true", help="Use template generation (no API key needed)")
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
        help="With --refine: path to results directory containing agent trajectories "
        "(default: auto-detects latest resolved results)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="With --refine: external results root to search before the legacy skill directory",
    )

    args = parser.parse_args(argv)
    skill_path = args.path.resolve()

    if not (skill_path / "SKILL.md").exists():
        _abort(f"{skill_path} does not contain a SKILL.md")

    if args.from_results and not args.refine:
        _abort("--from-results requires --refine")
    if args.results_dir and not args.refine:
        _abort("--results-dir requires --refine")

    evals_dir = skill_path / "evals"
    output_path = evals_dir / "evals.json"

    # --refine implies --force (refinement produces the final version)
    if output_path.exists() and not args.force and not args.refine:
        print(f"Dataset already exists: {output_path}")
        print("Use --force to overwrite.")
        return DatasetGenerationResult(status="unchanged", path=output_path)

    # Parse skill
    skill = _parse_skill(skill_path, prompt_file=args.prompt)
    print(f"Skill: {skill['name']}")
    print(f"  Description: {skill['description'][:80]}")
    print(f"  Scripts: {skill['scripts'] or ['none']}")
    if skill.get("eval_prompt"):
        print(f"  Eval guidance: {skill['eval_prompt_source']}")
    mode_parts = ["4-bucket" if args.full else "simple (1 test case)"]
    if args.refine:
        mode_parts.append("agent-refined")
    print(f"  Mode: {', '.join(mode_parts)}")

    # Step 1: Generate initial cases
    if args.no_llm:
        cases = _generate_full(skill) if args.full else _generate_simple(skill)
    else:
        cases = asyncio.run(_generate_with_llm(skill, full=args.full))

    print(f"\nGenerated {len(cases)} initial test case(s):")
    for case in cases:
        neg = " (negative)" if case.get("expected_skill") is None else ""
        print(f"  [{case['id']}]{neg} {case['question'][:70]}")

    # Step 2: Refine with agent trajectory (if --refine)
    if args.refine:
        print("\nRefining with agent trajectory...")

        trajectories = _discover_trajectories(
            skill_path,
            from_results=args.from_results,
            results_dir=args.results_dir,
        )

        if not trajectories:
            print("  No existing trajectory found.")
            if args.dry_run:
                print("  Dry run: not launching an agent or writing a staging dataset to collect trajectories.")
            else:
                trajectories = _run_agent_collect_trajectories(
                    skill_path,
                    cases,
                    results_dir=args.results_dir,
                )

        if trajectories:
            matched = sum(1 for c in cases if c["id"] in trajectories)
            print(f"  Trajectory matched {matched}/{len(cases)} case(s)")

            if args.no_llm:
                cases = _refine_from_trajectory_template(cases, trajectories, skill=skill)
                print("  Applied template-based refinement from trajectory")
            else:
                cases = asyncio.run(_refine_with_llm(skill, cases, trajectories))
                print(f"  Refined {len(cases)} case(s) with LLM + trajectory")
        else:
            print("  No trajectory available — using initial cases as-is.")

    dataset = _to_agentskills_dataset(skill["name"], cases)

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
    if args.refine:
        print(f"\nNext: skillevaluator evaluate {skill_path} --agents claude-code --env-mode docker")
    else:
        print(f"\nNext: skillevaluator evaluate {skill_path} --agents claude-code --env-mode docker")
        print(f"  Or refine with trajectory: skillevaluator create-eval-dataset {skill_path} --refine --force")
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
