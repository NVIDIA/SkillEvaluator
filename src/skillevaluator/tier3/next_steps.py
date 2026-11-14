# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-level Tier 3 next steps: the change or experiment to try next, and what would show it worked.

The per-case "How to improve" list says how to fix individual failed cases.
These steps cover what a single case cannot show: missing attempts, noisy
results, effects that depend on the agent, and gaps in the dataset. Each step
names its target, the observed problem, the evidence from this run, the action,
and the check that would show improvement. The rules are deterministic, so the
same steps appear when no LLM is configured.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from skillevaluator.reporting.terminal_styles import AMBER
from skillevaluator.tier3.case_summary import (
    _CLEAR_EFFECT_PP,
    _VARIANTS,
    ERROR,
    FAIL,
    PASS,
    _agent_rates,
    _case_ids,
    _cases,
    _comparable,
    _dataset_entries,
    _details_truncated,
    _execution_unavailable,
    _marks,
    _never_activated,
    _pct,
    _with_skill_skipped,
)

if TYPE_CHECKING:
    from rich.console import Console

# Observed directly in this run.
SUPPORTED = "supported"
# A possible cause or remedy to test before acting on it.
HYPOTHESIS = "hypothesis"
# Observed, but the run is incomplete, so the evidence is partial.
LIMITED = "limited"

MAX_NEXT_STEPS = 3
_MIN_SKILL_CASES = 5
_MAX_EVIDENCE_ITEMS = 4


def _routing(entry: Mapping[str, Any]) -> bool | None:
    """Mirror the grader: ``should_trigger`` wins, then a truthy ``expected_skill``; neither means unlabeled."""
    if "should_trigger" in entry:
        return bool(entry.get("should_trigger"))
    if "expected_skill" in entry:
        return bool(entry.get("expected_skill"))
    return None


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:_MAX_EVIDENCE_ITEMS])
    return shown + (f" (+{len(items) - _MAX_EVIDENCE_ITEMS} more)" if len(items) > _MAX_EVIDENCE_ITEMS else "")


def _step(
    kind: str, target: str, problem: str, evidence: str, action: str, check: str, strength: str
) -> dict[str, str]:
    return {
        "kind": kind,
        "target": target,
        "problem": problem,
        "evidence": evidence,
        "action": action,
        "check": check,
        "strength": strength,
    }


def _missing_attempts(agents: Mapping[str, Mapping[str, Any]]) -> tuple[int, list[str], list[str]]:
    missing = 0
    cases: list[str] = []
    unavailable: list[str] = []
    for name, agent in agents.items():
        if _execution_unavailable(agent) or _details_truncated(agent):
            unavailable.append(name)
        for key, label in _VARIANTS:
            for case_id, case in _cases(agent, key).items():
                if n := _marks(case).count(ERROR):
                    missing += n
                    cases.append(f"{name}/{case_id} ({label})")
    return missing, cases, unavailable


def _inconsistent_cases(agents: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], int]:
    mixed: list[str] = []
    most_attempts = 0
    for name, agent in agents.items():
        for key, label in _VARIANTS:
            for case_id, case in _cases(agent, key).items():
                planned = _marks(case)
                most_attempts = max(most_attempts, len(planned))
                marks = [m for m in planned if m in (PASS, FAIL)]
                if PASS in marks and FAIL in marks:
                    mixed.append(f"{name}/{case_id} {label} {''.join(marks)}")
    return mixed, most_attempts


def _pass_rate_changes(agents: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    changes: dict[str, float] = {}
    for name, agent in agents.items():
        if not _comparable(agent):
            continue
        rates = _agent_rates(agent)
        with_pct, without_pct = _pct(*rates["with_skill"]), _pct(*rates["without_skill"])
        if with_pct is not None and without_pct is not None:
            changes[name] = with_pct - without_pct
    return changes


def next_steps(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    fixes: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Return up to three distinct run-level next steps, most blocking first."""
    raw_agents = result.get("agents")
    raw_agents = raw_agents if isinstance(raw_agents, Mapping) else {}
    total_agent_count = len(raw_agents)
    agents = {
        str(name): agent
        for name, agent in raw_agents.items()
        if isinstance(agent, Mapping) and not _never_activated(agent) and not _with_skill_skipped(agent)
    }
    case_ids = _case_ids(agents)
    if not agents or not case_ids:
        return []
    new_case_fixes = any(
        isinstance(fix, Mapping) and fix.get("kind") == "eval case" and fix.get("case") == "all cases"
        for fix in fixes or []
    )
    has_run_error_fix = any(isinstance(fix, Mapping) and fix.get("kind") == "run error" for fix in fixes or [])
    steps: list[dict[str, str]] = []

    missing, missing_cases, unavailable = _missing_attempts(agents)
    partial = bool(missing or unavailable)
    observed = LIMITED if partial else SUPPORTED
    if partial and not has_run_error_fix:
        problem = (
            f"{missing} attempt(s) did not produce a score"
            if missing
            else f"execution is incomplete for {', '.join(unavailable)}"
        )
        steps.append(
            _step(
                "incomplete run",
                "evaluation",
                problem + ", so the verdict and the steps below rest on partial evidence",
                _listed(missing_cases) if missing_cases else _listed(unavailable),
                "Re-run the evaluation before changing the skill or the cases; "
                "if the same attempts fail again, check the agent and gateway logs in the run directory.",
                "Every planned attempt has a score and the verdict is no longer inconclusive.",
                SUPPORTED,
            )
        )

    mixed, most_attempts = _inconsistent_cases(agents)
    if mixed:
        attempt_count = max(3, most_attempts + 2)
        attempt_flags = f"--n-attempts {attempt_count}"
        attempt_policy = result.get("attempt_policy")
        if isinstance(attempt_policy, Mapping) and attempt_policy.get("stop_on_pass") is True:
            attempt_flags += " --no-stop-on-pass"
        steps.append(
            _step(
                "inconsistent results",
                "evaluation",
                f"{len(mixed)} case result(s) passed on some attempts and failed on others",
                _listed(mixed),
                f"Re-run with more attempts per case (`{attempt_flags}`) "
                "before editing the skill or the cases, so one lucky or unlucky attempt does not decide the change.",
                "Each case passes or fails consistently and the verdict keeps its direction.",
                observed,
            )
        )
    elif most_attempts == 1:
        steps.append(
            _step(
                "single attempt",
                "evaluation",
                "each case ran once, so a single attempt decides every pass or fail mark",
                f"{len(case_ids)} case(s), 1 attempt each",
                "Re-run with `--n-attempts 3` to see whether the results repeat before acting on them.",
                "The marks repeat across attempts and the verdict keeps its direction.",
                observed,
            )
        )

    changes = _pass_rate_changes(agents)
    if len(changes) >= 2:
        best = max(changes, key=changes.__getitem__)
        worst = min(changes, key=changes.__getitem__)
        spread = changes[best] - changes[worst]
        if spread >= 2 * _CLEAR_EFFECT_PP and changes[best] >= _CLEAR_EFFECT_PP:
            steps.append(
                _step(
                    "agent-dependent effect",
                    "agent",
                    f"the skill's effect depends on the agent: {best} {changes[best]:+.0f} pp, "
                    f"{worst} {changes[worst]:+.0f} pp",
                    ", ".join(f"{name} {delta:+.0f} pp" for name, delta in changes.items()),
                    f"Test the hypothesis that {worst} follows the skill differently: compare its failing "
                    f"trajectories with {best}'s on the same cases, and re-run {worst} alone with more attempts "
                    "before changing SKILL.md for every agent.",
                    f"{worst}'s pass-rate change moves toward {best}'s without lowering {best}'s.",
                    HYPOTHESIS,
                )
            )

    dataset = _dataset_entries(run_dir)
    routed = {case_id: _routing(dataset[case_id]) for case_id in case_ids if case_id in dataset}
    skill_cases = [case_id for case_id in case_ids if routed.get(case_id) is not False]
    if len(skill_cases) < _MIN_SKILL_CASES and not new_case_fixes:
        steps.append(
            _step(
                "more cases",
                "dataset",
                f"only {len(skill_cases)} skill case(s) were evaluated, so one case moves the pass rate by "
                f"{100 / max(len(skill_cases), 1):.0f} pp",
                _listed(skill_cases),
                "Write realistic requests and observable outcomes in <skill-dir>/evals/EVAL.md, covering "
                "inputs and phrasings the current cases miss. Preview drafts with "
                "`skillevaluator create-eval-dataset <skill-dir> --full --force --dry-run`, then add reviewed "
                "cases to <skill-dir>/evals/evals.json; keep the existing cases and their expectations.",
                f"With {_MIN_SKILL_CASES} or more skill cases the verdict keeps its direction.",
                observed,
            )
        )
    if routed and not any(value is False for value in routed.values()):
        steps.append(
            _step(
                "negative controls",
                "dataset",
                "the dataset has no negative-control case, so the run cannot show that the skill stays out of "
                "unrelated requests",
                f"{len(routed)} case(s), none with `expected_skill: null` or `should_trigger: false`",
                'Add one or two unrelated requests with `"expected_skill": null` to evals/evals.json; '
                "the agent should complete them without loading the skill.",
                "The negative controls pass with the skill installed.",
                observed,
            )
        )

    if total_agent_count == 1 and len(changes) == 1:
        [(name, delta)] = changes.items()
        if delta < _CLEAR_EFFECT_PP:
            steps.append(
                _step(
                    "agent experiment",
                    "agent",
                    f"only {name} was evaluated and the skill showed no clear gain ({delta:+.0f} pp), so it is "
                    "unknown whether the result depends on the agent or model",
                    f"{name} {delta:+.0f} pp",
                    f"As an experiment, run the same cases with a second agent (`--agents {name},<agent>`) or "
                    "another model (`--agent-model AGENT=MODEL`), keeping the dataset unchanged.",
                    "A second agent with the same direction points at the skill; "
                    "a different direction points at agent-specific guidance.",
                    HYPOTHESIS,
                )
            )

    return steps[:MAX_NEXT_STEPS]


def render_next_steps(
    steps: list[dict[str, str]],
    *,
    console: Console,
    safe: Callable[[object], str],
) -> None:
    """Print the run-level next steps under the case summary."""
    if not steps:
        return
    console.print(Text("Next steps", style="bold"))
    for i, step in enumerate(steps, 1):
        head = Text(f" {i}. ", style=f"bold {AMBER}")
        head.append(safe(step["problem"]))
        label = step["target"] if step["strength"] == SUPPORTED else f"{step['target']} · {step['strength']}"
        head.append(f"  [{safe(label)}]", style="dim")
        console.print(head)
        console.print(Text(f"    try: {safe(step['action'])}", style="green"))
        console.print(Text(f"    check: {safe(step['check'])}", style="dim"))
        if step.get("evidence"):
            console.print(Text(f"    evidence: {safe(step['evidence'])}", style="dim"))
    console.print()
