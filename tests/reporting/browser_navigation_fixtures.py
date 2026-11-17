# SPDX-License-Identifier: Apache-2.0
"""Render the HTML fixtures that ``browser_navigation_checks.py`` drives in a real browser.

Writes three reports into a directory:

- ``full.html``: every tier, with Tier 1 findings, a failing required quality check, a Tier 2
  similarity check and two Tier 3 agents (one with selector syntax in its name) with trials.
- ``stress.html``: the same tiers with long unbroken names, paths and messages, many findings
  and six agents, to catch clipped tiles and horizontal overflow.
- ``absent.html``: a Tier 1-only run that still displays Tiers 2 and 3 as not run.

Only project code is needed, so the fixtures match the checked-out templates.
"""

from pathlib import Path
from typing import Any

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

ALL_TIERS = (1, 2, 3)


def _finding(severity: str, message: str, file_path: str, line: int) -> Finding:
    return Finding(
        category="SECURITY",
        severity=severity,
        check_name="check",
        message=message,
        file_path=file_path,
        line_number=line,
    )


def _tier1_results(path: str = "scripts/fetch.py", repeat: int = 1, suffix: str = "") -> list[ValidationResult]:
    pii = ValidationResult(validator_name="PII Scan", validator_description="Detect PII")
    secrets = ValidationResult(validator_name="Secrets Detection", validator_description="Detect secrets")
    risk = ValidationResult(validator_name="Code Risk Analysis", validator_description="Static analysis")
    lint = ValidationResult(validator_name="SCRIPT_LINT", validator_description="Lint scripts")
    for index in range(repeat):
        line = 11 + index * 10
        pii.add_finding(_finding("critical", f"Hardcoded key in code{suffix}", path, line))
        secrets.add_finding(_finding("critical", f"API key found{suffix}", path, line))
        risk.add_finding(_finding("medium", f"Request without timeout{suffix}", path, line + 4))
        risk.add_finding(_finding("high", f"Shell injection{suffix}", path, line + 10))
        lint.add_finding(_finding("low", f"Line too long{suffix}", path, line + 2))
    quality = ValidationResult(validator_name="QUALITY", validator_description="Quality dimensions")
    quality.add_error("Quality score 66.2/100 is below the required 70/100")
    quality.metadata["quality_config"] = {"source": "packaged", "min_score": 70}
    quality.metadata["quality_scores"] = {
        "skill_name": "demo",
        "overall_score": 66.2,
        "grade": "D",
        "skill_type": "script-based",
        "dimensions": {"correctness": {"score": 66.2, "weight": 1.0, "issues_count": 0, "issues": []}},
    }
    clean = ValidationResult(validator_name="License Compliance", validator_description="Licenses")
    clean.add_success("license", "License found")
    return [pii, secrets, risk, lint, quality, clean]


def _similarity_result() -> ValidationResult:
    result = ValidationResult(validator_name="Similarity Check")
    result.add_success("similarity", "No duplicate skills found")
    result.metadata["gating"] = {"tier": 2, "blocking": True}
    return result


def _reward(trial_id: str, entry_id: str, score: float, steps: int) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "entry_id": entry_id,
        **dict.fromkeys(DEFAULT_METRICS, score),
        "_traj": {"steps": steps, "prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0},
    }


def _agent_result(skill_name: str, agent_names: list[str]) -> ValidationResult:
    cases = ("case-1", "case-2")
    with_skill = [_reward(f"{case}__attempt1", case, 0.8, 12) for case in cases]
    baseline = [_reward(f"{case}__baseline1", case, 0.4, 20) for case in cases]
    agents = {
        name: {
            "execution_status": "succeeded",
            "expected_attempts": len(with_skill) + len(baseline),
            "scored_attempts": len(with_skill) + len(baseline),
            "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
            "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
            "rewards": with_skill,
            "rewards_baseline": baseline,
            "num_trials": len(with_skill),
            "num_trials_baseline": len(baseline),
        }
        for name in agent_names
    }
    payload = build_agent_eval_payload(skill_name, agents, use_llm_judge=False)
    assert payload is not None
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = payload
    return result


def _render(skill_name: str, results: list[ValidationResult]) -> str:
    reporter = HTMLReporter(
        include_timestamp=False,
        explicit_skill_name=skill_name,
        requested_tiers=ALL_TIERS,
        displayed_tiers=ALL_TIERS,
    )
    return reporter.render_all(results)


def render_fixtures(directory: Path) -> Path:
    """Write full.html, stress.html and absent.html into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    full = [*_tier1_results(), _similarity_result(), _agent_result("demo", ['agent"[one]/雪', "agent.two"])]
    stress_name = "stress-" + "very-long-unbroken-skill-name-" * 4 + "end"
    stress_path = "scripts/" + "deeply/nested/" * 12 + "module_with_an_extremely_long_unbroken_file_name.py"
    stress = [
        *_tier1_results(stress_path, repeat=12, suffix=" " + "x" * 120),
        _similarity_result(),
        _agent_result(stress_name, [f"agent-{index}-" + "long-provider/model-name-" * 3 for index in range(6)]),
    ]
    absent_reporter = HTMLReporter(include_timestamp=False, requested_tiers=(1,), displayed_tiers=ALL_TIERS)
    schema = ValidationResult(validator_name="Schema")
    schema.add_success("schema", "valid")
    pages = {
        "full.html": _render("demo", full),
        "stress.html": _render(stress_name, stress),
        "absent.html": absent_reporter.render_all([schema]),
    }
    for name, html in pages.items():
        (directory / name).write_text(html, encoding="utf-8")
    return directory
