# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proofs L5, H1 and M15 (verifier follow-up): one status rule on every surface of a ``validate`` run.

A real failure outranks missing evidence. The footer, the terminal summary and BENCHMARK.md say FAIL when a check
found a blocking problem, and still say what did not complete. They say INCOMPLETE only when nothing failed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _f90_gate_cli import (
    EDGE05_INTERVAL,
    EDGE05_WITH,
    EDGE05_WITHOUT,
    callout,
    run_validate,
    scores,
    tier3_result,
)

from skillevaluator.evaluation.tier3_report import advisory_skip_result
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.validators.security import SecurityValidator

PARTIAL = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}

# check-10 mix of p01-agent-prompt-injection and n03-claude-benign-hooks: SkillSpector reports the injection and the
# credential paths in agents/exfil.md, and returns a partial analysis because of the hooks file.
_P01_HIGH = (
    ("Instruction Override (P1)", "Prompt Injection: Ignore all previous instructions", 8),
    ("Credential Access (PE3)", "Privilege Escalation: ~/.ssh/id_rsa", 9),
    ("Credential Access (PE3)", "Privilege Escalation: ~/.aws/credentials", 9),
)


def _security_scan(*, blocking: bool):
    """A SkillSpector result whose analysis was partial (the hooks file), with or without the p01 HIGH findings."""

    def validate_security_only(_self: SecurityValidator, _target: Path) -> ValidationResult:
        result = ValidationResult()
        for check, message, line in _P01_HIGH if blocking else ():
            result.add_finding(
                Finding(
                    category="SECURITY",
                    severity=Severity.HIGH,
                    check_name=check,
                    message=message,
                    file_path="agents/exfil.md",
                    line_number=line,
                )
            )
        result.add_finding(
            Finding(
                category="SECURITY",
                severity=Severity.LOW,
                check_name="Bundled hooks can execute when matching lifecycle events occur. (BH1)",
                message="Bundled Execution Surface: document:hooks/hooks.json",
                file_path="hooks/hooks.json",
                line_number=1,
            )
        )
        result.add_error("skillspector reported partial analysis; scanner diagnostics were redacted")
        result.mark_scan_incomplete("skillspector")
        return result

    return validate_security_only


def _tier_row(benchmark: str, tier: str) -> str:
    return next(line for line in benchmark.splitlines() if line.startswith(f"| {tier} |"))


# ---------------------------------------------------------------------------
# L5: a blocking finding outranks an incomplete scan in the same result
# ---------------------------------------------------------------------------


def test_blocking_finding_outranks_an_incomplete_scan_in_the_same_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SecurityValidator, "validate_security_only", _security_scan(blocking=True))

    exit_code, _report, benchmark, output = run_validate(
        monkeypatch, tmp_path, None, tiers="1", checks="schema,security", reports="cli,json"
    )

    assert exit_code == 1
    # The footer names the failure and still counts what did not complete.
    assert "✗ FAIL" in output
    assert "! INCOMPLETE" not in output
    assert "Security Scan failed in Tier 1 · 1 did not complete" in output
    # The terminal summary and BENCHMARK.md give the same verdict, and also say the scan did not complete.
    assert "[FAIL] Validation failed (evidence is also incomplete: skillspector did not complete)" in output
    assert "[INCOMPLETE] Validation evidence is incomplete" not in output
    assert "FAIL — Publication blocked" in callout(benchmark)
    assert "Evidence is also incomplete: skillspector did not complete." in benchmark
    tier1 = _tier_row(benchmark, "Tier 1")
    assert "**FAILED**" in tier1
    assert "3 blocking finding(s)" in tier1
    assert "skillspector" in tier1


def test_incomplete_scan_without_a_blocking_finding_stays_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # check-10 n03: a harmless hook makes the scan partial; the only finding is the LOW hook note.
    monkeypatch.setattr(SecurityValidator, "validate_security_only", _security_scan(blocking=False))

    exit_code, report, benchmark, output = run_validate(
        monkeypatch, tmp_path, None, tiers="1", checks="schema,security", reports="cli,json"
    )

    assert exit_code == 1
    assert report["overall_status"] == "incomplete"
    assert "! INCOMPLETE" in output
    assert "✗ FAIL" not in output
    assert "[INCOMPLETE] Validation evidence is incomplete" in output
    assert "INCOMPLETE — Required evidence is missing" in callout(benchmark)
    assert "**INCOMPLETE**" in _tier_row(benchmark, "Tier 1")


# ---------------------------------------------------------------------------
# L5: a gated Tier 3 run that never ran says FAIL on every surface
# ---------------------------------------------------------------------------


def test_gated_skip_says_fail_on_every_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    skipped = advisory_skip_result("Tier 3 live evaluation skipped: docker missing", skill_name="gate-demo")

    exit_code, report, benchmark, output = run_validate(
        monkeypatch, tmp_path, skipped, "--block-on-agent-eval", reports="cli,json"
    )

    assert exit_code == 1
    assert report["overall_status"] == "failed"
    assert "FAIL — Publication blocked" in callout(benchmark)
    tier3 = _tier_row(benchmark, "Tier 3")
    assert "**FAILED**" in tier3
    assert "docker missing" in tier3
    assert "Tier 3 live evaluation did not run" in benchmark
    assert re.search(r"AGENT_EVAL\s*│\s*FAIL\s*│", output)
    assert "[FAIL] Validation failed" in output
    assert "✗ FAIL Tier 3 live evaluation skipped: docker missing" in output
    assert "! INCOMPLETE" not in output


# ---------------------------------------------------------------------------
# H1: a partial plugin run with a FAIL verdict and no flag is an advisory FAIL, not "all tiers passed"
# ---------------------------------------------------------------------------


def test_partial_fail_without_the_flag_footer_says_the_fail_was_advisory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tier3 = tier3_result(scores(security=0.0), plugin_provenance=PARTIAL)

    exit_code, _report, benchmark, output = run_validate(monkeypatch, tmp_path, tier3, reports="cli,json")

    assert exit_code == 0
    assert "Tier 3 was advisory in this run" in callout(benchmark)
    assert "Tier 3 verdict FAIL is advisory" in output
    assert not re.search(r"all \d+ tiers? passed", output)
    assert re.search(r"AGENT_EVAL\s*│\s*FAIL \(advisory\)", output)
    assert "[PASS] Required validations passed (Tier 3 verdict FAIL is advisory" in output
    assert "**FAIL**" in _tier_row(benchmark, "Tier 3")


def test_partial_incomplete_without_the_flag_footer_does_not_say_all_tiers_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tier3 = tier3_result(scores(), plugin_provenance=PARTIAL)

    exit_code, _report, benchmark, output = run_validate(monkeypatch, tmp_path, tier3, reports="cli,json")

    assert exit_code == 0
    assert "INCOMPLETE — Required evidence is missing" in callout(benchmark)
    assert "Tier 3 INCOMPLETE is advisory" in output
    assert not re.search(r"all \d+ tiers? passed", output)


# ---------------------------------------------------------------------------
# M15: a confirmed Skill Lift regression fails a partial plugin run too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gated", [False, True])
def test_partial_run_with_a_confirmed_regression_is_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gated: bool
) -> None:
    tier3 = tier3_result(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL, plugin_provenance=PARTIAL)
    flags = ("--block-on-agent-eval",) if gated else ()

    exit_code, _report, benchmark, output = run_validate(monkeypatch, tmp_path, tier3, *flags)

    assert "Skill Lift regression" in benchmark
    if gated:
        assert exit_code == 1
        assert "FAIL — Publication blocked" in callout(benchmark)
        assert "✗ FAIL Tier 3 Skill Lift regression" in output
        assert "! INCOMPLETE" not in output
    else:
        assert exit_code == 0
        assert "FAIL — Not recommended for publication" in callout(benchmark)
        assert "Tier 3 Skill Lift regression is advisory" in output
    assert "**FAIL**" in _tier_row(benchmark, "Tier 3")
