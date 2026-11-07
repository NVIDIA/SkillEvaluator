# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.reporting.console_ui import ViewProgressReporter, _fit_row, tier3_case_rows, tier3_fix_rows
from skillevaluator.tier3.case_summary import live_case_marks


def _text(rows):
    return "\n".join(f"{r.label:<12}" + "".join(chunk for chunk, _ in r.segments) for r in rows)


def _marks(with_skill, without):
    ids = [f"pdf-00{i}" for i in range(1, len(with_skill) + 1)]
    return {
        "claude-code": {
            "with_skill": dict(zip(ids, with_skill, strict=True)),
            "without_skill": dict(zip(ids, without, strict=True)),
        }
    }


def test_final_rows_list_every_case_with_counts():
    rows = tier3_case_rows(_marks([["✓", "✓"], ["✗", "✓"], ["✓", "·"]], [["✗", "✗"], ["✓", "✓"], ["⊘", "✓"]]))
    text = _text(rows)
    assert "3 test cases (pdf-…) · k=2" in text
    assert "with skill  ✓ 001  ✗ 002  ✓ 003    2 passed · 1 failed" in text
    assert "without     ✗ 001  ✓ 002  ⊘ 003    1 passed · 1 failed · 1 did not run" in text


def test_live_rows_mark_running_and_waiting_cases():
    marks = _marks([["✓", "✓"], ["✓"], []], [["✗"], [], []])
    running = {("claude-code", "with_skill", "pdf-002")}
    text = _text(tier3_case_rows(marks, k=2, running=running, progress=(4, 12)))
    assert "4/12 attempts" in text
    assert "✓ 001  ▸ 002  · 003    1 passed · 0 failed · 2 to go" in text
    assert "✗ 001  · 002  · 003    0 passed · 1 failed · 2 to go" in text


def test_multi_agent_rows_and_long_case_lists_wrap():
    ids = [f"case-{i:03d}" for i in range(10)]
    agent = {"with_skill": {c: ["✓"] for c in ids}, "without_skill": {c: ["✗"] for c in ids}}
    rows = tier3_case_rows({"a": agent, "b": agent}, k=1)
    labels = [r.label for r in rows]
    assert labels == ["agent", "a", " with skill", "", " without", "", "b", " with skill", "", " without", ""]
    assert "10 passed · 0 failed" in _text(rows)


def test_no_cases_renders_nothing():
    assert tier3_case_rows({}) == []


def test_fix_rows():
    assert tier3_fix_rows(None) == []
    assert tier3_fix_rows([]) == []
    finding = {
        "case": "pdf-001, pdf-002",
        "kind": "goal not met",
        "scope": "claude-code",
        "what": "x: asked for a file",
        "action": "add the file",
    }
    rows = tier3_fix_rows([finding] * 4)
    text = _text(rows)
    assert rows[1].label == "what to fix"
    assert [r.label for r in rows if r.glyph in "123" and r.glyph != " "] == ["001, 002"] * 3
    assert "Goal not met: asked for a file" in text and "claude-code" not in text
    assert "→ add the file" in text and "+1 more in detailed reports" in text
    assert "(claude-code)" in _text(tier3_fix_rows([finding], agents=2))
    long = {
        **finding,
        "case": "pdf-001, pdf-002, pdf-003",
        "what": "x: " + "missing input " * 12,
        "action": "add the files " * 12,
    }
    rows = tier3_fix_rows([long])
    assert rows[2].label == "001-003"
    for room in (40, 76):
        lines = [line for r in rows[1:] for line in _fit_row(r, room)]
        assert all(len("".join(c for c, _ in line)) <= room for line in lines)
        assert "".join(c for c, _ in lines[0]) == "─" * room
    assert _text(rows).count("missing input") == 12


def test_fix_section_for_tier1_and_tier2():
    from skillevaluator.models.result import Finding, Severity, ValidationResult
    from skillevaluator.reporting.console_ui import summarize_tier1, summarize_tier2

    schema = ValidationResult(validator_name="Schema & Repository Governance", passed=False)
    schema.add_finding(
        Finding(
            "SCHEMA",
            Severity.HIGH,
            "author",
            "Author not specified in metadata",
            "SKILL.md",
            suggestion="Add 'metadata.author'",
        )
    )
    secrets = ValidationResult(
        validator_name="Secrets Detection", passed=False, metadata={"incomplete_scans": ["gitleaks"]}
    )
    secrets.warnings.append("gitleaks not installed - secrets scanning skipped. Install with: brew install gitleaks")
    clean = ValidationResult(validator_name="Version Check", passed=True)
    _ok, rows = summarize_tier1([schema, secrets, clean])
    text = _text(rows)
    assert "1/3 checks passed  ·  1 failed  ·  1 didn't run" in text
    assert "Author not specified in metadata\n" + " " * 12 + "→ Add 'metadata.author'" in text
    assert "secrets didn't run: gitleaks not installed\n" + " " * 12 + "→ brew install gitleaks" in text
    assert [r.label for r in rows if r.glyph in {"1", "2"}][1] == "setup"
    assert summarize_tier1([clean])[1][-1].label != ""

    dup = ValidationResult(validator_name="Context Deduplication", passed=False)
    dup.add_finding(
        Finding(
            "CONTENT_DEDUP",
            Severity.HIGH,
            "context_dedup",
            'Duplicate content found within forms.md:\n  "### A.4: Boxes" in forms.md (lines 1-2)\n  vs "## Step 2" in forms.md (lines 9-10)',
            "skill/forms.md",
            suggestion="Keep it under Step 2",
        )
    )
    rows = summarize_tier2([dup])[2]
    flat = " ".join("".join(c for c, _ in r.segments) for r in rows)
    assert 'within forms.md: "### A.4: Boxes" lines 1-2 · "## Step 2" lines 9-10' in flat
    assert "→ Keep it under Step 2" in _text(rows)
    assert next(r.label for r in rows if r.glyph == "1") == "forms.md"
    assert not any(r.label == "what to fix" for r in summarize_tier2([ValidationResult(validator_name="d")])[2])


def _trial(job: Path, name: str, result: dict | None) -> None:
    trial = job / name
    trial.mkdir(parents=True)
    if result is not None:
        (trial / "result.json").write_text(json.dumps(result))


def test_live_case_marks_reads_harbor_jobs(tmp_path):
    for variant in ("with", "without"):
        for case in ("pdf-001", "pdf-002"):
            (tmp_path / "_harbor-tasks" / "claude-code" / variant / case).mkdir(parents=True)
    jobs = tmp_path / "_harbor-jobs"
    done = {"finished_at": "t", "verifier_result": {"rewards": {"overall": 0.8}}}
    _trial(jobs / "pdf-claude-code-with", "pdf-001__a", done)
    _trial(jobs / "pdf-claude-code-with", "pdf-001__b", {**done, "verifier_result": {"rewards": {"overall": 0.2}}})
    _trial(jobs / "pdf-claude-code-with", "pdf-002__c", {"finished_at": None})
    _trial(jobs / "pdf-claude-code-without", "pdf-001__d", {"finished_at": "t", "exception_info": {"x": 1}})
    marks, running = live_case_marks(tmp_path, ["claude-code"])
    assert marks["claude-code"]["with_skill"] == {"pdf-001": ["✓", "✗"], "pdf-002": []}
    assert marks["claude-code"]["without_skill"] == {"pdf-001": ["⊘"], "pdf-002": []}
    assert running == {("claude-code", "with_skill", "pdf-002")}


def test_reporter_pushes_case_rows_once_run_dir_is_known(tmp_path):
    (tmp_path / "_harbor-tasks" / "claude-code" / "with" / "pdf-001").mkdir(parents=True)
    pushed = []
    reporter = ViewProgressReporter(lambda _lines: None, on_cases=pushed.append, poll_seconds=60)
    reporter.start(SimpleNamespace(output_dir=None, agents=("claude-code",)))
    assert pushed == []
    plan = SimpleNamespace(
        output_dir=str(tmp_path), agents=("claude-code",), baseline=False, attempts=2, matrix_trials=2
    )
    reporter.start(plan)
    reporter.close()
    assert "cases       1 test case · k=2" in _text(pushed[-1])
    assert "with skill  · pdf-001    0 passed · 0 failed · 1 to go" in _text(pushed[-1])
    assert "without" not in _text(pushed[-1])


def test_setup_fix_names_failed_download_and_recovery():
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.console_ui import _setup_fix

    secrets = ValidationResult(
        validator_name="Secrets Detection", passed=False, metadata={"incomplete_scans": ["gitleaks"]}
    )
    secrets.warnings.append(
        "couldn't download Gitleaks 8.30.0 (connection refused). Install with: brew install gitleaks"
    )
    item = _setup_fix([secrets])
    assert item.problem == "secrets didn't run: couldn't download Gitleaks 8.30.0 (connection refused)"
    assert item.actions == [
        "check network access to github.com, then rerun",
        "or install it yourself: brew install gitleaks",
        "or skip it: --skip secrets",
    ]


def test_setup_fix_names_failed_skillspector_setup():
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.console_ui import _setup_fix

    security = ValidationResult(
        validator_name="Security Scan",
        passed=False,
        metadata={"incomplete_scans": ["skillspector"]},
    )
    security.errors.append(
        "couldn't set up SkillSpector 2.13.3 (network is unreachable (os error 101)). "
        "Install with: uv tool install skillspector.whl"
    )
    item = _setup_fix([security])
    assert item.problem == (
        "security didn't run: couldn't set up SkillSpector 2.13.3 (network is unreachable (os error 101))"
    )
    assert item.actions == [
        "check network access to github.com and pypi.org, then rerun",
        "or install it yourself: uv tool install skillspector.whl",
        "or skip it: --skip security",
    ]


def _security_without_llm():
    from skillevaluator.models.result import ValidationResult

    security = ValidationResult(
        validator_name="Security Scan",
        passed=False,
        metadata={"incomplete_scans": ["skillspector-llm", "llm-verification"], "scanners": {"skillspector": "2.13.3"}},
    )
    security.errors.append("skillspector-llm failed with unexpected exit code 2: scanner diagnostics were redacted")
    return security


def test_llm_fix_asks_for_the_key_not_a_scanner_install(monkeypatch):
    from skillevaluator import provider_config
    from skillevaluator.reporting.console_ui import _llm_fix

    def no_key():
        raise provider_config.ProviderConfigurationError(
            "SKILL_EVAL_LLM_API_KEY is required for the selected provider.\n\n"
            "Set it in your shell:\n  export SKILL_EVAL_LLM_API_KEY='your-api-key'\n"
        )

    monkeypatch.setattr(provider_config, "resolve_llm_provider", no_key)
    item = _llm_fix([_security_without_llm()])
    assert item.problem == (
        "security LLM review didn't run: SKILL_EVAL_LLM_API_KEY is required for the selected provider"
    )
    assert item.actions == [
        "export SKILL_EVAL_LLM_API_KEY='your-api-key'",
        "or rerun without --llm / --llm-verify",
    ]


def test_llm_fix_without_a_provider_gives_every_export(monkeypatch):
    from skillevaluator import provider_config
    from skillevaluator.reporting.console_ui import _llm_fix

    monkeypatch.delenv("SKILL_EVAL_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_API_KEY", raising=False)
    for name in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(provider_config.ProviderConfigurationError):
        provider_config.resolve_llm_provider()
    item = _llm_fix([_security_without_llm()])
    assert item.problem == "security LLM review didn't run: No provider is configured"
    assert item.actions[0] == "export SKILL_EVAL_LLM_PROVIDER=nv_build NVIDIA_API_KEY='your-api-key'"


def test_llm_fix_points_at_doctor_when_the_provider_call_fails(monkeypatch):
    from skillevaluator import provider_config
    from skillevaluator.reporting.console_ui import _llm_fix

    monkeypatch.setattr(provider_config, "resolve_llm_provider", object)
    security = _security_without_llm()
    security.metadata["incomplete_scans"] = ["llm-verification"]
    item = _llm_fix([security])
    assert item.problem == "security LLM review didn't run: the LLM provider call failed"
    assert item.actions == [
        "check the key and model with: skillevaluator doctor --tier 2 --verify-models",
        "or rerun without --llm-verify",
    ]


def test_scanners_row_keeps_the_version_when_only_the_llm_stage_failed():
    from skillevaluator.reporting.console_ui import _only_llm_stages_missing, _scanners_row

    security = _security_without_llm()
    assert _only_llm_stages_missing(security)
    assert "SkillSpector 2.13.3, LLM didn't run" in _text(_scanners_row([security]))
    security.metadata["incomplete_scans"] = ["skillspector"]
    assert not _only_llm_stages_missing(security)


def test_tier1_summary_counts_named_checks_and_their_sub_checks():
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.console_ui import summarize_tier1

    results = [ValidationResult(validator_name=name, passed=True) for name in ("Code Risk Analysis", "Schema")]
    results.append(_security_without_llm())
    for result, performed in zip(results, (2, 10, 1), strict=True):
        result.summary.checks_performed = performed
    secrets = ValidationResult(validator_name="Secrets Detection", passed=False)
    secrets.errors.append("leaked token")
    results.append(secrets)
    _ok, rows = summarize_tier1(results, lineup=["code-integrity", "schema", "security"])
    assert "1/3 checks passed (13 sub-checks)  ·  1 failed  ·  1 didn't run" in _text(rows)
