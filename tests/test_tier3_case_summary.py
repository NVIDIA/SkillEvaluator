# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case-first Tier 3 summary and How-to-improve ranking."""

import json
from pathlib import Path

from rich.console import Console

from skillevaluator.tier3.case_summary import render_case_summary, verdict, what_to_fix
from skillevaluator.tier3.result_display import render_result


def _case(*passed: bool, missing: int = 0, prefix: str = "t") -> dict:
    return {
        "attempts": [
            {"attempt": i + 1, "trial": f"{prefix}{i}", "score": 0.9 if p else 0.2, "passed": p}
            for i, p in enumerate(passed)
        ],
        "attempts_missing": missing,
    }


def _agent(with_cases: dict, without_cases: dict) -> dict:
    return {
        "execution_status": "succeeded",
        "pass_at_k": {"with_skill": {"k": 2, "cases": with_cases}, "without_skill": {"k": 2, "cases": without_cases}},
        "lift": {"overall": {"with_skill": 0.7, "without_skill": 0.4, "delta": 0.3}},
        "dimensions_with_skill": {"effectiveness": {"score": 0.5}, "security": {"score": 1.0}},
        "dimensions_without_skill": {"effectiveness": {"score": 0.3}, "security": {"score": 1.0}},
    }


def _trial(run_dir: Path, agent: str, name: str, reward: dict, observation: str = "") -> None:
    trial = run_dir / agent / "with-skill" / "trials" / name
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(reward))
    steps = [{"source": "user", "message": "q"}, {"source": "agent", "message": "done", "observation": observation}]
    (trial / "trajectory.json").write_text(json.dumps({"steps": steps}))


def _reward(activated: bool, reason: str) -> dict:
    return {
        "goal_accuracy": 0.2,
        "details": {
            "skill_execution": {"activation": {"passed": activated}},
            "goal_accuracy": {"reason": reason},
            "behavior_check": {"results": [{"passed": False, "reason": reason}]},
        },
    }


def test_grid_shows_one_row_per_case_with_attempt_marks_per_agent() -> None:
    result = {
        "execution_status": "succeeded",
        "agents": {
            "claude-code": _agent({"c1": _case(True, True), "c2": _case(False, True)}, {"c1": _case(False, False)}),
            "codex": _agent({"c1": _case(True, False), "c2": _case(False, missing=1)}, {"c2": _case(True, True)}),
        },
    }
    output = render_result(result)
    rows = {line.split()[0]: line for line in output.splitlines() if line.startswith("c")}
    lines = output.splitlines()
    header = lines[next(i for i, line in enumerate(lines) if line.startswith("test case")) - 1]
    assert header.split() == ["claude-code", "codex"]
    assert "✗✗ → ✓✓  +100" in rows["c1"] and "- → ✓✗  ?" in rows["c1"]
    assert "- → ✗✓  ?" in rows["c2"] and "✓✓ → ✗⊘  ?" in rows["c2"]
    rates = next(line for line in output.splitlines() if line.startswith("success rate"))
    assert "0% → 75%  +75 pp" in rates and rates.rstrip().endswith("?")
    assert output.index("effectiveness") < output.index("security")
    assert output.count("(without → with skill)") == 2 and "change in pass rate" in output
    assert "skill helps   claude-code +75 pp · codex: 1 attempt(s) did not run" in output


def test_verdict_thresholds() -> None:
    helps = {"a": _agent({"c": _case(True, True)}, {"c": _case(False, False)})}
    flat = {"a": _agent({"c": _case(True, False)}, {"c": _case(False, True)})}
    hurts = {"a": _agent({"c": _case(False, False)}, {"c": _case(True, True)})}
    assert verdict(helps) == ("skill helps", "a +100 pp")
    assert verdict(flat)[0] == "no clear effect"
    assert verdict(hurts)[0] == "skill hurts"
    partial = {"a": _agent({"c": _case(True, missing=1)}, {"c": _case(False, False)})}
    assert verdict(partial) == ("inconclusive", "a: 1 attempt(s) did not run")


def test_what_to_fix_classifies_and_ranks_by_attempts_lost(tmp_path: Path) -> None:
    agents = {
        "opencode": _agent(
            {
                "unused": _case(False, False, prefix="u"),
                "env": _case(False, True, prefix="e"),
                "hurts": _case(True, False, prefix="h"),
                "broken": _case(missing=2),
                "fine": _case(True, True, prefix="f"),
            },
            {"hurts": _case(True, True), "unused": _case(False, False)},
        )
    }
    for name in ("u0", "u1"):
        _trial(tmp_path, "opencode", name, _reward(False, "never read SKILL.md"))
    _trial(tmp_path, "opencode", "e0", _reward(True, "no output"), "ModuleNotFoundError: No module named 'pypdf'")
    _trial(tmp_path, "opencode", "h1", _reward(True, "followed the wrong script"))
    for name in ("broken__a", "broken__b"):
        (tmp_path / "opencode" / "with-skill" / "trials" / name).mkdir(parents=True)
        (tmp_path / "opencode" / "with-skill" / "trials" / name / "failure.json").write_text(
            json.dumps({"error": {"type": "NonZeroAgentExitCodeError", "message": "EPIPE"}})
        )

    fixes = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert [(f["case"], f["kind"], f["attempts_lost"]) for f in fixes] == [
        ("unused", "skill not used", 2),
        ("env", "environment", 1),
        ("hurts", "skill hurts", 1),
    ]
    assert "No module named 'pypdf'" in fixes[1]["what"]
    assert fixes[0]["evidence"] == "opencode/with-skill/trials/u0/trajectory.json"
    assert all(f["action"] for f in fixes)


def test_run_errors_rank_after_skill_issues_and_are_not_blamed_on_the_skill(tmp_path: Path) -> None:
    agents = {"opencode": _agent({"broken": _case(missing=2)}, {})}
    trial = tmp_path / "opencode" / "with-skill" / "trials" / "broken__a"
    trial.mkdir(parents=True)
    (trial / "failure.json").write_text(json.dumps({"error": {"type": "AgentTimeoutError", "message": "900s"}}))

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert fix["kind"] == "run error"
    assert "AgentTimeoutError" in fix["what"]
    assert fix["action"].startswith("Not a skill problem")
    assert fix["evidence"] == "opencode/with-skill/trials/broken__a/failure.json"


def test_multi_agent_issue_scope_names_the_failing_agent(tmp_path: Path) -> None:
    agents = {
        "claude-code": _agent({"c": _case(True, True)}, {}),
        "codex": _agent({"c": _case(False, False, prefix="x")}, {}),
    }
    for name in ("x0", "x1"):
        _trial(tmp_path, "codex", name, _reward(True, "skipped validation step"))

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["scope"], fix["kind"]) == ("codex only", "skill")
    assert "skipped validation step" in fix["what"]


def test_trial_names_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    agents = {"opencode": _agent({"c": {"attempts": [{"trial": "../../etc", "passed": False}]}}, {})}
    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)
    assert fix["evidence"] == ""


def test_eval_prompt_that_restates_the_description_is_flagged_as_an_eval_problem(tmp_path: Path) -> None:
    description = "Use this skill whenever the user wants to create, read, edit, or manipulate Word documents."
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: docx\ndescription: {description}\n---\n# Docx\n")
    run = tmp_path / "run"
    cases = {f"c{i}": _case(False, prefix=f"c{i}_") for i in (1, 2)}
    entries = [{"id": f"c{i}", "question": f"I need help: {description.lower()}"} for i in (1, 2)]
    run.mkdir()
    (run / "dataset_snapshot.json").write_text(json.dumps({"dataset": entries}))
    for i in (1, 2):
        _trial(run, "claude-code", f"c{i}_0", _reward(False, "asked what to do"))

    [fix] = what_to_fix({"agents": {"claude-code": _agent(cases, {})}}, run_dir=run, skill_path=skill, use_llm=False)

    assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("all 2 cases", "eval case", 2)
    assert fix["action"].startswith("Not a skill problem: rewrite this prompt")


def test_setup_command_failure_reports_its_stdout_and_merges_by_error_type(tmp_path: Path) -> None:
    agents = {"claude-code": _agent({"a": _case(missing=1), "b": _case(missing=1)}, {})}
    for case, code in (("a", 35), ("b", 56)):
        trial = tmp_path / "claude-code" / "with-skill" / "trials" / f"{case}__x"
        trial.mkdir(parents=True)
        message = f"Command failed (exit 1): set -euo pipefail; curl -fsSL https://x | bash stdout: curl: ({code}) Recv failure  stderr: None"
        (trial / "failure.json").write_text(
            json.dumps({"error": {"type": "NetworkConnectionError", "message": message}})
        )

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert fix["case"] == "all 2 cases"
    assert fix["what"] == "2 attempt(s) never produced a score: NetworkConnectionError: curl: (35) Recv failure"


def test_failed_condition_rebuilds_case_rows_from_trial_rewards(tmp_path: Path) -> None:
    trials = tmp_path / "claude-code" / "with-skill" / "trials"
    for name, overall in (("ok__1", 0.9), ("low__1", 0.3)):
        (trials / name).mkdir(parents=True)
        (trials / name / "reward.json").write_text(json.dumps({"overall": overall}))
    (trials / "gone__1").mkdir()
    agent = _agent({}, {"ok": _case(False)})
    agent["pass_at_k"]["without_skill"]["pass_threshold"] = 0.5
    agent.update(output_dir=str(tmp_path / "claude-code"), trial_failures={"with_skill": [{"trial": "gone__1"}]})

    output = render_result({"execution_status": "failed", "agents": {"claude-code": agent}})

    assert "1/1 ✓ (0.90)" in output and "0/1 ✗ (0.30)" in output and "⊘" in output


def test_agent_asking_for_files_the_case_lacks_is_flagged_as_missing_input(tmp_path: Path) -> None:
    agents = {"claude-code": _agent({"app": _case(False, False, prefix="m")}, {})}
    for name in ("m0", "m1"):
        trial = tmp_path / "claude-code" / "with-skill" / "trials" / name
        _trial(tmp_path, "claude-code", name, _reward(True, "did not run Playwright"))
        steps = [{"source": "agent", "message": "The workspace appears to be empty. Could you share your app?"}]
        (trial / "trajectory.json").write_text(json.dumps({"steps": steps}))

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["kind"], fix["attempts_lost"]) == ("missing input", 2)
    assert "workspace appears to be empty" in fix["what"]
    assert fix["action"].startswith("Not a skill problem: add the files")


def test_baseline_passing_every_case_is_flagged_as_unable_to_show_lift(tmp_path: Path) -> None:
    agents = {"claude-code": _agent({"a": _case(True, True)}, {"a": _case(True, True)})}

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("all cases", "eval case", 0)
    assert fix["action"].startswith("Add cases the agent fails without the skill")
    console = Console(record=True, width=160)
    render_case_summary({"agents": agents}, console=console, fixes=[fix])
    assert "all cases  [eval case] · claude-code\n" in console.export_text()


def _scored(*scores: float) -> dict:
    return {"attempts": [{"attempt": i + 1, "score": s, "passed": s >= 0.5} for i, s in enumerate(scores)]}


def test_effect_column_names_what_the_skill_changed_per_case() -> None:
    with_cases = {
        "helps": _scored(0.8, 0.8),
        "hurts": _scored(0.3, 0.2),
        "both-fail": _scored(0.4, 0.3),
        "better": _scored(0.9, 0.9),
        "worse": _scored(0.6, 0.6),
        "same": _scored(0.9, 0.85),
        "partial": {**_scored(0.9), "attempts_missing": 1},
    }
    without_cases = {
        "helps": _scored(0.3, 0.6),
        "hurts": _scored(0.9, 0.9),
        "both-fail": _scored(0.2, 0.1),
        "better": _scored(0.6, 0.6),
        "worse": _scored(0.9, 0.8),
        "same": _scored(0.9, 0.9),
        "partial": _scored(0.9, 0.9),
    }
    output = render_result({"execution_status": "succeeded", "agents": {"a": _agent(with_cases, without_cases)}})
    effect = {
        line.split()[0]: line.rstrip().split("  ")[-1].strip() for line in output.splitlines() if line[:1].isalpha()
    }

    assert effect["helps"] == "helps"
    assert effect["hurts"] == "hurts"
    assert effect["both-fail"] == "both fail"
    assert effect["better"] == "better score (+0.30)"
    assert effect["worse"] == "worse score (-0.25)"
    assert effect["same"] == "no change"
    assert effect["partial"] == "inconclusive"
    assert "1/1 ✓⊘ (0.90)" in output and "(0.82) mean score 0-1" in output


def test_multi_agent_grid_wraps_instead_of_truncating_on_an_80_column_terminal() -> None:
    cases = {f"internal-comms-with-a-long-case-name-00{i}": _scored(0.81, 0.81) for i in (1, 2)}
    baseline = {f"internal-comms-with-a-long-case-name-00{i}": _scored(0.53, 0.2) for i in (1, 2)}
    console = Console(record=True, width=80)
    render_case_summary(
        {"agents": {"claude-code": _agent(cases, baseline), "codex": _agent(cases, baseline)}},
        console=console,
        fixes=[],
    )
    output = console.export_text()
    assert "…" not in output
    assert output.count("✓✗ → ✓✓  +50") == 4 and "codex" in output
    assert max(map(len, output.splitlines())) <= 80


def test_multi_agent_dimensions_read_without_then_with_skill() -> None:
    agent = _agent({"c1": _case(True, True)}, {"c1": _case(False, False)})
    console = Console(record=True, width=120)
    render_case_summary({"agents": {"a": agent, "b": agent}}, console=console, fixes=[])
    rows = {line.split()[0]: line for line in console.export_text().splitlines() if line.strip()}
    assert rows["effectiveness"].count("0.30 → 0.50  +0.20") == 2
    assert rows["security"].count("1.00 → 1.00  +0.00") == 2


def test_how_to_improve_says_nothing_to_fix_only_when_the_list_was_computed_and_empty() -> None:
    result = {"agents": {"a": _agent({"c1": _case(True, True)}, {"c1": _case(False, False)})}}
    for fixes, expected in (([], True), (None, False)):
        console = Console(record=True, width=120)
        render_case_summary(result, console=console, fixes=fixes)
        assert ("How to improve: nothing to fix." in console.export_text()) is expected


def test_passing_attempts_that_miss_the_goal_are_reported_with_a_prompt_fix(tmp_path: Path) -> None:
    agents = {
        name: _agent({"c1": _case(True, True, prefix=f"{name}1-"), "c2": _case(True, prefix=f"{name}2-")}, {})
        for name in ("a", "b")
    }
    asked = "I need a few quick details: team name, dates, progress."
    for name in ("a", "b"):
        for trial in (f"{name}1-0", f"{name}1-1", f"{name}2-0"):
            reward = {"goal_accuracy": 0.0, "details": {"goal_accuracy": {"reason": "no update", "end_state": "asked"}}}
            _trial(tmp_path, name, trial, reward)
            steps = [{"source": "user", "message": "q"}, {"source": "agent", "message": asked}]
            (tmp_path / name / "with-skill" / "trials" / trial / "trajectory.json").write_text(
                json.dumps({"steps": steps})
            )
    _trial(tmp_path, "a", "unused", {"goal_accuracy": 1.0})

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"], fix["scope"], fix["attempts_lost"]) == (
        "all 2 cases",
        "goal not met",
        "all agents",
        6,
    )
    assert fix["what"].endswith("goal accuracy 0.00: asked")
    assert fix["action"].startswith("The agent stopped to ask for input")


def test_goal_met_or_unscored_passes_are_not_flagged(tmp_path: Path) -> None:
    agents = {"a": _agent({"c1": _case(True, True)}, {})}
    _trial(tmp_path, "a", "t0", {"goal_accuracy": 0.6})
    _trial(tmp_path, "a", "t1", {})

    assert what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False) == []


def test_never_activated_scores_do_not_claim_a_skill_effect(tmp_path: Path) -> None:
    agent = _agent({"c": _case(False)}, {"c": _case(True)})
    agent["skill_activation"] = {"status": "never_activated"}
    _trial(tmp_path, "a", "t0", _reward(False, "the skill was not loaded"))
    assert verdict({"a": agent})[0] == "inconclusive"
    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
    assert all(fix["kind"] != "skill hurts" for fix in fixes)
    output = render_result({"execution_status": "succeeded", "agents": {"a": agent}})
    assert "skill hurts" not in output
    assert "inconclusive" in output


def test_truncated_attempts_cannot_produce_a_complete_effect_or_clean_bill(tmp_path: Path) -> None:
    case = {**_case(True), "attempt_details_total": 2, "attempt_details_truncated": True}
    agent = _agent({"c": case}, {"c": _case(False, False)})
    assert verdict({"a": agent})[0] == "inconclusive"
    console = Console(record=True, width=160)
    render_case_summary({"agents": {"a": agent}}, console=console, fixes=[])
    output = console.export_text()
    assert "inconclusive" in output and "nothing to fix" not in output


def test_truncated_case_sample_does_not_claim_baseline_saturation(tmp_path: Path) -> None:
    agent = _agent({"c": _case(True)}, {"c": _case(True)})
    agent["pass_at_k"]["without_skill"].update(case_details_truncated=True, total_cases=100)
    assert verdict({"a": agent})[0] == "inconclusive"
    assert what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False) == []


def test_new_summary_redacts_known_credentials_and_keeps_case_names_literal(tmp_path: Path, monkeypatch) -> None:
    secret = "review-fake-environment-credential-7135"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    case_id = f"[bold]case[/bold]-{secret}"
    agent = _agent({case_id: _case(False)}, {case_id: _case(False)})
    (tmp_path / "what_to_fix.json").write_text(
        json.dumps(
            [
                {
                    "case": case_id,
                    "scope": secret,
                    "kind": "skill",
                    "what": secret,
                    "action": secret,
                    "evidence": secret,
                    "attempts_lost": 1,
                }
            ]
        )
    )
    output = render_result({"execution_status": "succeeded", "run_dir": str(tmp_path), "agents": {secret: agent}})
    assert secret not in output
    assert "[bold]case[/bold]" in output


def test_trial_artifact_symlinks_cannot_supply_advice(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "reward.json").write_text(json.dumps(_reward(True, "outside-private-reason")))
    (outside / "trajectory.json").write_text(
        json.dumps({"steps": [{"source": "agent", "message": "outside-private-answer"}]})
    )
    run = tmp_path / "run"
    trials = run / "a" / "with-skill" / "trials"
    trials.mkdir(parents=True)
    (trials / "t0").symlink_to(outside, target_is_directory=True)
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=run, use_llm=False)
    assert "outside-private" not in json.dumps(fix)
    assert fix["evidence"] == ""


def test_symlinked_reward_file_cannot_supply_advice(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_reward(True, "outside-private-reason")))
    trial = tmp_path / "run" / "a" / "with-skill" / "trials" / "t0"
    trial.mkdir(parents=True)
    (trial / "reward.json").symlink_to(outside)
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path / "run", use_llm=False)
    assert "outside-private-reason" not in json.dumps(fix)


def test_parent_agent_component_cannot_escape_run_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    trial = tmp_path / "with-skill" / "trials" / "t0"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(_reward(True, "outside-private-reason")))
    [fix] = what_to_fix({"agents": {"..": _agent({"c": _case(False)}, {})}}, run_dir=run, use_llm=False)
    assert "outside-private-reason" not in json.dumps(fix)
    assert fix["evidence"] == ""


def test_advice_is_redacted_before_it_is_saved(tmp_path: Path, monkeypatch) -> None:
    secret = "review-fake-credential-in-judge-reason-8745"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _trial(tmp_path, "a", "t0", _reward(True, secret))
    fixes = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert secret not in json.dumps(fixes)
    assert "<redacted>" in json.dumps(fixes)


def test_issue_scope_names_a_subset_of_more_than_two_agents(tmp_path: Path) -> None:
    agents = {name: _agent({"c": _case(name == "c")}, {}) for name in ("a", "b", "c")}
    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)
    assert fix["scope"] == "a, b only"


def test_missing_attempts_suppress_the_single_agent_summary_delta() -> None:
    agent = _agent({"c": _case(True, missing=1)}, {"c": _case(False)})
    output = render_result({"execution_status": "succeeded", "agents": {"a": agent}})
    row = next(line for line in output.splitlines() if line.startswith("success rate"))
    assert "+100" not in row


def test_advisory_prompt_redacts_credentials_before_the_model_call(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    secret = "review-fake-advisory-prompt-secret-9375"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _trial(tmp_path, "a", "t0", _reward(True, secret))
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"# Skill\n\nPrivate setup: {secret}")
    prompts = []

    def call(prompt, **kwargs):
        prompts.append(prompt)
        return '["Improve the skill instructions."]', None

    monkeypatch.setattr(llm_judge, "call_public_llm", call)
    what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path, skill_path=skill)
    assert len(prompts) == 1 and secret not in prompts[0]
    assert "\n\nSKILL.md:\n# Skill\n" in prompts[0]


def test_long_error_logs_keep_the_root_error_from_the_bounded_tail(tmp_path: Path) -> None:
    trial = tmp_path / "a" / "with-skill" / "trials" / "c__1"
    trial.mkdir(parents=True)
    (trial / "failure.json").write_text(json.dumps({"error": {"type": "AgentExitError", "message": "wrapper failed"}}))
    (trial / "agent.txt").write_text("irrelevant log output\n" * 10000 + "Error: specific-tail-cause\n")
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(missing=1)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert "specific-tail-cause" in fix["what"]


def test_failed_condition_rebuild_does_not_follow_a_symlinked_trials_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "c__1"
    outside.mkdir(parents=True)
    (outside / "reward.json").write_text(json.dumps({"overall": 0.9}))
    agent_dir = tmp_path / "run" / "a"
    (agent_dir / "with-skill").mkdir(parents=True)
    (agent_dir / "with-skill" / "trials").symlink_to(outside.parent, target_is_directory=True)
    agent = _agent({}, {})
    agent["output_dir"] = str(agent_dir)
    console = Console(record=True, width=160)
    assert not render_case_summary({"agents": {"a": agent}}, console=console, fixes=None)


def test_different_condition_score_contracts_do_not_claim_a_causal_skill_effect(tmp_path: Path) -> None:
    for fields in (
        {"metrics_with_skill": ["security"], "metrics_without_skill": ["accuracy"]},
        {"score_policy_with_skill": "dimension-mean/v1", "score_policy_without_skill": "legacy-flat/v1"},
    ):
        agent = {**_agent({"c": _case(False)}, {"c": _case(True)}), **fields}
        assert verdict({"a": agent})[0] == "inconclusive"
        fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
        assert all(fix["kind"] != "skill hurts" for fix in fixes)
        output = render_result({"execution_status": "succeeded", "agents": {"a": agent}})
        assert "skill hurts" not in output
        rates = next(line for line in output.splitlines() if line.startswith("success rate"))
        assert "-100" not in rates


def test_disjoint_case_coverage_does_not_produce_an_overall_effect() -> None:
    agent = _agent(
        {"shared": _case(True), "with-only": _case(True)}, {"shared": _case(False), "baseline-only": _case(False)}
    )
    assert verdict({"a": agent})[0] == "inconclusive"


def test_logical_multistep_goal_counts_average_steps_once(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import goal_not_met_count

    for label, goals, expected in (("missed", [0.0, 0.0], 1), ("mixed", [0.0, 1.0], 0)):
        run = tmp_path / label
        trial_id = "case__attempt1"
        for i, goal in enumerate(goals, 1):
            _trial(run, "a", f"{trial_id}__step-{i}", {"trial_id": trial_id, "goal_accuracy": goal})
        case = {"attempts": [{"trial": trial_id, "passed": True, "score": 0.9}]}
        source = {"agents": {"a": _agent({"case": case}, {"case": _case(False)})}}
        assert goal_not_met_count(source, run_dir=run) == expected
        fixes = what_to_fix(source, run_dir=run, use_llm=False)
        assert sum(f["attempts_lost"] for f in fixes if f["kind"] == "goal not met") == expected


def test_failed_conditions_keep_rebuilt_scores_diagnostic(tmp_path: Path) -> None:
    agent = _agent({"c": _case(True)}, {"c": _case(False)})
    agent["conditions"] = {
        "with_skill": {"execution_status": "failed"},
        "without_skill": {"execution_status": "succeeded"},
    }
    assert verdict({"a": agent})[0] == "inconclusive"
    output = render_result({"execution_status": "failed", "agents": {"a": agent}})
    assert "skill helps" not in output


def test_saved_fixes_do_not_follow_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    content = json.dumps(
        [
            {
                "case": "private-external-case",
                "scope": "private",
                "kind": "skill",
                "what": "private",
                "action": "private",
                "evidence": "private",
                "attempts_lost": 1,
            }
        ]
    )
    outside.write_text(content)
    run = tmp_path / "run"
    run.mkdir()
    (run / "what_to_fix.json").symlink_to(outside)
    agent = _agent({"c": _case(True)}, {"c": _case(False)})
    output = render_result({"execution_status": "succeeded", "run_dir": str(run), "agents": {"a": agent}})
    assert "private-external-case" not in output
    assert outside.read_text() == content


def _goal_miss(run_dir: Path, agent: str, trial: str, end_state: str) -> None:
    _trial(run_dir, agent, trial, {"goal_accuracy": 0.0, "details": {"goal_accuracy": {"end_state": end_state}}})


def test_goal_not_met_uses_the_canned_fix_only_when_the_agent_stopped_for_input(tmp_path: Path) -> None:
    outcomes = {
        "The agent made HTTP requests to the API and returned the wrong totals.": False,
        "The agent fixed an import that was not found and wrote the report.": False,
        "The agent asked the user for the report file.": True,
        "report.pdf does not exist in the workspace, so nothing was extracted.": True,
    }
    for index, (outcome, stopped) in enumerate(outcomes.items()):
        run_dir = tmp_path / str(index)
        _goal_miss(run_dir, "a", "x0", outcome)

        [fix] = what_to_fix(
            {"agents": {"a": _agent({"c1": _case(True, prefix="x")}, {})}}, run_dir=run_dir, use_llm=False
        )

        assert fix["asks_for_input"] is stopped, outcome
        assert fix["action"].startswith("The agent stopped to ask for input") is stopped, outcome


def test_merged_goal_issue_keeps_the_canned_fix_only_if_every_agent_stopped_for_input(tmp_path: Path) -> None:
    agents = {name: _agent({"c1": _case(True, prefix=name)}, {}) for name in ("a", "b")}
    _goal_miss(tmp_path, "a", "a0", "The agent asked for the sprint details.")
    _goal_miss(tmp_path, "b", "b0", "The agent wrote the wrong section headings.")

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["scope"], fix["asks_for_input"]) == ("all agents", False)


def test_case_with_a_failure_also_reports_passes_that_missed_the_goal(tmp_path: Path) -> None:
    agents = {"a": _agent({"c1": _case(True, False, prefix="m")}, {})}
    _goal_miss(tmp_path, "a", "m0", "The agent asked for the data.")
    _trial(tmp_path, "a", "m1", _reward(True, "wrong totals"))

    kinds = {f["kind"] for f in what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)}

    assert kinds == {"skill", "goal not met"}


def test_fixes_redact_environment_secrets(tmp_path: Path, monkeypatch) -> None:
    secret = "review-fake-goal-credential-5521"
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", secret)
    agents = {"a": _agent({"c1": _case(True, prefix="s")}, {})}
    _goal_miss(tmp_path, "a", "s0", f"The agent printed {secret} and asked for the file.")

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert secret not in json.dumps(fix)
    assert "<redacted>" in fix["what"]


def test_goal_not_met_heading_counts_attempts_that_missed_the_goal() -> None:
    result = {"agents": {"a": _agent({"c1": _case(True)}, {"c1": _case(False)})}}
    fix = {"case": "c1", "kind": "goal not met", "scope": "a", "attempts_lost": 2, "what": "w", "action": "f"}
    console = Console(record=True, width=120)

    render_case_summary(result, console=console, fixes=[fix])

    text = console.export_text()
    assert "2 attempt(s) missed the goal" in text
    assert "lost 2" not in text
