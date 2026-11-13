# SPDX-License-Identifier: Apache-2.0
"""GitLab evidence selection must agree with provider-job admission."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())


@pytest.fixture(
    params=[
        ("push", "42", "true", "true", True),
        ("web", "42", "true", "true", True),
        ("schedule", "42", "true", "true", False),
        ("api", "42", "true", "true", False),
        ("trigger", "42", "true", "true", False),
        ("pipeline", "42", "true", "true", False),
        ("merge_request_event", "42", "false", "true", True),
        ("merge_request_event", "99", "false", "true", False),
        ("push", "42", "false", "true", False),
        ("push", "42", "true", "false", False),
    ],
    ids=[
        "push",
        "web",
        "schedule",
        "api",
        "trigger",
        "pipeline",
        "same-project-mr",
        "fork-mr",
        "unprotected",
        "keyless",
    ],
)
def admission_case(request):
    source, project, protected, llm, enriched = request.param
    return {
        "PATH": os.environ["PATH"],
        "CI_PIPELINE_SOURCE": source,
        "CI_PROJECT_ID": "42",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": project,
        "CI_COMMIT_BRANCH": "" if source == "merge_request_event" else "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": protected,
        "SKILL_EVAL_TIER1_LLM": llm,
    }, enriched


def job_enabled(job, env):
    # These rules use GitLab's string equality and boolean-expression subset,
    # which has the same semantics in bash's conditional expression syntax.
    conditions = []
    for rule in job["rules"]:
        expression = re.sub(r"\$([A-Z0-9_]+)", r'"${\1:-}"', rule["if"])
        status = 1 if rule.get("when") == "never" else 0
        conditions.append(f"if [[ {expression} ]]; then exit {status}; fi")
    result = subprocess.run(["bash", "-c", "\n".join([*conditions, "exit 1"])], env=env, check=False)
    assert result.returncode in {0, 1}
    return result.returncode == 0


def mr_checkout(workspace, env, *, skill_change=True):
    """Give an MR route the real checkout and pinned tools its skip check reads."""
    import sys

    git = ["git", "-C", str(workspace), "-c", "user.name=CI", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    (workspace / "skills" / "alpha").mkdir(parents=True)
    (workspace / "skills" / "alpha" / "SKILL.md").write_text("# alpha\n")
    subprocess.run([*git, "add", "skills"], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    base = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changed = workspace / ("skills/alpha/SKILL.md" if skill_change else ".gitlab-ci.yml")
    changed.write_text("# changed\n")
    subprocess.run([*git, "add", "--", str(changed.relative_to(workspace))], check=True)
    subprocess.run([*git, "commit", "-qm", "head"], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    bin_dir = workspace.parent / f"{workspace.name}-bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        SKILL_EVAL_TOOLS_DIR=str(ROOT),
        CI_PROJECT_DIR=str(workspace),
        DIFF_BASE_SHA=base,
        DIFF_HEAD_SHA=head,
        REPORT_COMMIT=head,
        SKILL_EVAL_PATHS="skills",
        SKILL_EVAL_SCAN_MODE="auto",
    )


def route_evidence(config, env, workspace):
    if env.get("CI_PIPELINE_SOURCE") == "merge_request_event" and "SKILL_EVAL_TOOLS_DIR" not in env:
        mr_checkout(workspace, env)
    script = "\n".join(config["skill-evaluation-aggregate"]["script"])
    routing = script[script.index("mkdir -p skill-evaluator-results/tier1") :]
    routing, separator, _ = routing.partition('python -I "$SKILL_EVAL_TOOLS_DIR/ci/run_tiers.py" aggregate')
    assert separator
    subprocess.run(["bash", "-euc", routing], cwd=workspace, env=env, check=True)


def test_enriched_job_is_scheduled_only_for_supported_provider_admission(config, admission_case):
    env, enriched = admission_case

    assert job_enabled(config["skill-evaluation"], env)
    assert job_enabled(config["skill-evaluation-tier1-llm"], env) is enriched


def test_aggregate_uses_keyless_evidence_when_provider_job_is_not_eligible(config, admission_case, tmp_path):
    env, enriched = admission_case
    baseline = tmp_path / "skill-evaluator-report"
    baseline.mkdir()
    (baseline / "evidence.txt").write_text("keyless")
    if enriched:
        authenticated = tmp_path / "skill-evaluator-enriched-tier1"
        authenticated.mkdir()
        (authenticated / "evidence.txt").write_text("enriched")

    route_evidence(config, env, tmp_path)

    assert (tmp_path / "skill-evaluator-results/tier1/evidence.txt").read_text() == (
        "enriched" if enriched else "keyless"
    )


def test_eligible_enriched_run_never_substitutes_keyless_evidence(config, tmp_path):
    env = {
        "PATH": os.environ["PATH"],
        "CI_PIPELINE_SOURCE": "push",
        "CI_PROJECT_ID": "42",
        "CI_COMMIT_BRANCH": "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": "true",
        "SKILL_EVAL_TIER1_LLM": "true",
    }
    baseline = tmp_path / "skill-evaluator-report"
    baseline.mkdir()
    (baseline / "evidence.txt").write_text("keyless")

    route_evidence(config, env, tmp_path)

    assert list((tmp_path / "skill-evaluator-results/tier1").iterdir()) == []


@pytest.mark.parametrize(("skill_change", "expected"), [(True, "enriched"), (False, "keyless")])
def test_same_project_mr_routes_by_recomputed_skill_change(config, tmp_path, skill_change, expected):
    """The real pinned-tools check, not an artifact, decides whether the paid scan was skipped."""
    env = {
        "PATH": os.environ["PATH"],
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_PROJECT_ID": "42",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
        "CI_DEFAULT_BRANCH": "main",
        "SKILL_EVAL_TIER1_LLM": "true",
    }
    mr_checkout(tmp_path, env, skill_change=skill_change)
    (tmp_path / "skill-evaluator-report").mkdir()
    (tmp_path / "skill-evaluator-report" / "evidence.txt").write_text("keyless")
    (tmp_path / "skill-evaluator-enriched-tier1").mkdir()
    (tmp_path / "skill-evaluator-enriched-tier1" / "evidence.txt").write_text("enriched")
    # A forged skip marker must not change the decision.
    (tmp_path / "skill-evaluator-enriched-tier1" / "enriched-tier1-skipped.json").write_text("{}")

    route_evidence(config, env, tmp_path)

    assert (tmp_path / "skill-evaluator-results/tier1/evidence.txt").read_text() == expected
