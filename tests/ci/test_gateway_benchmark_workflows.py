# SPDX-License-Identifier: Apache-2.0
"""Shared CI wiring for enriched security and independent live-agent models."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def workflow():
    return yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)


def test_enriched_tier1_requires_admission_and_keeps_the_keyless_path():
    document = workflow()
    assert document["on"]["workflow_call"]["inputs"]["tier1-llm"]["default"] == "false"
    baseline = document["jobs"]["tier1"]
    assert baseline["with"]["llm"] == "false"
    assert "always()" in baseline["if"]
    assert "needs.eligibility.outputs.allowed" in baseline["with"]["artifact-name"]
    assert baseline["with"]["artifact-name"].endswith("|| 'skill-evaluator-tier1' }}")
    enriched = document["jobs"]["tier1-llm"]
    # A native local Tier 3 runner need not have Docker. Tier 1 keeps the
    # separate image-capable runner used by its keyless baseline.
    assert enriched["runs-on"] == "ubuntu-24.04"
    assert "inputs.tier1-llm" in enriched["if"]
    assert "needs.eligibility.outputs.allowed == 'true'" in enriched["if"]
    assert enriched["environment"] == "${{ inputs.environment }}"
    check = next(step for step in enriched["steps"] if step.get("id") == "current-head")
    evaluate = next(step for step in enriched["steps"] if "run_tier1_image.py" in step.get("run", ""))
    assert enriched["steps"].index(check) < enriched["steps"].index(evaluate)
    assert "steps.current-head.outputs.allowed == 'true'" in evaluate["if"]
    assert "secrets." not in str(check)
    assert "tier1-llm" in document["jobs"]["aggregate"]["needs"]


def test_native_jobs_forward_gateway_settings_only_at_protected_execution_steps():
    document = workflow()
    inputs = document["on"]["workflow_call"]["inputs"]
    for name in ("base-url", "embedding-base-url", "tier3-provider", "tier3-model", "tier3-base-url", "agent-models"):
        assert inputs[name]["default"] == ""
    paid = document["jobs"]["paid-tiers"]
    assert paid["continue-on-error"] == "true"
    assert paid["env"]["SKILL_EVAL_TIER3_PROVIDER"] == "${{ inputs.tier3-provider }}"
    assert paid["env"]["SKILL_EVAL_AGENT_MODELS"] == "${{ inputs.agent-models }}"
    assert "secrets." not in str(paid["env"])
    for step in paid["steps"]:
        if "secrets.LLM_API_KEY" in str(step):
            assert "steps.current-head.outputs.allowed == 'true'" in step["if"]
            assert step["env"]["SKILL_EVAL_LLM_API_KEY"] == "${{ secrets.LLM_API_KEY }}"


def test_gitlab_enrichment_is_protected_and_does_not_need_docker_in_docker():
    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    job = document["skill-evaluation-tier1-llm"]
    assert job["environment"]["name"] == "$SKILL_EVAL_ENVIRONMENT"
    assert job["services"] == []
    assert "tags" not in job  # Use the existing untagged Tier 1 image runner.
    script = "\n".join(job["script"])
    assert script.index("eligible_gitlab.py") < script.index("run_tier1_image.py")
    assert "--in-image" in script
    rules = str(job["rules"])
    assert 'SKILL_EVAL_TIER1_LLM != "true"' in rules
    assert "CI_MERGE_REQUEST_SOURCE_PROJECT_ID != $CI_PROJECT_ID" in rules
    needs = document["skill-evaluation-aggregate"]["needs"]
    assert any(item["job"] == "skill-evaluation-tier1-llm" for item in needs)


def _aggregate_selects_enriched(env):
    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    script = "\n".join(document["skill-evaluation-aggregate"]["script"])
    selection = script[script.index("enriched=false") : script.index('if [ "$enriched" = true ]')]
    result = subprocess.run(
        ["sh", "-c", "set -eu\n" + selection + 'printf %s "$enriched"'],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout == "true"


@pytest.mark.parametrize(
    ("source", "admitted"),
    [("push", True), ("web", True), ("schedule", False), ("api", False), ("trigger", False), ("pipeline", False)],
)
def test_gitlab_enriched_tier1_matches_default_branch_admission(source, admitted):
    """Sources that admission rejects keep the keyless baseline instead of failing Tier 1."""
    env = {
        "SKILL_EVAL_TIER1_LLM": "true",
        "CI_PIPELINE_SOURCE": source,
        "CI_PROJECT_ID": "42",
        "CI_COMMIT_BRANCH": "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": "true",
    }
    assert _aggregate_selects_enriched(env) is admitted
    spec = importlib.util.spec_from_file_location("gate_sources", ROOT / "ci/eligible_gitlab.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gate._default_branch_candidate({**env, "CI_COMMIT_SHA": "a" * 40}) is admitted
    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    default_rule = document["skill-evaluation-tier1-llm"]["rules"][-1]["if"]
    assert '$CI_PIPELINE_SOURCE == "push" || $CI_PIPELINE_SOURCE == "web"' in default_rule


def test_gitlab_same_project_mr_still_selects_enriched_tier1():
    env = {
        "SKILL_EVAL_TIER1_LLM": "true",
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_PROJECT_ID": "42",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
        "CI_DEFAULT_BRANCH": "main",
    }
    assert _aggregate_selects_enriched(env)
    assert not _aggregate_selects_enriched({**env, "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "7"})


def test_benchmark_publisher_is_separate_from_read_only_pr_feedback():
    document = yaml.load((ROOT / ".github/workflows/publish-benchmarks.yml").read_text(), Loader=yaml.BaseLoader)
    job = document["jobs"]["benchmarks"]
    assert document["permissions"]["contents"] == "write"
    assert "workflow_run" in job["if"] and "push" in job["if"] and "workflow_dispatch" in job["if"]
    assert "head_repository.full_name == github.repository" in job["if"]
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ inputs.tools-ref }}"
    assert "pull_request.head" not in str(job)
    assert "secrets." not in str(job)
    feedback = yaml.load((ROOT / ".github/workflows/feedback-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    assert feedback["permissions"]["contents"] == "read"


def test_github_enriched_tier1_applies_the_keyless_scan_baselines():
    """The enriched gate evidence must honor security-baselines exactly like the keyless scan."""
    document = workflow()
    assert document["jobs"]["tier1"]["with"]["security-baselines"] == "${{ inputs.security-baselines }}"
    enriched = document["jobs"]["tier1-llm"]
    keyless = yaml.load((ROOT / ".github/workflows/evaluate-skills.yml").read_text(), Loader=yaml.BaseLoader)
    scan_env = next(
        step["env"] for step in keyless["jobs"]["tier1"]["steps"] if "POLICY_SOURCE_SHA" in step.get("env", {})
    )
    assert enriched["env"]["SECURITY_BASELINES"] == "${{ inputs.security-baselines }}"
    assert enriched["env"]["POLICY_SOURCE_SHA"] == scan_env["POLICY_SOURCE_SHA"]
    candidate = next(step for step in enriched["steps"] if step.get("with", {}).get("path") == "candidate")
    # The image cannot fetch, so baseline runs keep the policy commit's blobs.
    assert candidate["with"]["fetch-depth"] == "0"
    assert candidate["with"]["filter"] == "${{ !inputs.security-baselines && 'blob:none' || '' }}"


def _run_gitlab_enriched(tmp_path, extra_env):
    import os

    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    script = "\n".join(document["skill-evaluation-tier1-llm"]["script"])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        "  */eligible_gitlab.py) exit 0 ;;\n"
        f"  */prepare_security_baselines.py) printf '%s\\n' {'d' * 40}; exit 0 ;;\n"
        "esac\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --reports ]; then shift; report_dir=$1; fi\n'
        "  shift\n"
        "done\n"
        'printf \'%s\\n%s\\n\' "$SECURITY_BASELINES" "${POLICY_SOURCE_SHA:-}" > "$report_dir/baselines.txt"\n'
    )
    stub.chmod(0o755)
    outer = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CI_", "GITLAB_", "SKILL_EVAL_"))
        and key not in {"SECURITY_BASELINES", "POLICY_SOURCE_SHA"}
    }
    env = {
        **outer,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "CI_PROJECT_DIR": str(workspace),
        "SKILL_EVAL_TOOLS_DIR": str(tmp_path / "tools"),
        "SKILL_EVAL_IMAGE": "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64,
        **extra_env,
    }
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False)
    evidence = workspace / "skill-evaluator-enriched-tier1/baselines.txt"
    return result, evidence.read_text().splitlines() if evidence.exists() else None


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"SKILL_EVAL_SECURITY_BASELINES": "true"}, ["true", "d" * 40]),
        ({"SKILL_EVAL_SECURITY_BASELINES": "false", "POLICY_SOURCE_SHA": "a" * 40}, ["false", ""]),
        ({"POLICY_SOURCE_SHA": "a" * 40}, ["false", ""]),
    ],
)
def test_gitlab_enriched_tier1_applies_the_keyless_job_baselines(tmp_path, extra, expected):
    result, evidence = _run_gitlab_enriched(tmp_path, extra)
    assert result.returncode == 0, result.stderr
    assert evidence == expected


def test_gitlab_enriched_tier1_rejects_noncanonical_baseline_opt_in(tmp_path):
    result, evidence = _run_gitlab_enriched(tmp_path, {"SKILL_EVAL_SECURITY_BASELINES": "TRUE"})
    assert result.returncode != 0
    assert evidence is None


def _aggregate_tier1_source(tmp_path, env, *, recomputed, marker=True):
    """Run the aggregate's Tier 1 selection with a stubbed pinned-tools decision."""
    import os

    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    script = "\n".join(document["skill-evaluation-aggregate"]["script"])
    selection = script[script.index("skills_changed=true") : script.index("if [ -d skill-evaluator-native ]")]
    (tmp_path / "skill-evaluator-report").mkdir()
    (tmp_path / "skill-evaluator-report" / "source.txt").write_text("keyless")
    enriched = tmp_path / "skill-evaluator-enriched-tier1"
    enriched.mkdir()
    (enriched / "source.txt").write_text("enriched")
    if marker:  # A committed or forged file must never decide the gate.
        (enriched / "enriched-tier1-skipped.json").write_text('{"status": "skipped"}')
    (tmp_path / "skill-evaluator-results" / "tier1").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    stub = bin_dir / "python"
    stub.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n" + recomputed + "\n")
    stub.chmod(0o755)
    result = subprocess.run(
        ["sh", "-c", "set -eu\n" + selection + 'printf %s "$SKILL_EVAL_SKILLS_CHANGED" > exported.txt\n'],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
            "SKILL_EVAL_TOOLS_DIR": "/pinned/tools",
            "CI_PROJECT_DIR": str(tmp_path),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    copied = tmp_path / "skill-evaluator-results" / "tier1" / "source.txt"
    invoked = calls.read_text().splitlines() if calls.exists() else []
    exported = tmp_path / "exported.txt"
    _aggregate_tier1_source.exported = exported.read_text() if exported.exists() else None
    return result.returncode, copied.read_text() if copied.exists() else None, invoked


MR = {
    "SKILL_EVAL_TIER1_LLM": "true",
    "CI_PIPELINE_SOURCE": "merge_request_event",
    "CI_PROJECT_ID": "42",
    "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
    "CI_DEFAULT_BRANCH": "main",
}


@pytest.mark.parametrize(("recomputed", "expected"), [("echo false", "keyless"), ("echo true", "enriched")])
def test_gitlab_aggregate_recomputes_the_skill_change_skip(tmp_path, recomputed, expected):
    """No skill change keeps the keyless gate; a marker file alone never does."""
    code, source, invoked = _aggregate_tier1_source(tmp_path, MR, recomputed=recomputed)
    assert code == 0
    assert source == expected
    assert invoked == [f"-I /pinned/tools/ci/select_changed_skills.py --skills-changed --workspace {tmp_path}"]


def test_gitlab_aggregate_fails_closed_when_the_skip_cannot_be_recomputed(tmp_path):
    code, source, _ = _aggregate_tier1_source(tmp_path, MR, recomputed="exit 2")
    assert code != 0
    assert source is None


def test_gitlab_default_branch_aggregate_never_skips_enriched_evidence(tmp_path):
    env = {
        "SKILL_EVAL_TIER1_LLM": "true",
        "CI_PIPELINE_SOURCE": "push",
        "CI_COMMIT_BRANCH": "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": "true",
        "CI_PROJECT_ID": "42",
    }
    code, source, invoked = _aggregate_tier1_source(tmp_path, env, recomputed="echo false")
    assert (code, source, invoked) == (0, "enriched", [])


@pytest.mark.parametrize(
    ("recomputed", "exported"), [("echo false", "false"), ("echo true", "true"), ("exit 2", "unknown")]
)
def test_gitlab_aggregate_exports_the_paid_tier_decision(tmp_path, recomputed, exported):
    """Without enriched Tier 1, an unknown answer only keeps Tier 2/3 reported as incomplete."""
    env = {**MR, "SKILL_EVAL_TIER1_LLM": "false"}
    code, source, _ = _aggregate_tier1_source(tmp_path, env, recomputed=recomputed)
    assert (code, source) == (0, "keyless")
    assert _aggregate_tier1_source.exported == exported


def _run_native_tier(tmp_path, recomputed):
    import os

    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    script = "\n".join(document["skill-evaluation-native"]["script"])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_dir / "python").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        'case "$2" in */select_changed_skills.py) ' + recomputed + " ;; esac\n"
        "exit 3\n"  # Stop at admission, before any installation or provider work.
    )
    (bin_dir / "python").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "SKILL_EVAL_TIERS": "1,2,3",
        "SKILL_EVAL_TIER": "3",
        "SKILL_EVAL_TOOLS_DIR": "/pinned/tools",
        "CI_PROJECT_DIR": str(tmp_path),
    }
    result = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=False)
    return result, calls.read_text().splitlines()


def test_gitlab_paid_tier_skips_before_admission_when_no_skill_changed(tmp_path):
    result, calls = _run_native_tier(tmp_path, "echo false; exit 0")
    assert result.returncode == 0
    assert "Tier 3 did not run" in result.stdout
    assert calls == [f"-I /pinned/tools/ci/select_changed_skills.py --skills-changed --workspace {tmp_path}"]


@pytest.mark.parametrize("recomputed", ["echo true; exit 0", "exit 2"])
def test_gitlab_paid_tier_runs_when_a_skill_changed_or_the_answer_is_unknown(tmp_path, recomputed):
    result, calls = _run_native_tier(tmp_path, recomputed)
    assert result.returncode == 3  # Reached admission: the tier was not skipped.
    assert calls[-1].endswith("ci/eligible_gitlab.py")
