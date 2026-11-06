# SPDX-License-Identifier: Apache-2.0
"""Validate the security boundaries of the opt-in three-tier pipeline."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def load(path):
    return list(yaml.load_all((ROOT / path).read_text(), Loader=yaml.BaseLoader))[-1]


def test_required_tier1_reuses_hardened_workflow_and_parallel_paid_tiers():
    workflow = load(".github/workflows/evaluate-all-tiers.yml")
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert inputs["env-mode"]["default"] == "docker"
    assert inputs["tier3-required"]["default"] == "false"
    assert workflow["jobs"]["tier1"]["uses"] == "./.github/workflows/evaluate-skills.yml"
    paid = workflow["jobs"]["paid-tiers"]
    assert paid["strategy"]["fail-fast"] == "false"
    assert paid["strategy"]["matrix"]["tier"] == ["2", "3"]
    assert paid["environment"] == "${{ inputs.environment }}"
    assert "eligibility" in paid["needs"]
    assert "needs.eligibility.outputs.allowed" in paid["if"]
    checkouts = [s for s in paid["steps"] if s.get("uses", "").startswith("actions/checkout@")]
    assert {s["with"]["path"] for s in checkouts} == {"candidate", "evaluator"}
    tools = next(s for s in checkouts if s["with"]["path"] == "evaluator")
    assert tools["with"]["ref"] == "${{ inputs.tools-ref }}"
    assert tools["with"]["persist-credentials"] == "false"
    assert all("secrets." not in str(s) for s in workflow["jobs"]["eligibility"]["steps"])


def test_aggregate_retains_every_tier_after_failure():
    workflow = load(".github/workflows/evaluate-all-tiers.yml")
    aggregate = workflow["jobs"]["aggregate"]
    assert aggregate["if"] == "${{ always() }}"
    assert set(aggregate["needs"]) == {"tier1", "paid-tiers"}
    downloads = [s for s in aggregate["steps"] if s.get("uses", "").startswith("actions/download-artifact@")]
    assert [s["with"]["pattern"] for s in downloads] == ["skill-evaluator-tier[123]-attempt-*"]
    assert downloads[0]["with"]["path"] == "${{ runner.temp }}/skill-artifacts"
    upload = next(s for s in aggregate["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "${{ always() }}"
    assert upload["with"]["retention-days"] == "14"
    assert upload["with"]["name"] == "skill-evaluator-results-attempt-${{ github.run_attempt }}"


def test_aggregate_reuses_tier_evidence_from_jobs_not_rerun():
    import fnmatch

    workflow = load(".github/workflows/evaluate-all-tiers.yml")
    steps = workflow["jobs"]["aggregate"]["steps"]
    download = next(s for s in steps if s.get("uses", "").startswith("actions/download-artifact@"))
    # "Re-run failed jobs" leaves passing tier jobs' artifacts under their own earlier attempt.
    assert "github.run_attempt" not in str(download["with"])
    assert "name" not in download["with"]
    assert download["with"].get("merge-multiple", "false") == "false"
    pattern = download["with"]["pattern"]
    tier1 = load(".github/workflows/evaluate-skills.yml")["jobs"]["tier1"]["steps"]
    produced = [
        s["with"]["name"].replace("${{ inputs.artifact-name }}", workflow["jobs"]["tier1"]["with"]["artifact-name"])
        for s in tier1
        if s.get("uses", "").startswith("actions/upload-artifact@")
    ]
    produced += [
        s["with"]["name"].replace("${{ matrix.tier }}", tier)
        for s in workflow["jobs"]["paid-tiers"]["steps"]
        if s.get("uses", "").startswith("actions/upload-artifact@")
        for tier in workflow["jobs"]["paid-tiers"]["strategy"]["matrix"]["tier"]
    ]
    produced += [s["with"]["name"] for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")]
    for attempt in ("1", "2", "12"):
        names = [name.replace("${{ github.run_attempt }}", attempt) for name in produced]
        matched = sorted(name for name in names if fnmatch.fnmatchcase(name, pattern))
        assert matched == [f"skill-evaluator-tier{tier}-attempt-{attempt}" for tier in (1, 2, 3)]
    run = [s.get("run", "") for s in steps]
    select = next(i for i, script in enumerate(run) if "run_tiers.py select-attempts" in script)
    assert (
        steps.index(download) < select < next(i for i, script in enumerate(run) if "run_tiers.py aggregate" in script)
    )
    assert '--artifacts "$RUNNER_TEMP/skill-artifacts" --reports "$RUNNER_TEMP/skill-reports"' in run[select]
    assert "continue-on-error" not in steps[select]


def test_paid_tier_evaluation_stops_before_the_job_timeout():
    import subprocess

    paid = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]["paid-tiers"]
    assert paid["timeout-minutes"] == "${{ inputs.timeout-minutes }}"
    assert paid["env"]["SKILL_EVAL_TIMEOUT_MINUTES"] == "${{ inputs.timeout-minutes }}"
    step = next(s for s in paid["steps"] if s.get("name") == "Evaluate selected tier and retain native evidence")
    setup = step["run"].split("python -I", 1)[0]
    for minutes, expected in ((300, 280), (60, 45), (10, 8), (1, 1)):
        result = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", setup + 'printf %s "$SKILL_EVAL_TIMEOUT_SECONDS"'],
            env={"PATH": "/usr/bin:/bin", "SKILL_EVAL_TIMEOUT_MINUTES": str(minutes)},
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(result.stdout) == expected * 60


def test_paid_tiers_leave_the_embedding_default_to_the_runner():
    paid = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]["paid-tiers"]
    # An Anthropic chat provider defaults to OpenAI embeddings inside run_tiers.
    assert paid["env"]["SKILL_EVAL_EMBEDDING_PROVIDER"] == "${{ inputs.embedding-provider }}"


def test_feedback_checks_out_only_pinned_tools_and_uses_scoped_token():
    workflow = load(".github/workflows/feedback-all-tiers.yml")
    job = workflow["jobs"]["feedback"]
    assert job["permissions"]["pull-requests"] == "write"
    checkouts = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 1
    assert checkouts[0]["with"]["ref"] == "${{ inputs.tools-ref }}"
    assert "pull_request.head" not in str(checkouts)
    assert any("ci/feedback.py" in s.get("run", "") for s in job["steps"])


def test_gitlab_feedback_token_is_confined_to_trusted_default_branch_job():
    evaluation = load("ci/gitlab/all-tiers.yml")
    assert "SKILLEVALUATOR_GITLAB_TOKEN" not in str(evaluation)
    paid = evaluation["skill-evaluation-native"]
    assert paid["parallel"]["matrix"] == [{"SKILL_EVAL_TIER": ["2", "3"]}]
    assert paid["environment"]["name"] == "$SKILL_EVAL_ENVIRONMENT"
    assert evaluation["skill-evaluation-aggregate"]["when"] == "always"
    feedback = load("ci/gitlab/feedback.yml")["skill-evaluation-feedback"]
    assert all("CI_DEFAULT_BRANCH" in r["if"] and "SKILL_EVAL_FEEDBACK" in r["if"] for r in feedback["rules"])
    assert "SKILLEVALUATOR_GITLAB_TOKEN" in str(feedback)


def test_gitlab_tier1_runs_only_digest_pinned_legacy_image():
    job = load("ci/gitlab/all-tiers.yml")["skill-evaluation"]
    assert job["image"]["name"] == "$SKILL_EVAL_IMAGE"
    assert job["image"]["entrypoint"] == [""]
    assert "/opt/skillevaluator/ci/run_tier1.py" in str(job["script"])
    assert "python -m pip install" not in str(job["script"])


def test_both_paid_tiers_are_advisory_even_for_legacy_required_callers():
    workflow = load(".github/workflows/evaluate-all-tiers.yml")
    legacy = workflow["on"]["workflow_call"]["inputs"]["tier3-required"]
    assert legacy["type"] == "boolean"
    assert legacy["default"] == "false"
    assert "deprecated" in legacy["description"].lower()
    for name in ("paid-tiers", "aggregate"):
        job = workflow["jobs"][name]
        assert job["env"]["SKILL_EVAL_TIER3_REQUIRED"] == "false"
        assert "inputs.tier3-required" not in str(job)
    assert workflow["jobs"]["paid-tiers"]["continue-on-error"] == "true"


def test_new_candidate_callers_do_not_expose_a_paid_tier_requirement():
    github = load("ci/templates/all-tiers/skill-checks-all-tiers.yml")
    assert "tier3-required" not in github["jobs"]["skills"]["with"]
    for path in ("ci/templates/all-tiers/gitlab-all-tiers.yml", "ci/gitlab/all-tiers.yml"):
        assert "SKILL_EVAL_TIER3_REQUIRED" not in load(path)["variables"]


def test_trusted_aggregate_receipt_is_captured_before_untrusted_artifact_read():
    job = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]["aggregate"]
    assert job["permissions"]["id-token"] == "write"
    first = job["steps"][0]
    assert first["id"] == "provenance"
    assert "ACTIONS_ID_TOKEN_REQUEST_URL" in first["run"]
    assert "skill-evaluator-results-provenance-attempt-" in str(job["steps"])


def test_setup_configuration_changes_force_a_real_tier1_scan():
    jobs = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]
    assert jobs["tier1"]["needs"] == "eligibility"
    assert "needs.eligibility.outputs.scan-mode" in jobs["tier1"]["with"]["scan-mode"]
    gate = str(jobs["eligibility"]["steps"])
    assert "/files?per_page=100&page=" in gate
    assert ".github/" in gate and ".gitlab-ci.yml" in gate
    gitlab = str(load("ci/gitlab/all-tiers.yml")["skill-evaluation"]["script"])
    assert "from select_changed_skills import _git" in gitlab and 'print("all" if setup else "changed")' in gitlab


def test_gitlab_native_reports_are_outside_candidate_until_evaluation_completes():
    script = str(load("ci/gitlab/all-tiers.yml")["skill-evaluation-native"]["script"])
    assert "mktemp -d /tmp/skillevaluator-reports." in script
    assert '--reports "$reports_dir"' in script
    assert 'cp -R "$reports_dir/."' in script


def test_github_head_is_rechecked_after_environment_approval_before_provider_secrets():
    steps = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]["paid-tiers"]["steps"]
    check = next(s for s in steps if s.get("name") == "Recheck current head after environment approval")
    assert "secrets." not in str(check)
    assert "current.get('head', {}).get('sha')" in check["run"]
    assert "pulls/" in check["run"]
    first_secret = next(i for i, s in enumerate(steps) if "secrets." in str(s))
    assert steps.index(check) < first_secret


def test_provider_steps_cannot_run_when_post_approval_head_check_failed():
    steps = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]["paid-tiers"]["steps"]
    for step in steps:
        if "secrets." in str(step):
            assert "steps.current-head.outputs.allowed == 'true'" in step["if"]


def test_candidate_templates_do_not_change_legacy_bundle_generation():
    candidates = ROOT / "ci/templates/all-tiers"
    assert (candidates / "skill-checks-all-tiers.yml").is_file()
    assert not list((ROOT / "ci/templates").glob("*all-tiers.yml"))


def test_gitlab_privileged_bootstrap_uses_fixed_include_inputs():
    source = (ROOT / "ci/gitlab/feedback.yml").read_text()
    header = next(yaml.load_all(source, Loader=yaml.BaseLoader))
    assert set(header["spec"]["inputs"]) == {"tools-ref", "tools-repository", "api-url", "project-id"}
    job = load("ci/gitlab/feedback.yml")["skill-evaluation-feedback"]
    script = str(job["script"])
    assert "$SKILL_EVAL_TOOLS_REF" not in script
    assert "$SKILL_EVAL_TOOLS_REPOSITORY" not in script
    assert "$CI_API_V4_URL" not in script
    assert "$[[ inputs.tools-ref ]]" in script
    assert job["id_tokens"]["SKILL_EVAL_GITLAB_ID_TOKEN"]["aud"] == "urn:skillevaluator:feedback:v1"


def test_gitlab_feedback_request_uses_job_token_api_without_candidate_checkout():
    job = load("ci/gitlab/all-tiers.yml")["skill-evaluation-request-feedback"]
    assert "trigger" not in job
    assert "environment" not in job
    assert job["variables"] == {"GIT_STRATEGY": "none"}
    assert job["extends"] == ".skill-evaluation-tools"
    assert job["needs"] == [{"job": "skill-evaluation-aggregate", "artifacts": "false"}]
    assert job["when"] == "always"
    assert job["allow_failure"] == "true"
    script = "\n".join(job["script"])
    assert 'python -I "$SKILL_EVAL_TOOLS_DIR/ci/request_gitlab_feedback.py"' in script
    assert "uv" not in script and "bootstrap" not in script


def test_gitlab_bridge_rules_do_not_disable_their_own_feedback_flag():
    bridge = load("ci/gitlab/all-tiers.yml")["skill-evaluation-request-feedback"]
    assert all("SKILL_EVAL_FEEDBACK" not in rule["if"] for rule in bridge["rules"])


def test_gitlab_feedback_bootstrap_isolates_process_configuration():
    script = "\n".join(load("ci/gitlab/feedback.yml")["skill-evaluation-feedback"]["script"])
    assert "export PATH=/usr/local/bin:/usr/bin:/bin" in script
    assert "GIT_CONFIG_COUNT" in script and "GIT_EXEC_PATH" in script
    assert "python -I -m pip --isolated install" in script
    assert "env -i" in script


def test_gitlab_aggregate_enforces_required_gates_for_all_matrix_outcomes():
    jobs = load("ci/gitlab/all-tiers.yml")
    assert jobs["skill-evaluation-native"]["allow_failure"] == "true"
    assert jobs["skill-evaluation-aggregate"].get("allow_failure", "false") == "false"
    assert jobs["skill-evaluation-aggregate"]["when"] == "always"


def test_trusted_native_workflow_entrypoints_and_pip_ignore_consumer_import_paths():
    for file in (
        ".github/workflows/evaluate-all-tiers.yml",
        ".github/workflows/feedback-all-tiers.yml",
        "ci/gitlab/all-tiers.yml",
    ):
        document = load(file)
        jobs = document.get("jobs", document)
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            scripts = job.get("script", []) + [step["run"] for step in job.get("steps", []) if "run" in step]
            for script in scripts:
                for line in script.splitlines():
                    if "python " in line and "/opt/skillevaluator/ci/run_tier1.py" not in line:
                        assert "python -I " in line, (file, line)
                    if "-m pip" in line:
                        assert "-m pip --isolated install" in line, (file, line)


def test_candidate_caller_propagates_reusable_read_permissions():
    candidate = load("ci/templates/all-tiers/skill-checks-all-tiers.yml")
    assert candidate["permissions"]["actions"] == "read"
    assert candidate["permissions"]["pull-requests"] == "read"


def test_all_tier_runtimes_sync_the_reviewed_lock_with_pinned_uv():
    evaluation = load(".github/workflows/evaluate-all-tiers.yml")
    paid_install = next(
        s for s in evaluation["jobs"]["paid-tiers"]["steps"] if s.get("name", "").startswith("Install only")
    )["run"]
    assert "uv==0.11.27" in paid_install
    assert "sync --locked --no-dev" in paid_install
    assert '--extra "tier${TIER}"' in paid_install
    assert 'evaluator/.venv/bin" >> "$GITHUB_PATH"' in paid_install
    aggregate = "\n".join(s["run"] for s in evaluation["jobs"]["aggregate"]["steps"] if "run" in s)
    assert "sync --locked --no-dev" in aggregate
    assert "--default-index https://pypi.org/simple" in aggregate
    feedback = "\n".join(
        s["run"] for s in load(".github/workflows/feedback-all-tiers.yml")["jobs"]["feedback"]["steps"] if "run" in s
    )
    assert "--extra publisher" in feedback
    for name in ("skill-evaluation-native", "skill-evaluation-aggregate"):
        script = "\n".join(load("ci/gitlab/all-tiers.yml")[name]["script"])
        assert "uv==0.11.27" in script
        assert "sync --locked --no-dev" in script
        assert '--project "$SKILL_EVAL_TOOLS_DIR"' in script
        assert 'export PATH="$SKILL_EVAL_TOOLS_DIR/.venv/bin:$PATH"' in script
    privileged = "\n".join(load("ci/gitlab/feedback.yml")["skill-evaluation-feedback"]["script"])
    assert "env -i" in privileged and "--no-config" in privileged
    assert "sync --locked --no-dev --extra publisher" in privileged
    assert '"$tools_dir/.venv/bin/python" -I - ' in privileged


def test_github_native_and_aggregate_record_the_exact_tools_revision():
    jobs = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]
    for name in ("paid-tiers", "aggregate"):
        assert jobs[name]["env"]["SKILLEVALUATOR_TOOLS_REF"] == "${{ inputs.tools-ref }}"


def test_feedback_mutation_jobs_are_serialized_for_the_entire_repository():
    github = load(".github/workflows/feedback-all-tiers.yml")["jobs"]["feedback"]
    assert github["concurrency"] == {
        "group": "skillevaluator-feedback-${{ github.repository }}",
        "cancel-in-progress": "false",
    }
    assert load("ci/gitlab/feedback.yml")["skill-evaluation-feedback"]["resource_group"] == "skillevaluator-feedback"


def test_gitlab_dind_bootstrap_installs_client_compose_and_buildx_without_a_second_daemon():
    script = "\n".join(load("ci/gitlab/all-tiers.yml")["skill-evaluation-native"]["script"])
    assert "apt-get install -y --no-install-recommends docker-cli docker-compose docker-buildx" in script
    assert "--no-install-recommends docker.io" not in script
    assert "docker compose version" in script
    assert "docker buildx version" in script
    assert script.index("bootstrap_tiers.py") < script.index('run_tiers.py" evaluate')


def test_gitlab_shared_native_timeout_is_a_literal_duration():
    config = load("ci/gitlab/all-tiers.yml")
    assert config["skill-evaluation-native"]["timeout"] == "5 hours"
    assert "SKILL_EVAL_JOB_TIMEOUT" not in config["variables"]
    candidate = load("ci/templates/all-tiers/gitlab-all-tiers.yml")
    assert candidate["skill-evaluation-native"]["timeout"] == "5 hours"


def test_gitlab_setup_selector_uses_pinned_safe_git_reader():
    script = "\n".join(load("ci/gitlab/all-tiers.yml")["skill-evaluation"]["script"])
    assert 'ci_changed="$(git diff' not in script
    assert 'sys.path.insert(0, "/opt/skillevaluator/ci")' in script
    assert "from select_changed_skills import _git" in script
    assert "--name-only" in script and '"-z"' in script and "--no-renames" in script
    assert 're.fullmatch(r"[0-9a-f]{40}", value)' in script


def test_gitlab_read_token_is_removed_before_provider_bootstrap_and_feedback_is_advisory():
    config = load("ci/gitlab/all-tiers.yml")
    script = "\n".join(config["skill-evaluation-native"]["script"])
    assert (
        script.index("eligible_gitlab.py")
        < script.index("unset SKILLEVALUATOR_GITLAB_READ_TOKEN")
        < script.index("-m pip")
    )
    assert config["skill-evaluation-request-feedback"]["allow_failure"] == "true"
