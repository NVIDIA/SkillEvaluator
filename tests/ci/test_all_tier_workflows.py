# SPDX-License-Identifier: Apache-2.0
"""Validate the security boundaries of the opt-in three-tier pipeline."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def load(path):
    return list(yaml.load_all((ROOT / path).read_text(), Loader=yaml.BaseLoader))[-1]


def test_candidate_onboarding_uses_the_released_tier1_image():
    candidate = json.loads((ROOT / "src/skillevaluator/ci-release-all-tiers.json").read_text())
    released = json.loads((ROOT / "src/skillevaluator/ci-release.json").read_text())
    # Released schema capabilities, such as multi-valued metadata.domain,
    # must also work for new all-tier consumers of the same package.
    assert candidate["image"] == released["image"]


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
    # Eligibility tells the aggregate when the paid tiers were skipped for no skill change.
    assert set(aggregate["needs"]) == {"eligibility", "tier1", "tier1-llm", "paid-tiers"}
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
        # The default/fork path retains canonical static evidence. Protected
        # enrichment emits the same canonical name from its separate job.
        s["with"]["name"].replace("${{ inputs.artifact-name }}", "skill-evaluator-tier1")
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
    assert paid["parallel"]["matrix"] == [{"SKILL_EVAL_TIER": ["3"]}]
    assert evaluation["skill-evaluation-native-tier2"]["parallel"]["matrix"] == [{"SKILL_EVAL_TIER": ["2"]}]
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


def test_gitlab_tier2_has_no_docker_service_or_tls_settings():
    jobs = load("ci/gitlab/all-tiers.yml")
    tier2 = jobs["skill-evaluation-native-tier2"]
    assert tier2["extends"] == "skill-evaluation-native"
    assert tier2["services"] == []
    assert tier2["variables"] == {
        "DOCKER_HOST": "",
        "DOCKER_TLS_CERTDIR": "",
        "DOCKER_TLS_VERIFY": "",
        "DOCKER_CERT_PATH": "",
    }


def test_gitlab_native_caller_overrides_and_advisory_policy_apply_to_both_tiers():
    from skillevaluator._ci_onboarding_all_tiers import _gitlab_inherited

    jobs = load("ci/gitlab/all-tiers.yml")
    jobs["skill-evaluation-native"].update({"timeout": "45 minutes", "tags": ["native-linux"], "services": []})
    for name in ("skill-evaluation-native", "skill-evaluation-native-tier2"):
        job = jobs[name]
        assert _gitlab_inherited(jobs, job, "timeout") == "45 minutes"
        assert _gitlab_inherited(jobs, job, "tags") == ["native-linux"]
        assert _gitlab_inherited(jobs, job, "services") == []
        assert _gitlab_inherited(jobs, job, "allow_failure") == "true"
        assert _gitlab_inherited(jobs, job, "environment") == {"name": "$SKILL_EVAL_ENVIRONMENT", "action": "prepare"}
        assert _gitlab_inherited(jobs, job, "rules")[0] == {"if": '$SKILL_EVAL_FEEDBACK == "true"', "when": "never"}


def test_gitlab_aggregate_downloads_evidence_from_both_advisory_tiers():
    needs = load("ci/gitlab/all-tiers.yml")["skill-evaluation-aggregate"]["needs"]
    assert needs == [
        {"job": "skill-evaluation", "artifacts": "true", "optional": "true"},
        {"job": "skill-evaluation-native", "artifacts": "true", "optional": "true"},
        {"job": "skill-evaluation-native-tier2", "artifacts": "true", "optional": "true"},
        {"job": "skill-evaluation-tier1-llm", "artifacts": "true", "optional": "true"},
    ]


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


def test_feedback_mutation_jobs_are_serialized_per_pr_source_without_dropping_pending_runs():
    import re

    github = load(".github/workflows/feedback-all-tiers.yml")["jobs"]["feedback"]
    concurrency = github["concurrency"]
    # The default `queue: single` cancels an already pending run whenever another joins the group.
    assert concurrency["cancel-in-progress"] == "false"
    assert concurrency["queue"] == "max"
    assert set(concurrency) == {"group", "cancel-in-progress", "queue"}
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in github["if"]

    def rendered(repository, head_repository, head_branch, run_id, head_sha):
        # Unknown expressions raise KeyError; `workflow_run.pull_requests` is empty for fork PRs.
        event = {
            "github.repository": repository,
            "github.event.workflow_run.head_repository.full_name": head_repository,
            "github.event.workflow_run.head_branch": head_branch,
            "github.event.workflow_run.id": run_id,
            "github.event.workflow_run.head_sha": head_sha,
        }
        return re.sub(r"\$\{\{ ([^}]+) \}\}", lambda match: event[match.group(1)], concurrency["group"])

    first = rendered("acme/skills", "acme/skills", "feature-a", "101", "a" * 40)
    # Later producer runs for the same PR source serialize behind the comment update in progress.
    assert rendered("acme/skills", "acme/skills", "feature-a", "102", "b" * 40) == first
    # Other PRs, repositories and head repositories never compete for one pending slot.
    assert rendered("acme/skills", "acme/skills", "feature-b", "103", "c" * 40) != first
    assert rendered("acme/other", "acme/other", "feature-a", "104", "a" * 40) != first
    assert rendered("acme/skills", "fork/skills", "feature-a", "105", "a" * 40) != first
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


def _gitlab_minutes(duration):
    import re

    match = re.fullmatch(r"(\d+) (minutes|hours)", duration)
    assert match, duration
    return int(match[1]) * (60 if match[2] == "hours" else 1)


def test_gitlab_evaluator_budget_ends_before_the_native_job_timeout():
    for path in ("ci/gitlab/all-tiers.yml", "ci/templates/all-tiers/gitlab-all-tiers.yml"):
        config = load(path)
        minutes = _gitlab_minutes(config["skill-evaluation-native"]["timeout"])
        assert "$" not in config["skill-evaluation-native"]["timeout"]
        reserve = min(20, minutes // 4)
        assert config["variables"]["SKILL_EVAL_TIMEOUT_SECONDS"] == str((minutes - reserve) * 60) == "16800", path


def test_gitlab_dind_daemon_requires_tls_client_certificates():
    job = load("ci/gitlab/all-tiers.yml")["skill-evaluation-native"]
    assert job["services"] == [{"name": "docker:28.5.1-dind", "alias": "docker"}]
    variables = job["variables"]
    assert variables["DOCKER_HOST"] == "tcp://docker:2376"
    assert variables["DOCKER_TLS_CERTDIR"] == "/certs"
    assert variables["DOCKER_TLS_VERIFY"] == "1"
    assert variables["DOCKER_CERT_PATH"] == "/certs/client"
    assert "2375" not in str(job)
    docs = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    assert 'volumes = ["/certs/client"]' in docs


def test_gitlab_native_jobs_use_environment_without_recording_deployments():
    job = load("ci/gitlab/all-tiers.yml")["skill-evaluation-native"]
    assert job["environment"] == {"name": "$SKILL_EVAL_ENVIRONMENT", "action": "prepare"}
    docs = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    assert "`environment: action: prepare`" in docs and "Prevent outdated deployment jobs" in docs


def test_gitlab_feedback_job_uses_its_protected_environment_without_recording_deployments():
    # Scoped publisher token and deploy access still apply; no deployment means
    # "Prevent outdated deployment jobs" cannot drop an older MR's feedback.
    job = load("ci/gitlab/feedback.yml")["skill-evaluation-feedback"]
    assert job["environment"] == {"name": "skillevaluator-feedback", "action": "prepare"}
    # The publisher authenticates its job, pipeline and branch, never a deployment record.
    assert "/deployments" not in (ROOT / "ci/feedback.py").read_text()
    docs = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    assert "The native jobs and the trusted feedback job use `environment: action: prepare`" in docs
    # Some GitLab versions turn prepare jobs on an environment with approval
    # rules into manual jobs, so deploy access is the only safe restriction.
    assert "Deployment approval rules apply only to jobs that create deployments" not in docs
    assert "Do not add deployment approval rules to these environments" in docs
    assert "turn them into manual jobs" in docs and "`prevent_blocking_non_deployment_jobs`" in docs
    assert "Restrict who can run them with the protected environments' deploy access" in docs
    onboarding = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    assert "so deployment approvals do not gate them" not in onboarding
    assert "Do not add deployment approval rules to their environments" in onboarding
    assert "turn them into manual jobs" in onboarding


def test_gitlab_feedback_pipeline_guard_is_offered_and_documented():
    guard = {"if": '$SKILL_EVAL_FEEDBACK == "true"', "when": "never"}
    hidden = load("ci/gitlab/all-tiers.yml")[".skill-evaluation-not-feedback"]
    assert hidden["rules"][0] == guard
    assert hidden["rules"][1] == {"when": "on_success"}
    for path in ("ci/templates/all-tiers/README.md", "ci/templates/all-tiers/gitlab-all-tiers.yml"):
        text = (ROOT / path).read_text()
        assert "- if: '$SKILL_EVAL_FEEDBACK == \"true\"'" in text, path
    docs = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    assert "extends: .skill-evaluation-not-feedback" in docs
    assert "no comment is posted or updated" in docs
    assert "Merge these guards to the default branch before the setup MR" in docs
    # Included jobs use stage test, so a merged root's stages must keep it.
    assert "If the root defines `stages`, keep `test` in the list" in docs
    assert "neither `rules` nor `only`/`except`" in docs
    # Guards merge before the setup MR, while only all-tiers.yml defines the
    # hidden job, so they use rules; a job without rules needs a fallback rule.
    assert "gets the guard followed by `- when: on_success`" in docs
    assert "- when: on_success" in (ROOT / "ci/templates/all-tiers/README.md").read_text()
    assert "Write these guards as rules" in docs
    assert "until the setup MR merges, `extends: .skill-evaluation-not-feedback` fails validation" in docs
    assert ".skill-evaluation-not-feedback" not in (ROOT / "ci/gitlab/skill-evaluation.yml").read_text()
    template = " ".join((ROOT / "ci/templates/all-tiers/gitlab-all-tiers.yml").read_text().split())
    assert "If the root defines stages, keep test" in template
    assert "gets the guard followed by `- when: on_success`" in template
    assert "Write the guards as rules" in template
    assert "added after this file is on the default branch" in template
    # The upgrade path must order the guards before the setup MR as well.
    onboarding = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    guards = onboarding.index("merge those guards to the default branch")
    assert guards < onboarding.index("Then, in your setup MR")
    assert "keep `test` in any `stages` list" in onboarding


def test_pre_merge_paid_tier_advice_does_not_route_through_ci_demo():
    # `ci demo` needs the scan caller on the default branch (the onboarding tests
    # cover the rejection), so the docs must not offer it for an unmerged setup.
    docs = " ".join((ROOT / "ci/templates/all-tiers/README.md").read_text().split())
    assert "(`ci demo` opens one)" not in docs
    assert "`ci demo` cannot exercise an unmerged setup" in docs
    assert "rejects a setup that has not merged" in docs
    assert "Use `ci demo` after the setup merges." in docs
    advanced = " ".join((ROOT / "docs/advanced-ci.mdx").read_text().split())
    assert "`skillevaluator ci demo` works only after the setup has merged." in advanced


def _heredoc(script, marker):
    return script.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}\n", 1)[0] + "\n"


def _setup_literals(source):
    """Return the prefixes and exact paths that an inline setup predicate checks."""
    import ast

    prefixes, files = set(), set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "startswith":
            prefixes.update(ast.literal_eval(node.args[0]))
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "path":
            assert all(isinstance(op, ast.Eq) for op in node.ops)
            files.update(ast.literal_eval(value) for value in node.comparators)
    text = {value.decode() if isinstance(value, bytes) else value for value in prefixes}
    return text, {value.decode() if isinstance(value, bytes) else value for value in files}


def test_tier1_setup_predicates_equal_the_shared_native_predicate():
    from ci.select_changed_skills import SETUP_FILES, SETUP_PREFIXES

    jobs = load(".github/workflows/evaluate-all-tiers.yml")["jobs"]
    github = _heredoc(jobs["eligibility"]["steps"][0]["run"], "PY")
    gitlab = _heredoc("\n".join(load("ci/gitlab/all-tiers.yml")["skill-evaluation"]["script"]), "PYTHON")
    for source in (github, gitlab):
        assert _setup_literals(source) == (set(SETUP_PREFIXES), set(SETUP_FILES))
    # GitHub also widens when the PR file listing is too long to certify; the
    # paid tiers take that decision instead of recomputing a different one.
    paid = jobs["paid-tiers"]
    assert paid["env"]["SKILL_EVAL_SCAN_MODE"] == jobs["tier1"]["with"]["scan-mode"]
    assert "needs.eligibility.outputs.scan-mode" in paid["env"]["SKILL_EVAL_SCAN_MODE"]


def test_gitlab_tier1_and_native_tiers_select_the_same_skills(tmp_path):
    import os
    import subprocess
    import sys

    from ci import run_tiers

    heredoc = _heredoc("\n".join(load("ci/gitlab/all-tiers.yml")["skill-evaluation"]["script"]), "PYTHON")
    heredoc = heredoc.replace('"/opt/skillevaluator/ci"', repr(str(ROOT / "ci")))
    cases = {
        "skill only": ([("skills/alpha/SKILL.md", "# alpha v2\n")], []),
        "consumer ci script": ([("ci/build.sh", "echo build\n"), ("skills/alpha/SKILL.md", "# alpha v3\n")], []),
        "workflow": ([(".github/workflows/skills.yml", "name: skills\n")], []),
        "root pipeline": ([(".gitlab-ci.yml", "include: []\n")], []),
        "renamed feedback include": ([], [(".gitlab/skill-feedback.yml", "docs/skill-feedback.yml")]),
        "ignored setup gitlink": ([("skills/alpha/SKILL.md", "# alpha with setup gitlink\n")], []),
        "visible setup gitlink": ([("skills/alpha/SKILL.md", "# alpha with setup gitlink\n")], []),
    }
    for name, (writes, renames) in cases.items():
        repo = tmp_path / name.replace(" ", "-")
        repo.mkdir()
        git = ["git", "-C", str(repo), "-c", "user.name=CI", "-c", "user.email=ci@example.com"]
        subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
        for skill in ("alpha", "beta"):
            (repo / "skills" / skill).mkdir(parents=True)
            (repo / "skills" / skill / "SKILL.md").write_text(f"# {skill}\n")
        (repo / ".gitlab").mkdir()
        (repo / ".gitlab/skill-feedback.yml").write_text("include: []\n" * 20)
        gitlink = name in {"ignored setup gitlink", "visible setup gitlink"}
        if gitlink:
            ignore = "all" if name == "ignored setup gitlink" else "none"
            (repo / ".gitmodules").write_text(
                f'[submodule "actions"]\n path = .github/actions\n url = ./actions\n ignore = {ignore}\n'
            )
        subprocess.run([*git, "add", "-A"], check=True)
        if gitlink:
            subprocess.run([*git, "update-index", "--add", "--cacheinfo", f"160000,{'a' * 40},.github/actions"], check=True)
        subprocess.run([*git, "commit", "-qm", "base"], check=True)
        base = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        for relative, text in writes:
            (repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (repo / relative).write_text(text)
        for source, target in renames:
            (repo / target).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([*git, "mv", source, target], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        if gitlink:
            subprocess.run([*git, "update-index", "--add", "--cacheinfo", f"160000,{'b' * 40},.github/actions"], check=True)
        subprocess.run([*git, "commit", "-qm", "head"], check=True)
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        tier1 = subprocess.run(
            [sys.executable, "-I", "-"],
            input=heredoc,
            env={"PATH": os.environ["PATH"], "DIFF_BASE_SHA": base, "DIFF_HEAD_SHA": head, "CI_PROJECT_DIR": str(repo)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        selected, selection = run_tiers.select_skills(
            repo, {"SKILL_EVAL_PATHS": "skills", "DIFF_BASE_SHA": base, "REPORT_COMMIT": head}
        )
        native = "all" if selection.get("reason") == "ci_configuration_changed" else "changed"
        assert tier1 == native, name
        assert (native == "all") == (name not in {"skill only", "consumer ci script"}), name
        if native == "changed":
            assert [p.name for p in selected] == ["alpha"], name
        else:
            assert [p.name for p in selected] == ["alpha", "beta"], name
