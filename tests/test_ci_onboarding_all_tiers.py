"""Opt-in all-tier consumer onboarding, separate from released Tier 1 callers."""

import json
import subprocess

import pytest
import yaml
from click.testing import CliRunner

from skillevaluator.cli import cli
from test_ci_onboarding import RELEASE
from test_ci_onboarding import release_file as _release_file
from test_ci_onboarding import repository as _repository

repository = _repository
release_file = _release_file

ALL_RELEASE = {
    **RELEASE,
    "schema_version": 2,
    "capabilities": ["all-tiers-github", "all-tiers-gitlab"],
    "feedback_ref": "d" * 40,
    "gitlab_ref": "e" * 40,
    "tools_ref": "f" * 40,
}


@pytest.fixture
def all_release(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(ALL_RELEASE))
    return path


def invoke(repo, release, *args):
    return CliRunner().invoke(
        cli, ["ci", "init", "--repo", str(repo), "--release", str(release), "--tiers", "1,2,3", "--no-input", *args]
    )


def test_all_tiers_save_github_is_minimal_pinned_and_idempotent(repository, all_release):
    result = invoke(repository, all_release, "--mode", "save", "--paths", "skills")
    assert result.exit_code == 0, result.output
    scan_path = repository / ".github/workflows/skill-checks-all-tiers.yml"
    scan = yaml.safe_load(scan_path.read_text())
    job = scan["jobs"]["skills"]
    assert job["uses"].endswith("/evaluate-all-tiers.yml@" + ALL_RELEASE["workflow_ref"])
    assert job["with"]["env-mode"] == "docker"
    assert job["with"]["tiers"] == "1,2,3"
    assert job["with"]["tier3-required"] is False
    assert job["with"]["image"] == ALL_RELEASE["image"]
    feedback = yaml.safe_load((repository / ".github/workflows/feedback-all-tiers.yml").read_text())
    assert feedback["jobs"]["feedback"]["with"]["trusted-caller-blobs"]
    assert feedback["jobs"]["feedback"]["uses"].endswith("@" + ALL_RELEASE["feedback_ref"])
    assert not (repository / ".github/workflows/publish-skill-reports.yml").exists()
    assert invoke(repository, all_release, "--mode", "save", "--paths", "skills").exit_code == 0


def test_all_tiers_preview_offline_and_local_options(repository, all_release, monkeypatch):
    monkeypatch.setenv("GH_HOST", "unrelated.example")
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])
    result = invoke(
        repository,
        all_release,
        "--mode",
        "preview",
        "--env-mode",
        "local",
        "--provider",
        "anthropic",
        "--embedding-provider",
        "openai",
        "--model",
        "claude-test",
        "--embedding-model",
        "embed-test",
        "--agents",
        "native",
        "--runner",
        "self-hosted",
        "--show-files",
    )
    assert result.exit_code == 0, result.output
    assert "env-mode: local" in result.output
    assert "embedding-provider: openai" in result.output
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


def test_all_tiers_requires_capability_instead_of_using_legacy_refs(repository, release_file):
    result = invoke(repository, release_file, "--mode", "preview")
    assert result.exit_code != 0
    assert "all-tier" in result.output.lower()
    assert "release" in result.output.lower()


@pytest.mark.parametrize("args", [("--embedding-provider", "anthropic"), ("--n-concurrent", "17")])
def test_unsupported_runtime_configuration_is_rejected_before_files(repository, all_release, args):
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])
    result = invoke(repository, all_release, "--mode", "save", *args)
    assert result.exit_code != 0
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


def test_all_tiers_conflict_does_not_partially_save(repository, all_release):
    target = repository / ".github/workflows/feedback-all-tiers.yml"
    target.write_text("name: Keep me\n")
    result = invoke(repository, all_release, "--mode", "save")
    assert result.exit_code != 0
    assert not (repository / ".github/workflows/skill-checks-all-tiers.yml").exists()
    assert target.read_text() == "name: Keep me\n"


def test_gitlab_feedback_bootstrap_pins_are_literal_inputs(repository, all_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "git@gitlab.example.test:group/consumer.git"],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "save", "--paths", "skills")
    assert result.exit_code == 0, result.output
    feedback = yaml.safe_load((repository / ".gitlab/skill-feedback.yml").read_text())
    inputs = feedback["include"][0]["inputs"]
    assert inputs["tools-ref"] == ALL_RELEASE["tools_ref"]
    assert inputs["tools-repository"] == "NVIDIA/SkillEvaluator"
    assert inputs["api-url"] == "https://gitlab.example.test/api/v4"
    assert inputs["project-id"] == "group/consumer"


def test_gitlab_selfhosted_save_is_two_files_and_no_global_auth(repository, all_release):
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "set-url",
            "origin",
            "git@gitlab.example.test:group/nested/consumer.git",
        ],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "save", "--platform", "gitlab")
    assert result.exit_code == 0, result.output
    scan = yaml.safe_load((repository / ".gitlab-ci.yml").read_text())
    assert ALL_RELEASE["gitlab_ref"] in scan["include"][0]["remote"]
    assert scan["variables"]["SKILL_EVAL_ENV_MODE"] == "docker"
    feedback = (repository / ".gitlab/skill-feedback.yml").read_text()
    assert "feedback.yml" in feedback
    assert "protected" in result.output.lower()


@pytest.mark.parametrize("tiers", ["3", "1,4", "1,1,2,3"])
def test_invalid_tier_selection_fails_before_writes(repository, all_release, tiers):
    result = CliRunner().invoke(
        cli,
        [
            "ci",
            "init",
            "--repo",
            str(repository),
            "--release",
            str(all_release),
            "--tiers",
            tiers,
            "--no-input",
            "--mode",
            "save",
        ],
    )
    assert result.exit_code != 0
    assert not (repository / ".github/workflows/skill-checks-all-tiers.yml").exists()


def test_github_setup_pr_is_api_only_and_preserves_checkout(repository, all_release, monkeypatch):
    from skillevaluator import ci_onboarding as wizard

    mutations = []
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])

    def api(method, endpoint, payload=None, **kwargs):
        if method == "POST":
            mutations.append((endpoint, payload))
            if endpoint.endswith("/git/trees"):
                return {"sha": "a" * 40}
            if endpoint.endswith("/git/commits"):
                return {"sha": "b" * 40}
            if endpoint.endswith("/git/refs"):
                return {}
            if endpoint.endswith("/pulls"):
                return {"html_url": "https://github.com/example-org/consumer/pull/1"}
            raise AssertionError(endpoint)
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        if "/contents/" in endpoint or endpoint.endswith("/git/ref/heads/ci%2Fskillevaluator-all-tiers-setup"):
            return None
        if "/pulls?" in endpoint:
            return []
        if endpoint.endswith("/git/ref/heads/trunk"):
            return {"object": {"sha": "c" * 40}}
        if endpoint.endswith("/git/commits/" + "c" * 40):
            return {"tree": {"sha": "d" * 40}}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    # API commits upload raw bytes; local Git clean filters do not apply.
    monkeypatch.setattr(wizard, "committed_blob", lambda *_args: "0" * 40)
    result = invoke(repository, all_release, "--mode", "pr", "--paths", "skills")
    assert result.exit_code == 0, result.output
    assert "https://github.com/example-org/consumer/pull/1" in result.output
    tree = next(payload for endpoint, payload in mutations if endpoint.endswith("/git/trees"))
    assert {entry["path"] for entry in tree["tree"]} == {
        ".github/workflows/skill-checks-all-tiers.yml",
        ".github/workflows/feedback-all-tiers.yml",
    }
    texts = {entry["path"]: entry["content"] for entry in tree["tree"]}
    feedback = yaml.safe_load(texts[".github/workflows/feedback-all-tiers.yml"])
    assert feedback["jobs"]["feedback"]["with"]["trusted-caller-blobs"] == wizard.blob_id(
        texts[".github/workflows/skill-checks-all-tiers.yml"]
    )
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


def test_gitlab_token_api_rejects_host_mismatch_before_network(monkeypatch):
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_TOKEN", "private-test-value")
    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_API_URL", "https://different.example/api/v4")
    with pytest.raises(Exception, match="origin host") as error:
        all_tiers.gitlab_api("https://gitlab.example", "GET", "projects/test")
    assert "private-test-value" not in str(error.value)


def test_gitlab_demo_updates_existing_manifest_and_needs_merged_setup(repository, monkeypatch):
    import base64

    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    mutations = []

    def api(origin, method, endpoint, payload=None, **kwargs):
        if method == "POST":
            mutations.append((endpoint, payload))
            return (
                {"web_url": "https://gitlab.example.test/group/consumer/-/merge_requests/2"}
                if endpoint.endswith("/merge_requests")
                else {}
            )
        if endpoint == "projects/group%2Fconsumer":
            return {"default_branch": "trunk"}
        if "/merge_requests?" in endpoint:
            return []
        if "/repository/branches/" in endpoint:
            return None
        if "/repository/tree?" in endpoint:
            return [{"type": "blob", "path": "skills/a/SKILL.md"}]
        if "/repository/files/" in endpoint:
            content = (
                "variables:\n  SKILL_EVAL_PATHS: skills\n" if ".gitlab-ci.yml?" in endpoint else "# Example skill\n"
            )
            return {"encoding": "base64", "content": base64.b64encode(content.encode()).decode()}
        raise AssertionError(endpoint)

    monkeypatch.setattr(all_tiers, "gitlab_api", api)
    result = CliRunner().invoke(cli, ["ci", "demo", "--repo", str(repository), "--no-input"])
    assert result.exit_code == 0, result.output
    commit = next(payload for endpoint, payload in mutations if endpoint.endswith("/repository/commits"))
    assert commit["actions"][0]["action"] == "update"
    assert "automatic feedback" in commit["actions"][0]["content"]


def test_existing_setup_branch_with_extra_files_is_not_reused(repository, all_release, monkeypatch):
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers
    from skillevaluator import ci_onboarding as wizard

    client = all_tiers.Client(wizard, "github", "https://github.com", "example-org/consumer")

    def api(method, endpoint, payload=None, **kwargs):
        assert method == "GET"
        if "/pulls?" in endpoint:
            return [{"html_url": "https://github.com/example-org/consumer/pull/1"}]
        if "/git/ref/heads/" in endpoint:
            return {"object": {"sha": "a" * 40}}
        if "/contents/" in endpoint:
            import base64

            return {"encoding": "base64", "content": base64.b64encode(b"generated").decode()}
        if "/compare/" in endpoint:
            return {"files": [{"filename": "one.yml"}, {"filename": "unrelated.txt"}]}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    with pytest.raises(wizard.SetupError, match="other changes"):
        client.create("setup", "trunk", {"one.yml": "generated"}, "title", "body", setup=True)


def test_gitlab_local_render_sets_actual_runtime_inputs(repository, all_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "save", "--env-mode", "local", "--timeout-minutes", "45")
    assert result.exit_code == 0, result.output
    scan = yaml.safe_load((repository / ".gitlab-ci.yml").read_text())
    assert scan["variables"]["SKILL_EVAL_IMAGE"] == ALL_RELEASE["image"]
    assert scan["variables"]["SKILL_EVAL_TIMEOUT_SECONDS"] == "2700"
    assert "SKILL_EVAL_JOB_TIMEOUT" not in scan["variables"]
    assert scan["skill-evaluation-native"]["timeout"] == "45 minutes"
    assert scan["variables"]["SKILL_EVAL_RUNNER"] == "docker"
    assert scan["skill-evaluation-native"]["services"] == []


def test_github_feedback_identity_names_exact_reusable_workflow(repository, all_release):
    result = invoke(repository, all_release, "--mode", "save")
    assert result.exit_code == 0, result.output
    feedback = yaml.safe_load((repository / ".github/workflows/feedback-all-tiers.yml").read_text())
    expected = "NVIDIA/SkillEvaluator/.github/workflows/evaluate-all-tiers.yml@" + ALL_RELEASE["workflow_ref"]
    assert feedback["jobs"]["feedback"]["with"]["trusted-workflow-ref"] == expected


def test_live_all_tier_status_recognizes_scan_and_approval(repository, all_release, monkeypatch):
    import base64

    from skillevaluator import ci_onboarding as wizard

    assert invoke(repository, all_release, "--mode", "save").exit_code == 0

    def api(method, endpoint, payload=None, **kwargs):
        assert method == "GET"
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        for path in [".github/workflows/skill-checks-all-tiers.yml", ".github/workflows/feedback-all-tiers.yml"]:
            if "/contents/" + path + "?" in endpoint:
                return {"encoding": "base64", "content": base64.b64encode((repository / path).read_bytes()).decode()}
        if "/runs?" in endpoint:
            return {"workflow_runs": []}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    result = CliRunner().invoke(cli, ["ci", "status", "--repo", str(repository)])
    assert result.exit_code == 0, result.output
    assert "matches exact workflow" in result.output
    assert "not a verified scan" in result.output


def test_generated_github_caller_grants_every_reusable_job_permission(repository, all_release):
    from pathlib import Path

    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    caller = yaml.safe_load((repository / ".github/workflows/skill-checks-all-tiers.yml").read_text())
    reusable = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/evaluate-all-tiers.yml").read_text())
    levels = {"none": 0, "read": 1, "write": 2}
    grants = caller["permissions"]
    for job in reusable["jobs"].values():
        for permission, access in job.get("permissions", {}).items():
            assert levels[grants.get(permission, "none")] >= levels[access], permission


def test_generated_gitlab_root_is_accepted_by_trusted_feedback_and_has_root_rules(repository, all_release):
    from ci.feedback import validate_gitlab_config

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    document = yaml.safe_load((repository / ".gitlab-ci.yml").read_text())
    validate_gitlab_config(document)
    assert len(document["include"]) == 2
    root_rules = document["skill-evaluation"]["rules"]
    assert {"if": '$SKILL_EVAL_FEEDBACK == "true"', "when": "never"} in root_rules
    assert {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'} in root_rules
    assert {"if": "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH"} in root_rules
    assert {"if": '$CI_PIPELINE_SOURCE == "web"'} in root_rules
    workflow = document["workflow"]["rules"]
    assert {
        "if": '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $SKILL_EVAL_FEEDBACK == "true"'
    } in workflow


def test_all_tiers_without_release_uses_packaged_candidate_descriptor(repository, tmp_path, monkeypatch):
    from skillevaluator import ci_onboarding as wizard

    package = tmp_path / "package"
    package.mkdir()
    (package / "ci-release-all-tiers.json").write_text(json.dumps(ALL_RELEASE))
    (package / "ci-release.json").write_text(json.dumps(RELEASE))
    monkeypatch.setattr(wizard, "__file__", str(package / "ci_onboarding.py"))
    result = CliRunner().invoke(
        cli, ["ci", "init", "--repo", str(repository), "--tiers", "1,2,3", "--mode", "preview", "--no-input"]
    )
    assert result.exit_code == 0, result.output
    assert "Tier 1 + Tier 2 required" in result.output


def test_candidate_descriptor_is_declared_as_package_data():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert "ci-release-all-tiers.json" in project["tool"]["setuptools"]["package-data"]["skillevaluator"]


@pytest.mark.parametrize("minutes", [45, 300])
def test_gitlab_docker_render_sets_a_literal_job_timeout(repository, all_release, minutes):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "save", "--timeout-minutes", str(minutes))
    assert result.exit_code == 0, result.output
    scan = yaml.safe_load((repository / ".gitlab-ci.yml").read_text())
    assert scan["skill-evaluation-native"] == {"timeout": f"{minutes} minutes"}
    assert scan["variables"]["SKILL_EVAL_TIMEOUT_SECONDS"] == str(minutes * 60)
    assert "SKILL_EVAL_JOB_TIMEOUT" not in scan["variables"]
