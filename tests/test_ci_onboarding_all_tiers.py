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

REUSABLE_FEEDBACK = """# Preserve this reusable interface.
name: Trusted three-tier skill feedback
on:
  workflow_call:
    inputs:
      tools-ref:
        type: string
        required: true
jobs:
  feedback:
    runs-on: ubuntu-24.04
    steps:
      - run: echo reusable
"""


@pytest.fixture
def reusable_feedback(repository):
    path = repository / ".github/workflows/feedback-all-tiers.yml"
    path.write_text(REUSABLE_FEEDBACK)
    subprocess.run(["git", "-C", str(repository), "add", str(path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repository),
            "commit",
            "-m",
            "Add reusable feedback workflow",
        ],
        check=True,
        capture_output=True,
    )
    return path


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
    assert "tier3-required" not in job["with"]
    assert job["with"]["image"] == ALL_RELEASE["image"]
    feedback = yaml.safe_load((repository / ".github/workflows/feedback-all-tiers.yml").read_text())
    assert feedback["jobs"]["feedback"]["with"]["trusted-caller-blobs"]
    assert feedback["jobs"]["feedback"]["uses"].endswith("@" + ALL_RELEASE["feedback_ref"])
    assert not (repository / ".github/workflows/publish-skill-reports.yml").exists()
    assert invoke(repository, all_release, "--mode", "save", "--paths", "skills").exit_code == 0


def test_all_tiers_save_preserves_reusable_feedback_workflow(repository, all_release, reusable_feedback):
    before = {path.relative_to(repository).as_posix() for path in repository.glob(".github/workflows/*")}
    original = reusable_feedback.read_bytes()
    result = invoke(repository, all_release, "--mode", "save", "--paths", "skills")
    assert result.exit_code == 0, result.output
    after = {path.relative_to(repository).as_posix() for path in repository.glob(".github/workflows/*")}
    assert after - before == {
        ".github/workflows/skill-checks-all-tiers.yml",
        ".github/workflows/skill-feedback-all-tiers.yml",
    }
    assert reusable_feedback.read_bytes() == original
    alternate = repository / ".github/workflows/skill-feedback-all-tiers.yml"
    feedback = yaml.safe_load(alternate.read_text())
    assert feedback["jobs"]["feedback"]["uses"].endswith("@" + ALL_RELEASE["feedback_ref"])
    generated = alternate.read_bytes()
    result = invoke(repository, all_release, "--mode", "save", "--paths", "skills")
    assert result.exit_code == 0, result.output
    assert reusable_feedback.read_bytes() == original
    assert alternate.read_bytes() == generated


def test_reusable_feedback_does_not_bypass_conflicting_alternate(repository, all_release, reusable_feedback):
    alternate = repository / ".github/workflows/skill-feedback-all-tiers.yml"
    alternate.write_text("name: Keep this unrelated workflow\n")
    result = invoke(repository, all_release, "--mode", "save")
    assert result.exit_code != 0
    assert "skill-feedback-all-tiers.yml already exists" in " ".join(result.output.split())
    assert not (repository / ".github/workflows/skill-checks-all-tiers.yml").exists()
    assert reusable_feedback.read_text() == REUSABLE_FEEDBACK
    assert alternate.read_text() == "name: Keep this unrelated workflow\n"


def test_reusable_feedback_detection_does_not_read_symlinks(repository, all_release, tmp_path, monkeypatch):
    from pathlib import Path

    outside = tmp_path / "outside.yml"
    outside.write_text(REUSABLE_FEEDBACK)
    target = repository / ".github/workflows/feedback-all-tiers.yml"
    target.symlink_to(outside)
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        assert path != target, "A workflow symlink must be rejected before its contents are read"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    result = invoke(repository, all_release, "--mode", "preview")
    assert result.exit_code != 0
    assert "Symlink" in result.output
    assert not (repository / ".github/workflows/skill-feedback-all-tiers.yml").exists()


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


@pytest.mark.parametrize("content", ["name: Keep me\n", "true:\n  workflow_call:\n", "on:\n  workflow_dispatch:\n"])
def test_all_tiers_conflict_does_not_partially_save(repository, all_release, content):
    target = repository / ".github/workflows/feedback-all-tiers.yml"
    target.write_text(content)
    result = invoke(repository, all_release, "--mode", "save")
    assert result.exit_code != 0
    assert ".github/workflows/feedback-all-tiers.yml already exists" in " ".join(result.output.split())
    assert not (repository / ".github/workflows/skill-checks-all-tiers.yml").exists()
    assert target.read_text() == content


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


@pytest.mark.parametrize("has_reusable_feedback", [False, True])
@pytest.mark.parametrize("benchmark_pr", [False, True])
def test_github_setup_pr_is_api_only_and_preserves_checkout(
    repository, all_release, monkeypatch, request, has_reusable_feedback, benchmark_pr
):
    import base64

    from skillevaluator import ci_onboarding as wizard

    if has_reusable_feedback:
        request.getfixturevalue("reusable_feedback")
    feedback_path = ".github/workflows/" + (
        "skill-feedback-all-tiers.yml" if has_reusable_feedback else "feedback-all-tiers.yml"
    )
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
        if has_reusable_feedback and "/contents/.github/workflows/feedback-all-tiers.yml?" in endpoint:
            return {"encoding": "base64", "content": base64.b64encode(REUSABLE_FEEDBACK.encode()).decode()}
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
    if benchmark_pr:
        release = json.loads(all_release.read_text())
        release["capabilities"].append("ci-benchmark-pr-v1")
        all_release.write_text(json.dumps(release))
    result = invoke(
        repository, all_release, "--mode", "pr", "--paths", "skills", *(["--benchmark-pr"] if benchmark_pr else [])
    )
    assert result.exit_code == 0, result.output
    assert "https://github.com/example-org/consumer/pull/1" in result.output
    tree = next(payload for endpoint, payload in mutations if endpoint.endswith("/git/trees"))
    assert {entry["path"] for entry in tree["tree"]} == {
        ".github/workflows/skill-checks-all-tiers.yml",
        feedback_path,
    }
    texts = {entry["path"]: entry["content"] for entry in tree["tree"]}
    feedback = yaml.safe_load(texts[feedback_path])
    setup = next(payload for endpoint, payload in mutations if endpoint.endswith("/pulls"))
    assert "Tier 1 required" in setup["body"]
    assert "Tier 2 and Tier 3 advisory" in setup["body"]
    assert set(feedback["jobs"]) == ({"feedback", "benchmarks"} if benchmark_pr else {"feedback"})
    for job in feedback["jobs"].values():
        assert job["with"]["trusted-caller-blobs"] == wizard.blob_id(
            texts[".github/workflows/skill-checks-all-tiers.yml"]
        )
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


@pytest.mark.parametrize("live_content", [None, "name: Existing consumer feedback\n"])
def test_github_setup_pr_refuses_reusable_feedback_drift(
    repository, all_release, reusable_feedback, monkeypatch, live_content
):
    import base64

    from skillevaluator import ci_onboarding as wizard

    def api(method, endpoint, payload=None, **kwargs):
        assert method == "GET", "Conflicting default-branch feedback must stop before mutation"
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        if "/contents/.github/workflows/feedback-all-tiers.yml?" in endpoint and live_content is not None:
            return {"encoding": "base64", "content": base64.b64encode(live_content.encode()).decode()}
        if "/contents/" in endpoint:
            return None
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    result = invoke(repository, all_release, "--mode", "pr", "--paths", "skills")
    assert result.exit_code != 0
    assert "default branch" in result.output
    assert not (repository / ".github/workflows/skill-checks-all-tiers.yml").exists()
    assert reusable_feedback.read_text() == REUSABLE_FEEDBACK


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
    result = CliRunner().invoke(cli, ["ci", "demo", "--repo", str(repository), "--no-input", "--tiers", "1,2,3"])
    assert result.exit_code == 0, result.output
    commit = next(payload for endpoint, payload in mutations if endpoint.endswith("/repository/commits"))
    assert commit["actions"][0]["action"] == "update"
    assert "automatic feedback" in commit["actions"][0]["content"]
    review = next(payload for endpoint, payload in mutations if endpoint.endswith("/merge_requests"))
    assert "required Tier 1" in review["description"]
    assert "advisory Tier 2 and paired Tier 3" in review["description"]


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
    # 45 minutes minus min(20, 45 // 4) leaves time to retain evidence.
    assert scan["variables"]["SKILL_EVAL_TIMEOUT_SECONDS"] == "2040"
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


@pytest.mark.parametrize("alternate", ["generated", None, "name: Unrelated alternate\n"])
def test_github_status_recognizes_reusable_feedback_collision(repository, all_release, monkeypatch, alternate):
    import base64

    from skillevaluator import ci_onboarding as wizard

    # The live default branch has a reusable workflow even though this local
    # checkout has the ordinary consumer caller at the canonical feedback path.
    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    scan_path = ".github/workflows/skill-checks-all-tiers.yml"
    alternate_path = ".github/workflows/skill-feedback-all-tiers.yml"
    texts = {
        scan_path: (repository / scan_path).read_text(),
        ".github/workflows/feedback-all-tiers.yml": REUSABLE_FEEDBACK,
    }
    if alternate is not None:
        texts[alternate_path] = (
            (repository / ".github/workflows/feedback-all-tiers.yml").read_text()
            if alternate == "generated"
            else alternate
        )

    def api(method, endpoint, payload=None, **kwargs):
        assert method == "GET"
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        for path, text in texts.items():
            if "/contents/" + path + "?" in endpoint:
                return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}
        if "/contents/" + alternate_path + "?" in endpoint:
            return None
        if "/runs?" in endpoint:
            return {"workflow_runs": []}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    result = CliRunner().invoke(cli, ["ci", "status", "--repo", str(repository)])
    assert result.exit_code == 0, result.output
    assert alternate_path in " ".join(result.output.split())
    if alternate == "generated":
        assert "Trusted feedback: configured" in result.output
        assert "Scan approval: matches exact workflow" in result.output
    elif alternate is None:
        assert "Trusted feedback: missing" in result.output
    else:
        assert "not a trusted feedback configuration" in " ".join(result.output.split())
        assert "Trusted feedback: configured" not in result.output


@pytest.mark.parametrize(
    ("component", "content"),
    [
        (
            "checks",
            "jobs:\n  skills:\n    uses: NVIDIA/SkillEvaluator/.github/workflows/evaluate.yml@"
            + "a" * 40
            + "\n    with:\n      tiers: '1,2,3'\n",
        ),
        (
            "checks",
            "jobs:\n  skills:\n    uses: NVIDIA/SkillEvaluator/.github/workflows/evaluate-all-tiers.yml@main"
            "\n    with:\n      tiers: '1,2,3'\n",
        ),
        (
            "checks",
            "jobs:\n  skills:\n    uses: NVIDIA/SkillEvaluator/.github/workflows/evaluate-all-tiers.yml@"
            + "a" * 40
            + "\n    with:\n      tiers: '1'\n",
        ),
        ("checks", "jobs: ["),
        ("checks", {}),
        ("feedback", {}),
        ("feedback", "null\n"),
        ("feedback", "- not a workflow\n"),
        ("feedback", "jobs: ["),
        (
            "feedback",
            "jobs:\n  feedback:\n    uses: NVIDIA/SkillEvaluator/.github/workflows/feedback-all-tiers.yml@"
            + "a" * 40
            + "\n    with: []\n",
        ),
    ],
)
def test_github_status_does_not_claim_invalid_workflows_are_configured(
    repository, all_release, monkeypatch, component, content
):
    import base64

    from skillevaluator import ci_onboarding as wizard

    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    texts = {
        path: (repository / path).read_text()
        for path in [".github/workflows/skill-checks-all-tiers.yml", ".github/workflows/feedback-all-tiers.yml"]
    }
    target = ".github/workflows/" + (
        "skill-checks-all-tiers.yml" if component == "checks" else "feedback-all-tiers.yml"
    )
    if isinstance(content, dict):
        document = yaml.safe_load(texts[target])
        document["jobs"]["skills" if component == "checks" else "feedback"]["with"] = content
        texts[target] = yaml.safe_dump(document)
    else:
        texts[target] = content

    def api(method, endpoint, payload=None, **kwargs):
        assert method == "GET"
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        for path, text in texts.items():
            if "/contents/" + path + "?" in endpoint:
                return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}
        if "/runs?" in endpoint:
            return {"workflow_runs": []}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    result = CliRunner().invoke(cli, ["ci", "status", "--repo", str(repository), "--tiers", "1,2,3"])
    assert result.exit_code == 0, result.output
    label = "All-tier checks" if component == "checks" else "Trusted feedback"
    assert label + ": configured" not in result.output
    output = " ".join(result.output.split())
    assert "not an all-tier configuration" in output or "not a trusted feedback configuration" in output


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
    assert "Tier 1 required; Tiers 2 and 3 advisory" in result.output


@pytest.mark.parametrize("platform", ["github", "gitlab"])
def test_deprecated_tier3_required_flag_warns_and_never_changes_generated_policy(
    repository, all_release, tmp_path, platform
):
    if platform == "gitlab":
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "remote",
                "set-url",
                "origin",
                "https://gitlab.example.test/group/consumer.git",
            ],
            check=True,
        )
    receipt = tmp_path / "policy-receipt.json"
    result = invoke(repository, all_release, "--mode", "save", "--tier3-required", "--output", str(receipt))
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.output.lower() and "ignored" in result.output.lower()
    assert "Tier 1 required; Tiers 2 and 3 advisory" in result.output
    assert json.loads(receipt.read_text())["options"]["tier3_required"] is False
    if platform == "github":
        scan = yaml.safe_load((repository / ".github/workflows/skill-checks-all-tiers.yml").read_text())
        assert "tier3-required" not in scan["jobs"]["skills"]["with"]
    else:
        scan = yaml.safe_load((repository / ".gitlab-ci.yml").read_text())
        assert "SKILL_EVAL_TIER3_REQUIRED" not in scan["variables"]


def test_candidate_descriptor_is_declared_as_package_data():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert "ci-release-all-tiers.json" in project["tool"]["setuptools"]["package-data"]["skillevaluator"]


@pytest.mark.parametrize(("minutes", "seconds"), [(10, 480), (45, 2040), (80, 3600), (300, 16800)])
def test_gitlab_docker_render_sets_a_literal_job_timeout(repository, all_release, minutes, seconds):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "save", "--timeout-minutes", str(minutes))
    assert result.exit_code == 0, result.output
    scan = yaml.safe_load((repository / ".gitlab-ci.yml").read_text())
    assert scan["skill-evaluation-native"] == {"timeout": f"{minutes} minutes"}
    assert scan["variables"]["SKILL_EVAL_TIMEOUT_SECONDS"] == str(seconds)
    assert "SKILL_EVAL_JOB_TIMEOUT" not in scan["variables"]


@pytest.mark.parametrize("minutes", [1, 3, 9])
def test_job_timeouts_without_an_evidence_reserve_are_rejected(repository, all_release, minutes):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    # Below ten minutes the reserve min(20, minutes // 4) is under two minutes, or none.
    result = invoke(repository, all_release, "--mode", "save", "--timeout-minutes", str(minutes))
    assert result.exit_code == 2
    assert "--timeout-minutes" in result.output
    assert not (repository / ".gitlab-ci.yml").exists()


def test_gitlab_variable_scalars_are_explicit_strings_for_gitlab_yaml_parser(repository, all_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "save")
    assert result.exit_code == 0, result.output
    source = (repository / ".gitlab-ci.yml").read_text()
    document = yaml.compose(source)
    variables = next(value for key, value in document.value if key.value == "variables")
    assert all(value.style in {"'", '"'} for _key, value in variables.value)
    parsed = yaml.safe_load(source)
    assert parsed["variables"]["SKILL_EVAL_TIERS"] == "1,2,3"
    assert parsed["skill-evaluation-native"]["timeout"] == "300 minutes"
    feedback = yaml.safe_load((repository / ".gitlab/skill-feedback.yml").read_text())
    assert isinstance(feedback["include"][0]["inputs"]["tools-ref"], str)


@pytest.mark.parametrize(
    "version,required",
    [
        ("17.8.7-ee", True),
        ("18.3.0", True),
        ("18.4.0-ee", False),
        ("19.0.1", False),
        ("unknown", None),
        (None, None),
        (123, None),
        (True, None),
    ],
)
def test_gitlab_admission_prerequisite_checks_server_version(monkeypatch, version, required):
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    monkeypatch.setattr(all_tiers, "gitlab_api", lambda *_args, **_kwargs: {"version": version})
    result = all_tiers.gitlab_admission_prerequisite("https://gitlab.example.test", online=True)
    assert result["read_token_required"] is required
    assert result["verified"] is (required is not None)


def test_gitlab_offline_admission_prerequisite_does_not_call_api(monkeypatch):
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    monkeypatch.setattr(
        all_tiers, "gitlab_api", lambda *_args, **_kwargs: pytest.fail("Offline setup must remain offline")
    )
    result = all_tiers.gitlab_admission_prerequisite("https://gitlab.example.test", online=False)
    assert result["verified"] is False and result["read_token_required"] is None


@pytest.mark.parametrize(
    "version,phrase",
    [
        ("17.8.7-ee", "requires SKILLEVALUATOR_GITLAB_READ_TOKEN"),
        ("18.4.0", "supports CI_JOB_TOKEN"),
        ("unknown", "admission readiness is unverified"),
    ],
)
def test_gitlab_status_reports_admission_version_prerequisite(repository, monkeypatch, version, phrase):
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )

    def api(_origin, method, endpoint, payload=None, **_kwargs):
        assert method == "GET"
        if endpoint == "version":
            return {"version": version}
        if endpoint == "projects/group%2Fconsumer":
            return {"default_branch": "trunk"}
        if "/repository/files/" in endpoint:
            return None
        if "/pipelines?" in endpoint:
            return []
        raise AssertionError(endpoint)

    monkeypatch.setattr(all_tiers, "gitlab_api", api)
    result = CliRunner().invoke(
        cli, ["ci", "status", "--repo", str(repository), "--platform", "gitlab", "--tiers", "1,2,3"]
    )
    assert result.exit_code == 0, result.output
    assert phrase in result.output
    assert "credential validity is not verified" in result.output


@pytest.mark.parametrize("tiers", [None, "1,2,3"])
def test_preview_ignores_user_insteadof_rewrite_of_a_supported_origin(
    repository, release_file, all_release, tmp_path, monkeypatch, tiers
):
    # A developer-level rewrite to ssh:// must not hide the configured github.com origin.
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text('[url "ssh://git@github.com/"]\n\tinsteadOf = https://github.com/\n')
    monkeypatch.setenv("HOME", str(home))
    rewritten = subprocess.check_output(["git", "-C", str(repository), "remote", "get-url", "origin"], text=True)
    assert rewritten.startswith("ssh://")
    args = ["ci", "init", "--repo", str(repository), "--no-input", "--mode", "preview"]
    args += ["--release", str(release_file)] if tiers is None else ["--release", str(all_release), "--tiers", tiers]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output


def test_alias_origin_parses_after_insteadof_rewrite(repository, all_release):
    subprocess.run(["git", "-C", str(repository), "remote", "set-url", "origin", "gh:example-org/consumer"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "url.https://github.com/.insteadOf", "gh:"],
        check=True,
    )
    result = invoke(repository, all_release, "--mode", "preview")
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", [["status"], ["demo", "--no-input"]])
def test_unrecognized_origin_never_receives_generic_gitlab_token(repository, monkeypatch, command):
    import urllib.request

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://codeberg.org/someone/skills.git"],
        check=True,
    )
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-ambient-test-value")
    monkeypatch.delenv("SKILLEVALUATOR_GITLAB_TOKEN", raising=False)
    requests = []
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: requests.append(a) or pytest.fail("network"))
    result = CliRunner().invoke(cli, ["ci", *command, "--repo", str(repository)])
    assert result.exit_code != 0
    assert "--tiers 1,2,3" in result.output
    assert requests == []
    # An explicit GitLab opt-in still refuses the generic token.
    result = CliRunner().invoke(cli, ["ci", *command, "--repo", str(repository), "--platform", "gitlab"])
    assert result.exit_code != 0
    assert "SKILLEVALUATOR_GITLAB_TOKEN" in result.output
    assert "glpat-ambient-test-value" not in result.output
    assert requests == []


def _gitlab_status(repository, monkeypatch, root_config):
    import base64

    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    calls = []

    def api(origin, method, endpoint, payload=None, **kwargs):
        assert method == "GET"
        calls.append(endpoint)
        if endpoint == "version":
            return {"version": "18.4.0"}
        if endpoint == "projects/group%2Fconsumer":
            return {"default_branch": "trunk"}
        if "/repository/files/" in endpoint:
            content = root_config if ".gitlab-ci.yml?" in endpoint else "include: []\n"
            return {"encoding": "base64", "content": base64.b64encode(content.encode()).decode()}
        if "/pipelines?" in endpoint:
            return [{"status": "success", "web_url": "https://gitlab.example.test/p/1"}]
        raise AssertionError(endpoint)

    monkeypatch.setattr(all_tiers, "gitlab_api", api)
    result = CliRunner().invoke(cli, ["ci", "status", "--repo", str(repository), "--tiers", "1,2,3"])
    assert result.exit_code == 0, result.output
    return result.output, calls


def test_gitlab_status_requires_trusted_all_tier_root_config(repository, monkeypatch):
    output, calls = _gitlab_status(repository, monkeypatch, "build:\n  script: make\n")
    assert "All-tier checks: configured" not in output
    assert "not an all-tier configuration" in output
    assert not any("/pipelines" in call for call in calls)


def test_gitlab_status_accepts_generated_root_and_filters_default_branch(repository, all_release, monkeypatch):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    output, calls = _gitlab_status(repository, monkeypatch, (repository / ".gitlab-ci.yml").read_text())
    assert "All-tier checks: configured" in output
    assert any("/pipelines?ref=trunk" in call for call in calls)


@pytest.mark.parametrize(
    "document",
    [
        "build:\n  script: make\n",
        "include:\n  - remote: https://raw.githubusercontent.com/NVIDIA/SkillEvaluator/main/ci/gitlab/all-tiers.yml\n"
        "  - local: .gitlab/skill-feedback.yml\n",
        "include:\n  - remote: https://raw.githubusercontent.com/NVIDIA/SkillEvaluator/"
        + "a" * 40
        + "/ci/gitlab/all-tiers.yml\n"
        "  - local: .gitlab/other.yml\n",
        "- not a mapping\n",
    ],
)
def test_status_gitlab_validation_matches_trusted_feedback(document):
    from ci.feedback import FeedbackError, validate_gitlab_config

    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    with pytest.raises(FeedbackError):
        validate_gitlab_config(document.encode())
    assert all_tiers.gitlab_config_problem(document) is not None


@pytest.mark.parametrize(
    ("separator", "expected"),
    [(",", "matches exact workflow"), ("\n", "matches exact workflow"), (" ", "Malformed approval list")],
)
def test_status_parses_caller_blobs_like_trusted_feedback(repository, all_release, monkeypatch, separator, expected):
    import base64

    from skillevaluator import ci_onboarding as wizard

    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    scan = (repository / ".github/workflows/skill-checks-all-tiers.yml").read_bytes()
    feedback = yaml.safe_load((repository / ".github/workflows/feedback-all-tiers.yml").read_text())
    feedback["jobs"]["feedback"]["with"]["trusted-caller-blobs"] = "1" * 40 + separator + wizard.blob_id(scan.decode())
    texts = {
        ".github/workflows/skill-checks-all-tiers.yml": scan,
        ".github/workflows/feedback-all-tiers.yml": yaml.safe_dump(feedback).encode(),
    }

    def api(method, endpoint, payload=None, **kwargs):
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        for path, text in texts.items():
            if "/contents/" + path + "?" in endpoint:
                return {"encoding": "base64", "content": base64.b64encode(text).decode()}
        if "/runs?" in endpoint:
            return {"workflow_runs": []}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    result = CliRunner().invoke(cli, ["ci", "status", "--repo", str(repository)])
    assert result.exit_code == 0, result.output
    assert expected in result.output


def test_setup_pr_next_steps_name_the_all_tier_demo(repository, all_release, monkeypatch):
    from skillevaluator import ci_onboarding as wizard

    bodies = []

    def api(method, endpoint, payload=None, **kwargs):
        if method == "POST":
            if endpoint.endswith("/pulls"):
                bodies.append(payload["body"])
                return {"html_url": "https://github.com/example-org/consumer/pull/1"}
            return {"sha": "a" * 40}
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
    monkeypatch.setattr(wizard, "committed_blob", lambda *_args: "0" * 40)
    result = invoke(repository, all_release, "--mode", "pr", "--paths", "skills")
    assert result.exit_code == 0, result.output
    assert "ci demo --tiers 1,2,3" in bodies[0]
    assert "ci demo --tiers 1,2,3" in result.output


def test_all_tier_only_init_options_require_tiers(repository, release_file):
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])
    result = CliRunner().invoke(
        cli,
        [
            "ci",
            "init",
            "--repo",
            str(repository),
            "--release",
            str(release_file),
            "--no-input",
            "--mode",
            "save",
            "--checks-only",
            "--paths",
            "skills",
            "--provider",
            "anthropic",
            "--timeout-minutes",
            "45",
        ],
    )
    assert result.exit_code != 0
    assert "--provider" in result.output and "--timeout-minutes" in result.output
    assert "--tiers 1,2,3" in result.output
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


def test_all_tier_receipt_parent_is_created_before_saving(repository, all_release, tmp_path):
    receipt = tmp_path / "receipts" / "nested" / "plan.json"
    result = invoke(repository, all_release, "--mode", "save", "--paths", "skills", "--output", str(receipt))
    assert result.exit_code == 0, result.output
    assert json.loads(receipt.read_text())["action"] == "saved"


def test_all_tier_preview_lists_proposed_files_without_show_files(repository, all_release):
    result = invoke(repository, all_release, "--mode", "preview", "--paths", "skills")
    assert result.exit_code == 0, result.output
    assert ".github/workflows/skill-checks-all-tiers.yml" in result.output
    assert ".github/workflows/feedback-all-tiers.yml" in result.output
    assert "--show-files" in result.output
    assert "jobs:" not in result.output


def _gitlab_tier1_consumer(repository, extra=""):
    from pathlib import Path

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    released = (Path(__file__).parents[1] / "ci/bundles/0.7.2/gitlab-skill-checks.yml").read_text()
    root = repository / ".gitlab-ci.yml"
    root.write_text(released + extra)
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "commit.gpgsign=false", "commit", "-qm", "Tier 1"],
        check=True,
    )
    return root, root.read_bytes()


DEPLOY_JOB = "deploy:\n  script: ./deploy.sh\n  rules:\n    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'\n"


def test_gitlab_preview_prints_merge_steps_for_existing_root_config(repository, all_release):
    root, before = _gitlab_tier1_consumer(repository, DEPLOY_JOB)
    result = invoke(repository, all_release, "--mode", "preview", "--paths", "skills")
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert ".gitlab-ci.yml already exists; it will not be replaced" in output
    assert "Remove the Tier 1 skill-evaluation.yml include" in output
    assert "Jobs to guard: deploy." in output
    assert "Merge the guards to the default branch first" in output
    assert "- if: '$SKILL_EVAL_FEEDBACK == \"true\"' when: never" in output
    # The rendered root and feedback files are printed even without --show-files.
    assert ALL_RELEASE["gitlab_ref"] + "/ci/gitlab/all-tiers.yml" in output
    assert "skill-evaluation-native: timeout: 300 minutes" in output
    assert ".gitlab/skill-feedback.yml" in output and "/ci/gitlab/feedback.yml" in output
    assert root.read_bytes() == before
    assert not (repository / ".gitlab").exists()


def test_gitlab_merge_steps_keep_the_test_stage_and_limit_extends_to_jobs_without_only(repository, all_release):
    _gitlab_tier1_consumer(repository, "stages: [build, deploy]\nlegacy:\n  script: ./legacy.sh\n  only: [main]\n")
    result = invoke(repository, all_release, "--mode", "preview", "--paths", "skills")
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    # Every included job uses stage test; a root stages list without it rejects the pipeline.
    assert "If the root defines stages, keep test in the list" in output
    # GitLab rejects only/except beside the rules a job would inherit from the hidden guard.
    assert "a job without rules, only or except" in output
    assert "Jobs to guard: legacy." in output
    # Guards merge before the setup MR, when the hidden job is not yet defined.
    assert "guard followed by - when: on_success" in output
    assert "Write those guards as rules: extends: .skill-evaluation-not-feedback works only after" in output
    # Feedback triggers pass pipeline variables, which GitLab can restrict per project.
    assert "pass pipeline variables" in output


def test_gitlab_jobs_keeping_only_or_except_are_never_counted_as_guarded():
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    guard = "  rules:\n    - if: '$SKILL_EVAL_FEEDBACK == \"true\"'\n      when: never\n"
    root = "".join(
        [
            ".old:\n  only: [main]\n",
            "extended:\n  script: a\n  only: [main]\n  extends: .skill-evaluation-not-feedback\n",
            "inherited:\n  script: b\n  extends: [.old, .skill-evaluation-not-feedback]\n",
            "ruled:\n  script: c\n  except: [tags]\n" + guard,
            "guarded:\n  script: d\n  extends: .skill-evaluation-not-feedback\n",
            "plain:\n  script: e\n" + guard,
            # A null value drops an inherited key, and GitLab ignores null only/except beside rules.
            ".mid:\n  extends: .old\n  only: null\n",
            "dropped:\n  script: f\n  extends: .old\n  only: null\n" + guard + "    - when: on_success\n",
            "nulled:\n  script: g\n  except: null\n" + guard + "    - when: on_success\n",
            "middle:\n  script: h\n  extends: [.mid, .skill-evaluation-not-feedback]\n",
            # The last parent defining a key wins, so a later .old restores only.
            "restored:\n  script: i\n  extends: [.mid, .old, .skill-evaluation-not-feedback]\n",
        ]
    )
    assert all_tiers.gitlab_unguarded_jobs(root) == ["extended", "inherited", "ruled", "restored"]


def test_gitlab_pr_remote_root_conflict_points_to_an_updated_checkout(repository, all_release, monkeypatch):
    import base64

    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    remote_root = "build:\n  script: make\n"

    def api(origin, method, endpoint, payload=None, **kwargs):
        assert method == "GET", "A conflicting remote root must stop before any mutation"
        if endpoint == "version":
            return {"version": "18.4.0"}
        if endpoint == "projects/group%2Fconsumer":
            return {"default_branch": "trunk"}
        if endpoint == "projects/group%2Fconsumer/repository/files/.gitlab-ci.yml?ref=trunk":
            return {"encoding": "base64", "content": base64.b64encode(remote_root.encode()).decode()}
        raise AssertionError(endpoint)

    monkeypatch.setattr(all_tiers, "gitlab_api", api)
    # The stale checkout has no root, so preview alone would print no merge steps.
    result = invoke(repository, all_release, "--mode", "pr", "--paths", "skills")
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert ".gitlab-ci.yml differs on the default branch; it will not be replaced" in output
    assert "Update your checkout to the default branch, then run --mode preview" in output
    assert not (repository / ".gitlab-ci.yml").exists()
    # Following the hint reaches the merge steps.
    (repository / ".gitlab-ci.yml").write_text(remote_root)
    preview = invoke(repository, all_release, "--mode", "preview", "--paths", "skills")
    assert preview.exit_code == 0, preview.output
    assert "Merge the rendered files below by hand" in " ".join(preview.output.split())


@pytest.mark.parametrize("mode", ["save", "pr"])
def test_gitlab_save_and_pr_refuse_to_replace_existing_root_config(repository, all_release, monkeypatch, mode):
    from skillevaluator import _ci_onboarding_all_tiers as all_tiers

    monkeypatch.setattr(all_tiers, "gitlab_api", lambda *_a, **_k: pytest.fail("refused before any API call"))
    root, before = _gitlab_tier1_consumer(repository)
    result = invoke(repository, all_release, "--mode", mode, "--paths", "skills")
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert ".gitlab-ci.yml already exists with different content; it will not be replaced" in output
    assert "--mode preview" in output and "merge" in output
    assert root.read_bytes() == before
    assert not (repository / ".gitlab").exists()


def test_gitlab_interactive_save_refuses_existing_root_config(repository, all_release):
    root, before = _gitlab_tier1_consumer(repository)
    result = CliRunner().invoke(
        cli,
        ["ci", "init", "--repo", str(repository), "--release", str(all_release), "--tiers", "1,2,3"],
        input="2\nsave\n",
    )
    assert result.exit_code != 0
    assert "it will not be replaced" in result.output
    assert root.read_bytes() == before
    assert not (repository / ".gitlab").exists()


def test_gitlab_status_warns_about_root_jobs_without_feedback_guard(repository, all_release, monkeypatch):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    generated = (repository / ".gitlab-ci.yml").read_text()
    output, _calls = _gitlab_status(repository, monkeypatch, generated)
    assert "Warning" not in output
    guarded = (
        "lint:\n  script: make lint\n  rules:\n    - if: \"$SKILL_EVAL_FEEDBACK == 'true'\"\n      when: never\n"
        "    - when: on_success\n"
        "docs:\n  script: make docs\n  extends: .skill-evaluation-not-feedback\n"
        ".base:\n  rules:\n    - if: '$SKILL_EVAL_FEEDBACK == \"true\"'\n      when: never\n"
        "    - if: $CI_COMMIT_BRANCH\n"
        "package:\n  script: make package\n  extends: [.base]\n"
    )
    output, _calls = _gitlab_status(repository, monkeypatch, generated + guarded + DEPLOY_JOB + "test:\n  script: t\n")
    assert "All-tier checks: configured" in output
    normalized = " ".join(output.split())
    assert "also run deploy, test." in normalized
    assert "SKILL_EVAL_FEEDBACK" in normalized and ".skill-evaluation-not-feedback" in normalized
    assert "a job without rules, only or except can extend" in normalized


def test_gitlab_status_recognizes_tier2_job_overrides(repository, all_release, monkeypatch):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example.test/group/consumer.git"],
        check=True,
    )
    assert invoke(repository, all_release, "--mode", "save").exit_code == 0
    generated = (repository / ".gitlab-ci.yml").read_text()
    output, _calls = _gitlab_status(
        repository, monkeypatch, generated + "skill-evaluation-native-tier2:\n  tags: [cpu]\n"
    )
    assert "All-tier checks: configured" in output
    assert "Warning" not in output
