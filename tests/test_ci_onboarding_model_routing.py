"""Offline caller rendering and capability boundaries for CI model routing."""

import json
import subprocess

import pytest
import yaml
from click.testing import CliRunner

from skillevaluator import _ci_onboarding_all_tiers as all_tiers
from skillevaluator import ci_onboarding as wizard
from skillevaluator.cli import cli
from test_ci_onboarding import repository as _repository
from test_ci_onboarding_all_tiers import ALL_RELEASE, invoke

repository = _repository


@pytest.fixture
def routing_release(tmp_path, monkeypatch):
    path = tmp_path / "routing.json"
    path.write_text(json.dumps({**ALL_RELEASE, "capabilities": [*ALL_RELEASE["capabilities"], "ci-model-routing-v1"]}))
    monkeypatch.setattr(wizard, "github_api", lambda *_a, **_k: pytest.fail("offline setup contacted GitHub"))
    monkeypatch.setattr(all_tiers, "gitlab_api", lambda *_a, **_k: pytest.fail("offline setup contacted GitLab"))
    return path


ROUTING = (
    "--provider",
    "openai-compatible",
    "--base-url",
    "https://gateway.example/v1",
    "--embedding-provider",
    "openai-compatible",
    "--embedding-base-url",
    "https://embed.example/v1",
    "--tier1-llm",
    "--tier1-provider",
    "anthropic",
    "--tier1-model",
    "claude-tier1",
    "--tier3-provider",
    "openai-compatible",
    "--tier3-model",
    "judge-tier3",
    "--tier3-base-url",
    "https://judge.example/v1",
    "--agent-model",
    "codex=gpt-agent",
    "--agent-model",
    "claude-code=claude-agent",
    "--gateway-secret",
    "TEAM_GATEWAY_KEY",
)


def test_github_routing_preview_and_save_are_offline_and_thin(repository, routing_release):
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])
    preview = invoke(repository, routing_release, "--mode", "preview", "--show-files", *ROUTING)
    assert preview.exit_code == 0, preview.output
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before
    saved = invoke(repository, routing_release, "--mode", "save", *ROUTING)
    assert saved.exit_code == 0, saved.output
    scan = yaml.safe_load((repository / all_tiers.SCAN).read_text())["jobs"]["skills"]
    assert scan["with"]["tier1-llm"] is True
    assert scan["with"]["tier1-provider"] == "anthropic"
    assert scan["with"]["tier1-model"] == "claude-tier1"
    assert scan["with"]["base-url"] == "https://gateway.example/v1"
    assert scan["with"]["embedding-base-url"] == "https://embed.example/v1"
    assert scan["with"]["tier3-provider"] == "openai-compatible"
    assert scan["with"]["tier3-model"] == "judge-tier3"
    assert scan["with"]["tier3-base-url"] == "https://judge.example/v1"
    assert scan["with"]["agent-models"] == "codex=gpt-agent\nclaude-code=claude-agent"
    assert scan["secrets"]["LLM_API_KEY"] == "${{ secrets.TEAM_GATEWAY_KEY }}"
    assert scan["secrets"]["EMBEDDING_API_KEY"] == "${{ secrets.SKILL_EVAL_EMBEDDING_API_KEY }}"
    assert scan["secrets"]["OPENAI_API_KEY"] == "${{ secrets.OPENAI_API_KEY }}"
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard"], text=True
    )
    assert sorted(changed.splitlines()) == sorted([all_tiers.SCAN, all_tiers.FEEDBACK])
    assert "steps" not in scan


def test_gitlab_routing_uses_variables_and_custom_secret_alias(repository, routing_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, routing_release, "--mode", "save", *ROUTING)
    assert result.exit_code == 0, result.output
    variables = yaml.safe_load((repository / all_tiers.GL_SCAN).read_text())["variables"]
    assert variables["SKILL_EVAL_TIER1_LLM"] == "true"
    assert variables["SKILL_EVAL_TIER1_PROVIDER"] == "anthropic"
    assert variables["SKILL_EVAL_TIER1_MODEL"] == "claude-tier1"
    assert variables["SKILL_EVAL_BASE_URL"] == "https://gateway.example/v1"
    assert variables["SKILL_EVAL_EMBEDDING_BASE_URL"] == "https://embed.example/v1"
    assert variables["SKILL_EVAL_TIER3_PROVIDER"] == "openai-compatible"
    assert variables["SKILL_EVAL_TIER3_MODEL"] == "judge-tier3"
    assert variables["SKILL_EVAL_TIER3_BASE_URL"] == "https://judge.example/v1"
    assert variables["SKILL_EVAL_AGENT_MODELS"] == "codex=gpt-agent\nclaude-code=claude-agent"
    assert variables["SKILL_EVAL_LLM_API_KEY"] == "$TEAM_GATEWAY_KEY"


@pytest.mark.parametrize(
    "args",
    [
        ("--tier1-llm",),
        ("--tier3-model", "judge"),
        ("--base-url", "https://gateway.example/v1"),
        ("--agent-model", "codex=gpt"),
        ("--provider", "openai-compatible"),
    ],
)
def test_old_pins_refuse_new_routing_before_writes(repository, tmp_path, args):
    release = tmp_path / "old.json"
    release.write_text(json.dumps(ALL_RELEASE))
    result = invoke(repository, release, "--mode", "save", *args)
    assert result.exit_code != 0
    assert "ci-model-routing-v1" in result.output
    assert not (repository / all_tiers.SCAN).exists()


@pytest.mark.parametrize("secret", ["KEY }}\nX: injected", "bad-name", "1KEY", "secrets.KEY", "GITHUB_TOKEN"])
def test_secret_names_cannot_inject_caller_expressions(repository, routing_release, secret):
    result = invoke(repository, routing_release, "--mode", "save", "--gateway-secret", secret)
    assert result.exit_code != 0
    assert "secret" in result.output.lower()
    assert not (repository / all_tiers.SCAN).exists()


@pytest.mark.parametrize("value", ["missing-equals", "codex=", "unknown=model", "codex=model\nother=bad"])
def test_invalid_agent_models_fail_before_files(repository, routing_release, value):
    result = invoke(repository, routing_release, "--mode", "save", "--agent-model", value)
    assert result.exit_code != 0
    assert not (repository / all_tiers.SCAN).exists()


def test_new_settings_require_all_tier_opt_in(repository, routing_release):
    result = CliRunner().invoke(
        cli,
        [
            "ci",
            "init",
            "--repo",
            str(repository),
            "--release",
            str(routing_release),
            "--no-input",
            "--mode",
            "save",
            "--tier1-llm",
        ],
    )
    assert result.exit_code != 0
    assert "--tier1-llm" in result.output and "--tiers 1,2,3" in result.output


@pytest.mark.parametrize(
    "args, expected",
    [
        ((), {}),
        (("--tier1-llm",), {}),
        (
            ("--tier3-provider", "openai-compatible", "--tier3-base-url", "https://gateway.example/v1"),
            {"LLM_API_KEY": "${{ secrets.SKILL_EVAL_LLM_API_KEY }}"},
        ),
        (
            ("--embedding-provider", "openai-compatible", "--embedding-base-url", "https://embed.example/v1"),
            {"EMBEDDING_API_KEY": "${{ secrets.SKILL_EVAL_EMBEDDING_API_KEY }}"},
        ),
    ],
)
def test_generic_secrets_are_only_exposed_to_selected_routes(repository, routing_release, args, expected):
    result = invoke(repository, routing_release, "--mode", "save", *args)
    assert result.exit_code == 0, result.output
    job = yaml.safe_load((repository / all_tiers.SCAN).read_text())["jobs"]["skills"]
    assert {
        key: value for key, value in job["secrets"].items() if key in {"LLM_API_KEY", "EMBEDDING_API_KEY"}
    } == expected
    if not args:
        assert not set(job["with"]) & {"tier1-llm", "base-url", "tier3-model", "agent-models"}


def test_duplicate_agent_model_is_rejected_locally(repository, routing_release):
    result = invoke(
        repository, routing_release, "--mode", "save", "--agent-model", "codex=one", "--agent-model", "codex=two"
    )
    assert result.exit_code != 0
    assert "only one model" in result.output
    assert not (repository / all_tiers.SCAN).exists()


def test_native_routing_does_not_alias_an_unused_gateway_secret(repository, routing_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, routing_release, "--mode", "save", "--gateway-secret", "UNUSED_GATEWAY")
    assert result.exit_code == 0, result.output
    variables = yaml.safe_load((repository / all_tiers.GL_SCAN).read_text())["variables"]
    assert "SKILL_EVAL_LLM_API_KEY" not in variables


def test_default_gateway_key_needs_no_gitlab_self_alias(repository, routing_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example/group/consumer.git"],
        check=True,
    )
    result = invoke(
        repository,
        routing_release,
        "--mode",
        "save",
        "--provider",
        "openai-compatible",
        "--base-url",
        "https://gateway.example/v1",
    )
    assert result.exit_code == 0, result.output
    variables = yaml.safe_load((repository / all_tiers.GL_SCAN).read_text())["variables"]
    assert "SKILL_EVAL_LLM_API_KEY" not in variables


def test_tier1_llm_job_override_is_recognized_as_an_evaluator_job():
    assert all_tiers.gitlab_unguarded_jobs("skill-evaluation-tier1-llm:\n  tags: [cpu]\n") == []


@pytest.fixture
def benchmark_release(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({**ALL_RELEASE, "capabilities": [*ALL_RELEASE["capabilities"], "ci-benchmark-pr-v1"]}))
    return path


def test_benchmark_pr_adds_only_a_pinned_job_to_existing_feedback_file(repository, benchmark_release, monkeypatch):
    monkeypatch.setattr(wizard, "github_api", lambda *_a, **_k: pytest.fail("offline setup contacted GitHub"))
    result = invoke(repository, benchmark_release, "--mode", "save", "--benchmark-pr")
    assert result.exit_code == 0, result.output
    feedback = yaml.safe_load((repository / all_tiers.FEEDBACK).read_text())
    assert feedback["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "write"}
    assert set(feedback["jobs"]) == {"feedback", "benchmarks"}
    benchmarks = feedback["jobs"]["benchmarks"]
    assert benchmarks["uses"] == "NVIDIA/SkillEvaluator/.github/workflows/publish-benchmarks.yml@" + "d" * 40
    assert benchmarks["permissions"] == {"actions": "read", "contents": "write", "pull-requests": "write"}
    assert benchmarks["with"] == feedback["jobs"]["feedback"]["with"]
    assert benchmarks["with"]["tools-ref"] == "f" * 40
    assert benchmarks["with"]["trusted-workflow-ref"].endswith("@" + ALL_RELEASE["workflow_ref"])
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard"], text=True
    )
    assert sorted(changed.splitlines()) == sorted([all_tiers.SCAN, all_tiers.FEEDBACK])
    assert "benchmark-pr" not in yaml.safe_load((repository / all_tiers.SCAN).read_text())["jobs"]["skills"]["with"]


def test_benchmark_pr_requires_its_own_capability(repository, routing_release):
    result = invoke(repository, routing_release, "--mode", "save", "--benchmark-pr")
    assert result.exit_code != 0
    assert "ci-benchmark-pr-v1" in result.output
    assert not (repository / all_tiers.SCAN).exists()


def test_benchmark_pr_refuses_gitlab_without_writing(repository, benchmark_release):
    subprocess.run(
        ["git", "-C", str(repository), "remote", "set-url", "origin", "https://gitlab.example/group/consumer.git"],
        check=True,
    )
    result = invoke(repository, benchmark_release, "--mode", "save", "--benchmark-pr")
    assert result.exit_code != 0
    assert "benchmark-update PR publication currently requires GitHub" in result.output
    assert "GitLab cards retained in reports" in result.output
    assert not (repository / all_tiers.GL_SCAN).exists()


def test_benchmark_capability_alone_keeps_default_output_byte_identical(repository, benchmark_release, tmp_path):
    old_release = tmp_path / "old-default.json"
    old_release.write_text(json.dumps(ALL_RELEASE))
    before = invoke(repository, old_release, "--mode", "preview", "--show-files")
    after = invoke(repository, benchmark_release, "--mode", "preview", "--show-files")
    assert before.exit_code == after.exit_code == 0
    assert before.output == after.output
