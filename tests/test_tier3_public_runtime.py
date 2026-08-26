# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Tier 3 runtime boundaries."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.model_catalog import ModelCatalogFailureKind
from skillevaluator.provider_config import ProviderConfig, resolve_llm_provider
from skillevaluator.tier3 import commands as tier3_commands
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor.adapter import _EVALUATOR_MANAGED_RUNTIME_ENV, _write_task_toml
from skillevaluator.tier3.harbor.runner import (
    _check_prerequisites,
    _environment_extra_install_hint,
    _environment_kwarg_prerequisite_errors,
    _model_for_agent,
    _nvidia_build_agent_import_path,
    _provider_environment,
    _validate_agent_provider_credentials,
    build_harbor_run_command,
)
from skillevaluator.tier3.harbor.runtime_preflight import ModelProbeResult
from skillevaluator.tier3_environments import HARBOR_NATIVE_ENV_MODES


def _load_verifier_template():
    template_path = Path(__file__).resolve().parents[1] / "src/skillevaluator/tier3/harbor/templates/eval.py"
    spec = importlib.util.spec_from_file_location("skillevaluator_public_verifier_template", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_eval_exposes_only_harbor_native_environments() -> None:
    result = CliRunner().invoke(cli, ["evaluate", "--help"])

    assert result.exit_code == 0
    assert "docker" in result.output
    assert "e2b" in result.output
    assert "modal" in result.output
    assert "harbor-environment" not in result.output
    assert "k8s-sandbox" not in result.output
    assert "local" not in result.output
    assert "base-image-mode" not in result.output
    assert "--agent-runtime-preflight" in result.output


def test_public_config_accepts_runtime_controls(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(
        "schema_version: 1\n"
        "harbor:\n"
        "  base_image_mode: rebuild\n"
        "  n_attempts: 3\n"
        "  stop_on_pass: true\n"
        "  agent_runtime_preflight: false\n",
        encoding="utf-8",
    )

    config, _ = load_evals_config(tmp_path)

    assert config["harbor"]["base_image_mode"] == "rebuild"
    assert config["harbor"]["stop_on_pass"] is True
    assert config["harbor"]["agent_runtime_preflight"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [("base_image_mode", "sometimes"), ("stop_on_pass", "'yes'"), ("agent_runtime_preflight", "1")],
)
def test_public_config_validates_runtime_control_values(tmp_path: Path, key: str, value: str) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(f"schema_version: 1\nharbor:\n  {key}: {value}\n", encoding="utf-8")

    with pytest.raises(EvalsConfigError, match=rf"harbor\.{key}"):
        load_evals_config(tmp_path)


def test_public_config_still_rejects_sandbox_policy(tmp_path: Path) -> None:
    """The public engine has no consumer for a config-level sandbox policy."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  sandbox:\n    template: harbor-eval\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalsConfigError, match="unknown harbor key"):
        load_evals_config(tmp_path)


def test_native_environment_is_forwarded_to_harbor() -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="public-env-test",
        env_mode="e2b",
    )

    assert command[1] == "run"
    assert "--agent-import-path" not in command
    assert "--environment-import-path" not in command
    assert "-a" not in command
    assert command.count("--agent") == 1
    assert command[command.index("--agent") + 1] == "codex"
    assert command.count("--env") == 1
    assert command[command.index("--env") + 1] == "e2b"


@pytest.mark.parametrize("timeout_multiplier", [float("nan"), float("inf"), float("-inf")])
def test_harbor_command_rejects_nonfinite_timeout_multiplier(timeout_multiplier: float) -> None:
    with pytest.raises(ValueError, match="timeout_multiplier must be a finite number greater than 0"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="nonfinite-timeout",
            env_mode="docker",
            timeout_multiplier=timeout_multiplier,
        )


def test_harbor_command_rejects_overflowing_timeout_multiplier() -> None:
    with pytest.raises(ValueError, match="timeout_multiplier must be a finite number greater than 0"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="overflowing-timeout",
            env_mode="docker",
            timeout_multiplier=10**1000,
        )


def test_harbor_command_rejects_finite_multiplier_that_overflows_default_timeouts() -> None:
    with pytest.raises(ValueError, match="must yield finite Harbor timeouts"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="finite-overflowing-timeout",
            env_mode="docker",
            timeout_multiplier=1e308,
        )


def test_native_environment_kwargs_round_trip_through_real_harbor_parser() -> None:
    from harbor.cli.utils import parse_kwargs

    expected = {
        "region": "us-west-2",
        "security_group_ids": ["sg-123", "sg-456"],
        "use_public_ip": False,
        "root_volume_size_gb": 80,
    }
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="native-environment-kwargs",
        env_mode="ec2",
        environment_kwargs=expected,
    )

    encoded = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    assert parse_kwargs(encoded) == expected


@pytest.mark.parametrize("env_mode", sorted(HARBOR_NATIVE_ENV_MODES - {"docker"}))
def test_native_environment_kwargs_reject_unknown_harbor_022_names(env_mode: str) -> None:
    with pytest.raises(ValueError, match=rf"Harbor 0\.22\.0 environment '{env_mode}'.*totally_ignored"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="unknown-environment-kwarg",
            env_mode=env_mode,
            environment_kwargs={"totally_ignored": True},
        )


@pytest.mark.parametrize(
    ("env_mode", "name", "value"),
    [
        ("daytona", "connection_pool_maxsize", 32),
        ("modal", "modal_vm_runtime", True),
        ("novita", "dind_dockerd_start_cmd", "dockerd-entrypoint.sh dockerd"),
    ],
)
def test_native_environment_hidden_harbor_022_kwargs_remain_usable(
    env_mode: str,
    name: str,
    value: object,
) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="hidden-environment-kwarg",
        env_mode=env_mode,
        environment_kwargs={name: value},
    )

    assert command[command.index("--ek") + 1].startswith(f"{name}=")


def test_native_environment_kwargs_resolve_real_harbor_ec2_constructor(tmp_path: Path) -> None:
    from harbor.cli.utils import parse_kwargs
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
    from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    expected = {
        "region": "us-west-2",
        "launch_mode": "attach",
        "instance_id": "i-123",
    }
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="native-environment-constructor",
        env_mode="ec2",
        environment_kwargs=expected,
    )
    encoded = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    environment = EnvironmentFactory.create_environment_from_config(
        TrialEnvironmentConfig(type=EnvironmentType.EC2, kwargs=parse_kwargs(encoded)),
        environment_dir=environment_dir,
        environment_name="native-environment-constructor",
        session_id="test-session",
        trial_paths=TrialPaths(trial_dir),
        task_env_config=TaskEnvironmentConfig(),
    )

    assert type(environment).__name__ == "EC2Environment"
    assert environment.region == "us-west-2"
    assert environment.launch_mode == "attach"
    assert environment.instance_id == "i-123"


def test_ack_operator_kwargs_allow_safe_registry_and_scheduling_references() -> None:
    from harbor.cli.utils import parse_kwargs

    expected = {
        "namespace": "skill-evals",
        "image_pull_secret": "registry-credentials",
        "node_selector": {"pool": "sandbox"},
        "tolerations": [{"key": "sandbox", "operator": "Exists"}],
    }
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="ack-operator-kwargs",
        env_mode="ack",
        environment_kwargs=expected,
    )

    encoded = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    assert parse_kwargs(encoded) == expected


@pytest.mark.parametrize(
    ("env_mode", "environment_kwargs", "error"),
    [
        ("local", {"region": "us-west-2"}, "not supported for SkillEvaluator local mode"),
        ("local", {"totally_ignored": True}, "not supported for SkillEvaluator local mode"),
        ("docker", {"region": "us-west-2"}, "not supported for SkillEvaluator Docker mode"),
        ("docker", {"totally_ignored": True}, "not supported for SkillEvaluator Docker mode"),
        ("ec2", {"override_cpus": 999}, "reserved for Harbor runtime policy"),
        ("ec2", {"extra_docker_compose": ["escape.yml"]}, "reserved for Harbor runtime policy"),
        ("ec2", {"network_policy": {"network_mode": "public"}}, "reserved for Harbor runtime policy"),
        ("ack", {"pod_overrides": {"spec": {"hostNetwork": True}}}, "reserved for Harbor runtime policy"),
        ("ack", {"pod_privileged": True}, "reserved for Harbor runtime policy"),
        ("ack", {"extra_volumes": [{"hostPath": {"path": "/"}}]}, "reserved for Harbor runtime policy"),
    ],
)
def test_environment_kwargs_cannot_override_sandbox_or_runtime_policy(
    env_mode: str,
    environment_kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="untrusted-environment-kwargs",
            env_mode=env_mode,
            environment_kwargs=environment_kwargs,
        )


@pytest.mark.parametrize(
    ("env_mode", "name", "value"),
    [
        ("ack", "build_job_namespace", "privileged-builds"),
        ("ack", "buildkit_address", "tcp://buildkit.internal:1234"),
        ("ack", "dind_image", "untrusted/dind:latest"),
        ("ack", "memory_limit_multiplier", 0),
        ("ack", "pod_annotations", {"inject-sidecar": "enabled"}),
        ("ack", "pod_labels", {"network-policy": "bypass"}),
        ("ack", "sandbox_env_vars", {"LD_PRELOAD": "/escape.so"}),
        ("ack", "service_account", "cluster-admin"),
        ("ack", "use_buildkit", True),
        ("blaxel", "dind_extra_args", {"host": "tcp://0.0.0.0:2375"}),
        ("cua-cloud", "claim_spec", {"serviceAccountName": "cluster-admin"}),
        ("daytona", "network_block_all", False),
        ("ec2", "iam_instance_profile", "administrator"),
        ("ec2", "strict_host_key_checking", "no"),
        ("gke", "memory_limit_multiplier", 0),
        ("modal", "volumes", {"/workspace": "shared"}),
        ("opensandbox", "volumes", [{"host_path": "/"}]),
        ("openshift", "service_account_name", "cluster-admin"),
        ("singularity", "singularity_no_mount", ""),
        ("use-computer", "resources", {"cpu": 128, "memory": 1048576}),
        ("vercel", "ports", [22, 2375]),
    ],
)
def test_backend_aliases_cannot_bypass_sandbox_runtime_policy(
    env_mode: str,
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"reserved for Harbor runtime policy: {name}"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="backend-policy-alias",
            env_mode=env_mode,
            environment_kwargs={name: value},
        )


@pytest.mark.parametrize(
    ("env_mode", "environment_kwargs"),
    [
        (
            "ack",
            {
                "namespace": "skill-evals",
                "use_sandbox_claim": True,
                "sandbox_image": "registry.example/harbor-sandbox:v1",
                # SandboxSet template metadata is an intentional operator
                # integration surface, unlike legacy direct pod overrides.
                "sandbox_labels": {"pool": "eval"},
                "sandbox_annotations": {"owner": "operator"},
                "skip_image_check": False,
            },
        ),
        (
            "ec2",
            {
                "region": "us-west-2",
                "ami_id": "ami-123",
                "instance_type": "m7i-flex.large",
                "root_volume_size_gb": 80,
                "bootstrap_docker": False,
            },
        ),
        (
            "gke",
            {
                "cluster_name": "cluster",
                "region": "us-central1",
                "namespace": "skill-evals",
                "registry_location": "us-central1",
                "registry_name": "skill-evals",
                "cloud_build_machine_type": "E2_HIGHCPU_32",
                "cloud_build_disk_size_gb": 500,
            },
        ),
        (
            "opensandbox",
            {
                "entrypoint": ["/bin/sh", "-lc", "sleep infinity"],
                "extensions": {"provider.example/feature": "enabled"},
                "sandbox_timeout_sec": 7200,
            },
        ),
    ],
)
def test_backend_operator_functionality_outside_policy_boundary_remains_usable(
    env_mode: str,
    environment_kwargs: dict[str, object],
) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="allowed-backend-options",
        env_mode=env_mode,
        environment_kwargs=environment_kwargs,
    )

    assert command.count("--ek") == len(environment_kwargs)


def test_skill_config_cannot_supply_environment_kwargs(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "config.yml").write_text(
        "schema_version: 1\n"
        "harbor:\n"
        "  environment_kwargs:\n"
        "    extra_docker_compose:\n"
        "      - /tmp/privileged-compose.yml\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalsConfigError, match=r"unknown harbor key.*environment_kwargs"):
        load_evals_config(tmp_path)


@pytest.mark.parametrize(
    ("env_mode", "environment_kwargs", "expected"),
    [
        ("ec2", {}, "region"),
        ("ec2", {"region": "us-west-2"}, "ami_id"),
        ("ec2", {"region": "us-west-2", "launch_mode": "attach"}, "instance_id"),
        (
            "gke",
            {"cluster_name": "cluster", "region": "us-west1", "namespace": "evals"},
            "registry_location, registry_name",
        ),
        ("ack", {}, "namespace"),
    ],
)
def test_native_environment_required_kwargs_fail_before_mutation(
    env_mode: str,
    environment_kwargs: dict[str, object],
    expected: str,
) -> None:
    assert expected in _environment_kwarg_prerequisite_errors(env_mode, environment_kwargs)[0]


@pytest.mark.parametrize(
    ("env_mode", "environment_kwargs", "expected"),
    [
        (
            "gke",
            {
                "cluster_name": 1,
                "region": [],
                "namespace": {},
                "registry_location": False,
                "registry_name": "valid",
            },
            "cluster_name, region, namespace, registry_location",
        ),
        ("ack", {"namespace": []}, "namespace"),
        ("ec2", {"region": False, "ami_id": "ami-123"}, "region"),
        ("ec2", {"region": "us-west-2", "ami_id": 123}, "ami_id"),
        ("ec2", {"region": "us-west-2", "launch_mode": [], "instance_id": "i-123"}, "launch_mode"),
        ("ec2", {"region": "us-west-2", "launch_mode": "attach", "instance_id": {}}, "instance_id"),
    ],
)
def test_native_environment_required_kwargs_reject_non_string_values_without_crashing(
    env_mode: str,
    environment_kwargs: dict[str, object],
    expected: str,
) -> None:
    errors = _environment_kwarg_prerequisite_errors(env_mode, environment_kwargs)

    assert len(errors) == 1
    assert expected in errors[0]


def test_native_environment_required_kwargs_accept_valid_ec2_attach_configuration() -> None:
    assert (
        _environment_kwarg_prerequisite_errors(
            "ec2",
            {"region": "us-west-2", "launch_mode": "attach", "instance_id": "i-123"},
        )
        == []
    )


@pytest.mark.parametrize(
    "ssh_key_path",
    ["", "/definitely/missing/harbor-ssh-key", "~definitely-no-such-user-issue79/key"],
)
def test_ec2_environment_kwargs_reject_nonexistent_ssh_key_path(ssh_key_path: str) -> None:
    errors = _environment_kwarg_prerequisite_errors(
        "ec2",
        {"region": "us-west-2", "ami_id": "ami-123", "ssh_key_path": ssh_key_path},
    )

    assert len(errors) == 1
    assert "ssh_key_path" in errors[0]
    assert "existing regular file" in errors[0]


def test_ec2_environment_kwargs_accept_existing_ssh_key_path(tmp_path: Path) -> None:
    ssh_key = tmp_path / "id_ed25519"
    ssh_key.write_text("placeholder", encoding="utf-8")

    assert (
        _environment_kwarg_prerequisite_errors(
            "ec2",
            {"region": "us-west-2", "ami_id": "ami-123", "ssh_key_path": str(ssh_key)},
        )
        == []
    )


@pytest.mark.parametrize("subnet_id", [None, ""])
def test_ec2_private_ephemeral_environment_requires_subnet(subnet_id: str | None) -> None:
    environment_kwargs: dict[str, object] = {
        "region": "us-west-2",
        "ami_id": "ami-123",
        "use_public_ip": False,
    }
    if subnet_id is not None:
        environment_kwargs["subnet_id"] = subnet_id

    errors = _environment_kwarg_prerequisite_errors("ec2", environment_kwargs)

    assert len(errors) == 1
    assert "use_public_ip=False requires" in errors[0]
    assert "subnet_id" in errors[0]


def test_ec2_private_ephemeral_environment_accepts_nonempty_subnet() -> None:
    assert (
        _environment_kwarg_prerequisite_errors(
            "ec2",
            {
                "region": "us-west-2",
                "ami_id": "ami-123",
                "use_public_ip": False,
                "subnet_id": "subnet-123",
            },
        )
        == []
    )


@pytest.mark.parametrize(
    ("environment_kwargs", "subprocess_env", "ready"),
    [
        ({}, {}, False),
        ({}, {"OPENSANDBOX_DOMAIN": "sandbox.example.test"}, True),
        ({"domain": "sandbox.example.test"}, {}, True),
        ({"domain": None}, {}, False),
        ({"domain": None}, {"OPENSANDBOX_DOMAIN": "sandbox.example.test"}, True),
        ({"domain": ""}, {"OPENSANDBOX_DOMAIN": "sandbox.example.test"}, False),
        ({"domain": "   "}, {"OPENSANDBOX_DOMAIN": "sandbox.example.test"}, False),
    ],
)
def test_opensandbox_domain_preflight_uses_effective_child_configuration(
    monkeypatch: pytest.MonkeyPatch,
    environment_kwargs: dict[str, object],
    subprocess_env: dict[str, str],
    ready: bool,
) -> None:
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(EnvironmentFactory, "run_preflight", lambda *_args, **_kwargs: None)

    errors = _check_prerequisites(
        env_mode="opensandbox",
        agents=[],
        environment_kwargs=environment_kwargs,
        subprocess_env=subprocess_env,
    )

    assert (errors == []) is ready
    if not ready:
        assert len(errors) == 1
        assert "domain" in errors[0]


def test_native_environment_required_kwargs_reject_whitespace_padded_ec2_launch_mode() -> None:
    errors = _environment_kwarg_prerequisite_errors(
        "ec2",
        {"region": "us-west-2", "launch_mode": " attach ", "instance_id": "i-123"},
    )

    assert errors == ["Harbor environment 'ec2' requires launch_mode to be 'ephemeral' or 'attach'"]


def test_native_environment_install_hints_use_real_harbor_022_extra_names() -> None:
    assert "harbor[gke]==0.22.0" in _environment_extra_install_hint("ack")
    assert "harbor[cloud]==0.22.0" not in _environment_extra_install_hint("ack")
    assert "harbor[cua]==0.22.0" in _environment_extra_install_hint("cua-cloud")
    assert "no Python extra" in _environment_extra_install_hint("openshift")


def test_judge_model_overrides_are_forwarded_only_as_harbor_verifier_env() -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="judge-model-test",
        env_mode="docker",
        verifier_env={
            "LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}",
            "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
        },
    )

    forwarded = [command[index + 1] for index, value in enumerate(command) if value == "--verifier-env"]
    assert forwarded == [
        "LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}",
        "SKILL_EVAL_JUDGE_MODEL=${SKILL_EVAL_JUDGE_MODEL}",
    ]
    assert "--agent-env" not in command


def test_nvidia_build_agent_import_selection_includes_local_bridge_agents() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="openai/gpt-oss-120b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/openai/gpt-oss-120b",
    )

    assert _nvidia_build_agent_import_path(provider, "codex", "docker") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"
    )
    assert _nvidia_build_agent_import_path(provider, "claude-code", "docker") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode"
    )
    assert _nvidia_build_agent_import_path(provider, "opencode", "docker") is None
    assert _nvidia_build_agent_import_path(provider, "codex", "local") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildCodex"
    )
    assert _nvidia_build_agent_import_path(provider, "claude-code", "local") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildClaudeCode"
    )
    assert _nvidia_build_agent_import_path(provider, "opencode", "local") is None


def test_docker_bridge_command_combines_custom_agent_and_secure_environment() -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="bridge-test",
        env_mode="docker",
        agent_import_path=import_path,
    )

    assert "--agent-import-path" not in command
    assert "--environment-import-path" not in command
    assert "-a" not in command
    assert command[command.index("--agent") + 1] == import_path
    assert command[command.index("--env") + 1] == (
        "skillevaluator.tier3.harbor.secure_docker_environment:SkillEvaluatorSecureDockerEnvironment"
    )


def test_local_bridge_command_uses_custom_agent_import_path() -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildCodex"

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="local-bridge-test",
        env_mode="local",
        agent_import_path=import_path,
    )

    assert "--agent-import-path" not in command
    assert "--environment-import-path" not in command
    assert "-a" not in command
    assert command[command.index("--agent") + 1] == import_path
    assert command[command.index("--env") + 1] == (
        "skillevaluator.tier3.harbor.local_environment:SkillEvaluatorLocalEnvironment"
    )


def test_custom_agent_import_path_is_rejected_for_native_cloud() -> None:
    with pytest.raises(ValueError, match="agent_import_path is supported only with --env docker or local"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="bridge-test",
            env_mode="e2b",
            agent_import_path="example:Agent",
        )


def test_evaluate_forwards_native_environment_without_legacy_sandbox_configuration(monkeypatch, tmp_path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    captured: dict = {}
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )

    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(tier3_commands, "run_harbor_eval", lambda **kwargs: captured.update(kwargs) or {"ok": True})

    tier3_commands.evaluate(
        skill,
        agents="codex",
        env_mode="e2b",
        skip_baseline=False,
        n_attempts=None,
        pass_threshold=None,
        n_concurrent=None,
        max_agents=None,
        model=None,
        agent_model=(),
        custom_dockerfile_mode=None,
        skill_workspace_mode=None,
        include_skills=(),
        copy_repo=False,
        grading_mode="default_plus_custom",
        results_dir=None,
        harbor_keep_jobs=False,
        timeout_multiplier=None,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert captured["env_mode"] == "e2b"
    assert captured["grading_mode"] == "default_plus_custom"
    assert "sandbox_config" not in captured


def test_evaluate_forwards_claude_alias_as_canonical_agent(monkeypatch, tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    captured: dict = {}
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-5",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(
        tier3_commands,
        "run_harbor_eval",
        lambda **kwargs: (
            captured.update(kwargs) or {"execution_status": "succeeded", "execution_errors": [], "agents": {}}
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(skill),
            "--agents",
            "claude",
            "--agent-model",
            "claude=anthropic/claude-sonnet-4-5",
            "--progress",
            "off",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["agents"] == ["claude-code"]
    assert captured["agent_models"] == {"claude-code": ["anthropic/claude-sonnet-4-5"]}


def test_evaluate_rejects_repeated_model_override_before_engine(monkeypatch, tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-5",
    )
    engine = Mock()
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "run_harbor_eval", engine)

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(skill),
            "--agents",
            "claude-code",
            "--agent-model",
            "claude-code=first",
            "--agent-model",
            "claude-code=second",
            "--progress",
            "off",
        ],
    )

    assert result.exit_code != 0
    assert "specify only one model for claude-code" in result.output
    engine.assert_not_called()


def test_doctor_rejects_alias_model_collision_consistently(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-5",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "claude",
            "--agent-model",
            "claude=first",
            "--agent-model",
            "claude-code=second",
        ],
    )

    assert result.exit_code == 1
    normalized = " ".join(result.output.split())
    assert "refer to the same agent" in normalized
    assert "specify only one model for claude-code" in normalized


def test_doctor_ack_preflight_uses_the_exact_resolved_bedrock_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-test",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-test",
        region="us-west-2",
    )
    captured: dict[str, object] = {}
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "eks-exec-auth-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "eks-exec-auth-secret")
    monkeypatch.setenv("KUBECONFIG", "/config/eks")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "ambient-parent-only")
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        tier3_commands,
        "_check_prerequisites",
        lambda **kwargs: captured.update(kwargs) or [],
    )

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "claude-code",
            "--env-mode",
            "ack",
            "--environment-kwarg",
            "namespace=skill-evals",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["env_mode"] == "ack"
    assert captured["environment_kwargs"] == {"namespace": "skill-evals"}
    child_env = captured["subprocess_env"]
    assert isinstance(child_env, dict)
    assert child_env["KUBECONFIG"] == "/config/eks"
    assert child_env["AWS_ACCESS_KEY_ID"] == "eks-exec-auth-key"
    assert child_env["AWS_SECRET_ACCESS_KEY"] == "eks-exec-auth-secret"
    assert child_env["AWS_REGION"] == "us-west-2"
    assert child_env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert "ALIBABA_CLOUD_ACCESS_KEY_ID" not in child_env


def test_generated_task_stages_public_provider_variables_for_the_verifier(tmp_path) -> None:
    _write_task_toml(
        tmp_path,
        {"id": "provider-test", "expected_skill": "demo"},
        has_skill=True,
        runtime_env={
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
        },
    )

    task = (tmp_path / "task.toml").read_text(encoding="utf-8")
    assert 'SKILL_EVAL_LLM_PROVIDER = "${SKILL_EVAL_LLM_PROVIDER}"' in task
    assert 'NVIDIA_API_KEY = "${NVIDIA_API_KEY}"' in task
    assert 'OPENAI_API_KEY = "${OPENAI_API_KEY}"' in task


def test_generated_task_keeps_evaluator_provider_variables_out_of_agent_environment(tmp_path) -> None:
    _write_task_toml(
        tmp_path,
        {"id": "provider-test", "expected_skill": "demo"},
        has_skill=True,
        runtime_env={"SERVICE_API_TOKEN": "${SERVICE_API_TOKEN}"},
        verifier_env={
            "LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}",
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
            "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
        },
    )

    task = tomllib.loads((tmp_path / "task.toml").read_text(encoding="utf-8"))
    assert task["verifier"]["env"] == {
        "NVIDIA_API_KEY": "${NVIDIA_API_KEY}",
        "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
    }
    assert task["environment"]["env"] == {
        **_EVALUATOR_MANAGED_RUNTIME_ENV,
        "SERVICE_API_TOKEN": "${SERVICE_API_TOKEN}",
    }


def test_nvidia_build_provider_mapping_does_not_supply_an_openai_agent_credential() -> None:
    environment = _provider_environment(
        ProviderConfig(
            provider="nv_build",
            model="meta/llama-3.1-8b-instruct",
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            litellm_model="openai/meta/llama-3.1-8b-instruct",
        )
    )

    assert environment["NVIDIA_API_KEY"] == "test-key"
    assert "OPENAI_API_KEY" not in environment
    assert "OPENAI_BASE_URL" not in environment


def test_doctor_accepts_nvidia_build_codex_without_openai_runtime_credential(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = CliRunner().invoke(cli, ["doctor", "--agents", "codex"])

    assert result.exit_code == 0
    assert "Codex runtime credential" in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_doctor_nvidia_build_codex_ignores_incomplete_openai_runtime_credential(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "openai-runtime-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = CliRunner().invoke(cli, ["doctor", "--agents", "codex"])

    assert result.exit_code == 0
    assert "Codex runtime credential" in result.output
    assert "OPENAI_API_KEY + OPENAI_BASE_URL" not in result.output


def test_doctor_build_codex_ignores_native_pair_and_accepts_build_model(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "openai-runtime-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "codex",
            "--agent-model",
            "codex=nvidia/nemotron-3-super-120b-a12b",
        ],
    )

    assert result.exit_code == 0
    assert "Codex runtime credential" in result.output
    assert "pass" in result.output


def test_doctor_verify_models_warns_when_catalog_success_does_not_verify_credentials(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )
    probe = Mock(
        return_value=ModelProbeResult(
            True,
            "nv_build",
            "meta/llama-3.1-8b-instruct",
            "model is available",
        )
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", "opencode", "--env-mode", "docker", "--verify-models"],
    )

    assert result.exit_code == 0
    assert "warn" in result.output
    assert "does not verify runtime credentials" in " ".join(result.output.split()).lower()
    probe.assert_called_once()
    probed_provider = probe.call_args.args[0]
    assert probed_provider.provider == "nv_build"
    assert probed_provider.model == "meta/llama-3.1-8b-instruct"


@pytest.mark.parametrize(
    ("failure_kind", "http_status", "expected_status", "expected_exit_code"),
    [
        (ModelCatalogFailureKind.AUTHORIZATION, 403, "warn", 0),
        (ModelCatalogFailureKind.AUTHENTICATION, 401, "fail", 1),
    ],
)
def test_doctor_verify_models_uses_probe_disposition_for_catalog_failures(
    monkeypatch,
    failure_kind: ModelCatalogFailureKind,
    http_status: int,
    expected_status: str,
    expected_exit_code: int,
) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider="openai",
        model="gpt-test",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-test",
    )
    probe = Mock(
        return_value=ModelProbeResult(
            False,
            "openai",
            "gpt-test",
            f"model catalog returned HTTP {http_status}",
            failure_kind=failure_kind,
            http_status=http_status,
        )
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", "codex", "--env-mode", "docker", "--verify-models"],
    )

    assert result.exit_code == expected_exit_code
    assert expected_status in result.output


@pytest.mark.parametrize(
    ("provider_name", "base_url", "agent"),
    [
        ("openai", "https://gateway.example/v1", "codex"),
        ("anthropic", "https://gateway.example/v1", "claude-code"),
    ],
)
def test_doctor_verify_models_warns_for_custom_endpoint_catalog_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    base_url: str,
    agent: str,
) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider=provider_name,
        model="model-test",
        api_key="provider-key",
        base_url=base_url,
        litellm_model=f"{provider_name}/model-test",
    )
    probe = Mock(
        return_value=ModelProbeResult(
            False,
            provider_name,
            "model-test",
            "model catalog returned HTTP 401",
            failure_kind=ModelCatalogFailureKind.AUTHENTICATION,
            http_status=401,
        )
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", agent, "--env-mode", "docker", "--verify-models"],
    )

    assert result.exit_code == 0
    assert "warn" in result.output
    assert "HTTP 401" in result.output


def test_doctor_reports_missing_independent_cross_provider_credential(monkeypatch) -> None:
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agents", "claude-code", "--verify-models"],
        terminal_width=240,
    )

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_nvidia_build_docker_codex_uses_the_compatibility_bridge() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["codex"], {}, env_mode="docker") == []


def test_nvidia_build_rejects_agents_without_a_credential_contract() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="openai/gpt-oss-120b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/openai/gpt-oss-120b",
    )

    errors = _validate_agent_provider_credentials(provider, ["cursor-cli"], {})

    assert errors and "does not support live agent" in errors[0]


def test_nvidia_build_local_codex_uses_the_compatibility_bridge() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["codex"], {}, env_mode="local") == []


def test_nvidia_build_local_claude_uses_the_compatibility_bridge() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["claude-code"], {}, env_mode="local") == []


@pytest.mark.parametrize("agent", ["opencode", "codex", "claude-code"])
def test_nvidia_build_local_agents_require_network_access(
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="nvidia/nemotron-3-nano-30b-a3b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/nvidia/nemotron-3-nano-30b-a3b",
    )
    monkeypatch.setenv("SKILLEVALUATOR_LOCAL_ALLOW_NET", "0")

    errors = _validate_agent_provider_credentials(provider, [agent], {}, env_mode="local")

    assert errors and "network" in errors[0].lower()
    assert "SKILLEVALUATOR_LOCAL_ALLOW_NET" in errors[0]


def test_nvidia_build_claude_accepts_explicit_anthropic_credential_and_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert (
        _validate_agent_provider_credentials(
            provider,
            ["claude-code"],
            {"ANTHROPIC_API_KEY": "anthropic-key"},
            {"claude-code": "CLI"},
            env_mode="local",
        )
        == []
    )


def test_nvidia_build_opencode_default_model_is_prefixed_for_local_runtime() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _model_for_agent(
        "opencode",
        cli_model=None,
        config_agents={},
        provider=provider,
    ) == ("nvidia/meta/llama-3.1-8b-instruct", "public provider default")


@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [
        ("openai", "openai/test-model"),
        ("openai-compatible", "openai/test-model"),
        ("anthropic", "anthropic/test-model"),
    ],
)
def test_opencode_default_model_is_provider_qualified(provider_name: str, expected: str) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{provider_name}/test-model",
    )

    assert _model_for_agent("opencode", cli_model=None, config_agents={}, provider=provider) == (
        expected,
        "public provider default",
    )


@pytest.mark.parametrize(
    ("provider_name", "raw_model", "expected"),
    [
        ("nv_build", "nvidia/llama-test", "nvidia/nvidia/llama-test"),
        ("openai", "openai/vendor/model", "openai/openai/vendor/model"),
        ("anthropic", "anthropic/vendor/model", "anthropic/anthropic/vendor/model"),
    ],
)
def test_opencode_provider_default_preserves_raw_ids_that_begin_with_runtime_namespace(
    provider_name: str,
    raw_model: str,
    expected: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model=raw_model,
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model=f"openai/{raw_model}",
    )

    assert _model_for_agent("opencode", cli_model=None, config_agents={}, provider=provider) == (
        expected,
        "public provider default",
    )


@pytest.mark.parametrize(
    ("provider_name", "cli_model", "config_agents", "expected", "source"),
    [
        ("nv_build", "meta/llama-3.1-8b-instruct", {}, "meta/llama-3.1-8b-instruct", "CLI"),
        ("nv_build", "openai/gpt-oss-120b", {}, "openai/gpt-oss-120b", "CLI"),
        ("nv_build", "nvidia/openai/gpt-oss-120b", {}, "nvidia/openai/gpt-oss-120b", "CLI"),
        ("openai", "gpt-4.1-mini", {}, "gpt-4.1-mini", "CLI"),
        (
            "anthropic",
            None,
            {"opencode": {"model": "claude-sonnet-test"}},
            "claude-sonnet-test",
            "evals/config.yml",
        ),
        (
            "openai-compatible",
            None,
            {"opencode": {"model": "vendor/custom-model"}},
            "vendor/custom-model",
            "evals/config.yml",
        ),
    ],
)
def test_opencode_explicit_model_is_preserved_exactly(
    provider_name: str,
    cli_model: str | None,
    config_agents: dict,
    expected: str,
    source: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model="provider-default",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{provider_name}/provider-default",
    )

    assert _model_for_agent(
        "opencode",
        cli_model=cli_model,
        config_agents=config_agents,
        provider=provider,
    ) == (expected, source)


def test_doctor_explicit_opencode_runtime_model_probes_raw_catalog_id(monkeypatch) -> None:
    from skillevaluator.tier3.harbor import runtime_preflight

    provider = ProviderConfig(
        provider="nv_build",
        model="provider-default",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/provider-default",
    )
    probe = Mock(return_value=ModelProbeResult(True, "nv_build", "openai/gpt-oss-120b", "available"))
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    result = CliRunner().invoke(
        cli,
        [
            "doctor",
            "--agents",
            "opencode",
            "--env-mode",
            "docker",
            "--verify-models",
            "--agent-model",
            "opencode=nvidia/openai/gpt-oss-120b",
        ],
    )

    assert result.exit_code == 0
    probed_provider = probe.call_args.args[0]
    assert probed_provider.model == "openai/gpt-oss-120b"
    assert probed_provider.litellm_model == "openai/openai/gpt-oss-120b"


def test_nvidia_build_docker_opencode_uses_selected_provider_key() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    errors = _validate_agent_provider_credentials(provider, ["opencode"], {}, env_mode="docker")

    assert errors == []


def test_nvidia_build_local_opencode_uses_evaluator_provider_mapping() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["opencode"], {}, env_mode="local") == []


def test_nvidia_build_local_codex_uses_the_provider_default_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert _validate_agent_provider_credentials(provider, ["codex"], {}, env_mode="local") == []


def test_nvidia_build_codex_accepts_explicit_independent_credential_and_model() -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="meta/llama-3.1-8b-instruct",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/meta/llama-3.1-8b-instruct",
    )

    assert (
        _validate_agent_provider_credentials(
            provider,
            ["codex"],
            {"OPENAI_API_KEY": "openai-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
            {"codex": "CLI"},
            env_mode="local",
        )
        == []
    )


def test_generated_verifier_rejects_non_http_provider_base_urls(monkeypatch) -> None:
    verifier = _load_verifier_template()
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "file:///etc/passwd")

    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        verifier._resolve_url("openai")
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        verifier._anthropic_url()


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://gateway.example", "https://gateway.example/v1/messages"),
        ("https://gateway.example/", "https://gateway.example/v1/messages"),
        ("https://gateway.example/v1", "https://gateway.example/v1/messages"),
        ("https://gateway.example/team", "https://gateway.example/team/v1/messages"),
        ("https://gateway.example/team/v1", "https://gateway.example/team/v1/messages"),
        ("https://gateway.example:8443/team/v1/", "https://gateway.example:8443/team/v1/messages"),
        ("http://gateway.internal:8080/v1", "http://gateway.internal:8080/v1/messages"),
        ("http://anthropic_proxy:8000/v1", "http://anthropic_proxy:8000/v1/messages"),
        ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1/messages"),
        ("http://[::1]:8080/team/v1", "http://[::1]:8080/team/v1/messages"),
        (
            "http://[fe80::1%25eth0]:8080/team/v1",
            "http://[fe80::1%25eth0]:8080/team/v1/messages",
        ),
        ("https://xn--bcher-kva.example/v1", "https://xn--bcher-kva.example/v1/messages"),
        ("https://bücher.example/v1", "https://xn--bcher-kva.example/v1/messages"),
        ("https://faß.de/v1", "https://xn--fa-hia.de/v1/messages"),
        ("https://οδός.example/v1", "https://xn--pxavk3b.example/v1/messages"),
        (
            "https://bücher.example.:8443/bücher/v1",
            "https://xn--bcher-kva.example.:8443/b%C3%BCcher/v1/messages",
        ),
        ("https://gateway.example/caf%C3%A9/v1", "https://gateway.example/caf%C3%A9/v1/messages"),
        ("https://gateway.example/caf%c3%a9/v1", "https://gateway.example/caf%C3%A9/v1/messages"),
        ("https://gateway.example/opaque%ff/v1", "https://gateway.example/opaque%FF/v1/messages"),
        (
            "https://gateway.example/tenant%25west/v1",
            "https://gateway.example/tenant%25west/v1/messages",
        ),
        ("https://gateway.example/100%25/v1", "https://gateway.example/100%25/v1/messages"),
        ("https://gateway.example/team/%76%31", "https://gateway.example/team/v1/messages"),
        ("https://gateway.example/team/v1///", "https://gateway.example/team/v1/messages"),
        (
            "https://gateway.example/teams;v=1/@me+you/v1",
            "https://gateway.example/teams;v=1/@me+you/v1/messages",
        ),
    ],
)
def test_generated_verifier_builds_exactly_one_anthropic_native_messages_path(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    expected_url: str,
) -> None:
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)

    assert verifier._anthropic_url() == expected_url


@pytest.mark.parametrize(
    ("variable", "base_url"),
    [
        ("SKILL_EVAL_LLM_BASE_URL", "https://gateway.example/v1/messages"),
        ("ANTHROPIC_BASE_URL", "https://url-user:url-secret@gateway.example/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/v1?token=url-secret"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/v1#token=url-secret"),
        ("ANTHROPIC_BASE_URL", "https://:443/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example:not-a-port/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example:/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example\\team\\v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/v1\n"),
        ("ANTHROPIC_BASE_URL", "https:///team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway example/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway%2eexample/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example|evil/v1"),
        ("ANTHROPIC_BASE_URL", "https://999.1.1.1/v1"),
        ("ANTHROPIC_BASE_URL", "https://[v1.not-ipv6]/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%76%31/%6dessages"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%2Fv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%5cv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%0av1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%2"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%GG"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/../team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%2e%2e/team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%2576%2531/%256dessages"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/%252e%252e/team/v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%252Fv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%255Cv1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%250Av1"),
        (
            "ANTHROPIC_BASE_URL",
            "https://gateway.example/%25%37%36%25%33%31/%25%36%64essages",
        ),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team%25%32%46v1"),
        ("ANTHROPIC_BASE_URL", "https://gateway.example/team//v1"),
        ("ANTHROPIC_BASE_URL", "https://☃.example/v1"),
    ],
)
def test_generated_verifier_rejects_unsafe_anthropic_api_roots_without_echoing_credentials(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    base_url: str,
) -> None:
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv(variable, base_url)

    with pytest.raises(ValueError) as exc_info:
        verifier._anthropic_url()

    message = str(exc_info.value)
    assert variable in message
    assert base_url not in message
    assert "url-user" not in message
    assert "url-secret" not in message


def test_generated_verifier_uses_the_official_anthropic_messages_url_when_base_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    assert verifier._anthropic_url() == "https://api.anthropic.com/v1/messages"


@pytest.mark.parametrize(
    ("configured_path", "expected_path"),
    [
        ("bücher/v1", "/b%C3%BCcher/v1/messages"),
        ("opaque%FF/v1", "/opaque%FF/v1/messages"),
        ("tenant%25west/v1", "/tenant%25west/v1/messages"),
        ("100%25/v1", "/100%25/v1/messages"),
        ("team/v1///", "/team/v1/messages"),
    ],
)
def test_anthropic_sdk_and_bundled_verifier_use_the_same_ascii_path(
    monkeypatch: pytest.MonkeyPatch,
    configured_path: str,
    expected_path: str,
) -> None:
    from anthropic import Anthropic

    requested_paths: list[str] = []

    class RecordingHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requested_paths.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return None

    verifier = _load_verifier_template()
    with ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}/{configured_path}",
            }
        )
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url or "")
        try:
            with Anthropic(
                api_key="test-anthropic-key",
                base_url=provider.base_url,
                max_retries=0,
                timeout=5.0,
            ) as client:
                sdk_response = client.messages.create(
                    model="claude-test",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "hello"}],
                )
            verifier_response, verifier_error = verifier._call_anthropic("hello", "claude-test", 16, 0.0)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert sdk_response.content[0].text == "ok"
    assert verifier_response == "ok"
    assert verifier_error is None
    assert requested_paths == [expected_path, expected_path]


def test_anthropic_sdk_and_bundled_verifier_prepare_the_same_scoped_ipv6_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthropic import Anthropic

    sdk_urls: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        sdk_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    provider = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "ANTHROPIC_BASE_URL": "http://[fe80::1%25eth0]:8080/bücher/v1",
        }
    )
    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url or "")

    with (
        httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client,
        Anthropic(
            api_key="test-anthropic-key",
            base_url=provider.base_url,
            http_client=http_client,
        ) as client,
    ):
        client.messages.create(
            model="claude-test",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )

    verifier_url = verifier._anthropic_url()
    verifier_request = urllib.request.Request(verifier_url)
    expected_url = "http://[fe80::1%25eth0]:8080/b%C3%BCcher/v1/messages"
    assert sdk_urls == [expected_url]
    assert verifier_request.full_url == expected_url
    assert verifier_request.selector == "/b%C3%BCcher/v1/messages"


@pytest.mark.parametrize(
    ("unicode_host", "ascii_host"),
    [
        ("faß.de", "xn--fa-hia.de"),
        ("οδός.example", "xn--pxavk3b.example"),
    ],
)
def test_anthropic_idna_matches_httpx_sdk_and_bundled_verifier(
    monkeypatch: pytest.MonkeyPatch,
    unicode_host: str,
    ascii_host: str,
) -> None:
    from anthropic import Anthropic

    sdk_urls: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        sdk_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    configured_url = f"https://{unicode_host}/team/v1"
    provider = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "ANTHROPIC_BASE_URL": configured_url,
        }
    )
    expected_url = f"https://{ascii_host}/team/v1/messages"
    assert str(httpx.URL(configured_url)) == f"https://{ascii_host}/team/v1"

    verifier = _load_verifier_template()
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url or "")
    with (
        httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client,
        Anthropic(
            api_key="test-anthropic-key",
            base_url=provider.base_url,
            http_client=http_client,
        ) as client,
    ):
        client.messages.create(
            model="claude-test",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )

    assert sdk_urls == [expected_url]
    assert verifier._anthropic_url() == expected_url
