# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security policy regressions for user-supplied Harbor Docker Compose files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks, stage_native_harbor_tasks


def _write_skill(tmp_path: Path, compose: dict[str, Any]) -> Path:
    skill = tmp_path / "skill"
    evals_dir = skill / "evals"
    environment_dir = evals_dir / "environment"
    environment_dir.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    (evals_dir / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Run the test", "expected_skill": "skill"}]),
        encoding="utf-8",
    )
    (environment_dir / "docker-compose.yaml").write_text(yaml.safe_dump(compose), encoding="utf-8")
    return skill


def _generate(
    tmp_path: Path,
    compose: dict[str, Any],
    *,
    runtime_env: dict[str, str] | None = None,
) -> Path:
    task = generate_harbor_tasks(
        _write_skill(tmp_path, compose),
        tmp_path / "tasks",
        runtime_env=runtime_env,
    )[0]
    return task / "environment" / "docker-compose.yaml"


def _stage_native(tmp_path: Path, compose: dict[str, Any]) -> Path:
    skill = tmp_path / "native-skill"
    task = skill / "evals" / "harbor" / "case-001"
    environment = task / "environment"
    tests = task / "tests"
    environment.mkdir(parents=True)
    tests.mkdir()
    (skill / "SKILL.md").write_text("# Native test skill\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[environment]\n',
        encoding="utf-8",
    )
    (tests / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(yaml.safe_dump(compose), encoding="utf-8")

    staged = stage_native_harbor_tasks(
        skill,
        tmp_path / "native-tasks",
        grading_mode="custom_only",
    )[0]
    return staged / "environment" / "docker-compose.yaml"


_ESCAPE_SETTINGS = (
    ("privileged", True),
    ("devices", ["/dev/kvm:/dev/kvm"]),
    ("device_cgroup_rules", ["c 10:232 rwm"]),
    ("gpus", "all"),
    ("runtime", "nvidia"),
    ("use_api_socket", True),
    ("network_mode", "host"),
    ("pid", "host"),
    ("ipc", "host"),
    ("uts", "host"),
    ("userns_mode", "host"),
    ("cgroup", "host"),
    ("cgroup_parent", "/host.slice"),
    ("security_opt", ["seccomp=unconfined"]),
    ("cap_add", ["SYS_ADMIN"]),
    ("deploy", {"resources": {"reservations": {"devices": [{"capabilities": ["gpu"]}]}}}),
    ("post_start", [{"command": "id", "privileged": True}]),
    ("pre_stop", [{"command": "id", "privileged": True}]),
    ("container_name", "host-global-name"),
    ("develop", {"watch": [{"action": "sync", "path": "..", "target": "/workspace"}]}),
    ("extra_hosts", ["host.docker.internal:host-gateway"]),
    ("logging", {"driver": "syslog", "options": {"syslog-address": "unixgram:///dev/log"}}),
    ("driver_opts", {"com.docker.network.bridge.host_binding_ipv4": "127.0.0.1"}),
    ("models", ["host-model"]),
)


@pytest.mark.parametrize("service_name", ("main", "database"))
@pytest.mark.parametrize(("field", "value"), _ESCAPE_SETTINGS)
def test_custom_compose_rejects_host_escape_settings_on_every_service(
    tmp_path: Path,
    service_name: str,
    field: str,
    value: object,
) -> None:
    compose = {"services": {service_name: {"image": "postgres:16", field: value}}}

    with pytest.raises(ValueError, match=rf"service '{service_name}'.*{field}"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize("service_name", ("main", "database"))
@pytest.mark.parametrize(
    "volume",
    (
        "/:/host-root:ro",
        "./docker.sock:/var/run/docker.sock",
        "../secrets:/run/secrets:ro",
        "${HOST_DATA_DIR}:/data",
        r"C:\Users:/host-users:ro",
        {"type": "bind", "source": "/var/run/docker.sock", "target": "/var/run/docker.sock"},
    ),
)
def test_custom_compose_rejects_short_and_long_host_bind_mounts(
    tmp_path: Path,
    service_name: str,
    volume: object,
) -> None:
    compose = {"services": {service_name: {"image": "postgres:16", "volumes": [volume]}}}

    with pytest.raises(ValueError, match=rf"service '{service_name}'.*host bind mount"):
        _generate(tmp_path, compose)


def test_custom_compose_rejects_interpolated_short_volume_spec(tmp_path: Path) -> None:
    compose = {
        "services": {
            "database": {
                "image": "postgres:16",
                "volumes": ["${MOUNT_SPEC}"],
            }
        }
    }

    with pytest.raises(ValueError, match=r"service 'database'.*host bind mount"):
        _generate(tmp_path, compose, runtime_env={"MOUNT_SPEC": "${MOUNT_SPEC}"})


@pytest.mark.parametrize(
    "field",
    ("build", "image", "command", "entrypoint", "environment", "env_file", "ports", "working_dir"),
)
def test_custom_compose_rejects_main_service_runtime_overrides(tmp_path: Path, field: str) -> None:
    values: dict[str, object] = {
        "build": ".",
        "image": "alpine:3",
        "command": ["sleep", "infinity"],
        "entrypoint": ["sh"],
        "environment": {"HOST_SECRET": "${HOST_SECRET}"},
        "env_file": "../../.env",
        "ports": ["8080:80"],
        "working_dir": "/host",
    }

    with pytest.raises(ValueError, match=rf"service 'main'.*{field}"):
        _generate(tmp_path, {"services": {"main": {field: values[field]}}})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("driver", "local"),
        ("driver_opts", {"type": "none", "o": "bind", "device": "/"}),
        ("external", True),
        ("name", "shared-host-volume"),
    ),
)
def test_custom_compose_rejects_non_project_scoped_named_volumes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    compose = {
        "services": {"database": {"image": "postgres:16", "volumes": ["data:/var/lib/postgresql/data"]}},
        "volumes": {"data": {field: value}},
    }

    with pytest.raises(ValueError, match=rf"volume 'data'.*{field}"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("external", True),
        ("name", "shared-host-network"),
        ("driver", "host"),
        ("driver", "macvlan"),
        ("driver_opts", {"com.docker.network.bridge.host_binding_ipv4": "0.0.0.0"}),
        ("ipam", {"driver": "host-plugin"}),
    ),
)
def test_custom_compose_rejects_non_project_scoped_networks(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    compose = {
        "services": {"database": {"image": "postgres:16", "networks": ["backend"]}},
        "networks": {"backend": {field: value}},
    }

    with pytest.raises(ValueError, match=rf"network 'backend'.*{field}"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("env_file", "../../.env"),
        ("volumes_from", ["container:host-container:ro"]),
        ("external_links", ["host-container:database"]),
        ("extends", {"file": "/tmp/host-compose.yaml", "service": "database"}),
        ("credential_spec", {"file": "/tmp/host-credential.json"}),
        ("label_file", "../../host-labels.env"),
        ("configs", ["host-config"]),
        ("secrets", ["host-secret"]),
    ),
)
def test_custom_compose_rejects_indirect_host_file_and_container_access(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    compose = {"services": {"database": {"image": "postgres:16", field: value}}}

    with pytest.raises(ValueError, match=rf"service 'database'.*{field}"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize(
    "build",
    (
        "/host/source",
        "../host-source",
        "https://example.com/source.git",
        "git@example.com:project/source.git",
        "builder@example.com:project/source.git",
        {"context": "${HOST_SOURCE}"},
        {"context": ".", "network": "host"},
        {"context": ".", "privileged": True},
        {"context": ".", "ssh": ["default"]},
        {"context": ".", "secrets": ["host-secret"]},
        {"context": ".", "entitlements": ["security.insecure"]},
        {"context": ".", "cache_from": ["type=local,src=/tmp/host-cache"]},
        {"context": ".", "cache_to": ["type=local,dest=/tmp/host-cache"]},
        {"context": ".", "extra_hosts": ["host.docker.internal=host-gateway"]},
        {"context": ".", "isolation": "process"},
        {"context": ".", "tags": ["host-image:overwrite"]},
    ),
)
def test_custom_compose_rejects_sidecar_build_host_access(tmp_path: Path, build: object) -> None:
    compose = {"services": {"builder": {"build": build}}}

    with pytest.raises(ValueError, match=r"service 'builder'.*build"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize(
    "build",
    (
        {"context": ".", "dockerfile": "/tmp/HostDockerfile"},
        {"context": ".", "dockerfile": "../HostDockerfile"},
        {"context": ".", "dockerfile": "https://example.com/HostDockerfile"},
        {"context": ".", "additional_contexts": ["host=/"]},
        {"context": ".", "additional_contexts": {"host": "../host-source"}},
        {"context": ".", "additional_contexts": {"host": "https://example.com/source.git"}},
        {"context": ".", "additional_contexts": {"host": "git@example.com:project/source.git"}},
    ),
)
def test_custom_compose_rejects_sidecar_build_files_and_contexts_outside_staging(
    tmp_path: Path,
    build: object,
) -> None:
    compose = {"services": {"builder": {"build": build}}}

    with pytest.raises(ValueError, match=r"service 'builder'.*build"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("include", ["/tmp/host-compose.yaml"]),
        ("configs", {"host-config": {"file": "/tmp/host-config"}}),
        ("secrets", {"host-secret": {"file": "/tmp/host-secret"}}),
        ("name", "host-global-project"),
        ("models", {"host-model": {"model": "host-managed-model"}}),
    ),
)
def test_custom_compose_rejects_top_level_host_resource_sources(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    compose = {"services": {"database": {"image": "postgres:16"}}, field: value}

    with pytest.raises(ValueError, match=rf"top-level.*{field}"):
        _generate(tmp_path, compose)


def test_custom_compose_rejects_interpolation_not_declared_as_runtime_env(tmp_path: Path) -> None:
    compose = {
        "services": {
            "database": {
                "image": "postgres:16",
                "environment": {"STOLEN_PROVIDER_KEY": "${OPENAI_API_KEY}"},
            }
        }
    }

    with pytest.raises(ValueError, match=r"OPENAI_API_KEY.*harbor.runtime_env"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize(
    "interpolation",
    (
        "${DECLARED_VALUE:-${OPENAI_API_KEY}}",
        "$$${OPENAI_API_KEY}",
    ),
)
def test_custom_compose_rejects_hidden_undeclared_interpolation(
    tmp_path: Path,
    interpolation: str,
) -> None:
    compose = {
        "services": {
            "database": {
                "image": "postgres:16",
                "environment": {"STOLEN_PROVIDER_KEY": interpolation},
            }
        }
    }

    with pytest.raises(ValueError, match=r"OPENAI_API_KEY.*harbor.runtime_env"):
        _generate(tmp_path, compose, runtime_env={"DECLARED_VALUE": ""})


def test_custom_compose_allows_even_dollar_literal(tmp_path: Path) -> None:
    compose = {
        "services": {
            "database": {
                "image": "postgres:16",
                "environment": {"LITERAL_TEMPLATE": "$${NOT_A_HOST_VARIABLE}"},
            }
        }
    }

    staged = yaml.safe_load(_generate(tmp_path, compose).read_text(encoding="utf-8"))

    assert staged["services"]["database"]["environment"] == compose["services"]["database"]["environment"]


@pytest.mark.parametrize(
    "environment",
    (
        ["OPENAI_API_KEY"],
        {"OPENAI_API_KEY": None},
    ),
)
def test_custom_compose_rejects_undeclared_bare_environment_passthrough(
    tmp_path: Path,
    environment: object,
) -> None:
    compose = {"services": {"database": {"image": "postgres:16", "environment": environment}}}

    with pytest.raises(ValueError, match=r"OPENAI_API_KEY.*harbor.runtime_env"):
        _generate(tmp_path, compose)


@pytest.mark.parametrize("args", (["OPENAI_API_KEY"], {"OPENAI_API_KEY": None}))
def test_custom_compose_rejects_undeclared_bare_build_arg_passthrough(tmp_path: Path, args: object) -> None:
    compose = {"services": {"builder": {"build": {"context": ".", "args": args}}}}

    with pytest.raises(ValueError, match=r"OPENAI_API_KEY.*harbor.runtime_env"):
        _generate(tmp_path, compose)


def test_native_harbor_compose_uses_the_same_escape_policy(tmp_path: Path) -> None:
    compose = {"services": {"database": {"image": "postgres:16", "privileged": True}}}

    with pytest.raises(ValueError, match=r"service 'database'.*privileged"):
        _stage_native(tmp_path, compose)


def test_custom_compose_preserves_literal_and_declared_runtime_environment(tmp_path: Path) -> None:
    compose = {
        "services": {
            "database": {
                "image": "postgres:16",
                "environment": {
                    "LITERAL_SETTING": "safe-value",
                    "DECLARED_TOKEN": "${SERVICE_TOKEN}",
                    "CONTAINER_EXPANSION": "$${INSIDE_CONTAINER}",
                },
            }
        }
    }

    staged = yaml.safe_load(
        _generate(tmp_path, compose, runtime_env={"SERVICE_TOKEN": "${SERVICE_TOKEN}"}).read_text(encoding="utf-8")
    )

    assert staged["services"]["database"]["environment"] == compose["services"]["database"]["environment"]


def test_custom_compose_preserves_declared_bare_environment_and_build_args(tmp_path: Path) -> None:
    compose = {
        "services": {
            "database": {
                "build": {"context": ".", "args": ["SERVICE_TOKEN", "LITERAL_ARG=value"]},
                "environment": ["SERVICE_TOKEN", "LITERAL_SETTING=safe-value"],
            }
        }
    }

    staged = yaml.safe_load(
        _generate(tmp_path, compose, runtime_env={"SERVICE_TOKEN": "${SERVICE_TOKEN}"}).read_text(encoding="utf-8")
    )

    assert staged["services"]["database"]["build"]["args"] == compose["services"]["database"]["build"]["args"]
    assert staged["services"]["database"]["environment"] == compose["services"]["database"]["environment"]


def test_custom_compose_preserves_safe_sidecar_features_and_strips_host_ports(tmp_path: Path) -> None:
    compose = {
        "services": {
            "main": {"depends_on": {"database": {"condition": "service_healthy"}}},
            "database": {
                "image": "postgres:16",
                "environment": {"POSTGRES_PASSWORD": "test-only"},
                "healthcheck": {"test": ["CMD-SHELL", "pg_isready -U postgres"]},
                "ports": ["5432:5432"],
                "expose": ["5432"],
                "networks": ["backend"],
                "volumes": [
                    "database-data:/var/lib/postgresql/data",
                    {"type": "volume", "source": "database-cache", "target": "/cache"},
                    {"type": "tmpfs", "target": "/tmp"},
                ],
            },
        },
        "volumes": {"database-data": {}, "database-cache": {}},
        "networks": {"backend": {"internal": True}},
    }

    staged = yaml.safe_load(_generate(tmp_path, compose).read_text(encoding="utf-8"))

    assert staged["services"]["main"] == compose["services"]["main"]
    assert "ports" not in staged["services"]["database"]
    assert staged["services"]["database"]["expose"] == ["5432"]
    assert staged["services"]["database"]["healthcheck"] == compose["services"]["database"]["healthcheck"]
    assert staged["services"]["database"]["volumes"] == compose["services"]["database"]["volumes"]
    assert staged["networks"] == compose["networks"]
