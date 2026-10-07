# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SkillEvaluatorGKEEnvironment pod isolation, exec readiness, retry, and redaction."""

from __future__ import annotations

import asyncio
import http.server
import os
import shutil
import stat
import subprocess
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.gke import GKEEnvironment
from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException

import skillevaluator.tier3.harbor.gke_environment as gke_env_mod
from skillevaluator.tier3.harbor.gke_environment import (
    SkillEvaluatorGKEEnvironment,
    _build_in_pod_metadata_isolation_probe_script,
    _build_metadata_blocking_network_policy,
    _network_policy_blocks_metadata,
    _redact_exec_stderr,
)


def _make_pod(name: str = "eval-pod", namespace: str = "skill-eval") -> k8s_client.V1Pod:
    """Create a minimal V1Pod fixture for _create_pod unit tests."""
    return k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        spec=k8s_client.V1PodSpec(containers=[k8s_client.V1Container(name="main", image="ubuntu:24.04")]),
    )


def _make_weak_metadata_network_policy(namespace: str = "skill-eval") -> k8s_client.V1NetworkPolicy:
    """Return a weakened V1NetworkPolicy missing the Standard GKE 169.254.169.252/32 exclusion."""
    policy = _build_metadata_blocking_network_policy(namespace)
    policy.spec.egress[0].to[0].ip_block._except = ["169.254.169.254/32"]
    return policy


@pytest.fixture
def local_metadata_http_server() -> Iterator[Callable[[int], str]]:
    """Run an ephemeral local HTTP server returning configurable HTTP status codes for probe script tests."""
    status_holder = {"code": 200}

    class _MetadataHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            code = status_holder["code"]
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"access_token":"fake-token"}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _MetadataHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def _url_for_status(code: int) -> str:
        status_holder["code"] = code
        return f"http://{host}:{port}/computeMetadata/v1/instance/service-accounts/default/token"

    try:
        yield _url_for_status
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def make_gke_env(monkeypatch: pytest.MonkeyPatch) -> Callable[..., SkillEvaluatorGKEEnvironment]:
    """Provide a factory for SkillEvaluatorGKEEnvironment wired to fake CoreV1Api and NetworkingV1Api."""
    monkeypatch.delenv("SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY", raising=False)
    monkeypatch.setattr(GKEEnvironment, "_api", property(lambda self: self._fake_api))

    def _factory(
        *,
        namespace: str = "skill-eval",
        allow_workload_identity: bool = False,
        autopilot: bool | str | None = None,
        compose_mode: bool = False,
        fake_api: object | None = None,
        networking_api: object | None = None,
    ) -> SkillEvaluatorGKEEnvironment:
        env = object.__new__(SkillEvaluatorGKEEnvironment)
        env.pod_name = "pod-under-test"
        env.namespace = namespace
        env._compose_mode = compose_mode
        kwargs: dict[str, str] = {}
        if allow_workload_identity:
            kwargs["allow_workload_identity"] = "1"
        if autopilot is not None:
            kwargs["autopilot"] = str(autopilot)
        env._kwargs = kwargs
        env._allow_workload_identity = allow_workload_identity
        env._autopilot = autopilot
        env._persistent_env = {}
        env.default_user = None
        env.task_env_config = SimpleNamespace(workdir=None, user=None)
        env._fake_api = fake_api
        if networking_api is not None:
            env._networking_api = networking_api
        return env

    return _factory


def test_gke_environment_create_pod_fails_closed_on_service_account_api_error(
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
) -> None:
    """Fail closed when ServiceAccount inspection raises a Kubernetes API error without Workload Identity opt-in."""

    def raise_forbidden(**_kw: object) -> None:
        raise RuntimeError("403 Forbidden: serviceaccounts 'default' is forbidden")

    env = make_gke_env(fake_api=SimpleNamespace(read_namespaced_service_account=raise_forbidden))

    with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
        asyncio.run(env._create_pod(_make_pod("pod-forbidden")))


@pytest.mark.parametrize(
    (
        "allow_workload_identity",
        "annotations",
        "has_networking_api",
        "expected_metadata_host",
        "expected_automount",
        "expect_error",
    ),
    [
        # 1. Unannotated KSA WITHOUT NetworkingV1Api / metadata NetworkPolicy fails closed (offline reproduction)
        (False, {}, False, "127.0.0.1:1", False, True),
        # 2. Unannotated KSA WITH verified metadata NetworkPolicy succeeds and labels pod
        (False, {}, True, "127.0.0.1:1", False, False),
        # 3. Annotated KSA (iam.gke.io/gcp-service-account) fails closed even with NetworkingV1Api
        (
            False,
            {"iam.gke.io/gcp-service-account": "eval-sa@my-proj.iam.gserviceaccount.com"},
            True,
            "127.0.0.1:1",
            False,
            True,
        ),
        # 4. Direct WIF email annotation (iam.gke.io/return-principal-id-as-email=true) fails closed
        (
            False,
            {"iam.gke.io/return-principal-id-as-email": "true"},
            True,
            "127.0.0.1:1",
            False,
            True,
        ),
        # 5. Explicit opt-in allows bound KSA without metadata NetworkPolicy
        (
            True,
            {"iam.gke.io/gcp-service-account": "eval-sa@my-proj.iam.gserviceaccount.com"},
            False,
            None,
            None,
            False,
        ),
    ],
)
def test_gke_environment_create_pod_blocks_metadata_host_and_bound_ksa_when_unprivileged(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    allow_workload_identity: bool,
    annotations: dict[str, str],
    has_networking_api: bool,
    expected_metadata_host: str | None,
    expected_automount: bool | None,
    expect_error: bool,
) -> None:
    """Enforce metadata NetworkPolicy, trusted init + companion probe containers, token automount=False, and reject bound KSAs unless opted in."""
    created_pods: list[k8s_client.V1Pod] = []
    created_policies: list[k8s_client.V1NetworkPolicy] = []

    async def record_create_pod(self: GKEEnvironment, pod: k8s_client.V1Pod) -> None:
        created_pods.append(pod)

    monkeypatch.setattr(GKEEnvironment, "_create_pod", record_create_pod)

    networking_api = (
        SimpleNamespace(
            read_namespaced_network_policy=lambda **_kw: (_ for _ in ()).throw(RuntimeError("404 Not Found")),
            create_namespaced_network_policy=lambda namespace, body, **_kw: (
                created_policies.append(body) or _build_metadata_blocking_network_policy(namespace)
            ),
        )
        if has_networking_api
        else None
    )
    env = make_gke_env(
        allow_workload_identity=allow_workload_identity,
        fake_api=SimpleNamespace(
            read_namespaced_service_account=lambda **_kw: SimpleNamespace(
                metadata=SimpleNamespace(annotations=annotations)
            )
        ),
        networking_api=networking_api,
    )

    pod = _make_pod("pod-metadata-check")
    pod.spec.host_network = True
    if expect_error:
        with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
            asyncio.run(env._create_pod(pod))
        assert created_pods == []
        return

    asyncio.run(env._create_pod(pod))

    assert len(created_pods) == 1
    assert pod.spec.automount_service_account_token is expected_automount
    if not allow_workload_identity:
        assert pod.spec.host_network is False
        assert pod.metadata.labels.get("skillevaluator.nvidia.com/metadata-isolated") == "true"
        assert pod.metadata.annotations.get("autopilot.gke.io/primary-container") == "main"
        assert pod.metadata.annotations.get("kubectl.kubernetes.io/default-container") == "main"
        assert [c.name for c in (pod.spec.init_containers or [])] == ["skillevaluator-metadata-probe-init"]
        init_probe = pod.spec.init_containers[0]
        assert init_probe.image == "python:3.12-slim"
        assert init_probe.command == ["sh", "-c", gke_env_mod._IN_POD_METADATA_ISOLATION_PROBE_SCRIPT]
        assert init_probe.resources is None
        assert [c.name for c in pod.spec.containers] == ["main", "skillevaluator-metadata-probe"]
        probe_container = pod.spec.containers[1]
        assert probe_container.image == "python:3.12-slim"
        assert probe_container.command == ["sleep", "infinity"]
        assert len(created_policies) == 1
        assert len(created_policies[0].spec.egress) == 2
        dns_rule = created_policies[0].spec.egress[1]
        assert {(p.port, p.protocol) for p in dns_rule.ports} == {(53, "UDP"), (53, "TCP")}
    else:
        assert not getattr(pod.spec, "init_containers", None)
        assert [c.name for c in pod.spec.containers] == ["main"]
    assert env._persistent_env.get("GCE_METADATA_HOST") == expected_metadata_host
    container_env_map = {item.name: item.value for item in (pod.spec.containers[0].env or [])}
    assert container_env_map.get("GCE_METADATA_HOST") == expected_metadata_host


@pytest.mark.parametrize(
    ("mutate_policy", "expected_valid"),
    [
        # 1. Default two-rule policy (IPv4 egress excluding metadata + UDP/TCP 53 DNS exception) -> valid
        (lambda _p: None, True),
        # 2. Legacy single-rule policy (ipBlock only, no DNS exception rule) -> valid
        (lambda p: setattr(p.spec, "egress", [p.spec.egress[0]]), True),
        # 3. Missing 169.254.169.252/32 from IPv4 except -> invalid
        (lambda p: setattr(p.spec.egress[0].to[0].ip_block, "_except", ["169.254.169.254/32"]), False),
        # 4. DNS rule exposes HTTP port 80 to 169.254.169.254/32 -> invalid
        (
            lambda p: p.spec.egress[1].ports.append(k8s_client.V1NetworkPolicyPort(port=80, protocol="TCP")),
            False,
        ),
        # 5. DNS rule uses port range 53..80 -> invalid
        (
            lambda p: setattr(p.spec.egress[1].ports[0], "end_port", 80),
            False,
        ),
        # 6. DNS rule omits ports (allowing all ports to 169.254.169.254/32) -> invalid
        (
            lambda p: setattr(p.spec.egress[1], "ports", []),
            False,
        ),
        # 7. DNS rule targets 169.254.169.252/32 instead of 169.254.169.254/32 -> invalid
        (
            lambda p: setattr(p.spec.egress[1].to[0].ip_block, "cidr", "169.254.169.252/32"),
            False,
        ),
        # 8. DNS rule targets arbitrary pod selector instead of kube-dns in kube-system -> invalid
        (
            lambda p: setattr(p.spec.egress[1].to[1].pod_selector, "match_labels", {"app": "attacker"}),
            False,
        ),
        # 9. IPv6 ::/0 peer is rejected (causes Cilium/Dataplane V2 to grant reserved:world on IPv4 clusters) -> invalid
        (
            lambda p: p.spec.egress[0].to.append(
                k8s_client.V1NetworkPolicyPeer(ip_block=k8s_client.V1IPBlock(cidr="::/0"))
            ),
            False,
        ),
    ],
)
def test_network_policy_blocks_metadata_accepts_valid_and_rejects_weakened_rules(
    mutate_policy: Callable[[k8s_client.V1NetworkPolicy], None],
    expected_valid: bool,
) -> None:
    """Validate that _network_policy_blocks_metadata permits only UDP/TCP 53 DNS rules and rejects non-53 metadata exposure."""
    policy = _build_metadata_blocking_network_policy("skill-eval")
    mutate_policy(policy)
    assert _network_policy_blocks_metadata(policy) is expected_valid


@pytest.mark.parametrize(
    ("read_outcomes", "create_conflict", "expect_error"),
    [
        # Existing policy is weakened (missing 169.254.169.252/32) -> rejected
        ([_make_weak_metadata_network_policy("skill-eval")], False, True),
        # 404 on first read, 409 Conflict on create, re-read returns weakened policy -> rejected
        ([RuntimeError("404 Not Found"), _make_weak_metadata_network_policy("skill-eval")], True, True),
        # 404 on first read, 409 Conflict on create, re-read returns valid policy -> accepted
        ([RuntimeError("404 Not Found"), _build_metadata_blocking_network_policy("skill-eval")], True, False),
    ],
)
def test_gke_environment_create_pod_rejects_weakened_existing_network_policy(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    read_outcomes: list[object],
    create_conflict: bool,
    expect_error: bool,
) -> None:
    """Reject pod creation when an existing or concurrently created (409) NetworkPolicy does not block both GKE metadata IPs."""
    created_pods: list[k8s_client.V1Pod] = []

    async def record_create_pod(self: GKEEnvironment, pod: k8s_client.V1Pod) -> None:
        created_pods.append(pod)

    monkeypatch.setattr(GKEEnvironment, "_create_pod", record_create_pod)
    outcomes = list(read_outcomes)

    def fake_read_np(**_kw: object) -> object:
        item = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(item, Exception):
            raise item
        return item

    def fake_create_np(**_kw: object) -> object:
        if create_conflict:
            raise RuntimeError("409 Conflict: networkpolicies 'skillevaluator-block-gce-metadata' already exists")
        return _build_metadata_blocking_network_policy("skill-eval")

    env = make_gke_env(
        fake_api=SimpleNamespace(
            read_namespaced_service_account=lambda **_kw: SimpleNamespace(metadata=SimpleNamespace(annotations={}))
        ),
        networking_api=SimpleNamespace(
            read_namespaced_network_policy=fake_read_np,
            create_namespaced_network_policy=fake_create_np,
        ),
    )

    if expect_error:
        with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
            asyncio.run(env._create_pod(_make_pod("pod-weak-policy")))
        assert created_pods == []
    else:
        asyncio.run(env._create_pod(_make_pod("pod-conflict-valid")))
        assert len(created_pods) == 1


@pytest.mark.parametrize(
    (
        "storage_request",
        "storage_limit",
        "autopilot",
        "node_name",
        "gpu_request",
        "compose_mode",
        "custom_probe_image",
        "expected_main_request",
        "expected_main_limit",
        "expected_init_containers",
        "expected_containers",
        "expected_probe_image",
    ),
    [
        # 1. Autopilot (explicit flag) 10Gi (10240Mi) main storage is capped to 10176Mi so main + 64Mi probe == 10240Mi
        (
            "10240Mi",
            "10Gi",
            True,
            None,
            None,
            False,
            None,
            "10176Mi",
            "10176Mi",
            ["skillevaluator-metadata-probe-init"],
            ["main", "skillevaluator-metadata-probe"],
            "python:3.12-slim",
        ),
        # 2. Autopilot (auto-detected via gk3- node prefix) 10Gi (10240Mi) main storage is capped to 10176Mi
        (
            "10240Mi",
            "10Gi",
            None,
            "gk3-eval-cluster-nap-12345",
            None,
            False,
            None,
            "10176Mi",
            "10176Mi",
            ["skillevaluator-metadata-probe-init"],
            ["main", "skillevaluator-metadata-probe"],
            "python:3.12-slim",
        ),
        # 3. Standard GKE (non-Autopilot node) preserves 10240Mi without lowering ([P2])
        (
            "10240Mi",
            "10Gi",
            None,
            "gke-standard-cluster-default-pool-abc",
            None,
            False,
            None,
            "10240Mi",
            "10Gi",
            ["skillevaluator-metadata-probe-init"],
            ["main", "skillevaluator-metadata-probe"],
            "python:3.12-slim",
        ),
        # 4. Explicit >10Gi storage (20480Mi / 20Gi) is never lowered to 10176Mi even when Autopilot=True ([P2])
        (
            "20480Mi",
            "20Gi",
            True,
            "gk3-eval-cluster-nap-12345",
            None,
            False,
            None,
            "20480Mi",
            "20Gi",
            ["skillevaluator-metadata-probe-init"],
            ["main", "skillevaluator-metadata-probe"],
            "python:3.12-slim",
        ),
        # 5. Autopilot GPU pod preserves 10240Mi because accelerator pods support up to 56TiB ephemeral storage
        (
            "10240Mi",
            "10Gi",
            True,
            None,
            "1",
            False,
            None,
            "10240Mi",
            "10Gi",
            ["skillevaluator-metadata-probe-init"],
            ["main", "skillevaluator-metadata-probe"],
            "python:3.12-slim",
        ),
        # 6. Sub-ceiling storage (5120Mi) is untouched and custom probe image env var is honored on init + companion
        (
            "5120Mi",
            None,
            True,
            None,
            None,
            False,
            "us-central1-docker.pkg.dev/my-proj/eval/probe:v1",
            "5120Mi",
            None,
            ["skillevaluator-metadata-probe-init"],
            ["main", "skillevaluator-metadata-probe"],
            "us-central1-docker.pkg.dev/my-proj/eval/probe:v1",
        ),
        # 7. Compose mode (DinD) does not inject init/companion probe containers or cap dind storage
        (
            "10240Mi",
            None,
            True,
            None,
            None,
            True,
            None,
            "10240Mi",
            None,
            [],
            ["main"],
            None,
        ),
    ],
)
def test_gke_environment_create_pod_autopilot_storage_idempotency_and_custom_probe_image(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    storage_request: str,
    storage_limit: str | None,
    autopilot: bool | None,
    node_name: str | None,
    gpu_request: str | None,
    compose_mode: bool,
    custom_probe_image: str | None,
    expected_main_request: str,
    expected_main_limit: str | None,
    expected_init_containers: list[str],
    expected_containers: list[str],
    expected_probe_image: str | None,
) -> None:
    """Scope 10Gi->10176Mi storage adjustment to non-accelerator Autopilot pods, preserve >10Gi requests, and avoid duplicates on retry."""
    if custom_probe_image is not None:
        monkeypatch.setenv("SKILLEVALUATOR_GKE_METADATA_PROBE_IMAGE", custom_probe_image)

    async def noop_create_pod(self: GKEEnvironment, pod: k8s_client.V1Pod) -> None:
        _ = pod

    monkeypatch.setattr(GKEEnvironment, "_create_pod", noop_create_pod)
    fake_nodes = [SimpleNamespace(metadata=SimpleNamespace(name=node_name, labels={}))] if node_name is not None else []
    env = make_gke_env(
        autopilot=autopilot,
        compose_mode=compose_mode,
        fake_api=SimpleNamespace(
            read_namespaced_service_account=lambda **_kw: SimpleNamespace(metadata=SimpleNamespace(annotations={})),
            list_node=lambda **_kw: SimpleNamespace(items=fake_nodes),
        ),
        networking_api=SimpleNamespace(
            read_namespaced_network_policy=lambda **_kw: _build_metadata_blocking_network_policy("skill-eval"),
            create_namespaced_network_policy=lambda **_kw: _build_metadata_blocking_network_policy("skill-eval"),
        ),
    )

    pod = _make_pod("pod-storage-check")
    requests_dict: dict[str, str] = {"cpu": "1", "memory": "2048Mi", "ephemeral-storage": storage_request}
    limits_dict: dict[str, str] = {"ephemeral-storage": storage_limit} if storage_limit else {}
    if gpu_request is not None:
        requests_dict["nvidia.com/gpu"] = gpu_request
        limits_dict["nvidia.com/gpu"] = gpu_request
    pod.spec.containers[0].resources = k8s_client.V1ResourceRequirements(
        requests=requests_dict,
        limits=limits_dict or None,
    )

    # Call _create_pod twice to verify idempotency on retry/recreation
    asyncio.run(env._create_pod(pod))
    asyncio.run(env._create_pod(pod))

    assert [c.name for c in (pod.spec.init_containers or [])] == expected_init_containers
    assert [c.name for c in pod.spec.containers] == expected_containers
    assert pod.spec.containers[0].resources.requests["ephemeral-storage"] == expected_main_request
    if expected_main_limit is not None:
        assert pod.spec.containers[0].resources.limits["ephemeral-storage"] == expected_main_limit
    if expected_probe_image is not None:
        init_probe = pod.spec.init_containers[0]
        assert init_probe.image == expected_probe_image
        assert init_probe.resources is None
        probe_container = pod.spec.containers[1]
        assert probe_container.image == expected_probe_image
        assert probe_container.resources.requests["ephemeral-storage"] == "64Mi"
        assert probe_container.resources.limits["ephemeral-storage"] == "64Mi"


@pytest.mark.parametrize(
    ("init_exit_code", "init_waiting_reason", "pod_phase", "expected_error_snippet"),
    [
        # 1. Init probe succeeds (rc=0) and pod is Running/Ready -> succeeds
        (0, None, "Running", None),
        # 2. Init probe detects reachable metadata before main starts (rc=42) -> deletes pod and fails closed
        (42, None, "Failed", "in-pod init probe reached the GKE/GCE metadata server"),
        # 3. Init probe fails with missing tool/error (rc=43) while pod still Pending -> deletes pod and fails closed
        (43, None, "Pending", "in-pod metadata isolation init probe exited with status 43"),
        # 4. Init probe image fails to pull (ImagePullBackOff) -> deletes pod and fails closed without waiting 300s
        (None, "ImagePullBackOff", "Pending", "failed to pull metadata probe image"),
    ],
)
def test_gke_environment_wait_for_pod_ready_fails_closed_on_init_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    init_exit_code: int | None,
    init_waiting_reason: str | None,
    pod_phase: str,
    expected_error_snippet: str | None,
) -> None:
    """Inspect init_container_statuses during _wait_for_pod_ready so a failed pre-start probe deletes the pod immediately."""
    deleted_pods: list[tuple[str, str]] = []

    async def fast_sleep(_sec: float) -> None:
        return None

    monkeypatch.setattr(gke_env_mod.asyncio, "sleep", fast_sleep)

    init_state = SimpleNamespace(
        waiting=(
            SimpleNamespace(reason=init_waiting_reason, message="Back-off pulling image python:3.12-slim")
            if init_waiting_reason
            else None
        ),
        terminated=(
            SimpleNamespace(exit_code=init_exit_code, reason="Completed" if init_exit_code == 0 else "Error")
            if init_exit_code is not None
            else None
        ),
    )
    fake_pod = SimpleNamespace(
        status=SimpleNamespace(
            phase=pod_phase,
            reason=None,
            message=None,
            init_container_statuses=[
                SimpleNamespace(
                    name="skillevaluator-metadata-probe-init",
                    ready=(init_exit_code == 0),
                    state=init_state,
                )
            ],
            container_statuses=[
                SimpleNamespace(
                    name="main",
                    ready=(pod_phase == "Running"),
                    state=SimpleNamespace(waiting=None, terminated=None),
                ),
                SimpleNamespace(
                    name="skillevaluator-metadata-probe",
                    ready=(pod_phase == "Running"),
                    state=SimpleNamespace(waiting=None, terminated=None),
                ),
            ],
        )
    )

    env = make_gke_env(
        namespace="skilleval",
        fake_api=SimpleNamespace(
            read_namespaced_pod=lambda **_k: fake_pod,
            delete_namespaced_pod=lambda name, namespace, **_k: deleted_pods.append((name, namespace)),
        ),
    )
    env.pod_name = "pod-init-check"

    if expected_error_snippet is not None:
        with pytest.raises(RuntimeError, match=expected_error_snippet):
            asyncio.run(env._wait_for_pod_ready(timeout_sec=3))
        assert deleted_pods == [("pod-init-check", "skilleval")]
    else:
        asyncio.run(env._wait_for_pod_ready(timeout_sec=3))
        assert deleted_pods == []


@pytest.mark.parametrize(
    ("probe_rc", "compose_mode", "expect_error"),
    [
        (0, False, False),
        (0, True, False),
        (42, False, True),
        (43, False, True),
    ],
)
def test_gke_environment_wait_for_container_exec_ready_drains_stream_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    probe_rc: int,
    compose_mode: bool,
    expect_error: bool,
) -> None:
    """Drain the Kubernetes WSClient stream, retry readiness, and enforce the in-pod metadata isolation probe in the trusted probe container."""
    ws_outcomes: list[int | Exception] = [
        ValueError(
            "invalid literal for int() with base 10: "
            "'error sending request: Post \"https://10.128.0.47:10250/exec/skilleval/pod/main?command=true\"'"
        ),
        1,
        0,
        # 4th stream call is the in-pod metadata isolation probe
        probe_rc,
    ]
    ws_attempts = 0
    read_output_calls = 0
    deleted_pods: list[tuple[str, str]] = []
    recorded_calls: list[tuple[list[str], str | None]] = []

    class FakeWSClient:
        def __init__(self, outcome: int | Exception) -> None:
            self._outcome = outcome
            self.closed = False

        def is_open(self) -> bool:
            return True

        def run_forever(self, timeout: int | float | None = None) -> None:
            _ = timeout

        @property
        def returncode(self) -> int:
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return self._outcome

        def close(self) -> None:
            self.closed = True

    def fake_stream(
        _fn: object,
        *_args: object,
        command: list[str] | None = None,
        container: str | None = None,
        **_kwargs: object,
    ) -> FakeWSClient:
        nonlocal ws_attempts
        if command is not None:
            recorded_calls.append((list(command), container))
        outcome = ws_outcomes[min(ws_attempts, len(ws_outcomes) - 1)]
        ws_attempts += 1
        return FakeWSClient(outcome)

    async def fast_sleep(_sec: float) -> None:
        return None

    async def noop_check_terminated(self: GKEEnvironment) -> None:
        return None

    monkeypatch.setattr(gke_env_mod, "stream", fake_stream, raising=False)
    monkeypatch.setattr(gke_env_mod.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(GKEEnvironment, "_check_pod_terminated", noop_check_terminated)

    env = make_gke_env(
        namespace="skilleval",
        compose_mode=compose_mode,
        fake_api=SimpleNamespace(
            connect_get_namespaced_pod_exec=lambda *_a, **_k: None,
            delete_namespaced_pod=lambda name, namespace, **_k: deleted_pods.append((name, namespace)),
        ),
    )
    env.pod_name = "pod-ready-check"

    def record_read_output(_resp: object) -> None:
        nonlocal read_output_calls
        read_output_calls += 1

    env._read_exec_output = record_read_output  # type: ignore[method-assign]
    if expect_error:
        with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
            asyncio.run(env._wait_for_container_exec_ready(max_attempts=4))
        assert deleted_pods == [("pod-ready-check", "skilleval")]
        assert recorded_calls[-1][1] == "skillevaluator-metadata-probe"
        return

    asyncio.run(env._wait_for_container_exec_ready(max_attempts=4))
    assert ws_attempts == 4
    assert read_output_calls == 4
    assert deleted_pods == []
    assert any("169.254.169.254" in " ".join(cmd) for cmd, _ in recorded_calls)
    expected_readiness_container = "dind" if compose_mode else None
    expected_probe_container = "dind" if compose_mode else "skillevaluator-metadata-probe"
    assert [container for _, container in recorded_calls[:3]] == [expected_readiness_container] * 3
    assert recorded_calls[3][1] == expected_probe_container

    # Exhausting max_attempts raises RuntimeError
    ws_outcomes[:] = [RuntimeError("kubelet stream refused")]
    ws_attempts = 0
    with pytest.raises(RuntimeError, match="Container not ready for exec after 2 attempts"):
        asyncio.run(env._wait_for_container_exec_ready(max_attempts=2))
    assert ws_attempts == 2


@pytest.mark.parametrize(
    ("scenario", "http_status", "with_dead_proxy", "tool_mode", "expected_rc"),
    [
        # 1. Unreachable endpoint via python returns 0 (isolated)
        ("python_unreachable", None, False, "python", 0),
        # 2. Reachable 200 OK endpoint via python returns 42 even with dead http_proxy / HTTP_PROXY (SP-2)
        ("python_reachable_200_with_dead_proxy", 200, True, "python", 42),
        # 3. Reachable 403 Forbidden endpoint via python returns 42 (HTTPError proves TCP reachability, EC-4)
        ("python_reachable_403", 403, True, "python", 42),
        # 4. Reachable 404 Not Found endpoint via python returns 42 (EC-4)
        ("python_reachable_404", 404, False, "python", 42),
        # 5. Unreachable endpoint via curl fallback (--noproxy '*') returns 0
        ("curl_unreachable", None, True, "curl_only", 0),
        # 6. Reachable 200 OK endpoint via curl fallback returns 42 even with dead http_proxy (SP-2)
        ("curl_reachable_200_with_dead_proxy", 200, True, "curl_only", 42),
        # 7. Shimmed python3/python returning 0 without sentinel output returns 43 (SP-3)
        ("shimmed_python", 200, False, "shimmed_python", 43),
        # 8. Shimmed curl returning 0 without HTTP_CODE:000 output returns 43 (SP-3)
        ("shimmed_curl", 200, False, "shimmed_curl", 43),
        # 9. BusyBox wget (docker:dind) rejecting --no-proxy returns 0 when connection is refused ([P1])
        ("busybox_wget_unreachable", None, True, "busybox_wget", 0),
        # 10. BusyBox wget (docker:dind) rejecting --no-proxy returns 42 when endpoint responds HTTP/1.1 200 ([P1])
        ("busybox_wget_reachable_200_with_dead_proxy", 200, True, "busybox_wget", 42),
        # 11. Shimmed or broken wget printing unrecognized error text fails closed with 43 ([P1])
        ("shimmed_wget_unknown_output", 200, False, "shimmed_wget", 43),
    ],
)
def test_in_pod_metadata_isolation_probe_script_resists_proxy_and_binary_shims(
    tmp_path: Path,
    local_metadata_http_server: Callable[[int], str],
    scenario: str,
    http_status: int | None,
    with_dead_proxy: bool,
    tool_mode: str,
    expected_rc: int,
) -> None:
    """Verify the shell probe script ignores http_proxy variables, supports BusyBox wget, and rejects shims."""
    _ = scenario
    target_url = (
        "http://127.0.0.1:1/computeMetadata/v1/instance/service-accounts/default/token"
        if http_status is None
        else local_metadata_http_server(http_status)
    )
    script = _build_in_pod_metadata_isolation_probe_script(urls=(target_url,))
    env = dict(os.environ)
    if with_dead_proxy:
        env["http_proxy"] = "http://127.0.0.1:9"
        env["HTTP_PROXY"] = "http://127.0.0.1:9"
        env["all_proxy"] = "http://127.0.0.1:9"
        env["ALL_PROXY"] = "http://127.0.0.1:9"
    if tool_mode == "curl_only":
        real_curl = shutil.which("curl")
        if real_curl is None:
            pytest.skip("curl is not installed on host")
        bin_dir = tmp_path / "curl_only_bin"
        bin_dir.mkdir()
        (bin_dir / "curl").symlink_to(real_curl)
        env["PATH"] = str(bin_dir)
    elif tool_mode == "shimmed_python":
        shim_dir = tmp_path / "shims_py"
        shim_dir.mkdir()
        for tool_name in ("python3", "python"):
            shim_path = shim_dir / tool_name
            shim_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR)
        env["PATH"] = str(shim_dir)
    elif tool_mode == "shimmed_curl":
        shim_dir = tmp_path / "shims_curl"
        shim_dir.mkdir()
        shim_path = shim_dir / "curl"
        shim_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR)
        env["PATH"] = str(shim_dir)
    elif tool_mode in ("busybox_wget", "shimmed_wget"):
        shim_dir = tmp_path / f"shims_{tool_mode}"
        shim_dir.mkdir()
        real_grep = shutil.which("grep")
        if real_grep is not None:
            (shim_dir / "grep").symlink_to(real_grep)
        wget_path = shim_dir / "wget"
        if tool_mode == "busybox_wget":
            busybox_output = (
                "Connecting to 127.0.0.1:1 (127.0.0.1:1)\\nwget: can't connect to remote host (127.0.0.1): Connection refused\\n"
                if http_status is None
                else f"Connecting to 127.0.0.1 (127.0.0.1)\\n  HTTP/1.1 {http_status} OK\\n"
            )
            busybox_rc = 1 if http_status is None else 0
            wget_path.write_text(
                "#!/bin/sh\n"
                'for arg in "$@"; do\n'
                '  if [ "$arg" = "--no-proxy" ]; then\n'
                "    printf \"wget: unrecognized option '--no-proxy'\\nBusyBox v1.36.1 multi-call binary.\\n\" >&2\n"
                "    exit 1\n"
                "  fi\n"
                "done\n"
                'if [ -n "${http_proxy:-}${HTTP_PROXY:-}${all_proxy:-}${ALL_PROXY:-}" ]; then\n'
                '  printf "wget: proxy environment variable was not unset\\n" >&2\n'
                "  exit 1\n"
                "fi\n"
                f'printf "{busybox_output}" >&2\n'
                f"exit {busybox_rc}\n",
                encoding="utf-8",
            )
        else:
            wget_path.write_text(
                '#!/bin/sh\nprintf "wget: unexpected shim error\\n" >&2\nexit 1\n',
                encoding="utf-8",
            )
        wget_path.chmod(wget_path.stat().st_mode | stat.S_IXUSR)
        env["PATH"] = str(shim_dir)

    completed = subprocess.run(
        ["/bin/sh", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == expected_rc


@pytest.mark.parametrize(
    ("probe_container_rc", "expect_isolation_failure"),
    [
        # SC-1: Probe container confirms isolation (rc=0) while untargeted Harbor exec routes to "main"
        (0, False),
        # SC-2 / SP-1: Compromised "main" container spoofs rc=0, but trusted probe container detects reachable metadata (rc=42)
        (42, True),
    ],
)
def test_gke_environment_seam_routes_untargeted_exec_to_main_and_probes_trusted_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    probe_container_rc: int,
    expect_isolation_failure: bool,
) -> None:
    """Verify Harbor untargeted exec calls route to 'main' while metadata isolation verification runs in 'skillevaluator-metadata-probe'."""
    exec_dispatches: list[tuple[list[str], str | None]] = []
    deleted_pods: list[tuple[str, str]] = []

    class FakeWSResponse:
        def __init__(self, returncode: int, stdout: str = "") -> None:
            self.returncode = returncode
            self._stdout = stdout

        def is_open(self) -> bool:
            return False

        def run_forever(self, timeout: int | float | None = None) -> None:
            _ = timeout

        def close(self) -> None:
            return

    class FakeCoreV1Api:
        def __init__(self) -> None:
            self.api_client = object()

        def read_namespaced_service_account(self, **_kw: object) -> object:
            return SimpleNamespace(metadata=SimpleNamespace(annotations={}))

        def create_namespaced_pod(self, **_kw: object) -> object:
            return None

        def delete_namespaced_pod(self, name: str, namespace: str, **_kw: object) -> None:
            deleted_pods.append((name, namespace))

        def connect_get_namespaced_pod_exec(
            self,
            _name: str,
            _namespace: str,
            *,
            command: list[str] | None = None,
            container: str | None = None,
            **_kwargs: object,
        ) -> FakeWSResponse:
            cmd_list = list(command or [])
            exec_dispatches.append((cmd_list, container))
            if container == "skillevaluator-metadata-probe":
                return FakeWSResponse(returncode=probe_container_rc)
            # Even if a compromised "main" container always returns 0, the probe never trusts "main"
            return FakeWSResponse(returncode=0, stdout="from-main")

    def passthrough_stream(fn: Callable[..., FakeWSResponse], *args: object, **kwargs: object) -> FakeWSResponse:
        # Kubernetes stream() requires fn.__self__.api_client to exist on bound API methods
        assert hasattr(getattr(fn, "__self__", None), "api_client")
        return fn(*args, **kwargs)

    async def noop_check_terminated(self: GKEEnvironment) -> None:
        return None

    async def direct_ensure_client(self: GKEEnvironment) -> None:
        return None

    monkeypatch.setattr(gke_env_mod, "stream", passthrough_stream, raising=False)
    monkeypatch.setattr("harbor.environments.gke.stream", passthrough_stream, raising=False)
    monkeypatch.setattr(GKEEnvironment, "_check_pod_terminated", noop_check_terminated)
    monkeypatch.setattr(GKEEnvironment, "_ensure_client", direct_ensure_client)

    fake_api = FakeCoreV1Api()
    env = make_gke_env(
        namespace="skill-eval",
        fake_api=fake_api,
        networking_api=SimpleNamespace(
            read_namespaced_network_policy=lambda **_kw: _build_metadata_blocking_network_policy("skill-eval"),
            create_namespaced_network_policy=lambda **_kw: _build_metadata_blocking_network_policy("skill-eval"),
        ),
    )
    env.task_env_config = SimpleNamespace(workdir=None, user=None)
    env._read_exec_output = lambda _resp: ("from-main", "")  # type: ignore[method-assign]

    pod = _make_pod("pod-seam-test")
    asyncio.run(env._create_pod(pod))

    if expect_isolation_failure:
        with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
            asyncio.run(env._wait_for_container_exec_ready(max_attempts=2))
        assert deleted_pods == [("pod-under-test", "skill-eval")]
        assert [container for _, container in exec_dispatches] == ["main", "skillevaluator-metadata-probe"]
        return

    asyncio.run(env._wait_for_container_exec_ready(max_attempts=2))
    exec_result = asyncio.run(env.exec("echo hello"))
    assert exec_result.return_code == 0
    assert exec_result.stdout == "from-main"
    assert [container for _, container in exec_dispatches] == [
        "main",
        "skillevaluator-metadata-probe",
        "main",
    ]


def test_gke_environment_exec_retries_transient_kubelet_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry transient kubelet 10250 errors during exec(), but do not retry regular command failures."""

    async def fast_sleep(_sec: float) -> None:
        return None

    monkeypatch.setattr(gke_env_mod.asyncio, "sleep", fast_sleep)

    # 1. Transient kubelet error recovers on second attempt
    exec_calls = 0

    async def flaky_super_exec(
        self: GKEEnvironment, command: str, *args: object, env: dict[str, str] | None = None, **kwargs: object
    ) -> ExecResult:
        nonlocal exec_calls
        exec_calls += 1
        if exec_calls == 1:
            return ExecResult(
                stdout=None,
                stderr=(
                    "invalid literal for int() with base 10: "
                    "'error sending request: Post \"https://10.128.0.47:10250/exec/skilleval/pod/main?command=sh&command=-c&command=OPENAI_API_KEY=ya29.super-secret\"'"
                ),
                return_code=1,
            )
        return ExecResult(stdout="created", stderr="", return_code=0)

    monkeypatch.setattr(GKEEnvironment, "exec", flaky_super_exec)
    env = object.__new__(SkillEvaluatorGKEEnvironment)
    env._persistent_env = {}
    res_ok = asyncio.run(env.exec("mkdir -p /logs/agent", user="root"))
    assert res_ok.return_code == 0
    assert exec_calls == 2

    # 2. Non-transient command failure returns immediately without retrying
    non_transient_calls = 0

    async def regular_failure_super_exec(
        self: GKEEnvironment, command: str, *args: object, env: dict[str, str] | None = None, **kwargs: object
    ) -> ExecResult:
        nonlocal non_transient_calls
        non_transient_calls += 1
        return ExecResult(stdout="", stderr="pytest: 1 failed", return_code=1)

    monkeypatch.setattr(GKEEnvironment, "exec", regular_failure_super_exec)
    res_regular = asyncio.run(env.exec("bash /tests/test.sh"))
    assert res_regular.return_code == 1
    assert non_transient_calls == 1

    # 3. Exhausted transient retries scrub secrets from stderr before returning
    exhausted_calls = 0

    async def always_failing_super_exec(
        self: GKEEnvironment, command: str, *args: object, env: dict[str, str] | None = None, **kwargs: object
    ) -> ExecResult:
        nonlocal exhausted_calls
        exhausted_calls += 1
        return ExecResult(
            stdout=None,
            stderr=(
                "invalid literal for int() with base 10: "
                "'error sending request: Post \"https://10.128.0.47:10250/exec/skilleval/pod/main?command=sh&command=-c&command=OPENAI_API_KEY=ya29.super-secret-token\"'"
            ),
            return_code=1,
        )

    monkeypatch.setattr(GKEEnvironment, "exec", always_failing_super_exec)
    res_fail = asyncio.run(
        env.exec(
            "bash /tests/test.sh",
            env={"OPENAI_API_KEY": "ya29.super-secret-token"},
        )
    )
    assert res_fail.return_code == 1
    assert exhausted_calls == 4
    assert "ya29.super-secret-token" not in (res_fail.stderr or "")
    assert "?command=sh" not in (res_fail.stderr or "")
    assert "?command=<redacted>" in (res_fail.stderr or "")


@pytest.mark.parametrize(
    ("stderr", "env_values", "must_not_contain", "must_contain"),
    [
        # 1. ?command= as first query parameter
        (
            'Post "https://10.128.0.47:10250/exec/ns/pod/main?command=sh&command=-c&command=export%20OPENAI_API_KEY%3Dmy-secret-key"',
            {"OPENAI_API_KEY": "my-secret-key"},
            ["my-secret-key", "?command=sh"],
            ["?command=<redacted>"],
        ),
        # 2. &command= after container/stderr query parameters
        (
            'Post "https://10.128.0.47:10250/exec/ns/pod/main?container=main&stderr=true&command=sh&command=-c&command=OPENAI_API_KEY=my-secret-key"',
            {"OPENAI_API_KEY": "my-secret-key"},
            ["my-secret-key", "&command=sh"],
            ["?container=main&stderr=true&command=<redacted>"],
        ),
        # 3. URL-encoded secret outside ?command= query string
        (
            f"upstream proxy error: token={urllib.parse.quote('sec+ret/val=12345', safe='')}",
            {"OPENAI_API_KEY": "sec+ret/val=12345"},
            ["sec+ret/val=12345", urllib.parse.quote("sec+ret/val=12345", safe="")],
            ["<redacted>"],
        ),
        # 4. Non-sensitive environment values (CLOUD_ML_REGION, ANTHROPIC_VERTEX_PROJECT_ID) are NOT over-redacted
        (
            "Vertex request failed in region global for project my-vertex-project-123 with key secret-api-key-9999",
            {
                "CLOUD_ML_REGION": "global",
                "ANTHROPIC_VERTEX_PROJECT_ID": "my-vertex-project-123",
                "GCE_METADATA_HOST": "127.0.0.1:1",
                "OPENAI_API_KEY": "secret-api-key-9999",
            },
            ["secret-api-key-9999"],
            ["region global", "project my-vertex-project-123", "<redacted>"],
        ),
        # 5. CPython int() 200-char truncation cutting a secret mid-value before closing quote
        (
            "invalid literal for int() with base 10: 'kubelet auth error: custom_secret_prefix_abcdef'",
            {"ANTHROPIC_AUTH_TOKEN": "custom_secret_prefix_abcdef_0123456789_tail"},
            ["custom_secret_prefix_abcdef"],
            ["<redacted>'"],
        ),
        # 6. Pattern-based Google API key (AIza...) and truncated OAuth token (ya29....)
        (
            " ".join(("google", "AIza" + "SyDa1234567890abcdef", "ya29." + "a0AfB7c8D9")),
            {},
            ["AIza" + "SyDa1234567890abcdef", "ya29." + "a0AfB7c8D9"],
            ["google <redacted> <redacted>"],
        ),
    ],
)
def test_redact_exec_stderr_scrubs_queries_and_encoded_secrets_without_corrupting_config(
    stderr: str,
    env_values: dict[str, str],
    must_not_contain: list[str],
    must_contain: list[str],
) -> None:
    """Scrub query strings, encoded secrets, and truncated prefixes while preserving non-sensitive config strings."""
    redacted = _redact_exec_stderr(stderr, env_values)
    assert redacted is not None
    for forbidden in must_not_contain:
        assert forbidden not in redacted
    for expected in must_contain:
        assert expected in redacted


@pytest.mark.parametrize(
    ("persistent_env", "call_env", "minted_token", "expected_call_env_key", "expected_persistent_key"),
    [
        # 1. Refreshes OPENAI_API_KEY in _persistent_env when exec(env=None) is called
        (
            {
                "OPENAI_API_KEY": "stale-persistent-token",
                "OPENAI_BASE_URL": "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi",
                "SKILL_EVAL_LLM_CREDENTIAL_SOURCE": "ADC",
            },
            None,
            "fresh-adc-token",
            "fresh-adc-token",
            "fresh-adc-token",
        ),
        # 2. Refreshes OPENAI_API_KEY passed in explicit env dict
        (
            {},
            {
                "OPENAI_API_KEY": "stale-call-token",
                "OPENAI_BASE_URL": "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi",
                "SKILL_EVAL_LLM_CREDENTIAL_SOURCE": "ADC",
            },
            "fresh-adc-token",
            "fresh-adc-token",
            None,
        ),
        # 3. Setup/agent exec with env=None and no OPENAI_API_KEY in _persistent_env does NOT leak host OPENAI_API_KEY
        (
            {"_JAVA_OPTIONS": ""},
            None,
            "fresh-adc-token",
            None,
            None,
        ),
        # 4. Explicit empty OPENAI_API_KEY="" is NOT overwritten with a live token
        (
            {},
            {"OPENAI_API_KEY": ""},
            "fresh-adc-token",
            "",
            None,
        ),
        # 5. Transient None from token refresh preserves existing non-empty OPENAI_API_KEY
        (
            {},
            {"OPENAI_API_KEY": "still-valid-token"},
            None,
            "still-valid-token",
            None,
        ),
    ],
)
def test_gke_environment_exec_adc_token_refresh_and_isolation(
    monkeypatch: pytest.MonkeyPatch,
    persistent_env: dict[str, str],
    call_env: dict[str, str] | None,
    minted_token: str | None,
    expected_call_env_key: str | None,
    expected_persistent_key: str | None,
) -> None:
    """Refresh Vertex ADC tokens only when OPENAI_API_KEY is already active in the target execution environment."""
    captured_merged: dict[str, str] = {}
    captured_call_env: dict[str, str] | None = None

    async def capture_super_exec(
        self: GKEEnvironment, command: str, *args: object, env: dict[str, str] | None = None, **kwargs: object
    ) -> ExecResult:
        nonlocal captured_call_env
        captured_call_env = dict(env) if env is not None else None
        merged = BaseEnvironment._merge_env(self, env) or {}
        captured_merged.update(merged)
        return ExecResult(stdout="ok", stderr="", return_code=0)

    monkeypatch.setattr(GKEEnvironment, "exec", capture_super_exec)
    monkeypatch.setenv("OPENAI_API_KEY", "host-verifier-token")
    monkeypatch.setenv("SKILL_EVAL_LLM_CREDENTIAL_SOURCE", "ADC")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi",
    )
    monkeypatch.setattr("skillevaluator.provider_config._get_google_access_token", lambda **_kw: minted_token)

    env_obj = object.__new__(SkillEvaluatorGKEEnvironment)
    env_obj._persistent_env = dict(persistent_env)
    asyncio.run(env_obj.exec("bash /tests/test.sh", env=call_env))

    assert captured_merged.get("OPENAI_API_KEY") == expected_call_env_key
    assert env_obj._persistent_env.get("OPENAI_API_KEY") == expected_persistent_key
    if call_env is None and expected_persistent_key is None:
        assert captured_call_env is None


@pytest.mark.parametrize(
    ("read_effect", "should_raise", "expected_match"),
    [
        # 1. Running pod with healthy container -> no error
        (
            SimpleNamespace(
                status=SimpleNamespace(
                    phase="Running",
                    container_statuses=[
                        SimpleNamespace(
                            name="main",
                            state=SimpleNamespace(terminated=None),
                            last_state=SimpleNamespace(terminated=None),
                        )
                    ],
                )
            ),
            False,
            "",
        ),
        # 2. Pod deleted/preempted (404 Not Found) -> fail fast with RuntimeError
        (
            ApiException(status=404, reason="Not Found"),
            True,
            r"no longer exists|deleted or preempted",
        ),
        # 3. Transient API server error (503 Service Unavailable) -> ignored
        (
            ApiException(status=503, reason="Service Unavailable"),
            False,
            "",
        ),
        # 4. Terminal pod phase (Failed) -> raises RuntimeError
        (
            SimpleNamespace(
                status=SimpleNamespace(
                    phase="Failed",
                    container_statuses=[],
                )
            ),
            True,
            r"terminal phase 'Failed'",
        ),
    ],
)
def test_check_pod_terminated_fails_fast_when_pod_deleted_or_preempted(
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
    read_effect: object,
    should_raise: bool,
    expected_match: str,
) -> None:
    """Fail immediately when the evaluation pod is 404 Not Found while ignoring transient non-404 API errors."""

    def _read_pod(*, name: str, namespace: str) -> object:
        assert name == "pod-under-test"
        assert namespace == "default"
        if isinstance(read_effect, Exception):
            raise read_effect
        return read_effect

    env_obj = make_gke_env(
        namespace="default",
        fake_api=SimpleNamespace(read_namespaced_pod=_read_pod),
    )

    if should_raise:
        with pytest.raises(RuntimeError, match=expected_match):
            asyncio.run(env_obj._check_pod_terminated())
    else:
        asyncio.run(env_obj._check_pod_terminated())


def test_wait_for_container_exec_ready_aborts_immediately_on_gke_warden_missing_pod(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
) -> None:
    """Translate GKE Warden missing-pod WebSocket AttributeError into an immediate RuntimeError without 60x retry sleep."""
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.gke_environment.asyncio.sleep",
        _fake_sleep,
    )
    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.gke_environment.stream",
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )

    def _warden_missing_pod_exec(*_args: object, **_kwargs: object) -> object:
        warden_err = RuntimeError(
            'Handshake status 400 Bad Request: {"message":"Cannot connect to pod default/pod-under-test, not found."}'
        )
        raise AttributeError("'NoneType' object has no attribute 'decode'") from warden_err

    env_obj = make_gke_env(
        namespace="default",
        fake_api=SimpleNamespace(
            connect_get_namespaced_pod_exec=_warden_missing_pod_exec,
            read_namespaced_pod=lambda **_kw: SimpleNamespace(
                status=SimpleNamespace(phase="Running", container_statuses=[])
            ),
        ),
    )

    with pytest.raises(RuntimeError, match=r"no longer exists|not found"):
        asyncio.run(env_obj._wait_for_container_exec_ready(max_attempts=60))

    assert sleep_calls == []


def test_gke_environment_preflight_normalizes_multipath_kubeconfig(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept split KUBECONFIG in SkillEvaluatorGKEEnvironment.preflight() and raise SystemExit when missing."""
    monkeypatch.setattr("skillevaluator.tier3.harbor.gke_environment.shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    valid_kc = tmp_path / "valid_kubeconfig"
    valid_kc.write_text("apiVersion: v1\nclusters: []\n", encoding="utf-8")
    missing_kc = tmp_path / "missing_kubeconfig"
    monkeypatch.setenv("KUBECONFIG", f"{missing_kc}{os.pathsep}{valid_kc}")

    # Split KUBECONFIG with one valid file succeeds where stock GKEEnvironment.preflight() would fail
    SkillEvaluatorGKEEnvironment.preflight()

    # All-missing KUBECONFIG raises SystemExit
    monkeypatch.setenv("KUBECONFIG", str(missing_kc))
    with pytest.raises(SystemExit, match="Kubernetes credentials"):
        SkillEvaluatorGKEEnvironment.preflight()


def test_gke_environment_exec_handles_harbor_024_stream_closed_and_pod_not_found(
    monkeypatch: pytest.MonkeyPatch,
    make_gke_env: Callable[..., SkillEvaluatorGKEEnvironment],
) -> None:
    """Retry GKEExecStreamClosedError, redact persistent stream errors, and raise GKEPodNotFoundError on Warden missing pod."""
    from harbor.environments.base import ExecResult
    from harbor.environments.gke import GKEEnvironment, GKEExecStreamClosedError, GKEPodNotFoundError

    async def _no_sleep(_sec: float) -> None:
        return None

    monkeypatch.setattr("skillevaluator.tier3.harbor.gke_environment.asyncio.sleep", _no_sleep)
    secret_val = "sk-ant-api03-supersecretvalue123456"
    env_obj = make_gke_env(namespace="default")
    env_obj._persistent_env = {"ANTHROPIC_API_KEY": secret_val}

    # 1. Transient GKEExecStreamClosedError succeeds on retry
    attempts = 0

    async def _flaky_exec(self_inner: object, **kwargs: object) -> ExecResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise GKEExecStreamClosedError(f"WebSocket closed with {secret_val}")
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    monkeypatch.setattr(GKEEnvironment, "exec", _flaky_exec)
    res = asyncio.run(env_obj.exec("echo hi"))
    assert res.return_code == 0
    assert attempts == 2

    # 2. Persistent GKEExecStreamClosedError raises with redacted message
    async def _always_closed_exec(self_inner: object, **kwargs: object) -> ExecResult:
        raise GKEExecStreamClosedError(f"WebSocket closed: token={secret_val}")

    monkeypatch.setattr(GKEEnvironment, "exec", _always_closed_exec)
    with pytest.raises(GKEExecStreamClosedError) as exc_info:
        asyncio.run(env_obj.exec("echo hi"))
    assert secret_val not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)

    # 3. Warden 400 missing-pod in stderr raises GKEPodNotFoundError with redacted message
    async def _warden_missing_exec(self_inner: object, **kwargs: object) -> ExecResult:
        return ExecResult(
            stdout="",
            stderr=f'Handshake status 400 Bad Request: {{"message":"Cannot connect to pod default/pod-1, not found."}} key={secret_val}',
            return_code=1,
        )

    monkeypatch.setattr(GKEEnvironment, "exec", _warden_missing_exec)
    with pytest.raises(GKEPodNotFoundError) as pod_exc_info:
        asyncio.run(env_obj.exec("echo hi"))
    assert secret_val not in str(pod_exc_info.value)
    assert "<redacted>" in str(pod_exc_info.value)

