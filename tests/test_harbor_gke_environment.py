# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SkillEvaluatorGKEEnvironment pod isolation, exec readiness, retry, and redaction."""

from __future__ import annotations

import asyncio
import urllib.parse
from types import SimpleNamespace

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.gke import GKEEnvironment
from kubernetes import client as k8s_client

import skillevaluator.tier3.harbor.gke_environment as gke_env_mod
from skillevaluator.tier3.harbor.gke_environment import (
    SkillEvaluatorGKEEnvironment,
    _redact_exec_stderr,
)


def _make_pod(name: str = "eval-pod", namespace: str = "skill-eval") -> k8s_client.V1Pod:
    """Create a minimal V1Pod fixture for _create_pod unit tests."""
    return k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
        spec=k8s_client.V1PodSpec(containers=[k8s_client.V1Container(name="main", image="ubuntu:24.04")]),
    )


def test_gke_environment_create_pod_fails_closed_on_service_account_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when ServiceAccount inspection raises a Kubernetes API error without Workload Identity opt-in."""
    monkeypatch.delenv("SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY", raising=False)
    monkeypatch.setattr(GKEEnvironment, "_api", property(lambda self: self._fake_api))

    def raise_forbidden(**_kw: object) -> None:
        raise RuntimeError("403 Forbidden: serviceaccounts 'default' is forbidden")

    env = object.__new__(SkillEvaluatorGKEEnvironment)
    env.namespace = "skill-eval"
    env._kwargs = {}
    env._allow_workload_identity = False
    env._fake_api = SimpleNamespace(read_namespaced_service_account=raise_forbidden)

    with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
        asyncio.run(env._create_pod(_make_pod("pod-forbidden")))


@pytest.mark.parametrize(
    ("allow_workload_identity", "annotations", "expected_metadata_host", "expected_automount", "expect_error"),
    [
        (False, {}, "127.0.0.1:1", False, False),
        (
            False,
            {"iam.gke.io/gcp-service-account": "eval-sa@my-proj.iam.gserviceaccount.com"},
            "127.0.0.1:1",
            False,
            True,
        ),
        (
            True,
            {"iam.gke.io/gcp-service-account": "eval-sa@my-proj.iam.gserviceaccount.com"},
            None,
            None,
            False,
        ),
    ],
)
def test_gke_environment_create_pod_blocks_metadata_host_and_bound_ksa_when_unprivileged(
    monkeypatch: pytest.MonkeyPatch,
    allow_workload_identity: bool,
    annotations: dict[str, str],
    expected_metadata_host: str | None,
    expected_automount: bool | None,
    expect_error: bool,
) -> None:
    """Inject GCE_METADATA_HOST=127.0.0.1:1, disable token automount, and reject bound KSAs unless opted in."""
    created_pods: list[k8s_client.V1Pod] = []

    async def record_create_pod(self: GKEEnvironment, pod: k8s_client.V1Pod) -> None:
        created_pods.append(pod)

    monkeypatch.setattr(GKEEnvironment, "_create_pod", record_create_pod)
    monkeypatch.setattr(GKEEnvironment, "_api", property(lambda self: self._fake_api))
    monkeypatch.delenv("SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY", raising=False)

    env = object.__new__(SkillEvaluatorGKEEnvironment)
    env.namespace = "skill-eval"
    env._kwargs = {"allow_workload_identity": "1"} if allow_workload_identity else {}
    env._allow_workload_identity = allow_workload_identity
    env._persistent_env = {}
    env._fake_api = SimpleNamespace(
        read_namespaced_service_account=lambda **_kw: SimpleNamespace(metadata=SimpleNamespace(annotations=annotations))
    )

    pod = _make_pod("pod-metadata-check")
    if expect_error:
        with pytest.raises(RuntimeError, match="SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1"):
            asyncio.run(env._create_pod(pod))
        assert created_pods == []
        return

    asyncio.run(env._create_pod(pod))

    assert len(created_pods) == 1
    assert pod.spec.automount_service_account_token is expected_automount
    assert env._persistent_env.get("GCE_METADATA_HOST") == expected_metadata_host
    container_env_map = {item.name: item.value for item in (pod.spec.containers[0].env or [])}
    assert container_env_map.get("GCE_METADATA_HOST") == expected_metadata_host


def test_gke_environment_wait_for_container_exec_ready_drains_stream_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain the Kubernetes WSClient stream and retry until returncode is 0 or attempts are exhausted."""
    ws_outcomes: list[int | Exception] = [
        ValueError(
            "invalid literal for int() with base 10: "
            "'error sending request: Post \"https://10.128.0.47:10250/exec/skilleval/pod/main?command=true\"'"
        ),
        1,
        0,
    ]
    ws_attempts = 0
    read_output_calls = 0

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

    def fake_stream(*_args: object, **_kwargs: object) -> FakeWSClient:
        nonlocal ws_attempts
        outcome = ws_outcomes[min(ws_attempts, len(ws_outcomes) - 1)]
        ws_attempts += 1
        return FakeWSClient(outcome)

    async def fast_sleep(_sec: float) -> None:
        return None

    async def noop_check_terminated(self: GKEEnvironment) -> None:
        return None

    monkeypatch.setattr(gke_env_mod, "stream", fake_stream, raising=False)
    monkeypatch.setattr(gke_env_mod.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(GKEEnvironment, "_api", property(lambda self: self._fake_api))
    monkeypatch.setattr(GKEEnvironment, "_check_pod_terminated", noop_check_terminated)

    env = object.__new__(SkillEvaluatorGKEEnvironment)
    env.pod_name = "pod-ready-check"
    env.namespace = "skilleval"
    env._fake_api = SimpleNamespace(connect_get_namespaced_pod_exec=lambda *_a, **_k: None)

    def record_read_output(_resp: object) -> None:
        nonlocal read_output_calls
        read_output_calls += 1

    env._read_exec_output = record_read_output  # type: ignore[method-assign]
    asyncio.run(env._wait_for_container_exec_ready(max_attempts=4))
    assert ws_attempts == 3
    assert read_output_calls == 3

    # Exhausting max_attempts raises RuntimeError
    ws_outcomes[:] = [RuntimeError("kubelet stream refused")]
    ws_attempts = 0
    with pytest.raises(RuntimeError, match="Container not ready for exec after 2 attempts"):
        asyncio.run(env._wait_for_container_exec_ready(max_attempts=2))
    assert ws_attempts == 2


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
