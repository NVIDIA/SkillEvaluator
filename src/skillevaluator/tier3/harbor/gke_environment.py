# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GKE Harbor environment wrapper that enforces pod ServiceAccount and Workload Identity isolation."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import urllib.parse
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from harbor.environments.gke import GKEEnvironment, stream
from kubernetes import client as k8s_client

from skillevaluator.provider_config import (
    CREDENTIAL_EXPIRY_ENV,
    refresh_host_vertex_adc_environment,
    sync_refreshed_adc_persistent_env,
)
from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line
from skillevaluator.utils.redaction import is_sensitive_key, redact_sensitive_text

if TYPE_CHECKING:
    from harbor.environments.base import ExecResult

GKE_ALLOW_WORKLOAD_IDENTITY_ENV = "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY"
GKE_BOUND_SERVICE_ACCOUNT_ERROR_TEMPLATE = (
    "GKE namespace '{namespace}' ServiceAccount '{service_account}' is bound to GCP service account "
    "'{gcp_sa}' via Workload Identity, which exposes pod-level cloud credentials to evaluated skill "
    "commands. Restrict this mode to trusted skills by setting "
    "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1 or --ek allow_workload_identity=1."
)
GKE_HOST_UNVERIFIED_VERTEX_AUTH_DETAIL = (
    "host does not possess Vertex AI credentials; runtime authentication is unverified on host "
    "(pending in-pod GKE Workload Identity probe)"
)
GKE_METADATA_ISOLATION_ERROR_TEMPLATE = (
    "Failed to establish or verify GKE metadata server isolation in namespace '{namespace}' ({detail}). "
    "Unprivileged GKE evaluations must block egress to 169.254.169.254/32 and 169.254.169.252/32 to prevent "
    "direct Workload Identity Federation or node metadata credential exposure. Configure a NetworkPolicy-enforcing "
    "CNI with networking.k8s.io permissions or explicitly opt in for trusted skills via "
    "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1 or --ek allow_workload_identity=1."
)
GKE_METADATA_ISOLATION_LABEL_KEY = "skillevaluator.nvidia.com/metadata-isolated"
GKE_METADATA_ISOLATION_LABEL_VALUE = "true"
GKE_METADATA_NETWORK_POLICY_NAME = "skillevaluator-block-gce-metadata"
GKE_SERVICE_ACCOUNT_INSPECTION_ERROR_TEMPLATE = (
    "Failed to inspect GKE namespace '{namespace}' ServiceAccount '{service_account}' for Workload Identity "
    "isolation ({detail}). Grant 'get' permission on serviceaccounts or explicitly opt in via "
    "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1 or --ek allow_workload_identity=1."
)
SECURE_GKE_ENV_IMPORT_PATH = "skillevaluator.tier3.harbor.gke_environment:SkillEvaluatorGKEEnvironment"

_BLOCKED_GCE_METADATA_CIDRS = ("169.254.169.252/32", "169.254.169.254/32")
_BLOCKED_GCE_METADATA_HOST = "127.0.0.1:1"
_DEFAULT_K8S_REQUEST_TIMEOUT_SEC = 5.0
_EXEC_QUERY_COMMAND_RE = re.compile(r"(?P<sep>[?&])command=[^\s\"']+")
_GKE_DIRECT_WIF_PRINCIPAL_ANNOTATION = "iam.gke.io/return-principal-id-as-email"
_GKE_METADATA_PROBE_URLS = (
    "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
    "http://169.254.169.252:988/computeMetadata/v1/instance/service-accounts/default/token",
)
_GKE_WORKLOAD_IDENTITY_ANNOTATION = "iam.gke.io/gcp-service-account"
_GOOGLE_CREDENTIAL_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:ya29\.[A-Za-z0-9._-]{6,}|AIza[A-Za-z0-9_-]{12,})")
_MAX_TRANSIENT_EXEC_ATTEMPTS = 4
_METADATA_PROBE_HTTP_TIMEOUT_SEC = 2
_METADATA_PROBE_MISSING_TOOL_EXIT_CODE = 43
_METADATA_PROBE_STREAM_TIMEOUT_SEC = 8
_METADATA_REACHABLE_EXIT_CODE = 42
_MIN_EXACT_SECRET_LENGTH = 8
_READINESS_PROBE_STREAM_TIMEOUT_SEC = 5
_TRAILING_QUOTES_RE = re.compile(r"[\s\"']+\Z")
_TRANSIENT_KUBELET_EXEC_ERROR_SNIPPETS = (
    "error sending request:",
    "No agent available",
    "unable to upgrade connection",
)


def _build_in_pod_metadata_isolation_probe_script() -> str:
    """Build the shell script that verifies GKE/GCE metadata endpoints are unreachable inside the pod."""
    py_urls = ", ".join(f'"{url}"' for url in _GKE_METADATA_PROBE_URLS)
    sh_urls = " ".join(_GKE_METADATA_PROBE_URLS)
    return (
        "if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then "
        "PY=$(command -v python3 || command -v python); "
        f"$PY -c '"
        "import sys, urllib.error, urllib.request\n"
        f"for url in ({py_urls}):\n"
        '    req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})\n'
        "    try:\n"
        f"        with urllib.request.urlopen(req, timeout={_METADATA_PROBE_HTTP_TIMEOUT_SEC}):\n"
        f"            sys.exit({_METADATA_REACHABLE_EXIT_CODE})\n"
        "    except urllib.error.HTTPError:\n"
        f"        sys.exit({_METADATA_REACHABLE_EXIT_CODE})\n"
        "    except Exception:\n"
        "        pass\n"
        "sys.exit(0)"
        "'; "
        "elif command -v curl >/dev/null 2>&1; then "
        f"for u in {sh_urls}; do "
        f'code=$(curl -s -o /dev/null -m {_METADATA_PROBE_HTTP_TIMEOUT_SEC} -w "%{{http_code}}" '
        '-H "Metadata-Flavor: Google" "$u" 2>/dev/null || true); '
        f'if [ -n "$code" ] && [ "$code" != "000" ]; then exit {_METADATA_REACHABLE_EXIT_CODE}; fi; '
        "done; exit 0; "
        "elif command -v wget >/dev/null 2>&1; then "
        f"for u in {sh_urls}; do "
        f'if wget -S -T {_METADATA_PROBE_HTTP_TIMEOUT_SEC} --header="Metadata-Flavor: Google" '
        f'-O /dev/null "$u" 2>&1 | grep -q "HTTP/"; then exit {_METADATA_REACHABLE_EXIT_CODE}; fi; '
        "done; exit 0; "
        f"else exit {_METADATA_PROBE_MISSING_TOOL_EXIT_CODE}; fi"
    )


_IN_POD_METADATA_ISOLATION_PROBE_SCRIPT = _build_in_pod_metadata_isolation_probe_script()


def coerce_gke_opt_in_flag(value: object) -> bool:
    """Return True when an opt-in flag value represents an explicit truthy setting."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def is_gke_workload_identity_allowed(
    environment_kwargs: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return True only when the operator explicitly opts into GKE Workload Identity."""
    for candidate_env in (env, os.environ):
        if candidate_env is not None and coerce_gke_opt_in_flag(candidate_env.get(GKE_ALLOW_WORKLOAD_IDENTITY_ENV)):
            return True
    return bool(environment_kwargs and coerce_gke_opt_in_flag(environment_kwargs.get("allow_workload_identity")))


def _is_transient_kubelet_exec_error(stderr: str | None) -> bool:
    """Return True when stderr indicates a transient GKE Konnectivity or kubelet exec failure."""
    if not stderr:
        return False
    return any(snippet in stderr for snippet in _TRANSIENT_KUBELET_EXEC_ERROR_SNIPPETS)


def _collect_secret_variants(env_values: Mapping[str, str] | None) -> list[str]:
    """Extract sensitive environment values and their shell/URL-encoded representations."""
    if not env_values:
        return []
    variants: set[str] = set()
    for key, raw_value in env_values.items():
        if not raw_value or not is_sensitive_key(str(key)):
            continue
        secret = str(raw_value).strip()
        if len(secret) < _MIN_EXACT_SECRET_LENGTH:
            continue
        shell_quoted = shlex.quote(secret)
        for candidate in (
            secret,
            shell_quoted,
            shell_quoted.strip("'"),
            urllib.parse.quote(secret, safe=""),
            urllib.parse.quote_plus(secret),
            urllib.parse.quote(shell_quoted, safe=""),
            urllib.parse.quote_plus(shell_quoted),
        ):
            if len(candidate) >= _MIN_EXACT_SECRET_LENGTH:
                variants.add(candidate)
    return sorted(variants, key=len, reverse=True)


def _redact_truncated_secret_suffix(text: str, secret_variants: list[str]) -> str:
    """Scrub trailing secret prefixes when CPython int() error formatting truncates stderr at 200 chars."""
    if not secret_variants:
        return text
    trailing_match = _TRAILING_QUOTES_RE.search(text)
    if trailing_match:
        stem = text[: trailing_match.start()]
        suffix = text[trailing_match.start() :]
    else:
        stem = text
        suffix = ""
    for candidate in secret_variants:
        max_prefix = min(len(candidate) - 1, len(stem))
        for prefix_len in range(max_prefix, _MIN_EXACT_SECRET_LENGTH - 1, -1):
            if stem.endswith(candidate[:prefix_len]):
                return stem[: len(stem) - prefix_len] + "<redacted>" + suffix
    return text


def _redact_exec_stderr(stderr: str | None, env_values: Mapping[str, str] | None = None) -> str | None:
    """Scrub kubelet exec query strings, encoded secrets, and credential patterns from stderr."""
    if not stderr:
        return stderr
    redacted = _EXEC_QUERY_COMMAND_RE.sub(r"\g<sep>command=<redacted>", stderr)
    secret_variants = _collect_secret_variants(env_values)
    for secret in secret_variants:
        if secret in redacted:
            redacted = redacted.replace(secret, "<redacted>")
    redacted = _redact_truncated_secret_suffix(redacted, secret_variants)
    redacted = redact_secrets_in_log_line(redacted, extra_secret_values=secret_variants)
    redacted = redact_sensitive_text(redacted)
    return _GOOGLE_CREDENTIAL_RE.sub("<redacted>", redacted)


def _call_k8s_with_timeout(
    fn: Any,
    *,
    request_timeout: float = _DEFAULT_K8S_REQUEST_TIMEOUT_SEC,
    **kwargs: Any,
) -> Any:
    """Invoke a Kubernetes client method with _request_timeout, falling back when fakes omit the parameter."""
    try:
        return fn(**kwargs, _request_timeout=request_timeout)
    except TypeError:
        return fn(**kwargs)


def inspect_bound_gcp_service_account(
    api: Any,
    *,
    namespace: str,
    service_account: str = "default",
    request_timeout: float = _DEFAULT_K8S_REQUEST_TIMEOUT_SEC,
) -> str | None:
    """Return the bound GCP identity if the Kubernetes ServiceAccount has Workload Identity annotations."""
    read_sa = getattr(api, "read_namespaced_service_account", None)
    if not callable(read_sa):
        return None
    try:
        sa_obj = _call_k8s_with_timeout(
            read_sa,
            request_timeout=request_timeout,
            name=service_account,
            namespace=namespace,
        )
    except Exception as exc:
        raise RuntimeError(
            GKE_SERVICE_ACCOUNT_INSPECTION_ERROR_TEMPLATE.format(
                namespace=namespace,
                service_account=service_account,
                detail=str(exc),
            )
        ) from exc

    metadata = getattr(sa_obj, "metadata", None)
    annotations = getattr(metadata, "annotations", None)
    if isinstance(annotations, dict):
        bound = str(annotations.get(_GKE_WORKLOAD_IDENTITY_ANNOTATION, "")).strip()
        if bound:
            return bound
        if coerce_gke_opt_in_flag(annotations.get(_GKE_DIRECT_WIF_PRINCIPAL_ANNOTATION)):
            return f"{_GKE_DIRECT_WIF_PRINCIPAL_ANNOTATION}=true"
    return None


def _build_metadata_blocking_network_policy(namespace: str) -> k8s_client.V1NetworkPolicy:
    """Build a pod-scoped egress NetworkPolicy blocking GCE and GKE metadata server IPs."""
    return k8s_client.V1NetworkPolicy(
        metadata=k8s_client.V1ObjectMeta(
            name=GKE_METADATA_NETWORK_POLICY_NAME,
            namespace=namespace,
        ),
        spec=k8s_client.V1NetworkPolicySpec(
            pod_selector=k8s_client.V1LabelSelector(
                match_labels={GKE_METADATA_ISOLATION_LABEL_KEY: GKE_METADATA_ISOLATION_LABEL_VALUE}
            ),
            policy_types=["Egress"],
            egress=[
                k8s_client.V1NetworkPolicyEgressRule(
                    to=[
                        k8s_client.V1NetworkPolicyPeer(
                            ip_block=k8s_client.V1IPBlock(
                                cidr="0.0.0.0/0",
                                _except=list(_BLOCKED_GCE_METADATA_CIDRS),
                            )
                        ),
                        k8s_client.V1NetworkPolicyPeer(
                            ip_block=k8s_client.V1IPBlock(
                                cidr="::/0",
                            )
                        ),
                    ]
                )
            ],
        ),
    )


def _network_policy_blocks_metadata(policy: Any) -> bool:
    """Return True if a V1NetworkPolicy enforces egress isolation against both GKE metadata CIDRs."""
    spec = getattr(policy, "spec", None)
    if spec is None:
        return False
    policy_types = getattr(spec, "policy_types", None) or []
    if "Egress" not in policy_types:
        return False
    selector = getattr(spec, "pod_selector", None)
    match_labels = getattr(selector, "match_labels", None)
    if not isinstance(match_labels, dict):
        return False
    if match_labels.get(GKE_METADATA_ISOLATION_LABEL_KEY) != GKE_METADATA_ISOLATION_LABEL_VALUE:
        return False
    egress_rules = getattr(spec, "egress", None) or []
    if not egress_rules:
        return False
    required_cidrs = set(_BLOCKED_GCE_METADATA_CIDRS)
    saw_ipv4_rule = False
    for rule in egress_rules:
        peers = getattr(rule, "to", None) or []
        if not peers:
            return False
        for peer in peers:
            ip_block = getattr(peer, "ip_block", None)
            if ip_block is None:
                return False
            cidr = str(getattr(ip_block, "cidr", "") or "").strip()
            if cidr == "0.0.0.0/0":
                saw_ipv4_rule = True
                exceptions = {str(item).strip() for item in (getattr(ip_block, "_except", None) or [])}
                if not required_cidrs.issubset(exceptions):
                    return False
    return saw_ipv4_rule


def _is_k8s_http_status(exc: Exception, status_code: int, phrase: str) -> bool:
    """Return True when a Kubernetes exception matches the given HTTP status code or status phrase."""
    if getattr(exc, "status", None) == status_code:
        return True
    text = str(exc)
    return str(status_code) in text and phrase.lower() in text.lower()


def _validate_existing_metadata_network_policy(policy: Any, *, namespace: str) -> None:
    """Raise RuntimeError if a NetworkPolicy object does not block both GKE metadata CIDRs."""
    if not _network_policy_blocks_metadata(policy):
        raise RuntimeError(
            GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                namespace=namespace,
                detail=(
                    f"existing NetworkPolicy '{GKE_METADATA_NETWORK_POLICY_NAME}' does not block "
                    f"{', '.join(_BLOCKED_GCE_METADATA_CIDRS)}"
                ),
            )
        )


def ensure_gke_metadata_network_policy(
    networking_api: Any,
    *,
    namespace: str,
    request_timeout: float = _DEFAULT_K8S_REQUEST_TIMEOUT_SEC,
) -> None:
    """Verify or create the pod-scoped GKE metadata-blocking NetworkPolicy in the target namespace."""
    read_np = getattr(networking_api, "read_namespaced_network_policy", None)
    create_np = getattr(networking_api, "create_namespaced_network_policy", None)
    if not callable(read_np) or not callable(create_np):
        raise RuntimeError(
            GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                namespace=namespace,
                detail="Kubernetes NetworkingV1Api is unavailable to verify or create metadata NetworkPolicy",
            )
        )
    try:
        existing = _call_k8s_with_timeout(
            read_np,
            request_timeout=request_timeout,
            name=GKE_METADATA_NETWORK_POLICY_NAME,
            namespace=namespace,
        )
    except Exception as exc:
        existing = exc

    if not isinstance(existing, Exception):
        _validate_existing_metadata_network_policy(existing, namespace=namespace)
        return

    if not _is_k8s_http_status(existing, 404, "not found"):
        raise RuntimeError(
            GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                namespace=namespace,
                detail=f"failed to read NetworkPolicy '{GKE_METADATA_NETWORK_POLICY_NAME}': {existing}",
            )
        ) from existing

    policy_body = _build_metadata_blocking_network_policy(namespace)
    try:
        _call_k8s_with_timeout(
            create_np,
            request_timeout=request_timeout,
            namespace=namespace,
            body=policy_body,
        )
        return
    except Exception as exc:
        if not _is_k8s_http_status(exc, 409, "conflict"):
            raise RuntimeError(
                GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                    namespace=namespace,
                    detail=f"failed to create NetworkPolicy '{GKE_METADATA_NETWORK_POLICY_NAME}': {exc}",
                )
            ) from exc

    try:
        reread = _call_k8s_with_timeout(
            read_np,
            request_timeout=request_timeout,
            name=GKE_METADATA_NETWORK_POLICY_NAME,
            namespace=namespace,
        )
    except Exception as exc:
        raise RuntimeError(
            GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                namespace=namespace,
                detail=f"failed to re-read NetworkPolicy '{GKE_METADATA_NETWORK_POLICY_NAME}' after 409 Conflict: {exc}",
            )
        ) from exc
    _validate_existing_metadata_network_policy(reread, namespace=namespace)


class SkillEvaluatorGKEEnvironment(GKEEnvironment):
    """Enforce least-privilege pod identity, exec readiness, and ADC refresh for GKE evaluation pods."""

    def __init__(
        self,
        *args: Any,
        allow_workload_identity: bool | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GKE environment while recording Workload Identity opt-in state."""
        self._allow_workload_identity = (
            coerce_gke_opt_in_flag(allow_workload_identity) or is_gke_workload_identity_allowed()
        )
        super().__init__(*args, **kwargs)

    def _is_workload_identity_enabled(self) -> bool:
        """Return True when the operator explicitly opted into GKE Workload Identity."""
        if getattr(self, "_allow_workload_identity", False):
            return True
        extra_kwargs = getattr(self, "_kwargs", None)
        return is_gke_workload_identity_allowed(extra_kwargs if isinstance(extra_kwargs, Mapping) else None)

    async def _create_pod(self, pod: k8s_client.V1Pod) -> None:
        """Disable service account token automount, block metadata server, and reject GCP-bound KSAs unless opted in."""
        if not self._is_workload_identity_enabled():
            if getattr(pod, "metadata", None) is None:
                pod.metadata = k8s_client.V1ObjectMeta()
            labels = dict(getattr(pod.metadata, "labels", None) or {})
            labels[GKE_METADATA_ISOLATION_LABEL_KEY] = GKE_METADATA_ISOLATION_LABEL_VALUE
            pod.metadata.labels = labels

            if getattr(pod, "spec", None) is not None:
                pod.spec.automount_service_account_token = False
                pod.spec.host_network = False
                sa_name = (
                    getattr(pod.spec, "service_account_name", None)
                    or getattr(pod.spec, "service_account", None)
                    or "default"
                )
                for container in getattr(pod.spec, "containers", None) or []:
                    env_list = list(getattr(container, "env", None) or [])
                    if not any(getattr(item, "name", None) == "GCE_METADATA_HOST" for item in env_list):
                        env_list.append(k8s_client.V1EnvVar(name="GCE_METADATA_HOST", value=_BLOCKED_GCE_METADATA_HOST))
                        container.env = env_list
            else:
                sa_name = "default"
            persistent = getattr(self, "_persistent_env", None)
            if isinstance(persistent, dict):
                persistent.setdefault("GCE_METADATA_HOST", _BLOCKED_GCE_METADATA_HOST)
            api = getattr(self, "_api", None)
            namespace = getattr(self, "namespace", "default")
            if api is None:
                raise RuntimeError(
                    GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                        namespace=namespace,
                        detail="Kubernetes CoreV1Api client is not initialized",
                    )
                )
            bound_gcp_sa = await asyncio.to_thread(
                inspect_bound_gcp_service_account,
                api,
                namespace=namespace,
                service_account=sa_name,
            )
            if bound_gcp_sa:
                raise RuntimeError(
                    GKE_BOUND_SERVICE_ACCOUNT_ERROR_TEMPLATE.format(
                        namespace=namespace,
                        service_account=sa_name,
                        gcp_sa=bound_gcp_sa,
                    )
                )
            networking_api = getattr(self, "_networking_api", None)
            if networking_api is None:
                api_client = getattr(api, "api_client", None)
                if api_client is not None:
                    networking_api = k8s_client.NetworkingV1Api(api_client)
            await asyncio.to_thread(
                ensure_gke_metadata_network_policy,
                networking_api,
                namespace=namespace,
            )
        await super()._create_pod(pod)

    async def _delete_unisolated_pod_best_effort(self) -> None:
        """Delete the evaluation pod on metadata isolation verification failure."""
        api = getattr(self, "_api", None)
        delete_pod = getattr(api, "delete_namespaced_pod", None)
        if callable(delete_pod):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    delete_pod,
                    name=self.pod_name,
                    namespace=self.namespace,
                )

    async def _exec_pod_stream_command(self, command: list[str], *, timeout_sec: int) -> int | None:
        """Execute a command via Kubernetes WebSocket exec stream, draining output and returning returncode."""
        stream_kwargs: dict[str, Any] = {
            "command": command,
            "stderr": True,
            "stdin": False,
            "stdout": True,
            "tty": False,
            "_preload_content": False,
        }
        if getattr(self, "_compose_mode", False):
            stream_kwargs["container"] = "dind"

        resp = None
        try:
            resp = await asyncio.to_thread(
                stream,
                self._api.connect_get_namespaced_pod_exec,
                self.pod_name,
                getattr(self, "namespace", "default"),
                **stream_kwargs,
            )
            if hasattr(self, "_read_exec_output") and resp.is_open():
                await asyncio.to_thread(self._read_exec_output, resp)
            await asyncio.to_thread(resp.run_forever, timeout_sec)
            return resp.returncode
        finally:
            if resp is not None:
                with contextlib.suppress(Exception):
                    resp.close()

    async def _verify_in_pod_metadata_isolation(self) -> None:
        """Verify inside the running pod that GCE/GKE metadata endpoints are unreachable."""
        namespace = getattr(self, "namespace", "default")
        try:
            rc = await self._exec_pod_stream_command(
                ["sh", "-c", _IN_POD_METADATA_ISOLATION_PROBE_SCRIPT],
                timeout_sec=_METADATA_PROBE_STREAM_TIMEOUT_SEC,
            )
        except Exception as exc:
            await self._delete_unisolated_pod_best_effort()
            raise RuntimeError(
                GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                    namespace=namespace,
                    detail=f"in-pod metadata isolation probe failed to execute: {exc}",
                )
            ) from exc

        if rc not in (None, 0):
            await self._delete_unisolated_pod_best_effort()
            if rc == _METADATA_REACHABLE_EXIT_CODE:
                detail = (
                    "in-pod probe reached the GKE/GCE metadata server (169.254.169.254 or 169.254.169.252:988) "
                    "despite NetworkPolicy; cluster CNI may not enforce egress NetworkPolicy"
                )
            else:
                detail = (
                    f"in-pod metadata isolation probe exited with status {rc} (missing python/curl/wget or probe error)"
                )
            raise RuntimeError(
                GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                    namespace=namespace,
                    detail=detail,
                )
            )

    async def _wait_for_container_exec_ready(self, max_attempts: int = 60) -> None:
        """Wait until the GKE kubelet accepts exec streams and verify metadata server isolation."""
        for attempt in range(max_attempts):
            await self._check_pod_terminated()
            try:
                rc = await self._exec_pod_stream_command(
                    ["true"],
                    timeout_sec=_READINESS_PROBE_STREAM_TIMEOUT_SEC,
                )
                if rc not in (None, 0):
                    raise RuntimeError(f"Container readiness probe returned exit code {rc}")
                break
            except Exception as exc:
                if attempt >= max_attempts - 1:
                    raise RuntimeError(f"Container not ready for exec after {max_attempts} attempts: {exc}") from exc
                await asyncio.sleep(3)

        if not self._is_workload_identity_enabled():
            await self._verify_in_pod_metadata_isolation()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Refresh ADC-sourced Vertex OpenAPI credentials and retry transient kubelet exec errors."""
        merged = dict(self._merge_env(env) or {})
        fresh_token = await asyncio.to_thread(
            refresh_host_vertex_adc_environment,
            merged,
            fallback_env=os.environ,
            require_existing_api_key=True,
            fail_on_expired=True,
            update_os_environ=True,
        )
        merged.pop(CREDENTIAL_EXPIRY_ENV, None)
        updated_persistent = sync_refreshed_adc_persistent_env(
            getattr(self, "_persistent_env", None),
            fresh_token=fresh_token,
        )
        if env is not None and CREDENTIAL_EXPIRY_ENV in env:
            env = dict(env)
            env.pop(CREDENTIAL_EXPIRY_ENV, None)
        if fresh_token:
            merged["OPENAI_API_KEY"] = fresh_token
            if env is not None and env.get("OPENAI_API_KEY", "").strip():
                env = dict(env)
                env["OPENAI_API_KEY"] = fresh_token
            elif not updated_persistent:
                env = dict(env or {})
                env["OPENAI_API_KEY"] = fresh_token

        result: ExecResult | None = None
        for attempt in range(_MAX_TRANSIENT_EXEC_ATTEMPTS):
            result = await super().exec(
                command=command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )
            if result.return_code == 0 or not _is_transient_kubelet_exec_error(result.stderr):
                break
            if attempt < _MAX_TRANSIENT_EXEC_ATTEMPTS - 1:
                await asyncio.sleep(1.5 * (attempt + 1))

        assert result is not None
        if result.stderr:
            result.stderr = _redact_exec_stderr(result.stderr, merged)
        return result
