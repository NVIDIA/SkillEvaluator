# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GKE Harbor environment wrapper that enforces pod ServiceAccount and Workload Identity isolation."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import shutil
import subprocess
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
GKE_AUTOPILOT_ENV = "SKILLEVALUATOR_GKE_AUTOPILOT"
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
GKE_METADATA_PROBE_CONTAINER_NAME = "skillevaluator-metadata-probe"
GKE_METADATA_PROBE_INIT_CONTAINER_NAME = "skillevaluator-metadata-probe-init"
GKE_METADATA_PROBE_IMAGE_ENV = "SKILLEVALUATOR_GKE_METADATA_PROBE_IMAGE"
GKE_SERVICE_ACCOUNT_INSPECTION_ERROR_TEMPLATE = (
    "Failed to inspect GKE namespace '{namespace}' ServiceAccount '{service_account}' for Workload Identity "
    "isolation ({detail}). Grant 'get' permission on serviceaccounts or explicitly opt in via "
    "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1 or --ek allow_workload_identity=1."
)
SECURE_GKE_ENV_IMPORT_PATH = "skillevaluator.tier3.harbor.gke_environment:SkillEvaluatorGKEEnvironment"

_ACCELERATOR_RESOURCE_PREFIXES = ("nvidia.com/gpu", "google.com/tpu")
_AUTOPILOT_DETECTION_CACHE: dict[tuple[str, str], bool] = {}
_AUTOPILOT_MAX_EPHEMERAL_STORAGE_MIB = 10240
_BLOCKED_GCE_METADATA_CIDRS = ("169.254.169.252/32", "169.254.169.254/32")
_BLOCKED_GCE_METADATA_HOST = "127.0.0.1:1"
_DEFAULT_GKE_METADATA_PROBE_IMAGE = "python:3.12-slim"
_DEFAULT_K8S_REQUEST_TIMEOUT_SEC = 5.0
_DNS_PORT = 53
_DNS_PROTOCOLS = frozenset({"UDP", "TCP"})
_EXEC_QUERY_COMMAND_RE = re.compile(r"(?P<sep>[?&])command=[^\s\"']+")
_GKE_CLOUD_DNS_METADATA_CIDR = "169.254.169.254/32"
_GKE_DIRECT_WIF_PRINCIPAL_ANNOTATION = "iam.gke.io/return-principal-id-as-email"
_GKE_METADATA_PROBE_URLS = (
    "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
    "http://169.254.169.252:988/computeMetadata/v1/instance/service-accounts/default/token",
)
_GKE_WORKLOAD_IDENTITY_ANNOTATION = "iam.gke.io/gcp-service-account"
_GOOGLE_CREDENTIAL_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:ya29\.[A-Za-z0-9._-]{6,}|AIza[A-Za-z0-9_-]{12,})")
_KUBE_DNS_POD_LABEL_KEY = "k8s-app"
_KUBE_DNS_POD_LABEL_VALUE = "kube-dns"
_KUBE_SYSTEM_NAMESPACE_LABEL_KEY = "kubernetes.io/metadata.name"
_KUBE_SYSTEM_NAMESPACE_LABEL_VALUE = "kube-system"
_MAX_TRANSIENT_EXEC_ATTEMPTS = 4
_METADATA_PROBE_HTTP_TIMEOUT_SEC = 2
_METADATA_PROBE_ISOLATED_SENTINEL = "SKILLEVALUATOR_METADATA_ISOLATED_OK"
_METADATA_PROBE_MISSING_TOOL_EXIT_CODE = 43
_METADATA_PROBE_STREAM_TIMEOUT_SEC = 8
_METADATA_REACHABLE_EXIT_CODE = 42
_MIN_EXACT_SECRET_LENGTH = 8
_PROBE_CONTAINER_CPU_REQUEST = "10m"
_PROBE_CONTAINER_EPHEMERAL_STORAGE_MIB = 64
_PROBE_CONTAINER_MEMORY_REQUEST = "32Mi"
_READINESS_PROBE_STREAM_TIMEOUT_SEC = 5
_STORAGE_QUANTITY_RE = re.compile(r"^\s*(?P<amount>\d+)\s*(?P<unit>Mi|Gi|M|G)?\s*$")
_TRAILING_QUOTES_RE = re.compile(r"[\s\"']+\Z")
_TRANSIENT_KUBELET_EXEC_ERROR_SNIPPETS = (
    "error sending request:",
    "No agent available",
    "unable to upgrade connection",
)
_WGET_EXPECTED_FAILURE_PATTERN = (
    "Connection refused|timed out|can.t connect to remote host|Network is unreachable|No route to host|bad address"
)


def _build_in_pod_metadata_isolation_probe_script(
    urls: tuple[str, ...] = _GKE_METADATA_PROBE_URLS,
) -> str:
    """Build the shell script that verifies GKE/GCE metadata endpoints are unreachable inside the pod."""
    py_urls = ", ".join(f'"{url}"' for url in urls)
    sh_urls = " ".join(shlex.quote(url) for url in urls)
    return (
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY; "
        "if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then "
        "PY=$(command -v python3 || command -v python); "
        f"out=$($PY -I -c '"
        "import sys, urllib.error, urllib.request\n"
        "opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
        f"for url in [{py_urls}]:\n"
        "    try:\n"
        '        req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})\n'
        f"        with opener.open(req, timeout={_METADATA_PROBE_HTTP_TIMEOUT_SEC}):\n"
        f"            sys.exit({_METADATA_REACHABLE_EXIT_CODE})\n"
        "    except urllib.error.HTTPError:\n"
        f"        sys.exit({_METADATA_REACHABLE_EXIT_CODE})\n"
        "    except Exception:\n"
        "        pass\n"
        f'print("{_METADATA_PROBE_ISOLATED_SENTINEL}")\n'
        "sys.exit(0)"
        "' 2>/dev/null); rc=$?; "
        f'if [ "$rc" -eq {_METADATA_REACHABLE_EXIT_CODE} ]; then exit {_METADATA_REACHABLE_EXIT_CODE}; fi; '
        f'if [ "$rc" -ne 0 ] || [ "$out" != "{_METADATA_PROBE_ISOLATED_SENTINEL}" ]; then '
        f"exit {_METADATA_PROBE_MISSING_TOOL_EXIT_CODE}; fi; "
        "exit 0; "
        "elif command -v curl >/dev/null 2>&1; then "
        f"for u in {sh_urls}; do "
        f'code=$(curl --noproxy "*" -s -o /dev/null -m {_METADATA_PROBE_HTTP_TIMEOUT_SEC} '
        '-w "HTTP_CODE:%{http_code}" -H "Metadata-Flavor: Google" "$u" 2>/dev/null || true); '
        'case "$code" in '
        '"HTTP_CODE:000") ;; '
        f'"HTTP_CODE:"*) exit {_METADATA_REACHABLE_EXIT_CODE} ;; '
        f"*) exit {_METADATA_PROBE_MISSING_TOOL_EXIT_CODE} ;; "
        "esac; "
        "done; exit 0; "
        "elif command -v wget >/dev/null 2>&1; then "
        f"for u in {sh_urls}; do "
        f'out=$(wget -S -T {_METADATA_PROBE_HTTP_TIMEOUT_SEC} --header="Metadata-Flavor: Google" '
        '-O /dev/null "$u" 2>&1 || true); '
        'if [ -z "$out" ]; then '
        f"exit {_METADATA_PROBE_MISSING_TOOL_EXIT_CODE}; fi; "
        'if printf "%s\\n" "$out" | grep -q "HTTP/"; then '
        f"exit {_METADATA_REACHABLE_EXIT_CODE}; fi; "
        f'if ! printf "%s\\n" "$out" | grep -Eiq "{_WGET_EXPECTED_FAILURE_PATTERN}"; then '
        f"exit {_METADATA_PROBE_MISSING_TOOL_EXIT_CODE}; fi; "
        "done; exit 0; "
        f"else exit {_METADATA_PROBE_MISSING_TOOL_EXIT_CODE}; fi"
    )


_IN_POD_METADATA_ISOLATION_PROBE_SCRIPT = _build_in_pod_metadata_isolation_probe_script()


def _get_metadata_probe_image(env: Mapping[str, str] | None = None) -> str:
    """Return the trusted container image used for the GKE metadata isolation probe."""
    for candidate_env in (env, os.environ):
        if candidate_env is not None:
            configured = str(candidate_env.get(GKE_METADATA_PROBE_IMAGE_ENV) or "").strip()
            if configured:
                return configured
    return _DEFAULT_GKE_METADATA_PROBE_IMAGE


def _parse_storage_mib(quantity: object) -> int | None:
    """Parse a Kubernetes storage quantity string (Mi/Gi/M/G) into MiB."""
    if not isinstance(quantity, str):
        return None
    match = _STORAGE_QUANTITY_RE.match(quantity)
    if not match:
        return None
    amount = int(match.group("amount"))
    unit = match.group("unit") or "Mi"
    if unit in ("Gi", "G"):
        return amount * 1024
    return amount


def _has_accelerator_resource(resources: Any) -> bool:
    """Return True if a V1ResourceRequirements requests or limits GPUs or TPUs."""
    for attr_name in ("requests", "limits"):
        mapping = getattr(resources, attr_name, None)
        if not isinstance(mapping, dict):
            continue
        for key, val in mapping.items():
            key_str = str(key).strip()
            val_str = str(val or "").strip()
            if val_str and val_str != "0" and any(key_str.startswith(p) for p in _ACCELERATOR_RESOURCE_PREFIXES):
                return True
    return False


def _adjust_autopilot_ephemeral_storage_for_probe(main_container: Any) -> None:
    """Cap main container 10Gi ephemeral-storage so adding the 64Mi probe container stays within Autopilot's 10Gi ceiling."""
    resources = getattr(main_container, "resources", None)
    if resources is None or _has_accelerator_resource(resources):
        return
    max_main_mib = _AUTOPILOT_MAX_EPHEMERAL_STORAGE_MIB - _PROBE_CONTAINER_EPHEMERAL_STORAGE_MIB
    for attr_name in ("requests", "limits"):
        mapping = getattr(resources, attr_name, None)
        if not isinstance(mapping, dict):
            continue
        current_mib = _parse_storage_mib(mapping.get("ephemeral-storage"))
        if current_mib is not None and max_main_mib < current_mib <= _AUTOPILOT_MAX_EPHEMERAL_STORAGE_MIB:
            mapping["ephemeral-storage"] = f"{max_main_mib}Mi"


def _build_metadata_probe_init_container(env: Mapping[str, str] | None = None) -> k8s_client.V1Container:
    """Build the trusted init container that verifies GKE metadata isolation before any application container starts."""
    return k8s_client.V1Container(
        name=GKE_METADATA_PROBE_INIT_CONTAINER_NAME,
        image=_get_metadata_probe_image(env),
        command=["sh", "-c", _IN_POD_METADATA_ISOLATION_PROBE_SCRIPT],
        env=[k8s_client.V1EnvVar(name="GCE_METADATA_HOST", value=_BLOCKED_GCE_METADATA_HOST)],
        resources=None,
        volume_mounts=[],
    )


def _build_metadata_probe_container(env: Mapping[str, str] | None = None) -> k8s_client.V1Container:
    """Build the trusted companion container used to verify GKE metadata isolation in the pod network namespace."""
    probe_storage = f"{_PROBE_CONTAINER_EPHEMERAL_STORAGE_MIB}Mi"
    return k8s_client.V1Container(
        name=GKE_METADATA_PROBE_CONTAINER_NAME,
        image=_get_metadata_probe_image(env),
        command=["sleep", "infinity"],
        env=[k8s_client.V1EnvVar(name="GCE_METADATA_HOST", value=_BLOCKED_GCE_METADATA_HOST)],
        resources=k8s_client.V1ResourceRequirements(
            requests={
                "cpu": _PROBE_CONTAINER_CPU_REQUEST,
                "memory": _PROBE_CONTAINER_MEMORY_REQUEST,
                "ephemeral-storage": probe_storage,
            },
            limits={
                "memory": _PROBE_CONTAINER_MEMORY_REQUEST,
                "ephemeral-storage": probe_storage,
            },
        ),
        volume_mounts=[],
    )


def _extract_pod_missing_exec_detail(exc: BaseException) -> str | None:
    """Return a diagnostic message if an exception chain indicates the target pod no longer exists."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, Exception) and _is_k8s_http_status(cur, 404, "not found"):
            return str(cur)
        text = str(cur)
        lowered = text.lower()
        if "cannot connect to pod" in lowered and "not found" in lowered:
            return text
        if 'pods "' in lowered and "not found" in lowered:
            return text
        if "pod " in lowered and "no longer exists" in lowered:
            return text
        cur = cur.__cause__ or cur.__context__
    return None


def _ensure_default_exec_container_routing(api: Any, *, default_container: str = "main") -> None:
    """Ensure untargeted connect_get_namespaced_pod_exec calls default to the main task container."""
    orig = getattr(api, "connect_get_namespaced_pod_exec", None)
    if not callable(orig) or getattr(orig, "_skillevaluator_default_container_wrapped", False):
        return

    def _wrapped_exec(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("container") is None:
            kwargs["container"] = default_container
        try:
            return orig(*args, **kwargs)
        except Exception as exc:
            missing_detail = _extract_pod_missing_exec_detail(exc)
            if missing_detail is not None:
                pod_name = str(args[0] if len(args) > 0 else kwargs.get("name", "unknown"))
                namespace = str(args[1] if len(args) > 1 else kwargs.get("namespace", "default"))
                raise RuntimeError(
                    f"Pod {pod_name} in namespace {namespace} no longer exists (deleted or preempted): {missing_detail}"
                ) from exc
            raise

    _wrapped_exec._skillevaluator_default_container_wrapped = True  # type: ignore[attr-defined]
    _wrapped_exec.__self__ = getattr(orig, "__self__", api)  # type: ignore[attr-defined]
    with contextlib.suppress(Exception):
        api.connect_get_namespaced_pod_exec = _wrapped_exec


def parse_optional_bool_flag(value: object) -> bool | None:
    """Parse an optional boolean setting (1/true/yes vs 0/false/no), returning None when unset."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes"}:
        return True
    if cleaned in {"0", "false", "no"}:
        return False
    return None


def coerce_gke_opt_in_flag(value: object) -> bool:
    """Return True when an opt-in flag value represents an explicit truthy setting."""
    return bool(parse_optional_bool_flag(value))


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
    """Build a pod-scoped egress NetworkPolicy blocking GCE/GKE metadata HTTP while preserving DNS."""
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
                    ]
                ),
                k8s_client.V1NetworkPolicyEgressRule(
                    ports=[
                        k8s_client.V1NetworkPolicyPort(port=_DNS_PORT, protocol="UDP"),
                        k8s_client.V1NetworkPolicyPort(port=_DNS_PORT, protocol="TCP"),
                    ],
                    to=[
                        k8s_client.V1NetworkPolicyPeer(
                            ip_block=k8s_client.V1IPBlock(
                                cidr=_GKE_CLOUD_DNS_METADATA_CIDR,
                            )
                        ),
                        k8s_client.V1NetworkPolicyPeer(
                            namespace_selector=k8s_client.V1LabelSelector(
                                match_labels={_KUBE_SYSTEM_NAMESPACE_LABEL_KEY: _KUBE_SYSTEM_NAMESPACE_LABEL_VALUE}
                            ),
                            pod_selector=k8s_client.V1LabelSelector(
                                match_labels={_KUBE_DNS_POD_LABEL_KEY: _KUBE_DNS_POD_LABEL_VALUE}
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )


def _is_dns_only_egress_rule(rule: Any) -> bool:
    """Return True if an egress rule strictly permits only UDP/TCP port 53 to Cloud DNS or kube-dns."""
    ports = getattr(rule, "ports", None) or []
    if not ports:
        return False
    for port_obj in ports:
        port_val = getattr(port_obj, "port", None)
        if isinstance(port_val, bool) or port_val != _DNS_PORT:
            return False
        end_port = getattr(port_obj, "end_port", None)
        if end_port is not None and (isinstance(end_port, bool) or end_port != _DNS_PORT):
            return False
        protocol = str(getattr(port_obj, "protocol", "") or "TCP").strip().upper()
        if protocol not in _DNS_PROTOCOLS:
            return False

    peers = getattr(rule, "to", None) or []
    if not peers:
        return False
    for peer in peers:
        ip_block = getattr(peer, "ip_block", None)
        ns_sel = getattr(peer, "namespace_selector", None)
        pod_sel = getattr(peer, "pod_selector", None)
        if ip_block is not None:
            if ns_sel is not None or pod_sel is not None:
                return False
            cidr = str(getattr(ip_block, "cidr", "") or "").strip()
            if cidr != _GKE_CLOUD_DNS_METADATA_CIDR:
                return False
            if getattr(ip_block, "_except", None):
                return False
        else:
            if ns_sel is None or pod_sel is None:
                return False
            if getattr(ns_sel, "match_expressions", None) or getattr(pod_sel, "match_expressions", None):
                return False
            ns_labels = getattr(ns_sel, "match_labels", None)
            pod_labels = getattr(pod_sel, "match_labels", None)
            if not isinstance(ns_labels, dict) or not isinstance(pod_labels, dict):
                return False
            if ns_labels.get(_KUBE_SYSTEM_NAMESPACE_LABEL_KEY) != _KUBE_SYSTEM_NAMESPACE_LABEL_VALUE:
                return False
            if pod_labels.get(_KUBE_DNS_POD_LABEL_KEY) != _KUBE_DNS_POD_LABEL_VALUE:
                return False
    return True


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
        if _is_dns_only_egress_rule(rule):
            continue
        for peer in peers:
            if getattr(peer, "namespace_selector", None) is not None or getattr(peer, "pod_selector", None) is not None:
                return False
            ip_block = getattr(peer, "ip_block", None)
            if ip_block is None:
                return False
            cidr = str(getattr(ip_block, "cidr", "") or "").strip()
            if cidr != "0.0.0.0/0":
                return False
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


def _is_autopilot_node(node: Any) -> bool:
    """Return True if a Kubernetes V1Node exhibits GKE Autopilot naming or label conventions."""
    metadata = getattr(node, "metadata", None)
    name = str(getattr(metadata, "name", "") or "").strip()
    if name.startswith("gk3-"):
        return True
    labels = getattr(metadata, "labels", None)
    if isinstance(labels, Mapping):
        if any(str(k).startswith("autopilot.gke.io/") for k in labels):
            return True
        if parse_optional_bool_flag(labels.get("cloud.google.com/gke-autopilot")) is True:
            return True
    return False


class SkillEvaluatorGKEEnvironment(GKEEnvironment):
    """Enforce least-privilege pod identity, exec readiness, and ADC refresh for GKE evaluation pods."""

    def __init__(
        self,
        *args: Any,
        allow_workload_identity: bool | str | None = None,
        autopilot: bool | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GKE environment while recording Workload Identity and Autopilot settings."""
        self._allow_workload_identity = (
            coerce_gke_opt_in_flag(allow_workload_identity) or is_gke_workload_identity_allowed()
        )
        self._autopilot = parse_optional_bool_flag(autopilot)
        self._detected_autopilot: bool | None = None
        super().__init__(*args, **kwargs)

    def _is_workload_identity_enabled(self) -> bool:
        """Return True when the operator explicitly opted into GKE Workload Identity."""
        if getattr(self, "_allow_workload_identity", False):
            return True
        extra_kwargs = getattr(self, "_kwargs", None)
        return is_gke_workload_identity_allowed(extra_kwargs if isinstance(extra_kwargs, Mapping) else None)

    def _is_autopilot_cluster(self, api: Any = None) -> bool:
        """Return True when targeting a GKE Autopilot cluster via explicit override, node inspection, or gcloud."""
        explicit = parse_optional_bool_flag(getattr(self, "_autopilot", None))
        if explicit is not None:
            return explicit
        env_override = parse_optional_bool_flag(os.environ.get(GKE_AUTOPILOT_ENV))
        if env_override is not None:
            return env_override
        extra_kwargs = getattr(self, "_kwargs", None)
        if isinstance(extra_kwargs, Mapping):
            kw_override = parse_optional_bool_flag(extra_kwargs.get("autopilot"))
            if kw_override is not None:
                return kw_override
        cached = getattr(self, "_detected_autopilot", None)
        if isinstance(cached, bool):
            return cached

        list_node = getattr(api, "list_node", None)
        if callable(list_node):
            with contextlib.suppress(Exception):
                node_list = _call_k8s_with_timeout(
                    list_node,
                    request_timeout=_DEFAULT_K8S_REQUEST_TIMEOUT_SEC,
                    limit=1,
                )
                items = getattr(node_list, "items", None) or []
                if items:
                    is_ap = any(_is_autopilot_node(node) for node in items)
                    self._detected_autopilot = is_ap
                    return is_ap

        cluster_name = str(getattr(self, "cluster_name", "") or "").strip()
        region = str(getattr(self, "region", "") or "").strip()
        if cluster_name and region:
            cache_key = (cluster_name, region)
            if cache_key in _AUTOPILOT_DETECTION_CACHE:
                is_ap = _AUTOPILOT_DETECTION_CACHE[cache_key]
                self._detected_autopilot = is_ap
                return is_ap
            if shutil.which("gcloud"):
                with contextlib.suppress(Exception):
                    proc = subprocess.run(
                        [
                            "gcloud",
                            "container",
                            "clusters",
                            "describe",
                            cluster_name,
                            f"--location={region}",
                            "--format=value(autopilot.enabled)",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        is_ap = proc.stdout.strip().lower() == "true"
                        _AUTOPILOT_DETECTION_CACHE[cache_key] = is_ap
                        self._detected_autopilot = is_ap
                        return is_ap
        return False

    async def _create_pod(self, pod: k8s_client.V1Pod) -> None:
        """Disable service account token automount, block metadata server, and reject GCP-bound KSAs unless opted in."""
        if not self._is_workload_identity_enabled():
            if getattr(pod, "metadata", None) is None:
                pod.metadata = k8s_client.V1ObjectMeta()
            labels = dict(getattr(pod.metadata, "labels", None) or {})
            labels[GKE_METADATA_ISOLATION_LABEL_KEY] = GKE_METADATA_ISOLATION_LABEL_VALUE
            pod.metadata.labels = labels

            compose_mode = bool(getattr(self, "_compose_mode", False))
            if not compose_mode:
                annotations = dict(getattr(pod.metadata, "annotations", None) or {})
                annotations.setdefault("autopilot.gke.io/primary-container", "main")
                annotations.setdefault("kubectl.kubernetes.io/default-container", "main")
                pod.metadata.annotations = annotations

            if getattr(pod, "spec", None) is not None:
                pod.spec.automount_service_account_token = False
                pod.spec.host_network = False
                sa_name = (
                    getattr(pod.spec, "service_account_name", None)
                    or getattr(pod.spec, "service_account", None)
                    or "default"
                )
                containers = list(getattr(pod.spec, "containers", None) or [])
                for container in containers:
                    env_list = list(getattr(container, "env", None) or [])
                    if not any(getattr(item, "name", None) == "GCE_METADATA_HOST" for item in env_list):
                        env_list.append(k8s_client.V1EnvVar(name="GCE_METADATA_HOST", value=_BLOCKED_GCE_METADATA_HOST))
                        container.env = env_list
            else:
                sa_name = "default"
                containers = []

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
            if not compose_mode:
                _ensure_default_exec_container_routing(api, default_container="main")
                if getattr(pod, "spec", None) is not None:
                    init_containers = list(getattr(pod.spec, "init_containers", None) or [])
                    if not any(
                        getattr(c, "name", None) == GKE_METADATA_PROBE_INIT_CONTAINER_NAME for c in init_containers
                    ):
                        init_containers.append(_build_metadata_probe_init_container())
                        pod.spec.init_containers = init_containers
                    if not any(getattr(c, "name", None) == GKE_METADATA_PROBE_CONTAINER_NAME for c in containers):
                        if await asyncio.to_thread(self._is_autopilot_cluster, api):
                            for container in containers:
                                if getattr(container, "name", None) == "main":
                                    _adjust_autopilot_ephemeral_storage_for_probe(container)
                        containers.append(_build_metadata_probe_container())
                        pod.spec.containers = containers
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

    async def _wait_for_pod_ready(self, timeout_sec: int = 300) -> None:
        """Wait for pod readiness while failing fast if the metadata isolation init container fails."""
        namespace = getattr(self, "namespace", "default")
        max_polls = max(1, timeout_sec // 3)
        for attempt in range(max_polls):
            pod = await asyncio.to_thread(
                self._api.read_namespaced_pod,
                name=self.pod_name,
                namespace=namespace,
            )
            status = getattr(pod, "status", None)
            if not self._is_workload_identity_enabled():
                for init_cs in getattr(status, "init_container_statuses", None) or []:
                    if getattr(init_cs, "name", None) != GKE_METADATA_PROBE_INIT_CONTAINER_NAME:
                        continue
                    state = getattr(init_cs, "state", None)
                    terminated = getattr(state, "terminated", None)
                    if terminated is not None:
                        exit_code = getattr(terminated, "exit_code", None)
                        if exit_code not in (None, 0):
                            await self._delete_unisolated_pod_best_effort()
                            if exit_code == _METADATA_REACHABLE_EXIT_CODE:
                                detail = (
                                    "in-pod init probe reached the GKE/GCE metadata server "
                                    "(169.254.169.254 or 169.254.169.252:988) before starting main container "
                                    "despite NetworkPolicy; cluster CNI may not enforce egress NetworkPolicy"
                                )
                            else:
                                detail = (
                                    f"in-pod metadata isolation init probe exited with status {exit_code} "
                                    "(missing python/curl/wget or probe error)"
                                )
                            raise RuntimeError(
                                GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                                    namespace=namespace,
                                    detail=detail,
                                )
                            )
                    waiting = getattr(state, "waiting", None)
                    if waiting is not None and getattr(waiting, "reason", None) in (
                        "ImagePullBackOff",
                        "ErrImagePull",
                        "InvalidImageName",
                    ):
                        await self._delete_unisolated_pod_best_effort()
                        wait_msg = getattr(waiting, "message", None) or getattr(waiting, "reason", "ImagePullBackOff")
                        raise RuntimeError(
                            GKE_METADATA_ISOLATION_ERROR_TEMPLATE.format(
                                namespace=namespace,
                                detail=f"failed to pull metadata probe image: {wait_msg}",
                            )
                        )

            phase = getattr(status, "phase", None)
            if phase == "Running":
                statuses = getattr(status, "container_statuses", None) or []
                if statuses and all(getattr(cs, "ready", False) for cs in statuses):
                    logger = getattr(self, "logger", None)
                    if logger is not None:
                        logger.debug("Pod %s is ready (attempt %d)", self.pod_name, attempt + 1)
                    return
            elif phase in ("Failed", "Succeeded"):
                reason = getattr(status, "reason", None) or "unknown"
                msg = getattr(status, "message", None) or ""
                raise RuntimeError(f"Pod {self.pod_name} entered {phase} state: {reason} {msg}".strip())

            for cs in getattr(status, "container_statuses", None) or []:
                waiting = getattr(getattr(cs, "state", None), "waiting", None)
                if waiting is not None and getattr(waiting, "reason", None) in (
                    "ImagePullBackOff",
                    "ErrImagePull",
                    "CrashLoopBackOff",
                ):
                    raise RuntimeError(
                        f"Container {getattr(cs, 'name', 'unknown')} in pod {self.pod_name}: "
                        f"{waiting.reason} - {getattr(waiting, 'message', '')}"
                    )

            await asyncio.sleep(3)

        raise TimeoutError(f"Pod {self.pod_name} did not become ready within {timeout_sec}s")

    async def _exec_pod_stream_command(
        self,
        command: list[str],
        *,
        timeout_sec: int,
        container: str | None = None,
    ) -> int | None:
        """Execute a command via Kubernetes WebSocket exec stream, draining output and returning returncode."""
        stream_kwargs: dict[str, Any] = {
            "command": command,
            "stderr": True,
            "stdin": False,
            "stdout": True,
            "tty": False,
            "_preload_content": False,
        }
        if container is not None:
            stream_kwargs["container"] = container
        elif getattr(self, "_compose_mode", False):
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
        """Verify from the trusted probe container that GCE/GKE metadata endpoints are unreachable."""
        namespace = getattr(self, "namespace", "default")
        probe_container = "dind" if getattr(self, "_compose_mode", False) else GKE_METADATA_PROBE_CONTAINER_NAME
        try:
            rc = await self._exec_pod_stream_command(
                ["sh", "-c", _IN_POD_METADATA_ISOLATION_PROBE_SCRIPT],
                timeout_sec=_METADATA_PROBE_STREAM_TIMEOUT_SEC,
                container=probe_container,
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

    async def _check_pod_terminated(self) -> None:
        """Raise immediately if the pod was deleted/preempted or is in a terminal state."""
        namespace = getattr(self, "namespace", "default")
        try:
            pod = await asyncio.to_thread(
                self._api.read_namespaced_pod,
                name=self.pod_name,
                namespace=namespace,
            )
        except Exception as exc:
            if _is_k8s_http_status(exc, 404, "not found"):
                raise RuntimeError(
                    f"Pod {self.pod_name} in namespace {namespace} no longer exists (deleted or preempted) "
                    "and cannot accept exec."
                ) from exc
            return

        status = getattr(pod, "status", None)
        phase = getattr(status, "phase", None)
        if phase in ("Failed", "Succeeded"):
            raise RuntimeError(f"Pod {self.pod_name} is in terminal phase '{phase}' and cannot accept exec.")

        for cs in getattr(status, "container_statuses", None) or []:
            terminated = getattr(getattr(cs, "state", None), "terminated", None) or getattr(
                getattr(cs, "last_state", None), "terminated", None
            )
            if terminated is not None:
                reason = getattr(terminated, "reason", None) or ""
                exit_code = getattr(terminated, "exit_code", None)
                raise RuntimeError(
                    f"Container '{getattr(cs, 'name', 'unknown')}' in pod {self.pod_name} has terminated "
                    f"(reason={reason!r}, exit_code={exit_code}). Cannot exec into dead container."
                )

    async def _wait_for_container_exec_ready(self, max_attempts: int = 60) -> None:
        """Wait until the GKE kubelet accepts exec streams and verify metadata server isolation."""
        if not getattr(self, "_compose_mode", False):
            api = getattr(self, "_api", None)
            if api is not None:
                _ensure_default_exec_container_routing(api, default_container="main")

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
                if _extract_pod_missing_exec_detail(exc) is not None:
                    raise
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
