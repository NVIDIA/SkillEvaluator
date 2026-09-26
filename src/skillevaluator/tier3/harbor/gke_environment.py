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

from skillevaluator.provider_config import refresh_host_vertex_adc_environment
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
GKE_SERVICE_ACCOUNT_INSPECTION_ERROR_TEMPLATE = (
    "Failed to inspect GKE namespace '{namespace}' ServiceAccount '{service_account}' for Workload Identity "
    "isolation ({detail}). Grant 'get' permission on serviceaccounts or explicitly opt in via "
    "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1 or --ek allow_workload_identity=1."
)
SECURE_GKE_ENV_IMPORT_PATH = "skillevaluator.tier3.harbor.gke_environment:SkillEvaluatorGKEEnvironment"

_BLOCKED_GCE_METADATA_HOST = "127.0.0.1:1"
_DEFAULT_K8S_REQUEST_TIMEOUT_SEC = 5.0
_EXEC_QUERY_COMMAND_RE = re.compile(r"(?P<sep>[?&])command=[^\s\"']+")
_GKE_WORKLOAD_IDENTITY_ANNOTATION = "iam.gke.io/gcp-service-account"
_GOOGLE_CREDENTIAL_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:ya29\.[A-Za-z0-9._-]{6,}|AIza[A-Za-z0-9_-]{12,})")
_MAX_TRANSIENT_EXEC_ATTEMPTS = 4
_MIN_EXACT_SECRET_LENGTH = 8
_TRAILING_QUOTES_RE = re.compile(r"[\s\"']+\Z")
_TRANSIENT_KUBELET_EXEC_ERROR_SNIPPETS = (
    "error sending request:",
    "No agent available",
    "unable to upgrade connection",
)


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


def inspect_bound_gcp_service_account(
    api: Any,
    *,
    namespace: str,
    service_account: str = "default",
    request_timeout: float = _DEFAULT_K8S_REQUEST_TIMEOUT_SEC,
) -> str | None:
    """Return the bound GCP service account email if the Kubernetes ServiceAccount has Workload Identity."""
    read_sa = getattr(api, "read_namespaced_service_account", None)
    if not callable(read_sa):
        return None
    try:
        sa_obj = read_sa(name=service_account, namespace=namespace, _request_timeout=request_timeout)
    except TypeError:
        try:
            sa_obj = read_sa(name=service_account, namespace=namespace)
        except Exception as exc:
            raise RuntimeError(
                GKE_SERVICE_ACCOUNT_INSPECTION_ERROR_TEMPLATE.format(
                    namespace=namespace,
                    service_account=service_account,
                    detail=str(exc),
                )
            ) from exc
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
    return None


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
            if getattr(pod, "spec", None) is not None:
                pod.spec.automount_service_account_token = False
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
            if api is not None:
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
        await super()._create_pod(pod)

    async def _wait_for_container_exec_ready(self, max_attempts: int = 60) -> None:
        """Wait until the GKE kubelet accepts exec streams and returns exit code 0."""
        for attempt in range(max_attempts):
            await self._check_pod_terminated()
            resp = None
            try:
                resp = await asyncio.to_thread(
                    stream,
                    self._api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.namespace,
                    command=["true"],
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                if hasattr(self, "_read_exec_output") and resp.is_open():
                    await asyncio.to_thread(self._read_exec_output, resp)
                await asyncio.to_thread(resp.run_forever, 5)
                rc = resp.returncode
                if rc not in (None, 0):
                    raise RuntimeError(f"Container readiness probe returned exit code {rc}")
                return
            except Exception as exc:
                if attempt >= max_attempts - 1:
                    raise RuntimeError(f"Container not ready for exec after {max_attempts} attempts: {exc}") from exc
                await asyncio.sleep(3)
            finally:
                if resp is not None:
                    with contextlib.suppress(Exception):
                        resp.close()

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
            update_os_environ=True,
        )
        if fresh_token:
            merged["OPENAI_API_KEY"] = fresh_token
            persistent = getattr(self, "_persistent_env", None)
            updated_persistent = False
            if isinstance(persistent, dict) and persistent.get("OPENAI_API_KEY", "").strip():
                persistent["OPENAI_API_KEY"] = fresh_token
                updated_persistent = True
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
