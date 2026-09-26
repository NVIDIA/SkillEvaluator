# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GKE Harbor environment wrapper that enforces pod ServiceAccount and Workload Identity isolation."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from harbor.environments.gke import GKEEnvironment

from skillevaluator.provider_config import _get_google_access_token, _is_vertex_openapi_endpoint

if TYPE_CHECKING:
    from harbor.environments.base import ExecResult
    from kubernetes import client as k8s_client

SECURE_GKE_ENV_IMPORT_PATH = "skillevaluator.tier3.harbor.gke_environment:SkillEvaluatorGKEEnvironment"
GKE_ALLOW_WORKLOAD_IDENTITY_ENV = "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY"
_GKE_WORKLOAD_IDENTITY_ANNOTATION = "iam.gke.io/gcp-service-account"
GKE_BOUND_SERVICE_ACCOUNT_ERROR_TEMPLATE = (
    "GKE namespace '{namespace}' ServiceAccount '{service_account}' is bound to GCP service account "
    "'{gcp_sa}' via Workload Identity, which exposes pod-level cloud credentials to evaluated skill "
    "commands. Restrict this mode to trusted skills by setting "
    "SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1 or --ek allow_workload_identity=1."
)


def _coerce_opt_in_flag(value: object) -> bool:
    """Return True when an opt-in flag value represents an explicit truthy setting."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def inspect_bound_gcp_service_account(
    api: Any,
    *,
    namespace: str,
    service_account: str = "default",
) -> str | None:
    """Return the bound GCP service account email if the Kubernetes ServiceAccount has Workload Identity."""
    read_sa = getattr(api, "read_namespaced_service_account", None)
    if not callable(read_sa):
        return None
    try:
        sa_obj = read_sa(name=service_account, namespace=namespace, _request_timeout=5.0)
    except TypeError:
        try:
            sa_obj = read_sa(name=service_account, namespace=namespace)
        except Exception:
            return None
    except Exception:
        return None

    metadata = getattr(sa_obj, "metadata", None)
    annotations = getattr(metadata, "annotations", None)
    if isinstance(annotations, dict):
        bound = str(annotations.get(_GKE_WORKLOAD_IDENTITY_ANNOTATION, "")).strip()
        if bound:
            return bound
    return None


class SkillEvaluatorGKEEnvironment(GKEEnvironment):
    """Enforce least-privilege pod identity and ADC refresh for GKE evaluation pods."""

    def __init__(
        self,
        *args: Any,
        allow_workload_identity: bool | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GKE environment while recording Workload Identity opt-in state."""
        self._allow_workload_identity = _coerce_opt_in_flag(allow_workload_identity) or _coerce_opt_in_flag(
            os.environ.get(GKE_ALLOW_WORKLOAD_IDENTITY_ENV)
        )
        super().__init__(*args, **kwargs)

    def _is_workload_identity_enabled(self) -> bool:
        """Return True when the operator explicitly opted into GKE Workload Identity."""
        if getattr(self, "_allow_workload_identity", False):
            return True
        if _coerce_opt_in_flag(os.environ.get(GKE_ALLOW_WORKLOAD_IDENTITY_ENV)):
            return True
        extra_kwargs = getattr(self, "_kwargs", None)
        return bool(isinstance(extra_kwargs, dict) and _coerce_opt_in_flag(extra_kwargs.get("allow_workload_identity")))

    async def _create_pod(self, pod: k8s_client.V1Pod) -> None:
        """Disable service account token automount and block GCP-bound KSAs unless opted in."""
        if not self._is_workload_identity_enabled():
            if getattr(pod, "spec", None) is not None:
                pod.spec.automount_service_account_token = False
                sa_name = (
                    getattr(pod.spec, "service_account_name", None)
                    or getattr(pod.spec, "service_account", None)
                    or "default"
                )
            else:
                sa_name = "default"
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

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Refresh ADC-sourced Vertex OpenAPI credentials on the host before executing in-pod."""
        if env and (
            env.get("SKILL_EVAL_LLM_CREDENTIAL_SOURCE") == "ADC"
            or os.environ.get("SKILL_EVAL_LLM_CREDENTIAL_SOURCE") == "ADC"
        ):
            base_url = env.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            if _is_vertex_openapi_endpoint(base_url) and "OPENAI_API_KEY" in env:
                fresh_token = await asyncio.to_thread(_get_google_access_token)
                if fresh_token:
                    env = dict(env)
                    env["OPENAI_API_KEY"] = fresh_token
                    os.environ["OPENAI_API_KEY"] = fresh_token
        return await super().exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )
