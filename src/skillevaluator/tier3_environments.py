# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor environment names available in the public Tier 3 surface.

This deliberately lives outside the Tier 3 package so that the base CLI can
show its command help without importing Harbor or any optional dependencies.
Installed packages may add modes through the ``skillevaluator.env_modes``
entry-point group; listing them reads package metadata only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Any

#: The exact Harbor release whose CLI, models, and environment interfaces this
#: SkillEvaluator release targets; the ``tier3`` extra pins it.
HARBOR_VERSION = "0.24.0"

HARBOR_ENVIRONMENTS = (
    "docker",
    "kata",
    "daytona",
    "e2b",
    "modal",
    "runloop",
    "langsmith",
    "ec2",
    "gke",
    "ack",
    "openshift",
    "novita",
    "apple-container",
    "singularity",
    "islo",
    "tensorlake",
    "cwsandbox",
    "wandb",
    "use-computer",
    "blaxel",
    "beam",
    "skypilot",
    "hyperbrowser",
    "vercel",
    "runta",
    "mosaic",
    # Not a Harbor-native backend: SkillEvaluator's host execution mode, run
    # under an OS sandbox (bubblewrap on Linux, Seatbelt on macOS). Dispatched
    # by passing its custom import path through Harbor's unified --env flag.
    "local",
)
HARBOR_ENV_MODES = frozenset(HARBOR_ENVIRONMENTS)
#: Public env modes that Harbor accepts natively via ``--env`` (everything except ``local``).
HARBOR_NATIVE_ENV_MODES = frozenset(m for m in HARBOR_ENVIRONMENTS if m != "local")
#: Modes that run through SkillEvaluator's hardened Docker Compose backend on the
#: host's Docker engine. ``kata`` adds Harbor's Kata Containers runtime overlay.
HARBOR_DOCKER_ENGINE_ENV_MODES = frozenset({"docker", "kata"})
#: SkillEvaluator modes that run as a different Harbor environment type. Harbor
#: 0.24 merged its W&B sandbox into ``cwsandbox``, selected with ``auth=wandb``.
HARBOR_ENVIRONMENT_TYPE_ALIASES: dict[str, str] = {"wandb": "cwsandbox"}
#: Constructor kwargs SkillEvaluator supplies for an alias; operators cannot set them.
HARBOR_ENVIRONMENT_ALIAS_KWARGS: dict[str, dict[str, str]] = {"wandb": {"auth": "wandb"}}


def harbor_environment_type(env_mode: str) -> str:
    """Return the Harbor ``--env`` type that implements a SkillEvaluator mode."""
    return HARBOR_ENVIRONMENT_TYPE_ALIASES.get(env_mode, env_mode)


# Exact Harbor ``Provides-Extra`` names. ``ack`` reuses the Kubernetes
# dependencies supplied by ``gke``, and ``wandb`` the W&B-capable ``cwsandbox``
# extra. Of the five ``None`` entries, Docker and Kata need no additional Python
# extra and the other three are system-CLI backends.
HARBOR_ENVIRONMENT_EXTRAS: dict[str, str | None] = {
    "docker": None,
    "kata": None,
    "daytona": "daytona",
    "e2b": "e2b",
    "modal": "modal",
    "runloop": "runloop",
    "langsmith": "langsmith",
    "ec2": "ec2",
    "gke": "gke",
    "ack": "gke",
    "openshift": None,
    "novita": "novita",
    "apple-container": None,
    "singularity": None,
    "islo": "islo",
    "tensorlake": "tensorlake",
    "cwsandbox": "cwsandbox",
    "wandb": "cwsandbox",
    "use-computer": "use-computer",
    "blaxel": "blaxel",
    "beam": "beam",
    "skypilot": "skypilot",
    "hyperbrowser": "hyperbrowser",
    "vercel": "vercel",
    "runta": "runta",
    "mosaic": "mosaic",
}

# Constructor kwargs consumed by the pinned Harbor release. Keep this static so
# importing the base SkillEvaluator CLI never imports Harbor or optional provider
# SDKs. The packaging parity test AST-reads the pinned Harbor sources and catches
# additions, removals, and provider kwargs that are consumed through **kwargs.
# Registry-only entries (cua-cloud, opensandbox, hf-sandbox, podman, prime, smol)
# keep that parity exact; those modes are not exposed.
_HARBOR_BASE_ENVIRONMENT_KWARGS = frozenset(
    {
        "cpu_enforcement_policy",
        "enable_environment_dir_upload",
        "environment_dir",
        "environment_name",
        "extra_docker_compose",
        "logger",
        "memory_enforcement_policy",
        "mounts",
        "network_policy",
        "override_cpus",
        "override_gpus",
        "override_memory_mb",
        "override_storage_mb",
        "override_tpu",
        "persistent_env",
        "phase_network_policies",
        "session_id",
        "stream",
        "suppress_override_warnings",
        "task_env_config",
        "trial_paths",
    }
)

_HARBOR_BACKEND_ENVIRONMENT_KWARGS = {
    "docker": "keep_containers",
    "daytona": (
        "assume_global_snapshot auto_delete_interval_mins auto_labels auto_snapshot auto_stop_interval_mins "
        "connection_pool_maxsize dind_image dind_snapshot expose_sandbox_id labels network_block_all secrets "
        "snapshot_template_name"
    ),
    "e2b": "",
    "modal": (
        "app_name auto_labels dind_image keepalive labels modal_vm_runtime region registry_secret "
        "sandbox_idle_timeout_secs sandbox_timeout_secs secrets volumes"
    ),
    "runloop": "",
    "langsmith": (
        "api_key create_snapshot delete_after_stop_seconds delete_snapshot idle_ttl_seconds langsmith_api_key "
        "langsmith_endpoint poll_interval_seconds registry_id request_timeout_seconds sandbox_api_url snapshot_name "
        "startup_timeout_seconds ttl_seconds workdir"
    ),
    "ec2": (
        "ami_id aws_profile bootstrap_docker compose_up_timeout_sec docker_ready_timeout_sec iam_instance_profile "
        "instance_id instance_ready_timeout_sec instance_type key_name launch_mode region root_device_name "
        "root_volume_size_gb root_volume_type security_group_ids ssh_connect_timeout_sec ssh_key_path "
        "ssh_known_hosts_path ssh_port ssh_user strict_host_key_checking subnet_id tags use_public_ip"
    ),
    "gke": (
        "cloud_build_disk_size_gb cloud_build_machine_type cluster_name dind_image memory_limit_multiplier namespace "
        "project_id region registry_location registry_name"
    ),
    "ack": (
        "build_job_namespace build_timeout_sec buildkit_address claim_timeout context dind_image extra_env "
        "extra_volume_mounts extra_volumes image_pull_secret init_containers kubeconfig memory_limit_multiplier "
        "namespace node_selector pod_annotations pod_capabilities_add pod_capabilities_drop pod_labels pod_overrides "
        "pod_privileged pod_run_as_group pod_run_as_user registry sandbox_annotations sandbox_env_vars sandbox_image "
        "sandbox_labels sandboxset_replicas service_account skip_image_check tolerations use_buildkit use_sandbox_claim"
    ),
    "openshift": "namespace service_account_name",
    "novita": "dind_dockerd_start_cmd dind_template_alias",
    "apple-container": "keep_containers",
    "singularity": "singularity_force_pull singularity_image_cache_dir singularity_no_mount",
    "islo": "delete_after_seconds gateway gateway_profile",
    "tensorlake": "dind_image is_public preinstall_packages snapshot_id timeout_secs use_oci_image_build",
    "cwsandbox": (
        "auth base_url docker_image max_lifetime_seconds max_timeout_seconds mounts_json request_timeout_seconds "
        "secrets tags"
    ),
    "use-computer": (
        "api_key base_url device_type family gateway_url host keepalive_interval mode override_exec_timeout platform "
        "reservation_id resources runtime snapshot version"
    ),
    "cua-cloud": (
        "api_key base_url bind_timeout_sec claim_spec claim_ttl_sec claims_path kubeconfig namespace "
        "override_exec_timeout platform port_services ready_timeout_sec renew_interval startup_command sudo_password "
        "svc_auth svc_suffix svc_url template token_url warmpool"
    ),
    "blaxel": "deployment_timeout_sec dind_extra_args dind_image region sandbox_version ttl",
    "opensandbox": (
        "api_key domain entrypoint extensions health_check_poll_interval_sec image_auth metadata protocol "
        "ready_timeout_sec request_timeout_sec sandbox_timeout_sec skip_health_check use_server_proxy volumes"
    ),
    "beam": "keep_warm_seconds",
    "skypilot": "context_name namespace platform pool registry secrets",
    "hf-sandbox": "flavor forward_hf_token job_timeout",
    "hyperbrowser": "builder_cpus builder_memory_mib builder_scratch_mib image_id image_name region timeout_minutes",
    "vercel": (
        "builder_image create_timeout_sec credential_injection destroy_timeout_sec host_bootstrap host_image image "
        "ports post_cancel_command_timeout_sec process_cancel_grace_sec process_wait_retry_delay_sec project_name "
        "sandbox_lifetime_seconds task_bootstrap task_image transfer_timeout_sec"
    ),
    "podman": "keep_containers",
    "kata": "kata_dns kata_runtime keep_containers",
    "runta": "endpoint mode request_timeout_sec startup_timeout_sec token",
    "prime": "api_key compose_host_image gpu_type region team_id timeout_minutes",
    "mosaic": "build_args build_target enable_ssh metadata persist replicas secrets ttl_seconds volume",
    "smol": (
        "api_key auto_checkpoint base_url checkpoints fork_batch_size fork_batch_window_ms gpu_mode ready_timeout_sec "
        "target"
    ),
}
HARBOR_ENVIRONMENT_KWARGS: dict[str, frozenset[str]] = {
    mode: _HARBOR_BASE_ENVIRONMENT_KWARGS | frozenset(names.split())
    for mode, names in _HARBOR_BACKEND_ENVIRONMENT_KWARGS.items()
}
# An alias accepts its Harbor type's kwargs except the ones SkillEvaluator sets.
for _alias, _harbor_type in HARBOR_ENVIRONMENT_TYPE_ALIASES.items():
    HARBOR_ENVIRONMENT_KWARGS[_alias] = (
        HARBOR_ENVIRONMENT_KWARGS[_harbor_type] - HARBOR_ENVIRONMENT_ALIAS_KWARGS[_alias].keys()
    )
# SkillEvaluatorGKEEnvironment wrapper kwargs accepted on ``--env-mode gke`` in
# addition to stock Harbor's GKEEnvironment.__init__ parameters.
SKILLEVALUATOR_GKE_EXTRA_ENVIRONMENT_KWARGS = frozenset({"allow_workload_identity", "autopilot"})
ENV_MODE_LOCAL = "local"
DEFAULT_ENV_MODE = "docker"


# ---------------------------------------------------------------------------
# Custom execution backends
# ---------------------------------------------------------------------------

#: Entry-point group through which an installed package adds an ``--env-mode``.
#: The entry-point name is the mode name; its value names an ``EnvironmentBackendSpec``
#: instance, or a zero-argument callable that returns one.
ENV_MODE_ENTRY_POINT_GROUP = "skillevaluator.env_modes"
#: Set to ``1`` to ignore every installed custom execution backend.
DISABLE_CUSTOM_BACKENDS_ENV = "SKILLEVALUATOR_DISABLE_CUSTOM_BACKENDS"

_BACKEND_NAME_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_IMPORT_PATH_RE = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")
# Upper case only: SkillEvaluator compares host variable names case-insensitively.
_HOST_ENV_VAR_RE = re.compile(r"[A-Z_][A-Z0-9_]*")


class EnvironmentBackendError(ValueError):
    """An installed custom execution backend could not be loaded or is malformed."""


@dataclass(frozen=True)
class EnvironmentBackendSpec:
    """Describe one ``--env-mode`` that an installed package contributes.

    SkillEvaluator dispatches the mode to Harbor through ``--env IMPORT_PATH``
    and applies the same kwarg, credential, and runtime-env policy it applies to
    Harbor-native backends. The backend module is imported only when its mode is
    selected, so it may import Harbor at module level.
    """

    #: Harbor ``BaseEnvironment`` subclass, as ``package.module:ClassName``.
    import_path: str
    #: Constructor kwargs operators may pass with ``--ek`` beyond Harbor's base kwargs.
    environment_kwargs: frozenset[str] = frozenset()
    #: Kwargs that must be non-empty strings before a run starts.
    required_environment_kwargs: tuple[str, ...] = ()
    #: Accepted constructor kwargs that SkillEvaluator must never forward from operators.
    reserved_environment_kwargs: frozenset[str] = frozenset()
    #: Upper-case host variables forwarded to the Harbor process; skill ``runtime_env``
    #: cannot set them. Agent, provider, and other backends' credentials are refused.
    host_env_vars: frozenset[str] = frozenset()
    #: Members of ``host_env_vars`` that hold file paths; relative values are anchored to
    #: the caller's working directory because Harbor runs in a private directory.
    host_path_env_vars: frozenset[str] = frozenset()
    #: Members of ``environment_kwargs`` that hold host file paths, anchored the same way.
    host_path_environment_kwargs: frozenset[str] = frozenset()
    #: Per-agent Harbor agent import paths; unlisted agents use Harbor's stock agent.
    agent_import_paths: Mapping[str, str] = field(default_factory=dict)
    #: Whether the backend can run tasks that declare ``docker-compose.yaml`` sidecars.
    supports_docker_compose: bool = False
    #: Optional ``package.module:function`` called as ``fn(environment_kwargs) -> list[str]``
    #: before Harbor's own ``preflight``; each string is one blocking prerequisite error.
    prerequisites: str | None = None
    #: Shown when the environment class or a dependency it imports cannot be imported.
    install_hint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.import_path, str) or not _IMPORT_PATH_RE.fullmatch(self.import_path):
            raise EnvironmentBackendError("import_path must be 'package.module:ClassName'")
        if self.prerequisites is not None and not _IMPORT_PATH_RE.fullmatch(str(self.prerequisites)):
            raise EnvironmentBackendError("prerequisites must be 'package.module:function'")
        for label, names in (
            ("environment_kwargs", self.environment_kwargs),
            ("required_environment_kwargs", self.required_environment_kwargs),
            ("reserved_environment_kwargs", self.reserved_environment_kwargs),
        ):
            if isinstance(names, str) or not all(isinstance(name, str) and name.isidentifier() for name in names):
                raise EnvironmentBackendError(f"{label} must be a collection of Python keyword names")
        if missing := sorted(set(self.required_environment_kwargs) - set(self.environment_kwargs)):
            raise EnvironmentBackendError("required_environment_kwargs must also be accepted: " + ", ".join(missing))
        if isinstance(self.host_env_vars, str) or not all(
            isinstance(name, str) and _HOST_ENV_VAR_RE.fullmatch(name) for name in self.host_env_vars
        ):
            raise EnvironmentBackendError("host_env_vars must be a collection of upper-case environment variable names")
        if missing := sorted(set(self.host_path_env_vars) - set(self.host_env_vars)):
            raise EnvironmentBackendError("host_path_env_vars must also be host_env_vars: " + ", ".join(missing))
        if missing := sorted(set(self.host_path_environment_kwargs) - set(self.environment_kwargs)):
            raise EnvironmentBackendError(
                "host_path_environment_kwargs must also be environment_kwargs: " + ", ".join(missing)
            )
        if not isinstance(self.agent_import_paths, Mapping) or not all(
            isinstance(agent, str) and isinstance(path, str) and _IMPORT_PATH_RE.fullmatch(path)
            for agent, path in self.agent_import_paths.items()
        ):
            raise EnvironmentBackendError("agent_import_paths must map agent names to 'package.module:ClassName'")
        object.__setattr__(self, "environment_kwargs", frozenset(self.environment_kwargs))
        object.__setattr__(self, "required_environment_kwargs", tuple(self.required_environment_kwargs))
        object.__setattr__(self, "reserved_environment_kwargs", frozenset(self.reserved_environment_kwargs))
        object.__setattr__(self, "host_env_vars", frozenset(self.host_env_vars))
        object.__setattr__(self, "host_path_env_vars", frozenset(self.host_path_env_vars))
        object.__setattr__(self, "host_path_environment_kwargs", frozenset(self.host_path_environment_kwargs))
        object.__setattr__(self, "agent_import_paths", MappingProxyType(dict(self.agent_import_paths)))


def _custom_backends_disabled() -> bool:
    return os.environ.get(DISABLE_CUSTOM_BACKENDS_ENV, "").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def _custom_backend_entry_points() -> Mapping[str, tuple[EntryPoint, ...]]:
    """Return installed custom backend entry points by mode name, without importing them."""
    selected: dict[str, dict[tuple[str | None, str | None, str], EntryPoint]] = {}
    for entry_point in entry_points(group=ENV_MODE_ENTRY_POINT_GROUP):
        name = entry_point.name
        # Built-in modes keep their reviewed behavior; a custom backend can never replace one.
        if name in HARBOR_ENV_MODES or not _BACKEND_NAME_RE.fullmatch(name):
            continue
        # EntryPoint equality ignores the owning distribution. Key on it so two
        # packages registering the same target still count as two claims, while
        # one distribution seen twice on sys.path counts once.
        dist = getattr(entry_point, "dist", None)
        identity = (
            dist.name.lower().replace("_", "-") if dist is not None else None,
            dist.version if dist is not None else None,
            entry_point.value,
        )
        selected.setdefault(name, {}).setdefault(identity, entry_point)
    return MappingProxyType({name: tuple(points.values()) for name, points in sorted(selected.items())})


def _entry_point_source(entry_point: EntryPoint) -> str:
    dist = getattr(entry_point, "dist", None)
    return f"{dist.name} {dist.version}" if dist is not None else entry_point.value


def custom_backend_names() -> tuple[str, ...]:
    """Return the ``--env-mode`` names contributed by installed custom execution backends."""
    if _custom_backends_disabled():
        return ()
    return tuple(_custom_backend_entry_points())


def all_env_modes() -> tuple[str, ...]:
    """Return built-in modes followed by installed custom execution backends."""
    return (*HARBOR_ENVIRONMENTS, *custom_backend_names())


def is_env_mode(env_mode: str) -> bool:
    """Return True for a built-in or custom ``--env-mode``."""
    return env_mode in HARBOR_ENV_MODES or env_mode in custom_backend_names()


def load_custom_backend(env_mode: str) -> EnvironmentBackendSpec | None:
    """Load the custom backend behind ``env_mode``; return None for built-in or unknown modes."""
    if env_mode in HARBOR_ENV_MODES or env_mode not in custom_backend_names():
        return None
    return _load_custom_backend(env_mode)


@cache
def _load_custom_backend(env_mode: str) -> EnvironmentBackendSpec:
    claims = _custom_backend_entry_points()[env_mode]
    if len(claims) > 1:
        # Install order would otherwise decide which backend runs.
        raise EnvironmentBackendError(
            f"Custom execution backend '{env_mode}' is registered by more than one package: "
            + ", ".join(sorted(_entry_point_source(point) for point in claims))
            + ". Uninstall all but one"
        )
    entry_point = claims[0]
    try:
        loaded: Any = entry_point.load()
        backend = loaded if isinstance(loaded, EnvironmentBackendSpec) else loaded() if callable(loaded) else loaded
    except EnvironmentBackendError as exc:
        raise EnvironmentBackendError(f"Custom execution backend '{env_mode}' is invalid: {exc}") from None
    except Exception as exc:
        raise EnvironmentBackendError(
            f"Custom execution backend '{env_mode}' ({entry_point.value!r}) could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(backend, EnvironmentBackendSpec):
        raise EnvironmentBackendError(
            f"Custom execution backend '{env_mode}' ({entry_point.value!r}) must provide an EnvironmentBackendSpec"
        )
    return backend


def clear_custom_backend_cache() -> None:
    """Forget discovered custom backends, for tests and long-lived processes that install packages."""
    _custom_backend_entry_points.cache_clear()
    _load_custom_backend.cache_clear()
