# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in Harbor environment names available in the public Tier 3 surface.

This deliberately lives outside the Tier 3 package so that the base CLI can
show its command help without importing Harbor or any optional dependencies.
"""

from __future__ import annotations

HARBOR_ENVIRONMENTS = (
    "docker",
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
    "cua-cloud",
    "blaxel",
    "opensandbox",
    "beam",
    "skypilot",
    "hf-sandbox",
    "hyperbrowser",
    "vercel",
    # Not a Harbor-native backend: SkillEvaluator's host execution mode, run
    # under an OS sandbox (bubblewrap on Linux, Seatbelt on macOS). Dispatched
    # by passing its custom import path through Harbor's unified --env flag.
    "local",
)
HARBOR_ENV_MODES = frozenset(HARBOR_ENVIRONMENTS)
#: env modes that Harbor accepts natively via ``--env`` (everything except ``local``).
HARBOR_NATIVE_ENV_MODES = frozenset(m for m in HARBOR_ENVIRONMENTS if m != "local")
# Exact Harbor 0.22 ``Provides-Extra`` names. ``ack`` reuses the Kubernetes
# dependencies supplied by ``gke``. Of the four ``None`` entries, Docker needs
# no additional Python extra and the other three are system-CLI backends.
HARBOR_ENVIRONMENT_EXTRAS: dict[str, str | None] = {
    "docker": None,
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
    "wandb": "wandb",
    "use-computer": "use-computer",
    "cua-cloud": "cua",
    "blaxel": "blaxel",
    "opensandbox": "opensandbox",
    "beam": "beam",
    "skypilot": "skypilot",
    "hf-sandbox": "hf-sandbox",
    "hyperbrowser": "hyperbrowser",
    "vercel": "vercel",
}

# Constructor kwargs consumed by Harbor 0.22.0. Keep this static so importing
# the base SkillEvaluator CLI never imports Harbor or optional provider SDKs.
# The packaging parity test AST-reads the pinned Harbor sources and catches
# additions, removals, and provider kwargs that are consumed through **kwargs.
_HARBOR_V022_BASE_ENVIRONMENT_KWARGS = frozenset(
    {
        "cpu_enforcement_policy",
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
        "suppress_override_warnings",
        "task_env_config",
        "trial_paths",
    }
)

_HARBOR_V022_BACKEND_ENVIRONMENT_KWARGS = {
    "docker": "keep_containers",
    "daytona": (
        "assume_global_snapshot auto_delete_interval_mins auto_labels auto_snapshot auto_stop_interval_mins "
        "connection_pool_maxsize dind_image dind_snapshot expose_sandbox_id labels network_block_all secrets "
        "snapshot_template_name"
    ),
    "e2b": "",
    "modal": (
        "app_name auto_labels dind_image keepalive labels modal_sandbox_v2 modal_vm_runtime region registry_secret "
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
    "tensorlake": "is_public preinstall_packages snapshot_id timeout_secs use_oci_image_build",
    "cwsandbox": (
        "base_url docker_image max_lifetime_seconds max_timeout_seconds mounts_json request_timeout_seconds secrets tags"
    ),
    "wandb": (
        "base_url docker_image max_lifetime_seconds max_timeout_seconds mounts_json request_timeout_seconds secrets tags"
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
    "hyperbrowser": "image_id image_name region timeout_minutes",
    "vercel": (
        "builder_image create_timeout_sec credential_injection destroy_timeout_sec host_bootstrap host_image image "
        "ports post_cancel_command_timeout_sec process_cancel_grace_sec process_wait_retry_delay_sec project_name "
        "sandbox_lifetime_seconds task_bootstrap task_image transfer_timeout_sec"
    ),
}
HARBOR_V022_ENVIRONMENT_KWARGS: dict[str, frozenset[str]] = {
    mode: _HARBOR_V022_BASE_ENVIRONMENT_KWARGS | frozenset(names.split())
    for mode, names in _HARBOR_V022_BACKEND_ENVIRONMENT_KWARGS.items()
}
ENV_MODE_LOCAL = "local"
DEFAULT_ENV_MODE = "docker"
