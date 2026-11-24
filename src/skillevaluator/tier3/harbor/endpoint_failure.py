# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognise a run whose agents never got a response from the model endpoint.

When a proxy, firewall or DNS rule keeps the environment from reaching the
model endpoint, every trial's first model request hangs until Harbor's agent
timeout fires. The run then ends incomplete with one timeout per trial, and the
usual advice for a timeout (shorten the task, raise ``--timeout-multiplier``)
points the wrong way. Such a trial leaves a fingerprint in Harbor's own
``result.json``: an ``AgentTimeoutError`` and not a single input or output
token.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from skillevaluator.tier3.harbor.collector import _inspect_trial_directory, _read_json, _trial_output_tokens

#: Share of the run's trials that must match before the run is flagged.
ENDPOINT_FAILURE_THRESHOLD = 0.8
DIAGNOSTIC_KIND = "model_endpoint_unreachable"


def selected_agent_job_dirs(
    jobs_dir: Path,
    skill_name: str,
    agents: Iterable[str],
    *,
    skip_baseline: bool,
    skip_with_skill: bool,
) -> list[Path]:
    """Return the job directories this run wrote, one per agent and arm that ran."""
    variants = [*(() if skip_with_skill else ("with",)), *(() if skip_baseline else ("without",))]
    return [jobs_dir / f"{skill_name}-{agent}-{variant}" for agent in agents for variant in variants]


def _is_real_directory(path: Path) -> bool:
    return _inspect_trial_directory(path)[0] == "directory"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def detect_model_endpoint_failure(job_dirs: Iterable[Path]) -> dict[str, Any] | None:
    """Return a diagnostic when at least 80 percent of the trials timed out with no model tokens.

    Only the direct ``<trial>/result.json`` of each real trial directory counts:
    a recursive scan could also find copies kept inside trial artifacts and
    change the ratio without any new trial. Returns ``None`` when no trial is
    found or too few match.
    """
    total = 0
    matched = 0
    matched_agents: set[str] = set()
    for job_dir in job_dirs:
        if not _is_real_directory(job_dir):
            continue
        try:
            trial_dirs = list(job_dir.iterdir())
        except OSError:
            continue
        for trial_dir in trial_dirs:
            if not _is_real_directory(trial_dir):
                continue
            data = _read_json(trial_dir / "result.json")
            if not isinstance(data, dict) or data.get("trial_name") != trial_dir.name or "agent_info" not in data:
                continue
            total += 1
            timed_out = _mapping(data.get("exception_info")).get("exception_type") == "AgentTimeoutError"
            # The collector's own proof of model output also reads the trial's trajectory,
            # so a slow run that did produce tokens is not mistaken for an unreachable endpoint.
            if (
                not timed_out
                or _mapping(data.get("agent_result")).get("n_input_tokens")
                or _trial_output_tokens(trial_dir, data) is not None
            ):
                continue
            matched += 1
            name = _mapping(data["agent_info"]).get("name")
            if isinstance(name, str) and name:
                matched_agents.add(name)
    if total == 0 or matched / total < ENDPOINT_FAILURE_THRESHOLD:
        return None
    return {
        "kind": DIAGNOSTIC_KIND,
        "matched_trials": matched,
        "total_trials": total,
        "agents": sorted(matched_agents),
    }


def provider_hosts(base_urls: Iterable[str | None]) -> list[str]:
    """Return the distinct host names of the model endpoints, never user info, path or query."""
    hosts: list[str] = []
    for base_url in base_urls:
        try:
            host = urlsplit(base_url or "").hostname
        except ValueError:
            continue
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def endpoint_failure_warning(diagnostic: Mapping[str, Any], *, env_mode: str, hosts: Iterable[str]) -> str:
    """Return the console warning for a :func:`detect_model_endpoint_failure` diagnostic."""
    named = ", ".join(hosts)
    where = f" ({named})" if named else ""
    return (
        f"Likely model endpoint failure: {diagnostic['matched_trials']} of {diagnostic['total_trials']} trials "
        "ended in AgentTimeoutError without a single model token (0 input, 0 output), so the agents probably "
        f"never got a response from the model endpoint{where}. The timeouts most likely reflect the environment "
        f"rather than the skill. Check proxy, firewall and DNS access from the {env_mode} environment to the "
        "endpoint and the agent credential, then rerun."
    )
