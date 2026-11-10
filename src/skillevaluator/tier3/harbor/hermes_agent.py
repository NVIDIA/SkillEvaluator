# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned, Docker-only Hermes execution through NVIDIA Build's native provider."""

from __future__ import annotations

import json
import os
import shlex
from typing import TYPE_CHECKING

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.hermes import Hermes

from skillevaluator.tier3.harbor import HERMES_VERSION

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.trajectories import Trajectory

HERMES_COMMIT = "f97608f178d1ffeca59860195ab7da295f7c8e5f"
_HOME = "/tmp/hermes"


class SkillEvaluatorHermes(Hermes):
    """Keep Harbor's session/ATIF contract without its OpenRouter fallback."""

    def get_version_command(self) -> str:
        return 'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"; hermes --version'

    async def install(self, environment: BaseEnvironment) -> None:
        if self._version not in {None, HERMES_VERSION}:
            raise ValueError(f"SkillEvaluator Hermes requires {HERMES_VERSION}")
        self._version = HERMES_VERSION
        await self.ensure_system_dependencies(environment, ("curl", "git", "ripgrep", "xz"))
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/{HERMES_COMMIT}/scripts/install.sh "
                "-o /tmp/install-hermes.sh && "
                f"HERMES_HOME={_HOME} bash /tmp/install-hermes.sh --skip-setup --skip-browser "
                f"--skip-computer-use --no-skills --branch {HERMES_VERSION} --commit {HERMES_COMMIT} && "
                'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH" && hermes --version'
            ),
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:  # noqa: ARG002
        model = self.model_name or ""
        if model.count("/") != 1 or any(not part.strip() for part in model.split("/")):
            raise ValueError("Hermes requires a NVIDIA Build catalog model ID in publisher/model form")
        key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not key:
            raise ValueError("Hermes requires NVIDIA_API_KEY; OpenRouter fallback is disabled")
        self._last_run_was_resume = self._resume
        runtime_env = {
            "HERMES_HOME": _HOME,
            "TERMINAL_ENV": "local",
            "HERMES_BUNDLED_SKILLS": f"{_HOME}/empty-bundled-skills",
        }
        self.skills_dir = self.skills_dir or "/workspace/skills"
        config = self._build_config_yaml(model, self.options.max_turns)
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {_HOME}/empty-bundled-skills && cat > {_HOME}/config.yaml << 'EOF'\n{config}EOF",
            env=runtime_env,
            timeout_sec=10,
        )
        for command in (self._build_register_mcp_servers_command(), self._build_register_skills_command()):
            if command:
                await self.exec_as_agent(environment, command=command, env=runtime_env, timeout_sec=10)
        parts = [
            "hermes --yolo chat",
            '-q "$HARBOR_INSTRUCTION" -Q',
            f"--provider nvidia --model {shlex.quote(model)}",
        ]
        if self._resume:
            session_id = getattr(self, "_native_session_id", None)
            if not session_id:
                raise RuntimeError("Cannot resume Hermes: the previous run did not export a session ID")
            parts.append(f"--resume {shlex.quote(session_id)}")
        if self.options.toolsets:
            parts.append(f"--toolsets {shlex.quote(self.options.toolsets)}")
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH" && '
                    f"{' '.join(parts)} 2>&1 | stdbuf -oL tee /logs/agent/hermes.txt"
                ),
                env={**runtime_env, "NVIDIA_API_KEY": key, "HARBOR_INSTRUCTION": instruction},
            )
        finally:
            # Export even after a nonzero exit or timeout; never replace the original error.
            try:
                result = await self.exec_as_agent(
                    environment,
                    command=(
                        'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH" && '
                        "hermes sessions export /logs/agent/hermes-session.jsonl --source oneshot && "
                        "test -s /logs/agent/hermes-session.jsonl && head -n 1 /logs/agent/hermes-session.jsonl"
                    ),
                    env=runtime_env,
                    timeout_sec=30,
                )
                session_id = self._extract_native_session_id(result.stdout or "")
                if session_id:
                    self._native_session_id = session_id
            except Exception:
                self.logger.warning("Hermes session export failed; inspect hermes.txt for execution diagnostics")

    def _convert_hermes_session_to_atif(self, jsonl_text: str, session_id: str) -> Trajectory | None:
        trajectory = super()._convert_hermes_session_to_atif(jsonl_text, session_id)
        if trajectory is None:
            return None
        # Pinned Hermes exports SQLite session totals, not per-message usage.
        sessions = []
        for line in jsonl_text.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("messages"), list):
                sessions.append(record)
        if trajectory.final_metrics:
            for field, source in (
                ("total_prompt_tokens", "input_tokens"),
                ("total_completion_tokens", "output_tokens"),
            ):
                values = [record.get(source) for record in sessions]
                if (
                    getattr(trajectory.final_metrics, field) is None
                    and values
                    and all(type(value) is int and value >= 0 for value in values)
                    and sum(values) > 0
                ):
                    setattr(trajectory.final_metrics, field, sum(values))
        for step in trajectory.steps:
            results = {result.source_call_id: result for result in step.observation.results} if step.observation else {}
            for call in step.tool_calls or []:
                if call.function_name not in {"terminal", "skill_view", "read_file"}:
                    continue
                result = results.get(call.tool_call_id)
                try:
                    payload = json.loads(result.content) if result and isinstance(result.content, str) else {}
                except (ValueError, TypeError):
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                if call.function_name == "terminal":
                    call.function_name = "bash"
                    exit_code = payload.get("exit_code")
                    if result and (type(exit_code) is int or payload.get("error")):
                        result.extra = {
                            **(result.extra or {}),
                            "tool_result_is_error": bool(payload.get("error")) or exit_code != 0,
                        }
                elif call.function_name == "skill_view":
                    # Only an observed successful manifest read is activation evidence.
                    if payload.get("success") is True and payload.get("content") and not payload.get("error"):
                        name = str(call.arguments.get("name", ""))
                        file_path = str(call.arguments.get("file_path") or "SKILL.md")
                        call.function_name = "read"
                        call.arguments = {"path": f"{_HOME}/skills/{name}/{file_path}"}
                    elif result:
                        result.extra = {**(result.extra or {}), "tool_result_is_error": True}
                elif result:
                    result.extra = {
                        **(result.extra or {}),
                        "tool_result_is_error": bool(payload.get("error")) or payload.get("success") is False,
                    }
        return trajectory

    def populate_context_post_run(self, context: AgentContext) -> None:
        # A previous/resumed run's trajectory is not evidence for this execution.
        path = self.logs_dir / "trajectory.json"
        path.unlink(missing_ok=True)
        super().populate_context_post_run(context)
        trajectory = json.loads(path.read_text()) if path.is_file() else {}
        metrics = trajectory.get("final_metrics") or {}
        if metrics.get("total_prompt_tokens") is None:
            context.n_input_tokens = None
        if metrics.get("total_completion_tokens") is None:
            context.n_output_tokens = None
        if context.n_input_tokens is None or context.n_output_tokens is None:
            self._reported_session_metrics = None
        if not path.is_file():
            context.metadata = {
                **(context.metadata or {}),
                "hermes_session_error": "missing_or_invalid_session",
            }
            raise RuntimeError("Hermes produced no valid session trajectory; inspect hermes.txt and session export")
        if not any(step.get("source") == "agent" for step in trajectory.get("steps", [])):
            context.metadata = {
                **(context.metadata or {}),
                "hermes_session_error": "no_agent_response",
            }
            raise RuntimeError("Hermes session contains no agent response; inspect hermes.txt for provider errors")
