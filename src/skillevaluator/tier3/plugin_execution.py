# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private plugin evaluation lifecycle over the existing evaluation engine.

The runner consults :func:`active_runtime` at its with-arm seams only. It is
``None`` unless :func:`private_runtime` is active, so ordinary skill runs are
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from skillevaluator.tier3.plugin_native import (
    HARNESS_ADAPTERS,
    AgentLoadDecision,
    NativePluginSource,
    PluginLoadError,
    native_agent_import_path,
)

if TYPE_CHECKING:
    from skillevaluator.evaluation.options import EvaluationOptions


class PrivateRuntimeHooks(Protocol):
    """The with-arm extensions the runner calls while a private plugin run is active."""

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        """The Harbor agent for ``agent``'s with arm, or ``None`` to keep the arm's ordinary agent."""

    def stage_with_arm(self, agent: str, task_dirs: Sequence[Path]) -> None:
        """Stage per-agent runtime additions into the with arm's staged tasks."""


_ACTIVE_RUNTIME: ContextVar[PrivateRuntimeHooks | None] = ContextVar("skilleval_private_runtime", default=None)


def active_runtime() -> PrivateRuntimeHooks | None:
    """The private runtime hooks of the current run, or ``None`` for an ordinary run."""
    return _ACTIVE_RUNTIME.get()


@contextmanager
def private_runtime(hooks: PrivateRuntimeHooks) -> Iterator[PrivateRuntimeHooks]:
    """Activate ``hooks`` for the evaluation run inside the block."""
    token = _ACTIVE_RUNTIME.set(hooks)
    try:
        yield hooks
    finally:
        _ACTIVE_RUNTIME.reset(token)


#: The only failure a private run retries: Harbor's agent setup failed on the network before the task.
SETUP_RETRY_EXCEPTION = "AgentSetupNetworkError"
MAX_SETUP_RETRIES = 2


@dataclass
class NativeRuntimeHooks:
    """Native plugin loading for the with arm of the agents whose decision is native."""

    source: NativePluginSource
    decisions: Mapping[str, AgentLoadDecision]
    stagings: dict[str, Any] = field(default_factory=dict)

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        decision = self.decisions.get(agent)
        if decision is None or not decision.native:
            return None
        return native_agent_import_path(agent, base_import_path)

    def stage_with_arm(self, agent: str, task_dirs: Sequence[Path]) -> None:
        decision = self.decisions.get(agent)
        if decision is None or not decision.native:
            return
        from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle

        staging = self.stagings.get(agent) or build_native_task_staging(agent, HARNESS_ADAPTERS[agent], self.source)
        self.stagings[agent] = staging
        for task_dir in task_dirs:
            env_dir = task_dir / "environment"
            dockerfile = env_dir / "Dockerfile"
            if dockerfile.is_symlink() or not dockerfile.is_file():
                raise PluginLoadError(f"native plugin loading needs a task Dockerfile: {dockerfile}")
            lines = stage_native_bundle(env_dir, staging, excluded_roots=self.source.excluded_paths)
            content = dockerfile.read_text(encoding="utf-8")
            separator = "" if content.endswith("\n") else "\n"
            dockerfile.write_text(content + separator + "\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class PrivateRunOutcome:
    """A private plugin run's result, its evidence, and what happened around it."""

    result: dict[str, Any]
    attempts: int
    provenance: dict[str, Any]
    cleanup_error: str | None = None
    publication_error: str | None = None
    retry_reasons: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return str(self.result.get("execution_status") or "failed")


def setup_only_failure(result: Mapping[str, Any]) -> str | None:
    """The setup failure that makes ``result`` safe to rerun, or ``None``.

    Only a failed run whose every recorded error is the setup-network
    classification is rerun: that agent never received its task. Any other
    error (dispatch, agent, verifier, cancellation, timeout) may have run part
    of the task and is final.
    """
    if result.get("execution_status") == "succeeded":
        return None
    errors = result.get("error") or result.get("execution_errors") or []
    errors = [errors] if isinstance(errors, str) else [str(error) for error in errors]
    if errors and all(SETUP_RETRY_EXCEPTION in error for error in errors):
        return errors[0][:300]
    return None


def run_private_evaluation(
    options: EvaluationOptions,
    *,
    hooks: PrivateRuntimeHooks | None,
    provenance: Mapping[str, Any],
    evaluate: Callable[[EvaluationOptions], dict[str, Any]] | None = None,
    publish: Callable[[PrivateRunOutcome], None] | None = None,
    cleanup: Callable[[], None] | None = None,
    max_setup_retries: int = MAX_SETUP_RETRIES,
) -> PrivateRunOutcome:
    """Run the existing engine for a prepared plugin, then publish evidence, then clean up.

    Evidence is published before cleanup removes the operation-owned staging,
    and the run's own status survives a failed publication or cleanup.
    Cancellation (``KeyboardInterrupt``) is never retried and still cleans up.
    """
    if not 0 <= max_setup_retries <= MAX_SETUP_RETRIES:
        raise ValueError(f"max_setup_retries must be between 0 and {MAX_SETUP_RETRIES}")
    if evaluate is None:
        from skillevaluator.evaluation.service import EvaluationService

        evaluate = EvaluationService().evaluate
    outcome = PrivateRunOutcome(result={}, attempts=0, provenance=dict(provenance))
    try:
        while True:
            outcome.attempts += 1
            if hooks is None:
                outcome.result = evaluate(options)
            else:
                with private_runtime(hooks):
                    outcome.result = evaluate(options)
            reason = setup_only_failure(outcome.result)
            if reason is None or outcome.attempts > max_setup_retries:
                break
            outcome.retry_reasons.append(reason)
        if publish is not None:
            try:
                publish(outcome)
            except Exception as exc:
                outcome.publication_error = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        if cleanup is not None:
            try:
                cleanup()
            except Exception as exc:
                outcome.cleanup_error = f"{type(exc).__name__}: {exc}"[:600]
    return outcome
