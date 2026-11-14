# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private plugin evaluation lifecycle over the existing evaluation engine.

The runner consults :func:`active_runtime` on its calling thread. It is
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
    from skillevaluator.tier3.plugin_comparisons import ComparisonOptions
    from skillevaluator.tier3.plugin_evidence import EvidenceOptions, RuntimeEvidenceObserver
    from skillevaluator.tier3.plugin_prepare import PreparedRuntimePackage


class PrivateRuntimeHooks(Protocol):
    """The with-arm extensions the runner calls while a private plugin run is active."""

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        """The Harbor agent for ``agent``'s with arm, or ``None`` to keep the arm's ordinary agent."""

    def stage_with_arm(self, agent: str, task_dirs: Sequence[Path]) -> None:
        """Stage per-agent runtime additions into the with arm's staged tasks."""

    def baseline_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        """Classify pre-dispatch setup failures without adding plugin components."""

    def collect_evidence(self, jobs_dir: Path, result: dict[str, Any] | None) -> None:
        """Capture private evidence before the engine finalizes its artifacts."""

    def begin_attempt(self) -> None:
        """Start a new evidence scope before any attempt can return or fail."""


_ACTIVE_RUNTIME: ContextVar[PrivateRuntimeHooks | None] = ContextVar("skilleval_private_runtime", default=None)


def active_runtime() -> PrivateRuntimeHooks | None:
    """The private runtime hooks of the current run, or ``None`` for an ordinary run."""
    return _ACTIVE_RUNTIME.get()


def runtime_source_path(default: Path) -> Path:
    """Use the verified authored source when executing its generated package."""
    resolve = getattr(active_runtime(), "source_path", None)
    return resolve(default) if callable(resolve) else default


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
    package: PreparedRuntimePackage | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_error: str | None = None
    collected_jobs: set[Path] = field(default_factory=set)
    trial_censuses: dict[Path, Mapping[str, Any]] = field(default_factory=dict)
    trial_init_events: dict[Path, Mapping[str, Any]] = field(default_factory=dict)
    observer: RuntimeEvidenceObserver | None = None

    def configure_run(
        self,
        *,
        case_id_by_task_selector: Mapping[str, str] | None = None,
        arm_jobs: Mapping[str, Mapping[str, Path]] | None = None,
    ) -> None:
        """Supply coordinator-owned identities and additional private arm locations."""
        if self.observer is not None:
            self.observer.configure_run(case_id_by_task_selector=case_id_by_task_selector, arm_jobs=arm_jobs)

    def observe_collected_trials(
        self,
        agent: str,
        arm: str,
        job_dir: Path,
        collected: list[dict[str, Any]],
        scoreable: list[dict[str, Any]],
        selector_map: Mapping[str, str] | None,
    ) -> None:
        """Reuse the engine's reward reads; behavior evidence never participates in scoring."""
        if self.observer is not None:
            self.observer.observe_collected_trials(agent, arm, job_dir, collected, scoreable, selector_map)

    def begin_attempt(self) -> None:
        # Replace dictionaries: earlier results retain their own evidence even
        # when the next attempt fails before creating any artifacts.
        self.stagings = {}
        self.evidence = {}
        self.evidence_error = None
        self.collected_jobs = set()
        if self.observer is not None:
            self.observer.begin_attempt()

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        decision = self.decisions.get(agent)
        if decision is None or not decision.native:
            return self.baseline_agent_import_path(agent, base_import_path)
        return native_agent_import_path(agent, base_import_path)

    def baseline_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        from skillevaluator.tier3.plugin_native import private_agent_import_path

        return private_agent_import_path(agent, base_import_path)

    def source_path(self, default: Path) -> Path:
        """A prepared package binds the generated wrapper to this authored source."""
        return self.source.plugin_root if self.package is not None else default

    def stage_with_arm(self, agent: str, task_dirs: Sequence[Path]) -> None:
        decision = self.decisions.get(agent)
        if decision is None or not decision.native:
            for task_dir in task_dirs:
                _stage_task_mcp(task_dir, self.source)
            return
        import tomllib

        from skillevaluator.tier3.harbor.native_staging import (
            build_native_task_staging,
            stage_native_bundle,
            validate_native_mcp_source,
        )

        if self.package is not None:
            from skillevaluator.tier3.plugin_prepare import verify_source_identity

            verify_source_identity(self.package)
        for task_dir in task_dirs:
            task_config = task_dir / "task.toml"
            runtime_env = (
                tomllib.loads(task_config.read_text(encoding="utf-8")).get("environment", {}).get("env", {})
                if task_config.is_file()
                else {}
            )
            harness = HARNESS_ADAPTERS[agent]
            staging = self.stagings.get(agent)
            if staging is None:
                staging = build_native_task_staging(agent, harness, self.source, allowed_runtime_env=runtime_env)
                self.stagings[agent] = staging
            else:
                validate_native_mcp_source(harness, self.source, allowed_runtime_env=runtime_env)
            env_dir = task_dir / "environment"
            dockerfile = env_dir / "Dockerfile"
            if dockerfile.is_symlink() or not dockerfile.is_file():
                raise PluginLoadError(f"native plugin loading needs a task Dockerfile: {dockerfile}")
            _stage_native_skills(task_dir, staging)
            if staging.plugin_mcp_via_task:
                _stage_task_mcp(task_dir, self.source)
            lines = stage_native_bundle(env_dir, staging, excluded_roots=self.source.excluded_paths)
            content = dockerfile.read_text(encoding="utf-8")
            separator = "" if content.endswith("\n") else "\n"
            dockerfile.write_text(content + separator + "\n".join(lines) + "\n", encoding="utf-8")
        if self.package is not None:
            verify_source_identity(self.package)

    def collect_evidence(self, jobs_dir: Path, result: dict[str, Any] | None) -> None:
        if jobs_dir in self.collected_jobs:
            return
        self.collected_jobs.add(jobs_dir)
        from skillevaluator.tier3.plugin_prepare import wrapper_skill_name

        self.evidence = {}
        self.evidence_error = None
        if self.stagings:
            try:
                self.evidence = collect_native_evidence(self, jobs_dir, wrapper_skill_name(self.source.plugin_name))
            except Exception as exc:
                self.evidence_error = f"{type(exc).__name__}: {exc}"[:600]
        if result is not None:
            result["native_evidence"] = self.evidence
            if self.evidence_error:
                result["native_evidence_error"] = self.evidence_error
        if self.observer is not None:
            try:
                self.observer.collect(jobs_dir, result)
            except Exception as exc:
                self.observer.record_failure(exc)
                if result is not None:
                    result["runtime_evidence"] = self.observer.provenance()


def _stage_native_skills(task_dir: Path, staging: Any) -> None:
    """Project exactly the layout consumed by the selected native adapter."""
    import json
    import shutil

    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore, _sanitize_staged_runtime_skills
    from skillevaluator.tier3.harbor.secure_copy import copytree_secure
    from skillevaluator.tier3.plugin_prepare import wrapper_skill_name

    skills = task_dir / "environment" / "skills"
    wrapper = skills / wrapper_skill_name(staging.source.plugin_name)
    if not staging.stage_wrapper_skill and wrapper.exists():
        shutil.rmtree(wrapper)
    if staging.stage_member_skills:
        for source in staging.source.member_skills:
            target = skills / source.name
            if target.exists() or target.is_symlink():
                raise PluginLoadError(f"native member skill conflicts with staged skill: {source.name}")
            copytree_secure(
                source,
                target,
                allowed_root=staging.source.plugin_root,
                ignore=_runtime_skill_copy_ignore(source, staging.source.excluded_paths),
            )
        _sanitize_staged_runtime_skills(skills)
    entry_path = task_dir / "tests" / "entry.json"
    if entry_path.is_file():
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        entry["workspace_skill_names"] = sorted(
            set(entry.get("workspace_skill_names", []))
            | {path.name for path in staging.source.member_skills}
            | set(staging.workspace_skill_aliases())
        )
        entry_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def _stage_task_mcp(task_dir: Path, source: NativePluginSource) -> None:
    """Add portable MCP declarations only to the private with arm."""
    import json
    import tomllib

    from skillevaluator.tier3.harbor.adapter import _toml_quote, _toml_value, validate_mcp_server_declarations

    if not source.mcp_servers:
        return
    path = task_dir / "task.toml"
    content = path.read_text(encoding="utf-8")
    environment = tomllib.loads(content).get("environment", {})
    existing = environment.get("mcp_servers", [])
    servers = []
    names = {server["name"] for server in existing}
    for raw in source.mcp_servers:
        server = {
            key: value
            for key, value in raw.items()
            if key in {"name", "command", "args", "url", "env", "headers", "transport"}
        }
        if "transport" not in server:
            server["transport"] = raw.get("type") or ("stdio" if raw.get("command") else "http")
        if server["name"] in names:
            raise PluginLoadError(f"plugin MCP name conflicts with task environment: {server['name']}")
        names.add(server["name"])
        servers.append(server)
    validate_mcp_server_declarations(
        servers, allowed_runtime_env=environment.get("env", {}), source_label="private runtime MCP"
    )
    for server in servers:
        content += "\n[[environment.mcp_servers]]\n"
        content += "".join(f"{_toml_quote(key)} = {_toml_value(value)}\n" for key, value in server.items())
    tomllib.loads(content)
    path.write_text(content, encoding="utf-8")
    (task_dir / "mcp_servers.json").write_text(json.dumps([*existing, *servers], indent=2), encoding="utf-8")


@dataclass
class PrivateRunOutcome:
    """A private plugin run's result, its evidence, and what happened around it."""

    result: dict[str, Any]
    attempts: int
    provenance: dict[str, Any]
    cleanup_error: str | None = None
    publication_error: str | None = None
    retry_reasons: list[str] = field(default_factory=list)
    setup_attempt_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return str(self.result.get("execution_status") or "failed")


def setup_only_failure(result: Mapping[str, Any]) -> str | None:
    """The setup failure that makes ``result`` safe to rerun, or ``None``.

    Every expected trial in every executed arm must have a complete, typed
    setup failure record. Collector diagnostics about missing scores are
    consequences of that failure, not evidence that the agent ran. Strings
    alone, sampled details, successful arms and missing trials are insufficient.
    """
    if result.get("execution_status") != "failed":
        return None
    agents = result.get("agents")
    if not isinstance(agents, Mapping) or not agents:
        return None
    reason = None
    for info in agents.values():
        if not isinstance(info, Mapping):
            return None
        conditions = info.get("conditions", {})
        if not isinstance(conditions, Mapping) or not conditions:
            return None
        for arm, condition in conditions.items():
            if not isinstance(condition, Mapping):
                return None
            if condition.get("execution_status") == "skipped":
                continue
            expected = condition.get("expected_attempts")
            if condition.get("execution_status") != "failed" or condition.get("scored_attempts") != 0:
                return None
            if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
                return None
            failures = info.get("trial_failures", {}).get(arm, [])
            metadata = info.get("failure_detail_metadata", {}).get(f"{arm}_trials", {})
            if metadata.get("details_truncated") or metadata.get("details_total") != len(failures):
                return None
            if len(failures) != expected or len({f.get("trial") for f in failures}) != expected:
                return None
            for failure in failures:
                typed_reason = failure.get("reason", "")
                if not failure.get("trial") or not typed_reason.startswith(SETUP_RETRY_EXCEPTION + ":"):
                    return None
                reason = typed_reason[:300]
    return reason


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
    if type(max_setup_retries) is not int or not 0 <= max_setup_retries <= 5:
        raise ValueError("max_setup_retries must be an integer between 0 and 5")
    if evaluate is None:
        from skillevaluator.evaluation.service import EvaluationService

        evaluate = EvaluationService().evaluate
    outcome = PrivateRunOutcome(result={}, attempts=0, provenance=dict(provenance))
    try:
        while True:
            outcome.attempts += 1
            if begin_attempt := getattr(hooks, "begin_attempt", None):
                begin_attempt()
            if hooks is None:
                outcome.result = evaluate(options)
            else:
                with private_runtime(hooks):
                    outcome.result = evaluate(options)
            reason = setup_only_failure(outcome.result)
            if reason is None or outcome.attempts > max_setup_retries:
                break
            outcome.retry_reasons.append(reason)
            outcome.setup_attempt_results.append(outcome.result)
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


def evaluate_prepared_plugin(
    package: PreparedRuntimePackage,
    options: EvaluationOptions,
    *,
    work_dir: Path,
    load: str = "wrapper",
    evals_source: Path | None = None,
    evaluate: Callable[[EvaluationOptions], dict[str, Any]] | None = None,
    publish: Callable[[PrivateRunOutcome], None] | None = None,
    evidence_options: EvidenceOptions | None = None,
    observer_factory: Callable[..., RuntimeEvidenceObserver] | None = None,
    comparison_options: ComparisonOptions | None = None,
    max_setup_retries: int = MAX_SETUP_RETRIES,
) -> PrivateRunOutcome:
    """Evaluate a prepared plugin through the existing engine and remove the operation-owned staging.

    Results are written under the plugin's own ``evals/results`` (or the
    caller's ``results_dir``), never beneath the temporary wrapper, so they
    survive cleanup. Agents whose decision is native load the plugin through
    their harness in the with arm only; the baseline arm stays ordinary.
    """
    import shutil
    import tempfile
    from dataclasses import replace

    from skillevaluator.tier3.commands import parse_agents
    from skillevaluator.tier3.plugin_evidence import EvidenceOptions, RuntimeEvidenceObserver
    from skillevaluator.tier3.plugin_native import apply_native_refusals, resolve_plugin_load
    from skillevaluator.tier3.plugin_prepare import bind_runtime_results, stage_wrapper_skill, wrapper_skill_name

    if not package.runnable:
        # Nothing was evaluated: INCOMPLETE with its recorded reasons, never a successful skip.
        reasons = "; ".join(package.incomplete_reasons)
        raise PluginLoadError(
            "plugin has no runnable skill, rule, MCP server, or text component" + (f" ({reasons})" if reasons else "")
        )
    agents = parse_agents(options.agents)
    if load != "wrapper" and not agents:
        raise PluginLoadError("native plugin loading needs explicit agents")
    if comparison_options is not None:
        from skillevaluator.tier3.plugin_comparisons import ComparisonOptions, ComparisonRuntime

        if not isinstance(comparison_options, ComparisonOptions):
            raise ValueError("comparison_options must be private ComparisonOptions")
        if not agents:
            raise PluginLoadError("private comparisons need explicit agents")
    decisions = apply_native_refusals(
        load, resolve_plugin_load(load, agents, env_mode=options.env_mode), package.native_source
    )
    results_dir = options.results_dir or package.native_source.plugin_root / "evals" / "results"
    package = bind_runtime_results(package, results_dir)
    temp = Path(tempfile.mkdtemp(prefix="skilleval-plugin-", dir=work_dir))
    try:
        wrapper = stage_wrapper_skill(
            package, temp / wrapper_skill_name(package.plugin_name), evals_source=evals_source
        )
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    hooks = NativeRuntimeHooks(package.native_source, decisions, package=package)
    comparison_runtime = None
    try:
        hooks.observer = (observer_factory or RuntimeEvidenceObserver)(
            package, wrapper, hooks, evidence_options or EvidenceOptions()
        )
        if comparison_options is not None:
            comparison_runtime = ComparisonRuntime(
                hooks, package, comparison_options, agents, baseline=not options.skip_baseline
            )
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    provenance = {
        "plugin": package.plugin_name,
        "source": {
            "root": package.identity.root,
            "manifest": package.identity.manifest,
            "manifest_sha256": package.identity.manifest_sha256,
            "source_tree_sha256": package.identity.source_tree_sha256,
        },
        "wrapper_skill": wrapper.name,
        "case_sources": dict(wrapper.case_sources),
        "load_decisions": {agent: decision.to_dict() for agent, decision in decisions.items()},
        "audit": [vars(row) for row in package.audit],
        "incomplete_reasons": list(package.incomplete_reasons),
        "complete": package.complete,
    }
    if evaluate is None:
        from skillevaluator.evaluation.service import EvaluationService

        evaluate = EvaluationService().evaluate

    def publish_with_evidence(outcome: PrivateRunOutcome) -> None:
        if hooks.stagings:
            outcome.provenance["native_evidence"] = hooks.evidence
        if hooks.evidence_error:
            outcome.provenance["native_evidence_error"] = hooks.evidence_error
        if hooks.observer is not None:
            observed = hooks.observer.provenance()
            outcome.provenance.update(runtime_evidence=observed, mcp_proof=observed["mcp_proof"])
        if comparison_runtime is not None:
            outcome.provenance["private_comparisons"] = comparison_runtime.provenance()
        # Preserve a passive view while preparation provenance and all arm
        # summaries are still available. Its failure cannot suppress caller
        # publication or change the engine's status and cleanup semantics.
        try:
            from skillevaluator.reporting.plugin_runtime import project_plugin_runtime
            from skillevaluator.reporting.plugin_runtime_io import save_plugin_runtime_report

            report = project_plugin_runtime(
                outcome.result,
                provenance={**outcome.provenance, "attempts": outcome.attempts, "retry_reasons": outcome.retry_reasons},
            )
            if report is not None:
                outcome.provenance["runtime_report"] = report.payload()
                if run_dir := outcome.result.get("run_dir"):
                    save_plugin_runtime_report(report, Path(run_dir))
        except Exception as exc:
            from skillevaluator.utils.redaction import redact_printable_text

            error = redact_printable_text(f"Runtime report: {type(exc).__name__}: {exc}", max_len=600)
            outcome.provenance["runtime_report_error"] = error
            outcome.publication_error = error
        if publish is not None:
            publish(outcome)

    def cleanup() -> None:
        shutil.rmtree(temp)

    return run_private_evaluation(
        replace(options, skill_path=wrapper.path, results_dir=results_dir),
        hooks=comparison_runtime or hooks,
        provenance=provenance,
        evaluate=evaluate,
        publish=publish_with_evidence,
        cleanup=cleanup,
        max_setup_retries=max_setup_retries,
    )


_MAX_AGENT_LOG_BYTES = 8 * 1024 * 1024


def collect_native_evidence(hooks: NativeRuntimeHooks, jobs_dir: Path, skill_name: str) -> dict[str, Any]:
    """Per-agent load evidence of the native with arms, read from the retained Harbor jobs.

    A trial without a census file contributes only its staging plan, so a
    component is never reported loaded from staging alone; Claude Code's own
    ``system/init`` event confirms or refutes what the census listed.
    """
    from skillevaluator.tier3 import plugin_native as native

    evidence: dict[str, Any] = {}
    for agent, staging in sorted(hooks.stagings.items()):
        plan = staging.bundle.census_plan()
        declared = plan["declared"]
        job = jobs_dir / f"{skill_name}-{agent}-with"
        trials = sorted(p for p in job.iterdir() if (p / "agent").is_dir()) if job.is_dir() else []
        censuses = []
        for trial in trials:
            census = _trial_census(native, agent, trial / "agent", declared, plan, init_events=hooks.trial_init_events)
            hooks.trial_censuses[trial] = census
            censuses.append(census)
        summary = native.summarize_censuses(agent, "native", censuses, staged_hook_ids=plan["hook_ids"])
        evidence[agent] = {**summary, "trials": len(censuses), "job": job.name}
    return evidence


def _trial_census(
    native: Any,
    agent: str,
    logs: Path,
    declared: Any,
    plan: Mapping[str, Any],
    *,
    init_events: dict[Path, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = native.read_census_file(logs / native.LOAD_CENSUS_FILENAME)
    if raw is None:
        return {**native.fallback_census(agent, "native", declared), "fallback": True}
    census = native.restrict_census(raw, declared, agent=agent)
    harness = plan.get("harness") or {}
    log = logs / "claude-code.txt"
    if agent == "claude-code" and harness.get("plugin") and log.is_file() and not log.is_symlink():
        with log.open("rb") as handle:
            text = handle.read(_MAX_AGENT_LOG_BYTES).decode("utf-8", errors="replace")
        if (init := native.claude_init_event(text)) is not None:
            if init_events is not None:
                init_events[logs.parent] = init
            census = native.apply_claude_init_evidence(census, declared, init, plugin=harness["plugin"])
    return census
