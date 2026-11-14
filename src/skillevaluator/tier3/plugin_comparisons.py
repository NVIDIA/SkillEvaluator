# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private comparison planning over the existing runtime and its cached evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor import collector
from skillevaluator.tier3.harbor.loose_parts_arm import (
    LOAD_EVIDENCE_FILENAME,
    LoosePartsPlan,
    compact_load_evidence,
    plan_loose_parts,
)
from skillevaluator.tier3.harbor.metrics import finite_number
from skillevaluator.tier3.harbor.stats import (
    ArmObservations,
    TrialObservation,
    build_agent_statistics,
    integration_claim,
)
from skillevaluator.tier3.plugin_prepare import PreparationError, verify_source_identity
from skillevaluator.tier3.runtime_arms import ARM_LOOSE_PARTS, ARM_SUM_OF_PARTS, ARM_WITH, ARM_WITHOUT

VARIANTS = {ARM_WITH: "with", ARM_WITHOUT: "without", ARM_SUM_OF_PARTS: "sumofparts", ARM_LOOSE_PARTS: "looseparts"}
COMPARISON_FILENAME = "private_comparisons.json"


@dataclass(frozen=True)
class ComparisonOptions:
    """Explicit private controls; public options and the ordinary two-arm path stay separate."""

    member_skills: bool = True
    loose_parts: bool = False

    def __post_init__(self) -> None:
        if type(self.member_skills) is not bool or type(self.loose_parts) is not bool:
            raise ValueError("comparison options must be boolean")
        if self.loose_parts and not self.member_skills:
            raise ValueError("loose-parts comparisons require the member-skills arm")


def observation_usage(record: Any) -> dict[str, float]:
    """Normalize observed counters once; absent counters and actual zero remain distinct."""
    raw, first = record.usage or {}, record.first_turn_usage or {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "total_prompt_tokens"),
        "completion_tokens": ("completion_tokens", "total_completion_tokens"),
        "cached_tokens": ("cached_tokens", "total_cached_tokens"),
        "cost_usd": ("cost_usd", "total_cost_usd"),
    }
    usage = {}
    for name, candidates in aliases.items():
        for candidate in candidates:
            value = finite_number(raw.get(candidate), non_negative=True)
            if value is not None:
                usage[name] = value
                break
    excluded = getattr(record, "first_turn_excluded_reason", None)
    if first.get("first_turn_hosted_search") or excluded == "hosted_search":
        usage["first_turn_hosted_search"] = 1
    elif not excluded:
        value = finite_number(first.get("prompt_tokens", first.get("total_prompt_tokens")), non_negative=True)
        if value is not None:
            usage["first_turn_prompt_tokens"] = value
    return usage


def arm_observations(
    records: Sequence[Any],
    *,
    expected_case_ids: Sequence[str],
    n_attempts: int,
    stop_on_pass: bool,
    pass_threshold: float,
    job_failure: str = "",
    recovered_attempts: Sequence[dict[str, Any]] = (),
) -> ArmObservations:
    """Use the collector's existing score and attempt rules on cached logical records."""
    from skillevaluator.tier3.attempt_verdict import attempt_verdict

    expected = set(expected_case_ids)
    rewards, scored, unscored = [], [], []
    attempts: dict[str, list[tuple[int | None, float | None, bool]]] = {case: [] for case in expected_case_ids}
    identities: set[str] = set()
    complete = bool(expected)
    for record in records:
        case = record.case_id if record.case_id in expected else ""
        reward = dict(record.reward) if isinstance(record.reward, Mapping) else {}
        reward["entry_id"] = case
        identity = getattr(record, "trial_id", None)
        if identity:
            if identity in identities:
                complete = False
            identities.add(identity)
            reward.setdefault("_trial_root_name", identity)
        usage = observation_usage(record)
        if not case:
            usage.pop("first_turn_prompt_tokens", None)
        score = collector._overall_score(reward) if case else None
        if case:
            ordinal = collector._attempt_ordinal(reward) or collector._positive_attempt_ordinal(
                getattr(record, "attempt_ordinal", None)
            )
            passed = attempt_verdict(score, pass_threshold, reward=reward).passed
            attempts[case].append((ordinal, score, passed))
        else:
            complete = False
        observation = TrialObservation(
            case,
            reward,
            usage,
            score if reward.get("_logical_mixed_contracts") else None,
        )
        if score is None:
            unscored.append(observation)
        else:
            rewards.append(reward)
            scored.append(observation)
    common = {
        "expected_case_ids": list(expected_case_ids),
        "n_attempts": n_attempts,
        "stop_on_pass": stop_on_pass,
        "pass_threshold": pass_threshold,
        "recovered_attempts": list(recovered_attempts),
    }
    summary = collector._pass_summary(rewards, expected_cases=len(expected_case_ids), **common)
    execution = collector._condition_execution_summary(
        rewards,
        expected_cases=len(expected_case_ids),
        job_failure=job_failure,
        **common,
    )
    for case_attempts in attempts.values():
        ordinals = [ordinal for ordinal, _score, _passed in case_attempts]
        explicit = [ordinal for ordinal in ordinals if ordinal is not None]
        if explicit and len(explicit) != len(ordinals):
            complete = False
        ordered = sorted(case_attempts, key=lambda item: item[0] or 0) if explicit else case_attempts
        required = n_attempts
        if stop_on_pass:
            first_pass = next(
                (
                    (index, ordinal)
                    for index, (ordinal, score, passed) in enumerate(ordered, 1)
                    if score is not None and passed
                ),
                None,
            )
            if first_pass:
                required = first_pass[1] or first_pass[0]
        if required > n_attempts or len(case_attempts) != required:
            complete = False
        if explicit and set(explicit) != set(range(1, required + 1)):
            complete = False
    return ArmObservations(
        execution["execution_status"], tuple(scored), summary, job_failure, tuple(unscored), complete
    )


class ComparisonRuntime:
    """Delegate native execution, adding private arms and advisory comparison artifacts."""

    def __init__(self, native: Any, package: Any, options: ComparisonOptions, agents: Sequence[str], *, baseline: bool):
        self.native, self.package, self.options = native, package, options
        self.agents, self.baseline = tuple(agents), baseline
        self.comparison_plan = self
        self.task_sets: dict[str, dict[str, tuple[Path, ...]]] = {}
        self.jobs: dict[str, dict[str, Path]] = {}
        self.launch_errors: dict[str, dict[str, str]] = {agent: {} for agent in agents}
        self.expected_case_ids: tuple[str, ...] = ()
        self.n_attempts, self.stop_on_pass, self.pass_threshold = 1, False, 0.5
        self.data: dict[str, Any] = {"status": "not_collected", "agents": {}, "errors": []}
        self.members = package.native_source.member_skills if options.member_skills else ()
        self.loose_plan = LoosePartsPlan(requested=options.loose_parts, order=self.agents)

    def begin_attempt(self) -> None:
        """Start a fresh comparison attempt alongside the native runtime's reset."""
        begin = getattr(self.native, "begin_attempt", None)
        if begin is not None:
            begin()
        self.task_sets, self.jobs = {}, {}
        self.launch_errors = {agent: {} for agent in self.agents}
        self.expected_case_ids = ()
        self.data = {"status": "not_collected", "agents": {}, "errors": []}
        self.loose_plan = LoosePartsPlan(requested=self.options.loose_parts, order=self.agents)

    def source_path(self, default: Path) -> Path:
        resolve = getattr(self.native, "source_path", None)
        return resolve(default) if callable(resolve) else default

    def configure_agent_routes(self, paths: Mapping[str, str]) -> None:
        """Bind the coordinator-resolved routes once for this engine attempt."""
        self.task_sets = {}
        self.launch_errors = {agent: {} for agent in self.agents}
        self.loose_plan = plan_loose_parts(
            requested=self.options.loose_parts,
            agents=self.agents,
            decisions=self.native.decisions,
            source=self.package.native_source,
            agent_import_paths=paths,
            unavailable=None if self.baseline and self.members else "the ladder requires a baseline and member skills",
        )

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        return self.native.with_agent_import_path(agent, base_import_path)

    def baseline_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        return self.native.baseline_agent_import_path(agent, base_import_path)

    def stage_with_arm(self, agent: str, task_dirs: Sequence[Path]) -> None:
        self.native.stage_with_arm(agent, task_dirs)

    def observe_collected_trials(self, *args: Any, **kwargs: Any) -> None:
        observe = getattr(self.native, "observe_collected_trials", None)
        if observe is not None:
            observe(*args, **kwargs)

    def configure_run(self, **kwargs: Any) -> None:
        self.data = {"status": "not_collected", "agents": {}, "errors": []}
        self.jobs = kwargs.get("arm_jobs") or {}
        configure = getattr(self.native, "configure_run", None)
        if configure is not None:
            configure(**kwargs)

    def run_config(self) -> dict[str, Any]:
        return {"member_skills": self.options.member_skills, "loose_parts": self.loose_plan.run_config()}

    def stage_additional(
        self,
        agent: str,
        emit_control: Callable[[str, Sequence[Path]], Sequence[Path]],
        task_sets: Mapping[str, Sequence[Path]],
        base_import_path: str | None,
    ) -> tuple[tuple[str, Path, str | None], ...]:
        verify_source_identity(self.package)
        staged = {arm: tuple(paths) for arm, paths in task_sets.items()}
        if ARM_WITH not in staged:
            self.task_sets[agent] = staged
            self.loose_plan.arms.pop(agent, None)
            self.loose_plan.not_measured[agent] = "with-skill arm disabled; private comparison arms require it"
            return ()
        jobs = []
        if self.members:
            paths = tuple(emit_control(VARIANTS[ARM_SUM_OF_PARTS], self.members))
            if not paths:
                raise PreparationError("member-skills arm has no emitted tasks")
            staged[ARM_SUM_OF_PARTS] = paths
            jobs.append((VARIANTS[ARM_SUM_OF_PARTS], paths[0].parent, base_import_path))
        loose = self.loose_plan.arm_for(agent)
        if loose is not None:
            from skillevaluator.tier3.harbor.native_staging import stage_native_bundle

            paths = tuple(emit_control(VARIANTS[ARM_LOOSE_PARTS], self.members))
            if not paths:
                raise PreparationError("loose-parts arm has no emitted tasks")
            for path in paths:
                environment = path / "environment"
                dockerfile = environment / "Dockerfile"
                if dockerfile.is_symlink() or not dockerfile.is_file():
                    raise PreparationError("loose-parts staging needs an owned task Dockerfile")
                lines = stage_native_bundle(
                    environment, loose.staging, excluded_roots=self.package.native_source.excluded_paths
                )
                text = dockerfile.read_text(encoding="utf-8")
                dockerfile.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
            staged[ARM_LOOSE_PARTS] = paths
            jobs.append((VARIANTS[ARM_LOOSE_PARTS], paths[0].parent, loose.import_path))
        self.task_sets[agent] = staged
        verify_source_identity(self.package)
        return tuple(jobs)

    def bind_execution(
        self, case_ids: Sequence[str], *, n_attempts: int, stop_on_pass: bool, pass_threshold: float
    ) -> None:
        from skillevaluator.tier3.harbor.stats import MAX_COMPARISON_CASES

        if len(case_ids) > MAX_COMPARISON_CASES:
            raise ValueError("private comparison case limit exceeded")
        self.expected_case_ids = tuple(case_ids)
        self.n_attempts, self.stop_on_pass, self.pass_threshold = n_attempts, stop_on_pass, pass_threshold
        for agent, arms in self.task_sets.items():
            selected = {arm: tuple(path for path in paths if path.is_dir()) for arm, paths in arms.items()}
            self.task_sets[agent] = selected
            compared = {arm: selected.get(arm, ()) for arm in (ARM_WITH, ARM_SUM_OF_PARTS, ARM_LOOSE_PARTS)}
            self.loose_plan.hash_tasks(agent, compared)
            loose = self.loose_plan.arm_for(agent)
            if loose is not None and any(len(paths) != len(case_ids) for paths in compared.values()):
                loose.content.update(equal_across_arms=None, error="Expected emitted task coverage is incomplete")

    def _cache_init_events(self) -> None:
        """Read loose-arm startup evidence once, alongside the observer's artifact cache."""
        from skillevaluator.tier3 import plugin_native
        from skillevaluator.tier3.plugin_evidence import _trial_roots

        cache = getattr(self.native, "trial_init_events", None)
        if cache is None:
            return
        for agent in self.agents:
            if self.loose_plan.arm_for(agent) is None:
                continue
            for arm in (ARM_SUM_OF_PARTS, ARM_LOOSE_PARTS):
                job = self.jobs.get(agent, {}).get(arm)
                if job is None or not job.is_dir():
                    continue
                roots, _limited = _trial_roots(job)
                for root in roots:
                    if root in cache:
                        continue
                    text = collector._read_bounded_text(root / "agent" / "claude-code.txt", max_bytes=8 * 1024 * 1024)
                    cache[root] = plugin_native.claude_init_event(text) if text else None

    def collect_evidence(self, jobs_dir: Path, result: dict[str, Any] | None) -> None:
        try:
            self._cache_init_events()
        except Exception as exc:
            self._record_error(exc)
        self.native.collect_evidence(jobs_dir, result)
        if result is None:
            return
        try:
            self._collect_statistics(result, jobs_dir)
        except Exception as exc:
            self._record_error(exc)
        self.data["status"] = "incomplete" if self.data["errors"] else "recorded"
        self._write_artifact(jobs_dir.parent / COMPARISON_FILENAME, jobs_dir.parent, self.provenance)
        result["private_comparisons"] = self.provenance()

    def _record_error(self, exc: Exception, *, artifact: str = "") -> None:
        from skillevaluator.tier3.harbor.runner import _bounded_execution_error

        context = f"{artifact}: " if artifact else ""
        self.data["errors"].append(_bounded_execution_error(f"{context}{type(exc).__name__}: {exc}"))
        self.data["status"] = "incomplete"

    def _write_artifact(self, path: Path, output: Path, payload: Callable[[], Any]) -> None:
        """An advisory artifact cannot prevent the others from being collected."""
        try:
            if path.parent != output:
                collector._assert_safe_generated_output_path(path.parent, output, follow_target=False)
                path.parent.mkdir(exist_ok=True)
            collector._write_generated_root_json(path, output, payload())
        except Exception as exc:
            self._record_error(exc, artifact=path.relative_to(output).as_posix())

    def _collect_statistics(self, result: dict[str, Any], jobs_dir: Path) -> None:
        from skillevaluator.tier3.harbor.activation import summarize_skill_activation
        from skillevaluator.tier3.loose_parts import load_failures_by_case, loose_parts_load_evidence
        from skillevaluator.tier3.plugin_evidence import _trial_roots

        observer = getattr(self.native, "observer", None)
        if observer is None:
            raise ValueError("private comparisons have no runtime evidence observer")
        for agent in self.agents:
            arm_rows = observer.logical_trials.get(agent, {})
            arms, conditions, failure_details, evidence = {}, {}, {}, {}
            for arm, job in self.jobs.get(agent, {}).items():
                records = arm_rows.get(arm, ())
                rewards = [dict(record.reward) for record in records if isinstance(record.reward, Mapping)]
                recovered = (
                    collector._recovered_timeout_attempts(job, rewards, n_attempts=self.n_attempts)
                    if self.stop_on_pass and job.is_dir()
                    else []
                )
                ok, reason = collector.validate_harbor_job_result(
                    job / "result.json",
                    expected_trials=None if self.stop_on_pass else len(self.expected_case_ids) * self.n_attempts,
                )
                trial_failures = collector._extract_trial_failures(job) if recovered else []
                job_recovered = collector._recovered_job_failure(job, trial_failures, recovered) if recovered else False
                failure = self.launch_errors[agent].get(VARIANTS[arm], "") or (
                    "" if ok or job_recovered else collector._published_job_failure(reason)
                )
                observed = arm_observations(
                    records,
                    expected_case_ids=self.expected_case_ids,
                    n_attempts=self.n_attempts,
                    stop_on_pass=self.stop_on_pass,
                    pass_threshold=self.pass_threshold,
                    job_failure=failure,
                    recovered_attempts=recovered,
                )
                arms[arm] = observed
                conditions[arm] = collector._condition_execution_summary(
                    [dict(trial.reward) for trial in observed.trials],
                    expected_case_ids=list(self.expected_case_ids),
                    expected_cases=len(self.expected_case_ids),
                    n_attempts=self.n_attempts,
                    stop_on_pass=self.stop_on_pass,
                    pass_threshold=self.pass_threshold,
                    job_failure=failure,
                    recovered_attempts=recovered,
                )
                if arm in {ARM_SUM_OF_PARTS, ARM_LOOSE_PARTS}:
                    roots, limited = _trial_roots(job) if job.is_dir() else ([], False)
                    recovered_names = {row["trial"] for row in recovered}
                    failures = [
                        {"trial": collector._published_trial_label(root.name), "reason": reason}
                        for root in roots
                        if collector._published_trial_label(root.name) not in recovered_names
                        if (reason := collector._trial_failure_reason(root))
                    ]
                    metadata = collector._failure_list_metadata("details", failures)
                    if limited:
                        metadata["details_truncated"] = True
                    failure_details[arm] = (collector._public_failure_list(failures), metadata)
                loose = self.loose_plan.arm_for(agent)
                if loose is not None and arm in {ARM_WITH, ARM_SUM_OF_PARTS, ARM_LOOSE_PARTS}:
                    by_case = evidence.setdefault(arm, {})
                    for record in records:
                        if record.case_id not in self.expected_case_ids:
                            continue
                        signals = record.signals or {}
                        init = getattr(record, "init_event", None)
                        row = loose_parts_load_evidence(
                            [init] if init else [],
                            None,
                            loose.parts,
                            arm=arm,
                            mcp_calls=(signals.get("mcp_calls") or {}).get("by_server", {}),
                        )
                        by_case.setdefault(record.case_id, []).append(row)
            failures = load_failures_by_case(evidence)
            # Signal specs deliberately omit dataset classification fields.
            # Reuse the context's canonical two-member classification instead.
            cross_cases = getattr(observer.context, "cross_component_cases", None)
            stats = build_agent_statistics(
                arms,
                expected_case_ids=self.expected_case_ids,
                n_attempts=self.n_attempts,
                stop_on_pass=self.stop_on_pass,
                pass_threshold=self.pass_threshold,
                sum_of_parts_requested=bool(self.members),
                cross_component_case_ids=cross_cases,
                load_failures_by_case=failures,
                never_activated_arms={
                    arm
                    for arm, observed in arms.items()
                    if summarize_skill_activation(trial.reward for trial in observed.trials)["status"]
                    == "never_activated"
                },
            )
            loose = self.loose_plan.arm_for(agent)
            if loose is not None:
                decision = self.native.decisions.get(agent)
                stats["integration_claim"] = integration_claim(
                    stats["lift_ladder"],
                    loose.content,
                    stats["integration_completeness"],
                    agent=agent,
                    full_arm_load="native" if decision and decision.native else "wrapper",
                )
            info = result.get("agents", {}).get(agent)
            if isinstance(info, dict):
                info.setdefault("conditions", {}).update(
                    {arm: state for arm, state in conditions.items() if arm in {ARM_SUM_OF_PARTS, ARM_LOOSE_PARTS}}
                )
                for arm, (details, metadata) in failure_details.items():
                    info.setdefault("trial_failures", {})[arm] = details
                    info.setdefault("failure_detail_metadata", {})[f"{arm}_trials"] = metadata
                info.update(stats)
            self.data["agents"][agent] = {
                **stats,
                "conditions": conditions,
                "loose_parts": loose.info(evidence)
                if loose is not None
                else {
                    "measured": False,
                    "not_measured_reason": self.loose_plan.not_measured.get(agent)
                    or (
                        "not requested" if not self.options.loose_parts else "No supported loose-parts arm was prepared"
                    ),
                },
                "launch_errors": self.launch_errors[agent],
            }
            output = jobs_dir.parent
            directory = output / agent
            self._write_artifact(directory / "statistics.json", output, lambda stats=stats: stats)
            if loose is not None:
                self._write_artifact(
                    directory / LOAD_EVIDENCE_FILENAME,
                    output,
                    lambda evidence=evidence: compact_load_evidence(evidence),
                )

    def provenance(self) -> dict[str, Any]:
        return {
            "requested": {"member_skills": self.options.member_skills, "loose_parts": self.options.loose_parts},
            "source": vars(self.package.identity),
            "expected_case_ids": list(self.expected_case_ids),
            **self.data,
        }
