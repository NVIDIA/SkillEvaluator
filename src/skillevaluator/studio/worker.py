# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Studio job per process, using the public evaluator implementation."""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from skillevaluator.studio.adapter import is_native_custom_only, json_safe, normalize_evaluation, read_json


class EventReporter:
    """Native ProgressReporter sink with durable JSONL and readable stdout."""

    def __init__(self, job_dir: Path) -> None:
        from skillevaluator.tier3.harbor.progress import secret_values_from_environment

        self.path = job_dir / "events.jsonl"
        self._lock = threading.RLock()
        self._active = True
        self._secrets = secret_values_from_environment(os.environ)

    @property
    def is_active(self) -> bool:
        return self._active

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        self._secrets.update(value for value in values if value)

    def record(self, stage: str, state: str, detail: str | None = None, **fields: Any) -> None:
        from skillevaluator.tier3.harbor.progress import redact_progress_detail

        event = {
            "type": "progress",
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": stage,
            "state": state,
            **fields,
        }
        if detail is not None:
            event["detail"] = redact_progress_detail(detail, secret_values=self._secrets)
        safe = json_safe(event, secret_values=tuple(self._secrets))
        line = json.dumps(safe, ensure_ascii=True, allow_nan=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def start(self, plan: Any) -> None:
        self.record("plan", "running", type="plan", plan=asdict(plan))

    def emit(self, event: Any) -> None:
        self.record(**asdict(event))

    def heartbeat(self) -> None:
        self.record("heartbeat", "running")

    def close(self) -> None:
        self._active = False


class _RedactedStream:
    """Redact native print/log fragments before they leave the worker process."""

    def __init__(self, stream: TextIO, secrets: set[str]) -> None:
        self.stream = stream
        self.secrets = secrets
        self.pending = ""
        self._lock = threading.RLock()

    def write(self, value: str) -> int:
        with self._lock:
            self.pending += value
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self.stream.write(self._redact(line) + "\n")
            if len(self.pending) > 65536:
                # Drop an overlong unfinished line rather than expose half a
                # credential split across a bounded log-fragment boundary.
                self.stream.write("<overlong log fragment omitted>\n")
                self.pending = ""
        return len(value)

    def _redact(self, text: str) -> str:
        from skillevaluator.utils.redaction import redact_sensitive_text

        for secret in sorted(self.secrets, key=len, reverse=True):
            text = text.replace(secret, "<redacted>")
        return redact_sensitive_text(text)

    def flush(self) -> None:
        with self._lock:
            if self.pending:
                self.stream.write(self._redact(self.pending))
                self.pending = ""
            self.stream.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"


def _inside_job(value: Any, job_dir: Path, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path inside this job")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path inside this job")
    canonical = path.resolve()
    if canonical == job_dir or not canonical.is_relative_to(job_dir):
        raise ValueError(f"{label} must be inside this job directory")
    return canonical


def _bool_option(options: dict[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _int_option(options: dict[str, Any], name: str, default: int, maximum: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _guidance_text(guidance: Any) -> str:
    if not isinstance(guidance, dict):
        return ""
    sections = (("task", "Questions"), ("success", "Behaviors"), ("avoid", "Edge Cases"))
    return "\n\n".join(
        f"## {title}\n{value.strip()}"
        for key, title in sections
        if isinstance((value := guidance.get(key)), str) and value.strip()
    )


def _draft(request: dict[str, Any], skill: Path, job_dir: Path, reporter: EventReporter) -> dict[str, Any]:
    from skillevaluator.evaluation.results import DatasetGenerationError
    from skillevaluator.tier3.generate_dataset import main as generate_dataset

    options = request.get("options") or {}
    no_llm = _bool_option(options, "no_llm", True)
    if not no_llm:
        from skillevaluator.provider_config import resolve_llm_provider

        resolve_llm_provider()
    guidance = _guidance_text(request.get("guidance"))
    prompt = None
    if guidance:
        prompt = job_dir / "guidance.md"
        prompt.write_text(guidance, encoding="utf-8")
    reporter.record("dataset", "running", "Preparing a reviewable dataset preview.")
    output = _GenerationOutput(sys.stdout)
    arguments = [str(skill), "--full", "--dry-run", "--force"]
    if no_llm:
        arguments.append("--no-llm")
    if prompt is not None:
        arguments.extend(["--prompt", str(prompt)])
    try:
        with redirect_stdout(output):
            outcome = generate_dataset(arguments)
    except SystemExit as exc:
        diagnostic = getattr(exc, "diagnostic", None)
        message = (
            diagnostic
            if isinstance(diagnostic, str) and diagnostic.strip()
            else f"Dataset generation failed with exit code {exc.code}"
        )
        raise DatasetGenerationError(message) from exc
    dataset = outcome.dataset
    if not isinstance(dataset, dict) or not isinstance(dataset.get("evals"), list):
        raise ValueError("Dataset generator returned no reviewable cases")
    notes = ["Review expected answers and assertions before saving or evaluating these cases."]
    template = no_llm or output.template_fallback
    if template:
        notes.append("Template starters are not independently verified reference answers.")
    if output.template_fallback:
        notes.append("The evaluator fell back to template generation after the requested model generation failed.")
    return {
        "kind": "dataset",
        "generation_mode": "template" if template else "llm",
        "cases": dataset["evals"],
        "review_required": True,
        "notes": notes,
        "raw": asdict(outcome),
    }


class _GenerationOutput:
    """Relay native generator output and preserve its announced fallback mode."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.tail = ""
        self.template_fallback = False

    def write(self, value: str) -> int:
        self.tail = (self.tail + value)[-4096:]
        if "Using deterministic template mode." in self.tail:
            self.template_fallback = True
        return self.stream.write(value)

    def flush(self) -> None:
        self.stream.flush()


def _validate(skill: Path) -> dict[str, Any]:
    from skillevaluator.tier3.evals_spec import validate_harbor_contract, validate_skillevaluators

    checks = validate_skillevaluators(skill)
    evals = skill / "evals"
    if any((evals / name).exists() for name in ("harbor", "grader.py", "grader.sh")):
        checks.extend(validate_harbor_contract(skill))
        # Same authored native/custom-only exemption as `tier3 validate`,
        # without importing the live-run command module on a base install.
        if is_native_custom_only(skill):
            checks = [
                check
                for check in checks
                if not (
                    check.path == "evals/evals.json"
                    and check.status == "error"
                    and "REQUIRED dataset file missing" in check.message
                )
            ]
    issues = [{"path": item.path, "status": item.status, "message": item.message} for item in checks]
    return {
        "kind": "validation",
        "valid": not any(item["status"] in {"error", "missing"} for item in issues),
        "issues": issues,
    }


def _check(skill: Path) -> dict[str, Any]:
    from skillevaluator.tier1.commands import run_validation

    checks = run_validation(skill, checks="schema,quality", use_llm=False)
    raw = [{**item.to_dict(), "metadata": item.metadata} for item in checks]
    issues = [{**finding.to_dict(), "validator": item.validator_name} for item in checks for finding in item.findings]
    for item in checks:
        issues.extend(
            {"message": message, "severity": "high", "validator": item.validator_name} for message in item.gate_failures
        )
    quality = next(
        (item.metadata["quality_scores"] for item in checks if isinstance(item.metadata.get("quality_scores"), dict)),
        None,
    )
    return {
        "kind": "check",
        "passed": all(item.passed and not item.is_incomplete for item in checks),
        "quality": quality,
        "issues": issues,
        "raw": raw,
        "scope": "Deterministic skill structure and quality; does not run agents or a complete security scan.",
    }


def _optional_checks(request: dict[str, Any], skill: Path, reporter: EventReporter) -> dict[str, Any] | None:
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.tier1.commands import run_validation

    options = request.get("options") or {}
    selection = {name: _bool_option(options, name, False) for name in ("tier1", "tier1_llm", "dedup")}
    if selection["tier1_llm"] and not selection["tier1"]:
        raise ValueError("Tier 1 LLM analysis requires Tier 1 checks")
    if not any(selection.values()):
        return None
    results = []

    def incomplete(label: str, stage: str, exc: Exception) -> None:
        result = ValidationResult(validator_name=label)
        result.mark_scan_incomplete(stage)
        result.add_error(
            f"{label} could not complete ({type(exc).__name__}). Check installed tools and provider settings."
        )
        results.append(result)

    if selection["tier1"]:
        use_llm = selection["tier1_llm"]
        if use_llm:
            from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

            try:
                resolve_llm_provider()
            except ProviderConfigurationError as exc:
                incomplete("Tier 1 LLM security analysis", "skillspector-llm", exc)
                use_llm = False
        try:
            results.extend(
                run_validation(
                    skill,
                    use_llm=use_llm,
                    continue_on_failure=True,
                    on_check=lambda name: reporter.record("tier1", "running", name),
                )
            )
        except Exception as exc:
            incomplete("Tier 1 checks", "tier1", exc)
        reporter.record("tier1", "complete")
    if selection["dedup"]:
        reporter.record("dedup", "running", "Checking duplication within this skill.")
        try:
            from skillevaluator.tier2.commands import run_dedup_scan

            results.extend(run_dedup_scan(skill))
        except Exception as exc:
            incomplete("Within-skill duplication", "context-optimization", exc)
        reporter.record("dedup", "complete")
    return {
        "status": "incomplete"
        if any(item.is_incomplete for item in results)
        else "passed"
        if all(item.passed for item in results)
        else "failed",
        "selection": selection,
        "results": [
            {**item.to_dict(), "metadata": item.metadata, "errors": item.errors, "warnings": item.warnings}
            for item in results
        ],
        "findings": [
            {**finding.to_dict(), "validator": item.validator_name} for item in results for finding in item.findings
        ],
        "scope": "Selected checks on this job's skill snapshot; independent of agent results and lift.",
        "artifact": "optional-checks.json",
    }


def _write_result(path: Path, result: dict[str, Any], reporter: EventReporter) -> dict[str, Any]:
    safe = json_safe(result, secret_values=tuple(reporter._secrets))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".worker-result-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(safe, stream, indent=2, ensure_ascii=True, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return safe


def _evaluate(request: dict[str, Any], skill: Path, job_dir: Path, reporter: EventReporter) -> dict[str, Any]:
    from skillevaluator.evaluation import EvaluationOptions, EvaluationService

    options = request.get("options") or {}
    env_mode = options.get("env_mode", "local")
    if env_mode not in {"local", "docker"}:
        raise ValueError("Studio supports local and Docker evaluation")
    agent = options.get("agent")
    if agent not in {None, "opencode", "codex", "claude-code"}:
        raise ValueError("Select OpenCode, Codex, or Claude Code")
    results = _inside_job(request.get("results_dir", str(job_dir / "results")), job_dir, "results_dir")
    kwargs = {
        "agents": agent,
        "env_mode": env_mode,
        "n_attempts": _int_option(options, "n_attempts", 1, 20),
        "n_concurrent": _int_option(options, "n_concurrent", 1, 4),
        "max_agents": 1,
        "stop_on_pass": False,
        "skip_baseline": _bool_option(options, "skip_baseline", False),
        "harbor_keep_jobs": True,
        "results_dir": results,
    }
    for name in ("model", "grading_mode", "timeout_multiplier"):
        if name in options and options[name] not in (None, ""):
            kwargs[name] = options[name]
    if agent == "opencode" and kwargs.get("model"):
        from skillevaluator.provider_config import resolve_llm_provider

        # Settings accepts a provider catalog ID; OpenCode needs its own
        # namespace before that ID, including NVIDIA's publisher prefix.
        namespace = {"nv_build": "nvidia", "openai": "openai", "anthropic": "anthropic"}[
            resolve_llm_provider().provider
        ]
        kwargs["agent_model"] = (f"opencode={namespace}/{kwargs['model']}",)
    if "timeout_multiplier" in kwargs:
        timeout = kwargs["timeout_multiplier"]
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not 0 < timeout <= 20:
            raise ValueError("timeout_multiplier must be positive and at most 20")
    service = EvaluationService()
    raw = service.evaluate(EvaluationOptions(skill_path=skill, **kwargs), progress_reporter=reporter)
    if not isinstance(raw, dict):
        raise ValueError("Evaluator returned an invalid result")
    result = normalize_evaluation(raw, job_dir=job_dir, skill_path=skill)
    failure = service.failure_reason(raw)
    if failure:
        result["error"] = failure
        result["execution_status"] = "failed"
    return result


def run_job(job_dir: Path) -> int:
    """Execute a prepared job and persist results even on expected failures."""
    job_dir = job_dir.resolve(strict=True)
    reporter = EventReporter(job_dir)
    action = "unknown"
    exit_code = 0
    optional_checks = None
    try:
        request = read_json(job_dir / "request.json", root=job_dir)
        if not isinstance(request, dict):
            raise ValueError("request.json must contain an object")
        requested_action = request.get("action")
        if not isinstance(requested_action, str) or requested_action not in {
            "draft",
            "validate",
            "check",
            "evaluate",
            "improve",
            "compare_improvement",
        }:
            raise ValueError("Unknown Studio action")
        action = requested_action
        options = request.get("options")
        if options is not None and not isinstance(options, dict):
            raise ValueError("options must contain an object")
        skill = _inside_job(request.get("skill_path"), job_dir, "skill_path")
        if not skill.is_dir() or not (skill / "SKILL.md").is_file():
            raise ValueError("The job source must contain a SKILL.md file")
        reporter.record(action, "running")
        if action == "draft":
            result = _draft(request, skill, job_dir, reporter)
        elif action == "validate":
            result = _validate(skill)
        elif action in {"improve", "compare_improvement"}:
            from skillevaluator.studio import improvements
            from skillevaluator.studio.store import Workspace

            workspace = Workspace(job_dir.parent.parent)
            result = (
                improvements.prepare(workspace, request, skill, job_dir)
                if action == "improve"
                else improvements.comparison(workspace, request)
            )
        else:
            optional_checks = _optional_checks(request, skill, reporter)
            if optional_checks is not None:
                _write_result(job_dir / "optional-checks.json", optional_checks, reporter)
            if action == "evaluate":
                result = _evaluate(request, skill, job_dir, reporter)
                if request.get("validation_of"):
                    from skillevaluator.studio import improvements
                    from skillevaluator.studio.store import Workspace

                    _write_result(job_dir / "result.json", result, reporter)
                    comparison_request = {
                        "proposal_job_id": request["validation_of"],
                        "before_job_id": request["before_job_id"],
                        "after_job_id": job_dir.name,
                        "outcome_metrics": request.get("outcome_metrics", []),
                    }
                    try:
                        compared = improvements.comparison(Workspace(job_dir.parent.parent), comparison_request)
                        _write_result(job_dir / "improvement-comparison.json", compared, reporter)
                        result["improvement_comparison"] = compared
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        result["improvement_comparison"] = {
                            "kind": "improvement_comparison",
                            "error": str(exc),
                            "execution_status": "failed",
                        }
            elif optional_checks is None:
                result = _check(skill)
            else:
                result = {
                    "kind": "check",
                    "passed": optional_checks["status"] == "passed",
                    "scope": optional_checks["scope"],
                    "quality": next(
                        (
                            item["metadata"]["quality_scores"]
                            for item in optional_checks["results"]
                            if "quality_scores" in item["metadata"]
                        ),
                        None,
                    ),
                }
        result.setdefault("execution_status", "succeeded")
        if result["execution_status"] == "failed":
            exit_code = 1
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        exit_code = 130 if interrupted else 1
        kind = {
            "draft": "dataset",
            "validate": "validation",
            "check": "check",
            "evaluate": "evaluation",
            "improve": "improvement",
            "compare_improvement": "improvement_comparison",
        }.get(action, "error")
        result = {
            "kind": kind,
            "execution_status": "interrupted" if interrupted else "failed",
            "error": "Run interrupted before completion." if interrupted else f"{type(exc).__name__}: {exc}",
        }
        if action == "improve" and (job_dir / "improvement" / "receipt.json").is_file():
            result["receipt"] = read_json(job_dir / "improvement" / "receipt.json", root=job_dir)
        if kind == "evaluation":
            result.update(
                {
                    "cases": [],
                    "suggestions": [],
                    "suggestions_status": "not_generated",
                    "artifacts": [],
                    "summary": {
                        "expected_attempts": None,
                        "scored_attempts": None,
                        "with_skill": None,
                        "without_skill": None,
                        "lift": None,
                        "precision": None,
                    },
                }
            )
    if optional_checks is not None:
        result["optional_checks"] = optional_checks
    safe = _write_result(job_dir / "result.json", result, reporter)
    reporter.record(
        action, "failed" if exit_code else "complete", safe.get("error") if isinstance(safe.get("error"), str) else None
    )
    reporter.close()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    from skillevaluator.tier3.harbor.progress import secret_values_from_environment

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("Usage: python -m skillevaluator.studio.worker /absolute/job-directory\n")
        return 2
    stdout, stderr = sys.stdout, sys.stderr
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    secrets = secret_values_from_environment(os.environ)
    sys.stdout = _RedactedStream(stdout, secrets)
    sys.stderr = _RedactedStream(stderr, secrets)
    try:
        return run_job(Path(args[0]))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = stdout, stderr
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
