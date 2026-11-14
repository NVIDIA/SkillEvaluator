# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- trusted reusable CI entrypoint.
"""Evaluate one CI tier, then aggregate retained evidence without running source code.

The local onboarding wizard never calls this module. Reusable CI workflows do;
all outputs live outside the consumer checkout and provider subprocesses never
inherit repository or feedback credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

# Isolated Python deliberately omits the script directory. These siblings are
# from the pinned tools checkout; the consumer working directory is never added.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from skillevaluator.reporting.run_status import resolve_run_status
from skillevaluator.utils.secret_redaction import Redactor

try:
    from diagnostics import TierBudget, retain_diagnostics
    from github_api import strict_json
    from model_config import CREDENTIAL_KEYS, agent_model_overrides, model_environment, required_credentials
    from run_tier1 import discover_skills
    from secret_redaction import collect_sensitive_values
    from select_changed_skills import (
        SelectionError,
        _git,
        changed_paths,
        select_changed_skills,
        setup_changed,
        skill_roots,
    )
except ModuleNotFoundError:
    from ci.diagnostics import TierBudget, retain_diagnostics
    from ci.github_api import strict_json
    from ci.model_config import CREDENTIAL_KEYS, agent_model_overrides, model_environment, required_credentials
    from ci.run_tier1 import discover_skills
    from ci.secret_redaction import collect_sensitive_values
    from ci.select_changed_skills import (
        SelectionError,
        _git,
        changed_paths,
        select_changed_skills,
        setup_changed,
        skill_roots,
    )

SHA = re.compile(r"[0-9a-f]{40}")
MAX_JSON_BYTES = 32 * 1024 * 1024
SUCCESS = {"passed", "skipped"}
# Worker diagnostics explain execution problems, not complete failed or neutral verdicts.
EXECUTION_FAILURES = {"incomplete", "not_run"}
ATTEMPT = re.compile(r"[1-9][0-9]{0,7}")
TIER_ARTIFACT = re.compile(r"skill-evaluator-tier([123])-attempt-([1-9][0-9]{0,7})")


class ConfigurationError(ValueError):
    """A fixed diagnostic that contains no credential or untrusted command output."""


def integer(env: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = str(env.get(key, default))
    if not re.fullmatch(r"[0-9]{1,8}", value) or not minimum <= int(value) <= maximum:
        raise ConfigurationError(f"{key} must be between {minimum} and {maximum}.")
    return int(value)


def boolean(env: dict, key: str, default: bool = False) -> bool:
    value = str(env.get(key, str(default))).lower()
    if value not in {"true", "false"}:
        raise ConfigurationError(f"{key} must be true or false.")
    return value == "true"


def child_environment(env: dict[str, str], tier: int = 2) -> dict[str, str]:
    """Forward explicit selected provider configuration without CI credentials."""
    try:
        return model_environment(env, tier)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None


def select_skills(workspace: Path, env: dict) -> tuple[list[Path], dict]:
    roots = env.get("SKILL_EVAL_PATHS", "skills")
    mode = env.get("SKILL_EVAL_SCAN_MODE", "auto")
    if mode not in {"auto", "all"}:
        raise ConfigurationError("SKILL_EVAL_SCAN_MODE must be auto or all.")
    base, head = env.get("DIFF_BASE_SHA", ""), env.get("REPORT_COMMIT") or env.get("DIFF_HEAD_SHA", "")
    actual = _git(workspace, "rev-parse", "HEAD").decode().strip()
    if not SHA.fullmatch(head) or actual != head:
        raise ConfigurationError("The evaluated commit must match the checkout HEAD.")
    if base and mode == "auto":
        selected, selection = select_changed_skills(workspace, roots, base, head)
        # Tier 1 applies the same setup predicate, so every tier that runs selects the same skills.
        # The shared list is the one the paid-job skip reads (gitlinks included).
        changed = changed_paths(workspace, env, head)
        if not setup_changed(changed):
            return selected, selection
        reason = "ci_configuration_changed"
    else:
        selection, reason = {}, "full_scan"
    configured = skill_roots(workspace, roots)
    if _git(
        workspace, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching", "--", *configured
    ):
        raise ConfigurationError("Configured skill trees must match the evaluated commit.")
    selected = discover_skills(workspace, roots)
    return selected, {
        **selection,
        "reason": reason,
        "selected": [p.relative_to(workspace).as_posix() for p in selected],
    }


def context(env: dict) -> dict:
    mode = env.get("SKILL_EVAL_ENV_MODE", "docker")
    if mode not in {"docker", "local"}:
        raise ConfigurationError("Tier 3 onboarding supports docker or local.")
    platform_name = "gitlab" if env.get("CI_PIPELINE_ID") else "github"
    return {
        "schema_version": 1,
        "report_contract": "skillevaluator-ci-v1",
        "commit": env.get("REPORT_COMMIT") or env.get("DIFF_HEAD_SHA", ""),
        "base_commit": env.get("DIFF_BASE_SHA") or None,
        "repository": env.get("REPORT_REPOSITORY")
        or env.get("GITHUB_REPOSITORY")
        or env.get("CI_PROJECT_PATH", "local/consumer"),
        "platform": platform_name,
        "run_id": env.get("CI_PIPELINE_ID") or env.get("GITHUB_RUN_ID", "1"),
        "run_attempt": env.get("GITHUB_RUN_ATTEMPT", "1"),
        "environment": mode,
        "runtime": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "evaluator": importlib.metadata.version("skillevaluator"),
            "tools_ref": env.get("SKILLEVALUATOR_TOOLS_REF", ""),
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ConfigurationError("Report outputs must be regular files.")
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ConfigurationError("A required report is missing, linked or oversized.")
    value = strict_json(path.read_bytes(), ConfigurationError, "Report JSON is invalid.")
    if not isinstance(value, dict):
        raise ConfigurationError("Report JSON must be an object.")
    return value


def run_command(args: list[str], *, env: dict, cwd: Path, timeout: float, log: Path) -> int:
    """Bound subprocess lifetime and stop the entire process group on timeout."""
    if timeout <= 0:
        return 124
    with log.open("wb") as stream:
        process = subprocess.Popen(
            args, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if sys.exc_info()[0] is subprocess.TimeoutExpired:
                return 124
            raise


def skill_command(tier: int, skill: Path, output: Path, env: dict, *, catalog: Path | None = None) -> list[str]:
    args = [sys.executable, "-I", "-m", "skillevaluator.cli"]
    if tier == 1:
        args += ["tier1", str(skill), "--no-llm", "--min-score", str(integer(env, "SKILL_EVAL_MIN_SCORE", 70, 0, 100))]
    elif tier == 2:
        args += ["tier2", str(skill)]
        if catalog:
            args += ["--catalog", str(catalog)]
    else:
        return [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "native-tier3",
            "--skill",
            str(skill),
            "--output",
            str(output),
            "--config",
            str(output / "ci-options.json"),
            "--env-mode",
            env.get("SKILL_EVAL_ENV_MODE", "docker"),
        ]
    return [*args, "-r", "json", "-o", str(output)]


def tier3_options(skill: Path, output: Path, env: dict):
    """Use the native service directly: combined validate requires Tier 1."""
    from skillevaluator.evaluation.options import EvaluationOptions

    source = {
        "repository": env.get("REPORT_REPOSITORY") or env.get("GITHUB_REPOSITORY") or env.get("CI_PROJECT_PATH"),
        "commit": env.get("REPORT_COMMIT"),
    }
    revision = env.get("SKILLEVALUATOR_TOOLS_REF") or env.get("SKILL_EVAL_TOOLS_REF")
    if revision and SHA.fullmatch(revision):
        source["evaluator_container_revision"] = revision
    return EvaluationOptions(
        skill_path=skill,
        env_mode=env.get("SKILL_EVAL_ENV_MODE", "docker"),
        agents=env.get("SKILL_EVAL_AGENTS") or None,
        # CI's shared chat/judge model is carried in the provider environment.
        # Native resolution adds adapter namespaces and honors authored agent
        # overrides; a global CLI override would bypass that resolution.
        model=None,
        agent_model=agent_model_overrides(env),
        skip_baseline=False,
        skip_with_skill=False,
        max_cases=None,
        stop_on_pass=False,
        n_attempts=integer(env, "SKILL_EVAL_N_ATTEMPTS", 1, 1, 100),
        n_concurrent=integer(env, "SKILL_EVAL_N_CONCURRENT", 1, 1, 16),
        max_agents=1,
        results_dir=output / "harbor",
        evaluated_source=source,
    )


def native_tier3(skill: Path, output: Path, env: dict) -> int:
    """Run both native Harbor arms and emit their canonical typed JSON."""
    from skillevaluator.evaluation.service import EvaluationService
    from skillevaluator.evaluation.tier3_report import advisory_skip_result, agent_eval_result_from_run
    from skillevaluator.reporting.json_reporter import JSONReporter

    dataset_preflight(skill)
    service = EvaluationService()
    options = tier3_options(skill, output, env)
    engine = service.evaluate(options)
    result = agent_eval_result_from_run(
        skill, results_dir=options.results_dir, env_mode=options.env_mode, engine_result=engine
    )
    if result is None:
        reason = service.failure_reason(engine) or "Tier 3 did not retain a canonical live evaluation result."
        secrets = {os.environ[key] for key in CREDENTIAL_KEYS if os.environ.get(key)}
        result = advisory_skip_result(Redactor(secrets).text(reason)[:2000], skill_name=skill.name)
        result.mark_scan_incomplete("Tier 3 execution")
    result.metadata["gating"] = {"tier": 3, "blocking": False}
    result.metadata["evaluated_source"] = options.evaluated_source
    if service.failure_reason(engine):
        result.mark_scan_incomplete("Tier 3 execution")
    document = JSONReporter(expected_skill_name=skill.name, explicit_skill_name=skill.name).render_all([result])
    (output / "skillevaluator-tier3.json").write_text(document, encoding="utf-8")
    return int(resolve_run_status([result]).headline.value != "passed")


def dataset_preflight(skill: Path) -> dict[str, str]:
    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.evals_spec import validate_tier3_source

    source, findings = validate_tier3_source(skill)
    if source in {"invalid", "missing"} or any(f.status in {"error", "warning", "missing"} for f in findings):
        raise ConfigurationError("A committed Tier 3 dataset/task source failed strict validation.")
    configuration, _ = load_evals_config(skill)
    if (configuration.get("harbor") or {}).get("max_cases") is not None:
        raise ConfigurationError("CI requires the complete Tier 3 dataset; remove harbor.max_cases.")
    authored = skill / "evals"
    digests = {}
    total = 0
    for path in sorted(authored.rglob("*")):
        if "results" in path.relative_to(authored).parts:
            continue
        if path.is_symlink():
            raise ConfigurationError("Tier 3 authored inputs cannot contain symlinks.")
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_JSON_BYTES or len(digests) >= 10000:
                raise ConfigurationError("Tier 3 authored inputs exceed CI limits.")
            digests[path.relative_to(skill).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _combined(statuses: list[str]) -> str:
    return next(
        (
            status
            for status in ("failed", "incomplete", "needs_action", "not_run", "neutral", "passed", "skipped")
            if status in statuses
        ),
        "not_run",
    )


def _retain(
    raw: Path, destination: Path, tier: int, tokens: set[str], skill: Path, code: int, target_path: str | None = None
) -> dict:
    from skillevaluator.reporting.retained import retained_results

    try:
        from native_reports import render_native_retained_report
        from tier_manifest import tier3_execution_complete
    except ModuleNotFoundError:
        from ci.native_reports import render_native_retained_report
        from ci.tier_manifest import tier3_execution_complete
    documents = sorted(raw.glob("skillevaluator*.json"))
    if len(documents) != 1:
        raise ConfigurationError("The evaluator did not produce one canonical JSON report.")
    report = Redactor(tokens).object(read_json(documents[0]))
    if target_path is not None and isinstance(report.get("target_path"), str):
        # The evaluator records the target as the CI passed it, an absolute path
        # in the runner's workspace; the retained report keeps the repository path.
        report["target_path"] = target_path
    results = retained_results(report, allowed_tiers={tier})
    if tier == 2:
        # Standalone Tier 2 gates its CLI exit; CI keeps the same evidence and
        # tier verdict while declaring that verdict advisory in every format.
        for result, row in zip(results, report["results"], strict=True):
            result.metadata["gating"] = {**result.metadata.get("gating", {}), "tier": 2, "blocking": False}
            row["gating"] = result.metadata["gating"]
        report["gating"] = {
            "tiers": {
                "2": {
                    "blocking": False,
                    "validators": [r.validator_name for r in results],
                    # Keep the reporter's per-check policy list, now advisory too.
                    "validator_policies": [{"validator": r.validator_name, "blocking": False} for r in results],
                }
            }
        }
    run_status = resolve_run_status(results)
    if tier == 2:
        report.update(overall_passed=run_status.headline.value == "passed", overall_status=run_status.headline.value)
    status = run_status.tiers[tier].status.value
    # A confirmed failure outranks incomplete evidence in the status, so record
    # completeness on its own: the caller keeps diagnostics for incomplete runs.
    complete = tier3_execution_complete(report) if tier == 3 else not any(result.is_incomplete for result in results)
    if status == "not_run" or (code and status == "passed"):
        status = "incomplete"
    if tier == 3:
        if status == "passed" and not complete:
            status = "incomplete"
        elif status == "passed" and report.get("tier3", {}).get("verdict") == "neutral":
            status = "neutral"
    destination.mkdir(parents=True, exist_ok=True)
    name = f"skillevaluator-tier{tier}"
    write_json(destination / f"{name}.json", report)
    (destination / f"{name}.html").write_text(
        render_native_retained_report(
            report,
            skill_name=skill.name,
            expected_skill_name=skill.name,
            context={"report_index_url": "../../index.html", "canonical_json_url": f"{name}.json"},
        ),
        encoding="utf-8",
        errors="backslashreplace",
    )
    from skillevaluator.reporting.markdown import MarkdownReporter

    (destination / f"{name}.md").write_text(
        MarkdownReporter().render_all(results), encoding="utf-8", errors="backslashreplace"
    )
    metrics = {"findings": sum(len(result.findings) for result in results)}
    if tier == 3:
        payload = report.get("tier3") or {}
        summary = payload.get("summary") or payload
        metrics.update(
            {
                "score": summary.get("overall_score"),
                "skill_lift": summary.get("overall_lift"),
                "coverage": f"{summary.get('scored_attempts', 0)}/{summary.get('expected_attempts', 0)}",
                "execution_complete": complete,
            }
        )
    return {"status": status, "metrics": metrics, "complete": complete}


def _build_catalog(workspace: Path, selected: list[Path], path: Path) -> None:
    from skillevaluator.embedding.client import EmbeddingClient
    from skillevaluator.embedding.registry import EmbeddingRegistry

    registry = EmbeddingRegistry(EmbeddingClient())
    registry.build_from_paths(workspace, selected)
    registry.save_catalog(path)


def evaluate(tier: int, workspace: Path, reports: Path, env: dict[str, str]) -> int:
    if tier not in {1, 2, 3}:
        raise ConfigurationError("Select Tier 1, 2 or 3.")
    workspace = workspace.resolve(strict=True)
    reports = reports.resolve()
    if reports.is_relative_to(workspace) or any(p.is_symlink() for p in [reports, *reports.parents]):
        raise ConfigurationError("Reports must be outside the consumer checkout in an unlinked directory.")
    document = context(env)
    required = tier == 1
    selected, selection = select_skills(workspace, env)
    row = {
        "tier": tier,
        "required": required,
        "status": "skipped" if not selected else "incomplete",
        "skills": [
            {
                "path": skill.relative_to(workspace).as_posix(),
                "status": "incomplete",
                "report_json": None,
                "report_html": None,
            }
            for skill in selected
        ],
    }
    document.update({"tiers": [row], "expected_tiers": [tier], "selection": selection})
    destination = reports / f"tier{tier}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "ci-tier.json").exists():
        raise ConfigurationError("Use a fresh report directory for each run attempt.")

    # Reserve honest initial evidence before external work starts.
    def retain():
        document["overall_passed"] = not required or row["status"] in SUCCESS
        write_json(destination / "ci-tier.json", document)

    retain()
    if not selected:
        row["reason"] = "No selected skills changed; evaluation did not run."
        retain()
        return 0
    if env.get("SKILL_EVAL_DATASET_ROOT"):
        row["reason"] = "Use native per-skill evals directories; central dataset mappings require explicit conversion."
        retain()
        return int(required)
    child = child_environment(env, tier)
    provider_tokens = {child[key] for key in CREDENTIAL_KEYS if child.get(key)}
    needed = required_credentials(child, tier)
    if tier > 1 and not all(child.get(key) for key in needed):
        row["reason"] = "A required provider credential is missing; configure CI secrets before evaluation."
        retain()
        return int(required)
    deadline = time.monotonic() + integer(env, "SKILL_EVAL_TIMEOUT_SECONDS", 18000, 1, 86400)
    budget = TierBudget()
    with tempfile.TemporaryDirectory(prefix="skillevaluator-ci-") as temporary:
        scratch = Path(temporary).resolve()
        child["HOME"] = str(scratch / "home")
        Path(child["HOME"]).mkdir(mode=0o700)
        child["TMPDIR"] = str(scratch)
        child["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"] = str(scratch / "output-provenance.key")
        catalog = None
        if tier == 2:
            roots = env.get("SKILL_EVAL_PATHS", "skills")
            all_skills = discover_skills(workspace, roots)
            if len(all_skills) > 1:
                catalog = scratch / "catalog.json"
                # A helper subprocess preserves the same timeout and secret boundary as checks.
                request = scratch / "catalog-input.json"
                write_json(request, {"workspace": str(workspace), "skills": [str(p) for p in all_skills]})
                args = [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "catalog",
                    "--input",
                    str(request),
                    "--output",
                    str(catalog),
                ]
                if run_command(
                    args, env=child, cwd=workspace, timeout=deadline - time.monotonic(), log=scratch / "catalog.log"
                ):
                    row["reason"] = (
                        "The complete comparison catalog could not be built; inspect provider and embedding configuration."
                    )
                    diagnostics = scratch / "catalog-diagnostics"
                    diagnostics.mkdir()
                    retain_diagnostics(
                        diagnostics,
                        destination / "catalog",
                        provider_tokens,
                        command_log=scratch / "catalog.log",
                        budget=budget,
                    )
                    retain()
                    return int(required)
        doctor_done = False
        for index, skill in enumerate(selected, 1):
            skill_row = row["skills"][index - 1]
            raw = scratch / f"skill-{index:03}"
            raw.mkdir()
            leaf = destination / f"skill-{index:03}"
            tokens = provider_tokens
            command_log = scratch / f"skill-{index:03}.log"
            code = None
            complete = True
            try:
                digests = dataset_preflight(skill) if tier == 3 else {}
                tokens = collect_sensitive_values(skill) | provider_tokens
                if tier == 3 and not doctor_done:
                    args = [
                        sys.executable,
                        "-I",
                        "-m",
                        "skillevaluator.cli",
                        "doctor",
                        "--tier",
                        "3",
                        "--env-mode",
                        document["environment"],
                    ]
                    if env.get("SKILL_EVAL_AGENTS"):
                        args += ["--agents", env["SKILL_EVAL_AGENTS"]]
                    for model in agent_model_overrides(env):
                        args += ["--agent-model", model]
                    command_log = scratch / "doctor.log"
                    if run_command(
                        args,
                        env=child,
                        cwd=workspace,
                        timeout=min(180, deadline - time.monotonic()),
                        log=command_log,
                    ):
                        raise ConfigurationError(
                            "Tier 3 runtime preflight failed; check the selected mode, agent and provider."
                        )
                    doctor_done = True
                if tier == 3:
                    config_keys = (
                        "SKILL_EVAL_ENV_MODE",
                        "SKILL_EVAL_AGENTS",
                        "SKILL_EVAL_AGENT_MODELS",
                        "SKILL_EVAL_MODEL",
                        "SKILL_EVAL_N_ATTEMPTS",
                        "SKILL_EVAL_N_CONCURRENT",
                        "REPORT_REPOSITORY",
                        "REPORT_COMMIT",
                        "GITHUB_REPOSITORY",
                        "CI_PROJECT_PATH",
                        "SKILLEVALUATOR_TOOLS_REF",
                        "SKILL_EVAL_TOOLS_REF",
                    )
                    write_json(raw / "ci-options.json", {key: env[key] for key in config_keys if key in env})
                command_log = scratch / f"skill-{index:03}.log"
                code = run_command(
                    skill_command(tier, skill, raw, env, catalog=catalog),
                    env=child,
                    cwd=workspace,
                    timeout=deadline - time.monotonic(),
                    log=command_log,
                )
                if code == 124:
                    skill_row["error"] = "Evaluation exceeded the configured time budget."
                retained = _retain(raw, leaf, tier, tokens, skill, code, skill_row["path"])
                complete = retained.pop("complete")
                skill_row.update(retained)
                if tier == 3 and dataset_preflight(skill) != digests:
                    skill_row.update(status="incomplete", error="Authored evaluation inputs changed during execution.")
                skill_row["dataset_digests"] = digests
                skill_row["report_json"] = (leaf / f"skillevaluator-tier{tier}.json").relative_to(reports).as_posix()
                skill_row["report_html"] = (leaf / f"skillevaluator-tier{tier}.html").relative_to(reports).as_posix()
            except (OSError, ValueError, ImportError) as exc:
                # A failure after _retain must not leave a clean status beside an execution error.
                skill_row["status"] = "incomplete"
                # SDK and provider exception messages can carry credentials. Only fixed local diagnostics are safe.
                # An earlier specific diagnostic, such as the time budget, explains the missing report better.
                skill_row["error"] = skill_row.get("error") or (
                    str(exc)
                    if isinstance(exc, ConfigurationError)
                    else "Evaluation evidence could not be safely retained; check the runtime and provider."
                )
            finally:
                # Incomplete execution keeps diagnostics even behind a failed verdict. Exit
                # codes other than pass (0) and fail (1) mean the worker itself went wrong.
                if skill_row["status"] in EXECUTION_FAILURES or not complete or code not in {None, 0, 1}:
                    try:
                        retain_diagnostics(
                            raw,
                            leaf,
                            tokens,
                            command_log=command_log if command_log.is_file() else None,
                            budget=budget,
                        )
                    except (OSError, ValueError):
                        # Diagnostic failures must never replace the evaluation
                        # outcome or expose raw worker exception messages.
                        skill_row["diagnostics_error"] = "Worker diagnostics could not be safely retained."
            row["status"] = _combined([r["status"] for r in row["skills"]])
            retain()
            print(f"Tier {tier}: completed {index}/{len(selected)} selected skills; {skill_row['status']}.", flush=True)
    row["status"] = _combined([r["status"] for r in row["skills"]])
    retain()
    return int(required and row["status"] not in SUCCESS)


def select_attempts(artifacts: Path, reports: Path, current: int) -> dict[str, str]:
    """Place each tier's newest artifact from this or an earlier attempt of the run.

    GitHub's "Re-run failed jobs" reruns only failed jobs and their dependents.
    Tier jobs that already completed keep artifacts named for their own attempt.
    """
    chosen: dict[int, tuple[int, Path]] = {}
    if artifacts.is_dir():
        for path in artifacts.iterdir():
            match = TIER_ARTIFACT.fullmatch(path.name)
            if not match or int(match[2]) > current:
                continue
            if path.is_symlink() or not path.is_dir():
                raise ConfigurationError("Downloaded tier artifacts must be regular directories.")
            tier, attempt = int(match[1]), int(match[2])
            if attempt > chosen.get(tier, (0, path))[0]:
                chosen[tier] = (attempt, path)
    reports.mkdir(parents=True, exist_ok=True)
    selected = {}
    for tier, (attempt, path) in sorted(chosen.items()):
        # Tier 1 uploads its report root; Tier 2/3 upload a root containing tierN/.
        source = path if tier == 1 else path / f"tier{tier}"
        target = reports / f"tier{tier}"
        if source.is_symlink() or not source.is_dir():
            continue
        if target.exists() or target.is_symlink():
            raise ConfigurationError("Use a fresh report directory for each aggregate.")
        source.rename(target)
        selected[str(tier)] = str(attempt)
        print(f"Tier {tier}: using retained evidence from run attempt {attempt}.", flush=True)
    write_json(reports / "source-attempts.json", selected)
    return selected


def _source_attempts(reports: Path, current: int) -> dict[int, str]:
    path = reports / "source-attempts.json"
    if not path.exists() and not path.is_symlink():
        return {}
    recorded = read_json(path)
    if not all(
        key in {"1", "2", "3"} and isinstance(value, str) and ATTEMPT.fullmatch(value) and int(value) <= current
        for key, value in recorded.items()
    ):
        raise ConfigurationError("Recorded tier attempts are invalid.")
    return {int(key): value for key, value in recorded.items()}


def _keep_selected(row: dict, selected) -> None:
    """Record which skills the tier selected, so a summary can tell 'not selected' from missing."""
    row.pop("selected", None)
    if isinstance(selected, list) and all(isinstance(path, str) for path in selected):
        row["selected"] = selected


def aggregate(reports: Path, tiers: tuple[int, ...], env: dict) -> int:
    try:
        from tier_manifest import validate_manifest, write_reports
    except ModuleNotFoundError:
        from ci.tier_manifest import validate_manifest, write_reports
    result = context(env)
    result["expected_tiers"] = list(tiers)
    result["tiers"] = []
    current = integer(env, "GITHUB_RUN_ATTEMPT", 1, 1, 99_999_999)
    recorded = _source_attempts(reports, current)
    for tier in tiers:
        required = tier == 1
        path = reports / f"tier{tier}" / "ci-tier.json"
        if path.exists():
            incoming = validate_manifest(read_json(path))
            for field in ("commit", "base_commit", "repository", "platform", "run_id", "environment"):
                if incoming.get(field) != result[field]:
                    raise ConfigurationError("Tier artifacts do not identify the same evaluation run.")
            # Evidence from an earlier attempt of this run stays valid after "Re-run failed jobs".
            attempt = incoming["run_attempt"]
            if int(attempt) > current or recorded.get(tier, attempt) != attempt:
                raise ConfigurationError("Tier artifacts do not identify the same evaluation run.")
            rows = incoming.get("tiers")
            if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("tier") != tier:
                raise ConfigurationError("A tier artifact does not identify its expected tier.")
            row = {**rows[0], "required": required, "source_attempt": attempt}
            selection = incoming.get("selection", {})
            selected = selection.get("selected") if isinstance(selection, dict) else None
            # A retained checkpoint must cover every selected skill before it can claim completion.
            if (
                row.get("status") in {"passed", "neutral", "skipped"}
                and isinstance(selected, list)
                and set(selected) != {skill.get("path") for skill in row.get("skills", [])}
            ):
                row.update(
                    status="incomplete", reason="Results are missing for selected skills; the tier did not complete."
                )
            _keep_selected(row, selected)
        elif tier == 1 and (reports / "tier1/summary.json").is_file():
            legacy = read_json(reports / "tier1/summary.json")
            try:
                from publish_reports import validate_report, validate_summary
            except ModuleNotFoundError:
                from ci.publish_reports import validate_report, validate_summary
            validate_summary(legacy)
            if (
                legacy.get("overall_status") in {"passed", "skipped"}
                and legacy.get("execution", {}).get("commit") != result["commit"]
            ):
                raise ConfigurationError("Legacy Tier 1 evidence identifies a different evaluated commit.")
            status = legacy.get("overall_status", "incomplete")
            row = {"tier": 1, "required": True, "status": "incomplete" if status == "error" else status, "skills": []}
            if status == "skipped":
                row["reason"] = "No selected skills changed; Tier 1 evaluation did not run."
            elif legacy.get("error"):
                row["reason"] = legacy["error"]
            for skill in legacy.get("skills", []):
                directory = skill.get("report_dir", "")
                if not re.fullmatch(r"skill-[0-9]{3}", directory):
                    raise ConfigurationError("A legacy Tier 1 report directory is invalid.")
                native = reports / "tier1" / directory
                json_path, html_path = native / "skillevaluator-tier1.json", native / "skillevaluator-tier1.html"
                skill_status = skill.get("overall_status", "incomplete")
                if json_path.is_file() and not json_path.is_symlink() and skill_status != "error":
                    validate_report(read_json(json_path), skill)
                elif skill_status == "passed":
                    skill_status = "incomplete"
                row["skills"].append(
                    {
                        "path": skill["path"],
                        "status": "incomplete" if skill_status == "error" else skill_status,
                        "report_json": json_path.relative_to(reports).as_posix() if json_path.is_file() else None,
                        "report_html": html_path.relative_to(reports).as_posix() if html_path.is_file() else None,
                    }
                )
            if row["skills"]:
                row["status"] = _combined([s["status"] for s in row["skills"]])
            selection = legacy.get("selection")
            _keep_selected(row, selection.get("selected") if isinstance(selection, dict) else None)
            if tier in recorded:
                row["source_attempt"] = recorded[tier]
        elif not required and env.get("SKILL_EVAL_SKILLS_CHANGED") == "false":
            # The trusted workflow decided no skill changed, so this paid
            # advisory tier did not run. That is a skip, not a failed job.
            row = {
                "tier": tier,
                "required": False,
                "status": "skipped",
                "skills": [],
                "reason": "No file under a configured skill root changed; this paid tier did not run.",
            }
        else:
            row = {
                "tier": tier,
                "required": required,
                "status": "incomplete",
                "skills": [],
                "reason": "The tier did not produce retained evidence; inspect its failed or cancelled CI job.",
            }
        result["tiers"].append(row)
    result["overall_passed"] = all(not r["required"] or r["status"] in SUCCESS for r in result["tiers"])
    try:
        from benchmark_reports import write_benchmark_reports
    except ModuleNotFoundError:
        from ci.benchmark_reports import write_benchmark_reports
    result = validate_manifest(result)
    result["benchmarks"] = write_benchmark_reports(reports, result)
    write_reports(reports, validate_manifest(result))
    return int(not result["overall_passed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--tier", type=int, required=True)
    evaluate_parser.add_argument("--workspace", type=Path, required=True)
    evaluate_parser.add_argument("--reports", type=Path, required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--reports", type=Path, required=True)
    aggregate_parser.add_argument("--tiers", default="1,2,3")
    attempts_parser = commands.add_parser("select-attempts")
    attempts_parser.add_argument("--artifacts", type=Path, required=True)
    attempts_parser.add_argument("--reports", type=Path, required=True)
    catalog_parser = commands.add_parser("catalog")
    catalog_parser.add_argument("--input", type=Path, required=True)
    catalog_parser.add_argument("--output", type=Path, required=True)
    native_parser = commands.add_parser("native-tier3")
    native_parser.add_argument("--skill", type=Path, required=True)
    native_parser.add_argument("--output", type=Path, required=True)
    native_parser.add_argument("--config", type=Path, required=True)
    native_parser.add_argument("--env-mode", choices=["local", "docker"], required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return evaluate(args.tier, args.workspace, args.reports, dict(os.environ))
        if args.command == "select-attempts":
            current = integer(os.environ, "GITHUB_RUN_ATTEMPT", 1, 1, 99_999_999)
            select_attempts(args.artifacts, args.reports, current)
            return 0
        if args.command == "catalog":
            request = read_json(args.input)
            _build_catalog(Path(request["workspace"]), [Path(p) for p in request["skills"]], args.output)
            return 0
        if args.command == "native-tier3":
            config = read_json(args.config)
            config["SKILL_EVAL_ENV_MODE"] = args.env_mode
            return native_tier3(args.skill, args.output, config)
        tiers = tuple(int(tier) for tier in args.tiers.split(","))
        if not tiers or len(set(tiers)) != len(tiers) or not set(tiers) <= {1, 2, 3}:
            raise ConfigurationError("Select unique tiers from 1,2,3.")
        return aggregate(args.reports, tiers, dict(os.environ))
    except (OSError, ValueError, KeyError) as exc:
        print(
            str(exc)
            if isinstance(exc, ConfigurationError | SelectionError)
            else "CI execution could not complete safely.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
