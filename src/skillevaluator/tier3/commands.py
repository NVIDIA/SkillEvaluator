# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 command implementations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import webbrowser
from pathlib import Path
from typing import Any

import yaml
from rich.box import SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillevaluator import __version__
from skillevaluator.evaluation.results import DatasetGenerationError, DatasetGenerationResult
from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
from skillevaluator.tier3.case_ids import safe_child, validate_case_id, validate_case_ids
from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS, load_dataset_entries_with_format
from skillevaluator.tier3.evals_config import CONFIG_FILENAMES, _validate_config, load_evals_config
from skillevaluator.tier3.evals_spec import validate_harbor_contract, validate_skillevaluators
from skillevaluator.tier3.harbor import HARBOR_AGENTS, HARBOR_AGENTS_SUPPORTED, canonical_agent_name
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, LEGACY_METRICS
from skillevaluator.tier3.harbor.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    Tier3RunPlan,
    safe_progress_reporter,
    secret_values_from_environment,
)
from skillevaluator.tier3.harbor.runner import (
    _check_prerequisites,
    _harbor_bin,
    _model_for_agent,
    _resolve_agent_runtime_plan,
    run_harbor_eval,
)
from skillevaluator.tier3.harbor.secure_copy import copytree_secure
from skillevaluator.tier3.results_location import (
    iter_candidate_results_roots,
    resolve_latest_results,
    resolve_results_root,
)
from skillevaluator.tier3.toml_utils import toml_quote

console = Console()

_HARBOR_RESERVED_CASE_NAMES = frozenset({"dataset.toml", "readme.md", "metric.py", "results"})


def _engine_env_mode(value: str) -> str:
    return value


def _starter_evals_entry(skill_path: Path, case_id: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "prompt": "Replace this with the eval prompt for the agent.",
        "expected_output": "Replace this with the expected outcome.",
        "expected_skill": skill_path.name,
        "assertions": [
            "The agent addresses the task using the relevant skill workflow.",
        ],
    }


def _ensure_starter_evals_json(evals_dir: Path, skill_path: Path, case_id: str) -> None:
    existing = [evals_dir / f"evals{extension}" for extension in DATASET_EXTENSIONS]
    existing = [path for path in existing if path.exists()]
    if len(existing) > 1:
        raise ValueError("multiple eval datasets found: " + ", ".join(str(path) for path in existing))
    if existing:
        return
    evals_file = evals_dir / "evals.json"
    payload = {"skill_name": skill_path.name, "evals": [_starter_evals_entry(skill_path, case_id)]}
    evals_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validate_existing_scaffold_state(private_root: Path, evals_dir: Path) -> None:
    """Validate authored dataset/config inputs before changing a private snapshot."""

    datasets = [evals_dir / f"evals{extension}" for extension in DATASET_EXTENSIONS]
    datasets = [path for path in datasets if path.exists()]
    if len(datasets) > 1:
        raise ValueError("multiple eval datasets found: " + ", ".join(str(path) for path in datasets))
    if datasets:
        entries, dataset_format = load_dataset_entries_with_format(datasets[0])
        if not entries:
            raise ValueError(f"eval dataset is empty: {datasets[0]}")
        validate_case_ids(entry.get("id") for entry in entries)
        for index, entry in enumerate(entries):
            required_text = ("prompt", "expected_output") if dataset_format == "agentskills" else ("question",)
            for field_name in required_text:
                value = entry.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid eval dataset entry {index}: {field_name} must be a non-empty string")
            if dataset_format == "agentskills":
                for authored_name, runtime_name in (
                    ("prompt", "question"),
                    ("expected_output", "ground_truth"),
                    ("assertions", "expected_behavior"),
                ):
                    if authored_name in entry and runtime_name in entry and entry[authored_name] != entry[runtime_name]:
                        raise ValueError(
                            f"invalid eval dataset entry {index}: conflicting {authored_name} and {runtime_name}"
                        )
            assertions = entry.get("assertions")
            if assertions is not None and (
                not isinstance(assertions, list)
                or any(not isinstance(assertion, str) or not assertion.strip() for assertion in assertions)
            ):
                raise ValueError(f"invalid eval dataset entry {index}: assertions must be a list of non-empty strings")
            question = entry.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"invalid eval dataset entry {index}: question must be a non-empty string")
            ground_truth = entry.get("ground_truth")
            if ground_truth is not None and (
                not isinstance(ground_truth, str) or (dataset_format == "agentskills" and not ground_truth.strip())
            ):
                raise ValueError(f"invalid eval dataset entry {index}: ground_truth must be a non-empty string")
            expected_behavior = entry.get("expected_behavior")
            if expected_behavior is not None and (
                not isinstance(expected_behavior, list)
                or any(not isinstance(behavior, str) or not behavior.strip() for behavior in expected_behavior)
            ):
                raise ValueError(
                    f"invalid eval dataset entry {index}: expected_behavior must be a list of non-empty strings"
                )

    configs = [evals_dir / name for name in CONFIG_FILENAMES]
    configs = [path for path in configs if path.exists()]
    if len(configs) > 1:
        raise ValueError("multiple eval config files found: " + ", ".join(str(path) for path in configs))
    load_evals_config(private_root)


def _write_evals_config(evals_dir: Path, *, grading_mode: str, task_source: str) -> None:
    candidates = [evals_dir / name for name in CONFIG_FILENAMES]
    existing = [path for path in candidates if path.exists()]
    if len(existing) > 1:
        raise ValueError("multiple eval config files found: " + ", ".join(str(path) for path in existing))
    config_file = existing[0] if existing else candidates[0]
    if config_file.exists():
        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            raise ValueError(f"invalid eval config in {config_file}: top level must be a mapping")
    else:
        data = {}

    data["schema_version"] = data.get("schema_version", 1)
    harbor = data.setdefault("harbor", {})
    if not isinstance(harbor, dict):
        raise ValueError(f"invalid eval config in {config_file}: harbor must be a mapping")
    harbor["task_source"] = task_source
    harbor.setdefault("n_attempts", 1)
    harbor.setdefault("pass_threshold", 0.60)
    harbor.setdefault("stop_on_pass", False)

    grading = data.setdefault("grading", {})
    if not isinstance(grading, dict):
        raise ValueError(f"invalid eval config in {config_file}: grading must be a mapping")
    grading["mode"] = grading_mode

    _validate_config(data, config_file)
    config_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _python_grader_template() -> str:
    return '''#!/usr/bin/env python3
"""User-owned SkillEvaluator custom grader.

Contract:
- Read /logs/agent/trajectory.json for the agent trajectory.
- Read /tests/entry.json for eval case metadata.
- Write /logs/verifier/reward.json.
- Write /logs/verifier/reward.txt with a numeric 0.0-1.0 score.

In default_plus_custom mode, default grading runs first. Put domain-specific scores
under custom_metrics and do not overwrite reserved metric names:
security, skill_execution, skill_efficiency, accuracy, goal_accuracy,
behavior_check, overall, details, metrics, metric_set, entry_id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TRAJECTORY_JSON = Path(os.environ.get("HARBOR_ATIF_PATH", "/logs/agent/trajectory.json"))
ENTRY_JSON = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
REWARD_JSON = Path(os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
REWARD_TXT = Path(os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))


def main() -> None:
    _trajectory = json.loads(TRAJECTORY_JSON.read_text(encoding="utf-8")) if TRAJECTORY_JSON.exists() else {}
    _entry = json.loads(ENTRY_JSON.read_text(encoding="utf-8")) if ENTRY_JSON.exists() else {}

    score = 1.0
    reward = {
        "overall": score,
        "custom_metrics": {
            "custom_check": score,
        },
        "details": {
            "custom_check": {
                "score": score,
                "reason": "Replace this with real custom grading logic.",
            }
        },
    }
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    REWARD_TXT.write_text(str(score), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


def _shell_grader_template() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

# User-owned SkillEvaluator custom grader.
#
# Contract:
# - Read /logs/agent/trajectory.json for the agent trajectory.
# - Read /tests/entry.json for eval case metadata.
# - Write /logs/verifier/reward.json.
# - Write /logs/verifier/reward.txt with a numeric 0.0-1.0 score.
#
# In default_plus_custom, default grading runs first. Put domain-specific scores
# under custom_metrics and do not overwrite reserved metric names.

REWARD_JSON="${HARBOR_REWARD_JSON:-/logs/verifier/reward.json}"
REWARD_TXT="${HARBOR_REWARD_TXT:-/logs/verifier/reward.txt}"
score="1.0"

mkdir -p "$(dirname "$REWARD_JSON")"
cat > "$REWARD_JSON" <<EOF
{
  "overall": ${score},
  "custom_metrics": {
    "custom_check": ${score}
  },
  "details": {
    "custom_check": {
      "score": ${score},
      "reason": "Replace this with real custom grading logic."
    }
  }
}
EOF

printf "%s" "$score" > "$REWARD_TXT"
"""


def _write_grader_file(path: Path, *, language: str) -> None:
    if language == "shell":
        path.write_text(_shell_grader_template(), encoding="utf-8")
    else:
        path.write_text(_python_grader_template(), encoding="utf-8")
    path.chmod(0o755)


def _native_test_sh(language: str) -> str:
    command = 'bash "${tests_dir}/grader.sh"' if language == "shell" else 'python3 "${tests_dir}/grader.py"'
    return f'#!/bin/bash\nset -euo pipefail\ntests_dir="${{HARBOR_TESTS_DIR:-/tests}}"\n{command}\n'


def _write_harbor_dataset(harbor_dir: Path, case_id: str) -> None:
    """Add one quoted task to a private Harbor dataset manifest."""
    dataset_file = harbor_dir / "dataset.toml"
    task_name = f"nvidia/{case_id}"
    if dataset_file.exists():
        content = dataset_file.read_text(encoding="utf-8")
        try:
            parsed = tomllib.loads(content)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"invalid Harbor dataset manifest {dataset_file}: {exc}") from exc
        tasks = parsed.get("tasks", [])
        if not isinstance(tasks, list) or any(
            not isinstance(task, dict) or not isinstance(task.get("name"), str) for task in tasks
        ):
            raise ValueError(f"invalid Harbor dataset manifest {dataset_file}: tasks must contain string names")
        if any(task["name"] == task_name for task in tasks):
            return
        separator = "" if not content or content.endswith("\n\n") else "\n" if content.endswith("\n") else "\n\n"
        content = f"{content}{separator}[[tasks]]\nname = {toml_quote(task_name)}\n"
    else:
        content = (
            "[dataset]\n"
            f"name = {toml_quote('custom-skillevaluator')}\n"
            f"description = {toml_quote('User-owned Harbor task set for SkillEvaluator')}\n\n"
            "[[tasks]]\n"
            f"name = {toml_quote(task_name)}\n"
        )
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"generated invalid Harbor dataset manifest: {exc}") from exc
    dataset_file.write_text(content, encoding="utf-8")


def parse_agents(raw_agents: str | None) -> list[str]:
    """Parse a comma-separated Harbor agent list."""
    if not raw_agents:
        return ["codex"]
    agents = [canonical_agent_name(item.strip()) for item in raw_agents.split(",") if item.strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for agent in agents:
        if agent not in seen:
            deduped.append(agent)
            seen.add(agent)
    return deduped


def parse_agent_model_overrides(raw_overrides: tuple[str, ...]) -> dict[str, list[str]]:
    """Parse ``--agent-model agent=model`` values."""
    overrides: dict[str, list[str]] = {}
    authored_names: dict[str, str] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError("--agent-model must be in AGENT=MODEL form")
        agent, model = raw.split("=", 1)
        authored_name = agent.strip()
        agent = canonical_agent_name(authored_name)
        model = model.strip()
        if not agent or not model:
            raise ValueError("--agent-model must include both agent and model")
        previous = authored_names.get(agent)
        if previous is not None:
            raise ValueError(
                f"--agent-model names {previous} and {authored_name} refer to the same agent "
                f"({agent}); specify only one model for {agent}"
            )
        overrides.setdefault(agent, []).append(model)
        authored_names[agent] = authored_name
    return overrides


def validate_agents(agents: list[str]) -> list[str]:
    """Return unsupported agent names."""
    return [agent for agent in agents if agent not in HARBOR_AGENTS]


def create_dataset(
    skill_path: Path,
    *,
    full: bool = False,
    no_llm: bool = False,
    dry_run: bool = False,
    force: bool = False,
    prompt: Path | None = None,
    refine: bool = False,
    from_results: Path | None = None,
    results_dir: Path | None = None,
) -> DatasetGenerationResult:
    """Generate a synthetic evaluation dataset using the migrated generator."""
    from skillevaluator.tier3 import generate_dataset

    argv = [str(skill_path.resolve())]
    if full:
        argv.append("--full")
    if no_llm:
        argv.append("--no-llm")
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    if prompt:
        argv.extend(["--prompt", str(prompt.resolve())])
    if refine:
        argv.append("--refine")
    if from_results:
        argv.extend(["--from-results", str(from_results.resolve())])
    if results_dir:
        argv.extend(["--results-dir", str(results_dir.expanduser().resolve())])

    try:
        return generate_dataset.main(argv)
    except SystemExit as exc:
        diagnostic = getattr(exc, "diagnostic", None)
        if isinstance(diagnostic, str) and diagnostic.strip():
            raise DatasetGenerationError(diagnostic) from exc
        raise DatasetGenerationError(f"Dataset generation failed with exit code {exc.code}") from exc


def init_custom_grader(
    skill_path: Path,
    *,
    mode: str,
    language: str,
    force: bool,
    no_config: bool,
) -> int:
    """Create a BYOG custom grader starter under evals/."""
    skill_path = skill_path.expanduser().resolve()
    evals_dir = skill_path / "evals"
    if evals_dir.is_symlink():
        raise ValueError(f"evals directory must not be a symlink: {evals_dir}")
    if os.path.lexists(evals_dir) and not evals_dir.is_dir():
        raise ValueError(f"evals path must be a directory: {evals_dir}")

    with tempfile.TemporaryDirectory(prefix="skillevaluator-custom-grader-") as temporary:
        private_root = Path(temporary).resolve(strict=True)
        private_evals = private_root / "evals"
        if evals_dir.exists():
            copytree_secure(evals_dir, private_evals, allowed_root=skill_path)
        else:
            private_evals.mkdir(mode=0o700)

        grader_py = private_evals / "grader.py"
        grader_sh = private_evals / "grader.sh"
        existing = [path for path in (grader_py, grader_sh) if path.exists()]
        if existing and not force:
            names = ", ".join(str(evals_dir / path.name) for path in existing)
            console.print(f"[red]Error:[/red] custom grader already exists: {names}. Re-run with --force to overwrite.")
            return 1
        if force:
            for existing_path in existing:
                existing_path.unlink()

        private_grader = grader_sh if language == "shell" else grader_py
        _write_grader_file(private_grader, language=language)
        _ensure_starter_evals_json(private_evals, skill_path, "case-001")
        if not no_config:
            _write_evals_config(private_evals, grading_mode=mode, task_source="evals_json")

        copytree_secure(private_evals, evals_dir, replace_existing=True, allowed_root=private_root)

    grader_path = evals_dir / private_grader.name
    console.print(f"Created SkillEvaluator custom grader starter at [cyan]{grader_path}[/cyan]")
    return 0


def init_harbor_task(
    skill_path: Path,
    *,
    force: bool,
    case_id: str,
    mode: str,
    language: str,
    with_config: bool,
) -> int:
    """Create a BYOT Harbor starter template under evals/harbor/."""
    requested_skill_path = skill_path.expanduser()
    if requested_skill_path.is_symlink():
        raise ValueError(f"skill path must not be a symlink: {requested_skill_path}")
    skill_path = requested_skill_path.resolve()
    skill_file = skill_path / "SKILL.md"
    if not skill_path.is_dir() or not skill_file.is_file() or skill_file.is_symlink():
        raise ValueError(f"skill path must be a directory containing a regular SKILL.md: {skill_path}")

    case_id = validate_case_id(case_id)
    if case_id.casefold() in _HARBOR_RESERVED_CASE_NAMES:
        raise ValueError(f"case id {case_id!r} is reserved in the Harbor dataset root")

    evals_dir = skill_path / "evals"
    if evals_dir.is_symlink():
        raise ValueError(f"evals directory must not be a symlink: {evals_dir}")
    if os.path.lexists(evals_dir) and not evals_dir.is_dir():
        raise ValueError(f"evals path must be a directory: {evals_dir}")

    with tempfile.TemporaryDirectory(prefix="skillevaluator-scaffold-") as temporary:
        private_root = Path(temporary).resolve(strict=True)
        private_evals = private_root / "evals"
        if evals_dir.exists():
            copytree_secure(evals_dir, private_evals, allowed_root=skill_path)
        else:
            private_evals.mkdir(mode=0o700)
        _validate_existing_scaffold_state(private_root, private_evals)

        private_harbor = private_evals / "harbor"
        if os.path.lexists(private_harbor) and not private_harbor.is_dir():
            raise ValueError(f"Harbor scaffold root must be a directory: {evals_dir / 'harbor'}")
        private_harbor.mkdir(exist_ok=True)
        collision = next(
            (
                child.name
                for child in private_harbor.iterdir()
                if child.name != case_id and child.name.casefold() == case_id.casefold()
            ),
            None,
        )
        if collision is not None:
            raise ValueError(f"case id {case_id!r} conflicts with existing Harbor entry {collision!r}")
        private_case = safe_child(private_harbor, case_id)
        public_case = evals_dir / "harbor" / case_id
        if os.path.lexists(private_case):
            if not force:
                console.print(f"[red]Error:[/red] {public_case} already exists. Re-run with --force to overwrite.")
                return 1
            if not private_case.is_dir() or private_case.is_symlink():
                raise ValueError(f"existing Harbor case must be a real directory: {public_case}")
            shutil.rmtree(private_case)

        tests_dir = private_case / "tests"
        env_dir = private_case / "environment"
        tests_dir.mkdir(parents=True)
        env_dir.mkdir()

        _ensure_starter_evals_json(private_evals, skill_path, case_id)
        if with_config:
            _write_evals_config(private_evals, grading_mode=mode, task_source="native_harbor")
        _write_harbor_dataset(private_harbor, case_id)

        task_content = f"""schema_version = {toml_quote("1.3")}

[task]
name = {toml_quote(f"nvidia/{case_id}")}
description = {toml_quote("Custom SkillEvaluator Harbor task")}

[metadata]
entry_id = {toml_quote(case_id)}

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 180.0

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 2048
network_mode = {toml_quote("public")}
skills_dir = {toml_quote("/workspace/skills")}
"""
        tomllib.loads(task_content)
        (private_case / "task.toml").write_text(task_content, encoding="utf-8")
        (private_case / "instruction.md").write_text(
            "Replace this with the user-facing task instruction for the agent.\n", encoding="utf-8"
        )
        (env_dir / "Dockerfile").write_text(
            """FROM python:3.12-slim

RUN apt-get -o Acquire::Retries=3 update && \\
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends bash curl git jq ripgrep && \\
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/skills /logs/agent /logs/verifier
WORKDIR /workspace
""",
            encoding="utf-8",
        )
        grader_path = tests_dir / ("grader.sh" if language == "shell" else "grader.py")
        _write_grader_file(grader_path, language=language)
        test_sh = tests_dir / "test.sh"
        test_sh.write_text(_native_test_sh(language), encoding="utf-8")
        test_sh.chmod(0o755)
        (private_harbor / "README.md").write_text(
            """# SkillEvaluator BYOT Harbor Tasks

This directory is user-owned. SkillEvaluator stages copies under eval
results and does not mutate these source files in place.

Keep the result contract stable:

- trajectory path: `/logs/agent/trajectory.json`
- reward JSON: `/logs/verifier/reward.json`
- reward text: `/logs/verifier/reward.txt`
- score range: `0.0` to `1.0`
- case IDs should match `evals/evals.json` when using `default` or `default_plus_custom`
""",
            encoding="utf-8",
        )

        copytree_secure(private_evals, evals_dir, replace_existing=True, allowed_root=private_root)

    console.print(f"Created Harbor BYOT starter at [cyan]{evals_dir / 'harbor'}[/cyan]")
    return 0


def evaluate(
    skill_path: Path,
    *,
    agents: str | None,
    env_mode: str,
    skip_baseline: bool,
    n_attempts: int | None,
    pass_threshold: float | None,
    stop_on_pass: bool | None = None,
    n_concurrent: int | None,
    max_agents: int | None,
    model: str | None,
    agent_model: tuple[str, ...],
    custom_dockerfile_mode: str | None,
    skill_workspace_mode: str | None,
    include_skills: tuple[Path, ...],
    copy_repo: bool,
    grading_mode: str | None,
    results_dir: Path | None,
    harbor_keep_jobs: bool,
    agent_runtime_preflight: bool | None = None,
    timeout_multiplier: float | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Run Harbor live-agent evaluation for a skill."""
    env_mode = _engine_env_mode(env_mode)

    agent_list = parse_agents(agents)
    reporter = safe_progress_reporter(progress_reporter or NullProgressReporter())
    engine_started = False
    try:
        reporter.set_secret_values(secret_values_from_environment(os.environ))
        reporter.start(
            Tier3RunPlan(
                skill_name=skill_path.name,
                environment=env_mode,
                agents=tuple(agent_list),
                baseline=not skip_baseline,
                attempts=n_attempts,
                concurrency=n_concurrent,
                max_agents=max_agents,
                timeout_multiplier=timeout_multiplier,
            )
        )
        reporter.emit(ProgressEvent(stage="configuration", state="running"))
        unknown = validate_agents(agent_list)
        if unknown:
            supported = ", ".join(sorted(HARBOR_AGENTS_SUPPORTED))
            raise ValueError(f"Unknown agent(s): {', '.join(unknown)}. Supported agents: {supported}")

        try:
            resolve_llm_provider()
        except ProviderConfigurationError as exc:
            raise ValueError(f"A public LLM provider is required for live evaluation: {exc}") from exc

        agent_models = parse_agent_model_overrides(agent_model)
        unknown_model_agents = sorted(set(agent_models) - set(agent_list))
        if unknown_model_agents:
            raise ValueError(
                "--agent-model provided for agent(s) not selected by -a/--agents: " + ", ".join(unknown_model_agents)
            )

        output_dir = resolve_results_root(skill_path, results_dir)
        engine_started = True
        return run_harbor_eval(
            skill_path=skill_path.resolve(),
            agents=agent_list,
            skip_baseline=skip_baseline,
            n_attempts=n_attempts,
            pass_threshold=pass_threshold,
            stop_on_pass=stop_on_pass,
            n_concurrent=n_concurrent,
            max_agents=max_agents,
            model=model,
            agent_models=agent_models or None,
            custom_dockerfile_mode=custom_dockerfile_mode,
            skill_workspace_mode=skill_workspace_mode,
            include_skills=[p.resolve() for p in include_skills] or None,
            copy_repo=copy_repo,
            grading_mode=grading_mode,
            output_dir=output_dir,
            keep_harbor_jobs=harbor_keep_jobs,
            agent_runtime_preflight=agent_runtime_preflight,
            env_mode=env_mode,
            env_mode_source="CLI",
            timeout_multiplier=timeout_multiplier,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            override_storage_mb=override_storage_mb,
            progress_reporter=reporter,
        )
    except Exception as exc:
        if not engine_started:
            reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=str(exc)))
        raise
    finally:
        reporter.close()


def doctor(
    *,
    agents: str | None,
    env_mode: str,
    verify_models: bool = False,
    agent_model: tuple[str, ...] = (),
) -> int:
    """Check whether live evaluation dependencies are available."""
    env_mode = _engine_env_mode(env_mode)
    agent_list = parse_agents(agents)
    rows: list[tuple[str, str, str]] = []
    rows.append(("CLI package", "pass", f"skillevaluator {__version__}"))

    provider = None
    model_resolution: dict[str, tuple[str, str]] = {}
    runtime_plans: dict[str, Any] = {}
    try:
        provider = resolve_llm_provider()
    except ProviderConfigurationError as exc:
        rows.append(("Public LLM provider", "fail", str(exc)))
    else:
        rows.append(("Public LLM provider", "pass", f"{provider.provider} / {provider.model}"))
        try:
            overrides = parse_agent_model_overrides(agent_model)
        except ValueError as exc:
            overrides = {}
            rows.append(("Agent model plan", "fail", str(exc)))
        unknown_model_agents = sorted(set(overrides) - set(agent_list))
        if unknown_model_agents:
            rows.append(
                (
                    "Agent model plan",
                    "fail",
                    "--agent-model provided for agent(s) not selected by -a/--agents: "
                    + ", ".join(unknown_model_agents),
                )
            )
        model_resolution = {
            agent: _model_for_agent(
                agent,
                cli_model=(overrides.get(agent) or [None])[0],
                config_agents={},
                provider=provider,
            )
            for agent in agent_list
        }
        plan_error: str | None = None
        if not unknown_model_agents and not validate_agents(agent_list):
            try:
                runtime_plans = _resolve_agent_runtime_plan(
                    provider=provider,
                    agents=agent_list,
                    models={agent: details[0] for agent, details in model_resolution.items()},
                    configured_runtime_env={},
                    env_mode=env_mode,
                    model_sources={agent: details[1] for agent, details in model_resolution.items()},
                )
            except ValueError as exc:
                plan_error = str(exc)
        if provider.provider == "nv_build":
            labels = {"codex": "Codex", "claude-code": "Claude Code", "opencode": "OpenCode"}
            credential_label = (
                f"{labels[agent_list[0]]} runtime credential"
                if len(agent_list) == 1 and agent_list[0] in labels
                else "Agent runtime credential"
            )
            if plan_error is not None:
                rows.append((credential_label, "fail", plan_error))
            elif runtime_plans:
                rows.append((credential_label, "pass", "operator credential and model plan resolved"))
        elif plan_error is not None:
            rows.append(("Agent runtime credential", "fail", plan_error))

    unknown = validate_agents(agent_list)
    if unknown:
        rows.append(("Harbor agents", "fail", f"Unknown: {', '.join(unknown)}"))
    else:
        rows.append(("Harbor agents", "pass", ", ".join(agent_list)))

    prereq_errors = _check_prerequisites(env_mode=env_mode, agents=agent_list)
    if prereq_errors:
        for error in prereq_errors:
            rows.append((f"{env_mode} prerequisite", "fail", error))
    else:
        rows.append((f"{env_mode} prerequisite", "pass", "ready"))

    if verify_models:
        if provider is None or not runtime_plans:
            rows.append(("provider model", "fail", "provider model resolution was unavailable"))
        else:
            from skillevaluator.tier3.harbor.runtime_preflight import probe_model

            for agent in agent_list:
                probe = probe_model(runtime_plans[agent].provider)
                rows.append((f"{agent} model", "pass" if probe.ok else "fail", probe.detail))

    table = Table(title="SkillEvaluator Doctor", box=SIMPLE, show_edge=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    for name, status, detail in rows:
        style = "green" if status == "pass" else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", detail)
    console.print(table)
    return 0 if all(row[1] == "pass" for row in rows) else 1


def validate_evals(skill_path: Path, *, as_json: bool, strict: bool, harbor_contract: bool) -> int:
    """Validate the Tier 3 evals/ directory and optional Harbor contract."""
    results = validate_skillevaluators(skill_path)
    if harbor_contract:
        results.extend(validate_harbor_contract(skill_path))
        if _allows_missing_evals_json_for_native_custom_only(skill_path):
            results = [
                result
                for result in results
                if not (
                    result.path == "evals/evals.json"
                    and result.status == "error"
                    and "REQUIRED dataset file missing" in result.message
                )
            ]

    if as_json:
        console.print_json(
            json.dumps(
                [{"path": r.path, "status": r.status, "message": r.message} for r in results],
                indent=2,
            )
        )
    else:
        _print_validate_results(skill_path, results)

    has_errors = any(r.status == "error" for r in results)
    has_warnings = any(r.status == "warning" for r in results)
    return 1 if has_errors or (strict and has_warnings) else 0


def _allows_missing_evals_json_for_native_custom_only(skill_path: Path) -> bool:
    try:
        config, _ = load_evals_config(skill_path)
    except Exception:
        return False

    harbor = config.get("harbor") if isinstance(config, dict) else None
    grading = config.get("grading") if isinstance(config, dict) else None
    return (
        isinstance(harbor, dict)
        and isinstance(grading, dict)
        and harbor.get("task_source") == "native_harbor"
        and grading.get("mode") == "custom_only"
        and (skill_path / "evals" / "harbor").is_dir()
    )


def _print_validate_results(skill_path: Path, results: list[Any]) -> None:
    status_icon = {
        "ok": ("[ok]", "green"),
        "missing": ("[MISSING]", "red bold"),
        "warning": ("[warn]", "yellow"),
        "error": ("[ERROR]", "red bold"),
    }

    body = Text()
    for result in results:
        icon, style = status_icon.get(result.status, ("[ ? ]", "dim"))
        body.append(f"  {icon:<10s}", style=style)
        path_style = "bold" if result.status in ("error", "missing") else "white"
        body.append(f"{result.path:<38s}", style=path_style)
        body.append(f"{result.message}\n", style="dim")

    n_ok = sum(1 for result in results if result.status == "ok")
    n_warn = sum(1 for result in results if result.status == "warning")
    n_err = sum(1 for result in results if result.status in ("error", "missing"))

    summary = Text()
    summary.append(f"  {n_ok} ok", style="green")
    if n_warn:
        summary.append(f"  {n_warn} warning(s)", style="yellow")
    if n_err:
        summary.append(f"  {n_err} error(s)", style="red bold")
    if not n_err and not n_warn:
        summary.append("  all checks passed", style="green bold")

    console.print()
    console.print(
        Panel(
            body,
            title=f"[bold]Validate: {skill_path.name}/evals/[/bold]",
            border_style="cyan" if not n_err else "red",
            padding=(1, 1),
        )
    )
    console.print(summary)
    if n_err:
        console.print()
        console.print("  [dim]See expected structure:[/dim] skillevaluator tier3 validate --help")
    console.print()


def view_results(skill_path: Path, *, results_dir: Path | None = None) -> Path:
    """Open or generate the latest HTML report."""
    latest = resolve_latest_results(skill_path, results_dir)
    if not latest.exists():
        searched = ", ".join(str(p) for p in iter_candidate_results_roots(skill_path, results_dir))
        raise FileNotFoundError(
            f"No results found. Searched: {searched}. Run: skillevaluator evaluate {skill_path} -a codex"
        )

    target = latest.resolve() if latest.is_symlink() else latest
    report_path = target / "report.html"
    if not report_path.exists():
        report_path = render_agent_eval_html_report(skill_path, target)
    console.print(f"Opening: [cyan]{report_path}[/cyan]")
    webbrowser.open(report_path.as_uri())
    return report_path


def harbor_view(jobs_dir: Path) -> int:
    """Open retained Harbor job artifacts with Harbor's trajectory browser."""
    cmd = [_harbor_bin(), "view", str(jobs_dir.resolve())]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        console.print("[red]Error: harbor binary not found.[/red]")
        return 1
    except PermissionError:
        console.print("[red]Error: harbor binary is not executable.[/red]")
        return 1


def compare_results(skill_path: Path, *, results_dir: Path | None = None) -> int:
    """Compare latest Harbor result scores across agents."""
    candidate_roots = iter_candidate_results_roots(skill_path, results_dir)
    resolved_results_dir = next((root for root in candidate_roots if root.exists()), candidate_roots[0])
    if not resolved_results_dir.exists():
        searched = ", ".join(str(p) for p in candidate_roots)
        console.print(f"[red]Error: No results found. Searched: {searched}[/red]")
        return 1

    agent_with: dict[str, dict[str, float]] = {}
    agent_without: dict[str, dict[str, float]] = {}
    agent_meta: dict[str, dict[str, Any]] = {}

    for ts_dir in sorted(resolved_results_dir.iterdir(), reverse=True):
        if not ts_dir.is_dir() or ts_dir.name.startswith("_") or ts_dir.name == "latest":
            continue
        for agent_dir in sorted(ts_dir.iterdir()):
            if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
                continue
            agent_name = agent_dir.name
            if agent_name in agent_with:
                continue
            summary = agent_dir / "with-skill" / "summary.json"
            if not summary.exists():
                summary = agent_dir / "summary.json"
            if not summary.exists():
                continue
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            scores = _summary_scores(data)
            if scores:
                agent_with[agent_name] = scores
                agent_meta[agent_name] = {
                    "timestamp": ts_dir.name,
                    "path": str(agent_dir),
                    "num_trials": data.get("num_trials", "?"),
                }
                wo_summary = agent_dir / "without-skill" / "summary.json"
                if wo_summary.exists():
                    try:
                        wo_scores = _summary_scores(json.loads(wo_summary.read_text(encoding="utf-8")))
                        if wo_scores:
                            agent_without[agent_name] = wo_scores
                    except (ValueError, OSError):
                        pass

    if not agent_with:
        console.print("[red]No agent results found. Run skillevaluator evaluate first.[/red]")
        return 1

    agents = sorted(agent_with)
    display_metrics, overall_metrics = _display_metrics(agent_with)

    table = Table(show_header=True, header_style="bold dim", box=SIMPLE, padding=(0, 1), show_edge=False, expand=True)
    table.add_column("Evaluator", style="white", min_width=18, no_wrap=True)
    for agent in agents:
        table.add_column(f"{agent}\nscore", justify="right", min_width=7)
        if agent in agent_without:
            table.add_column("\nlift", justify="right", min_width=7)

    for metric in display_metrics:
        row: list[str | Text] = [Text(metric)]
        for agent in agents:
            with_score = _safe_score(agent_with[agent], metric)
            row.append(Text(f"{with_score:.2f}", style=f"bold {_score_style(with_score)}"))
            if agent in agent_without:
                without_score = _safe_score(agent_without[agent], metric)
                delta = with_score - without_score
                if delta > 0:
                    row.append(Text(f"+{delta:.2f}", style="green"))
                elif delta < 0:
                    row.append(Text(f"{delta:.2f}", style="red"))
                else:
                    row.append(Text("  -", style="dim"))
        table.add_row(*row)

    table.add_row(*[""] * (1 + sum(2 if agent in agent_without else 1 for agent in agents)))
    overall_row: list[str | Text] = [Text("Overall", style="bold")]
    for agent in agents:
        with_avg = sum(_safe_score(agent_with[agent], metric) for metric in overall_metrics) / len(overall_metrics)
        overall_row.append(Text(f"{with_avg:.2f}", style=f"bold {_score_style(with_avg)}"))
        if agent in agent_without:
            without_avg = sum(_safe_score(agent_without[agent], metric) for metric in overall_metrics) / len(
                overall_metrics
            )
            delta = with_avg - without_avg
            delta_text = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            delta_style = "bold green" if delta > 0 else ("bold red" if delta < 0 else "bold dim")
            overall_row.append(Text(delta_text, style=delta_style))
    table.add_row(*overall_row)

    console.print()
    console.print(
        Panel(table, title=f"[bold]Skill Evaluation - {skill_path.name}[/bold]", border_style="cyan", padding=(1, 1))
    )
    console.print()
    for agent in agents:
        meta = agent_meta[agent]
        console.print(f"  [dim]{agent:<16s} {meta['timestamp']} (Harbor, {meta.get('num_trials', '?')} trials)[/dim]")
    console.print()
    return 0


def _summary_scores(data: dict[str, Any]) -> dict[str, float]:
    if data.get("execution_status") != "succeeded":
        return {}
    scores: dict[str, float] = {}
    raw_scores = data.get("scores", data)
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                scores[str(key)] = float(value)
    custom_scores = data.get("custom_scores")
    if isinstance(custom_scores, dict):
        for key, value in custom_scores.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                scores[f"custom: {key}"] = float(value)
    return scores


def _display_metrics(agent_with: dict[str, dict[str, float]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    has_default_scores = any(any(metric in scores for metric in DEFAULT_METRICS) for scores in agent_with.values())
    if has_default_scores:
        default_metrics = (
            DEFAULT_METRICS if any("security" in scores for scores in agent_with.values()) else LEGACY_METRICS
        )
        custom_metrics = tuple(
            sorted(metric for scores in agent_with.values() for metric in scores if metric.startswith("custom: "))
        )
        return (*default_metrics, *custom_metrics), default_metrics
    metrics = tuple(sorted({metric for scores in agent_with.values() for metric in scores}))
    return metrics, metrics


def _safe_score(scores: dict[str, float], metric: str) -> float:
    value = scores.get(metric, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _score_style(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


__all__ = [
    "compare_results",
    "create_dataset",
    "doctor",
    "evaluate",
    "harbor_view",
    "init_custom_grader",
    "init_harbor_task",
    "parse_agents",
    "validate_evals",
    "view_results",
]
