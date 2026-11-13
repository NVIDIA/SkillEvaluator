# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public plugin workflow over shared preparation and execution."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_yaml

if TYPE_CHECKING:
    from skillevaluator.evaluation.options import EvaluationOptions, PluginOptions

MAX_CONFIG_BYTES = 256 * 1024


def _load_config(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    from skillevaluator.tier3.evals_config import CONFIG_FILENAMES, EvalsConfigError, _validate_config

    config: dict[str, Any] = {}
    controls: dict[str, bytes] = {}
    try:
        with SecureRoot(root) as reader:
            for name in CONFIG_FILENAMES:
                relative = Path("evals") / name
                if os.path.lexists(root / relative):
                    raw, _ = reader.read_bytes(relative, MAX_CONFIG_BYTES)
                    parsed = load_bounded_yaml(raw.decode("utf-8-sig"))
                    if not isinstance(parsed, dict):
                        raise EvalsConfigError("Plugin eval config must be a nonempty mapping")
                    config = _validate_config(parsed, root / relative)
                    break
            for name in ("grader.py", "grader.sh", "tests/grader.py", "tests/grader.sh"):
                relative = Path("evals") / name
                if os.path.lexists(root / relative):
                    controls[name], _ = reader.read_bytes(relative, MAX_CONFIG_BYTES)
    except (SecurePathError, OSError, UnicodeError, StructuredDataError) as exc:
        raise EvalsConfigError("Cannot read bounded plugin eval configuration or grader") from exc
    return config, controls


def evaluate_plugin(
    options: EvaluationOptions,
    *,
    plugin_options: PluginOptions,
    evaluate: Callable[[EvaluationOptions], dict[str, Any]],
) -> dict[str, Any]:
    """Prepare, run and retain one plugin using the existing engine boundary."""
    import yaml

    from skillevaluator.models.result import Severity
    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.provider_config import resolve_llm_provider
    from skillevaluator.tier3.commands import parse_agents, resolve_agents, validate_agents
    from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS
    from skillevaluator.tier3.nemo_gym.backend import HARBOR_BACKEND, normalize_backend
    from skillevaluator.tier3.plugin_comparisons import ComparisonOptions
    from skillevaluator.tier3.plugin_evidence import EvidenceOptions
    from skillevaluator.tier3.plugin_execution import evaluate_prepared_plugin
    from skillevaluator.tier3.plugin_prepare import SourceIdentity, prepare_runtime_package
    from skillevaluator.tier3.results_location import plugin_results_directory
    from skillevaluator.validators.private_plugin_schema import validate_inventory_schema

    if normalize_backend(options.backend) != HARBOR_BACKEND:
        raise ValueError("Plugin evaluation requires --backend harbor; other backends cannot load plugin components")
    if options.skip_with_skill:
        raise ValueError("Plugin runs require the with-plugin arm; omit --skip-with-skill")
    inventory = plugin_inventory_for_root(
        options.skill_path,
        allowed_private_hosts=plugin_options.allowed_private_hosts,
        hook_allowed_urls=plugin_options.hook_allowed_urls,
    )
    if inventory is None:
        raise ValueError("No supported plugin manifest was found")
    expected = SourceIdentity.of(inventory)
    config, controls = _load_config(inventory.root)
    authored = config.get("plugin", {})
    load = plugin_options.plugin_load or authored.get("load", "wrapper")
    mode = plugin_options.lift_mode or authored.get("lift_mode", "effectiveness")
    retries = plugin_options.trial_retries
    if retries is None:
        retries = config.get("harbor", {}).get("trial_retries", 0)
    if config.get("harbor", {}).get("task_source") == "native_harbor":
        raise ValueError("Plugin runs require an evals dataset or member datasets; native_harbor is unsupported")
    if plugin_options.probe_mcp_env and not plugin_options.probe_mcp:
        raise ValueError("--probe-mcp-env requires --probe-mcp")
    validation = validate_inventory_schema(inventory, repo_root=plugin_options.repo_root)
    blocking = [finding for finding in validation.findings if finding.severity in {Severity.CRITICAL, Severity.HIGH}]
    package = prepare_runtime_package(inventory, expected=expected, blocking_findings=blocking)
    agents = parse_agents(options.agents)
    if options.agents is None:
        agents = resolve_agents(None, provider=resolve_llm_provider().provider)
    if not agents or validate_agents(agents):
        raise ValueError("Select supported agents with --agents")
    options = replace(
        options, agents=",".join(agents), results_dir=plugin_results_directory(inventory.root, options.results_dir)
    )
    dataset = next(
        (
            inventory.root / "evals" / f"evals{extension}"
            for extension in DATASET_EXTENSIONS
            if os.path.lexists(inventory.root / "evals" / f"evals{extension}")
        ),
        None,
    )
    comparison = (
        None if mode == "effectiveness" else ComparisonOptions(member_skills=True, loose_parts=mode == "ladder")
    )
    if mode == "integration":
        options = replace(options, skip_baseline=True)
    evidence = EvidenceOptions(
        probe_mcp=bool(plugin_options.probe_mcp),
        env_grants=plugin_options.probe_mcp_env,
        allowed_private_hosts=plugin_options.allowed_private_hosts,
    )
    # The engine resolves relative paths against the skill it runs, which is the temporary wrapper.
    # Resolve them against the authored source first: ``include`` and ``decoy_manifest`` from the
    # source's parent directory, ``price_file`` from the source directory, as for a skill.
    workspace_config = config.get("skill_workspace", {})
    if include := workspace_config.get("include"):
        workspace_config["include"] = [
            str((inventory.root.parent / Path(path).expanduser()).absolute()) for path in include
        ]
    if manifest := workspace_config.get("decoy_manifest"):
        workspace_config["decoy_manifest"] = str((inventory.root.parent / Path(manifest).expanduser()).absolute())
    harbor_config = config.get("harbor", {})
    if price_file := harbor_config.get("price_file"):
        harbor_config["price_file"] = str((inventory.root / Path(price_file).expanduser()).absolute())
    staged_config = yaml.safe_dump(config).encode("utf-8") if config else None

    def run_prepared(prepared: EvaluationOptions) -> dict[str, Any]:
        if staged_config is not None:
            (prepared.skill_path / "evals" / "config.yml").write_bytes(staged_config)
        for name, raw in controls.items():
            target = prepared.skill_path / "evals" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if name.endswith(".sh"):
                target.chmod(0o700)
        environment = inventory.root / "evals" / "environment"
        if os.path.lexists(environment):
            from skillevaluator.tier3.harbor.secure_copy import copytree_secure
            from skillevaluator.tier3.plugin_prepare import verify_source_identity

            verify_source_identity(package)
            copytree_secure(
                environment,
                prepared.skill_path / "evals" / "environment",
                replace_existing=True,
                allowed_root=inventory.root,
            )
            verify_source_identity(package)
        return evaluate(prepared)

    with tempfile.TemporaryDirectory(prefix="skilleval-public-plugin-") as temporary:
        outcome = evaluate_prepared_plugin(
            package,
            options,
            work_dir=Path(temporary),
            load=load,
            evals_source=dataset,
            evaluate=run_prepared,
            comparison_options=comparison,
            evidence_options=evidence,
            max_setup_retries=retries,
        )
    result = outcome.result
    result["plugin_inventory"] = validation.metadata.get("plugin", {})
    result["plugin_provenance"] = {
        **{key: value for key, value in outcome.provenance.items() if key != "runtime_report"},
        "attempts": outcome.attempts,
        "retry_reasons": outcome.retry_reasons,
        "load": load,
        "lift_mode": mode,
    }
    if payload := outcome.provenance.get("runtime_report"):
        result["plugin_runtime_report"] = payload
    for key in ("publication_error", "cleanup_error"):
        if error := getattr(outcome, key):
            result["plugin_provenance"][key] = error
    if result.get("run_dir") and isinstance(result.get("agents"), dict) and result["agents"]:
        from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
        from skillevaluator.utils.redaction import redact_printable_text

        try:
            run_dir = Path(result["run_dir"])
            refreshed = render_agent_eval_html_report(
                inventory.root,
                run_dir,
                output_path=run_dir / "report.html",
                env_mode=options.env_mode,
                engine_result=result,
                use_llm_judge=False,
                include_benchmark=True,
            )
            result["report_path"] = str(refreshed)
            result["benchmark_path"] = str(refreshed.with_name("tier3-benchmark.md"))
            result["report_status"] = "complete"
        except Exception as exc:
            warning = redact_printable_text(
                f"Runtime report refresh unavailable: {type(exc).__name__}: {exc}", max_len=600
            )
            result["plugin_provenance"]["runtime_report_render_error"] = warning
            result.setdefault("warnings", []).append(warning)
            result["report_status"] = "degraded"
            for key in ("report_path", "benchmark_path"):
                if (path := result.get(key)) and not Path(path).is_file():
                    result.pop(key, None)
    return result
