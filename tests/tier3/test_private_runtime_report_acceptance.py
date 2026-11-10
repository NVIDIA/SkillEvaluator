# SPDX-License-Identifier: Apache-2.0
"""Private reports traverse the service and replay without live execution."""

from __future__ import annotations

import json
import os
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.evaluation.options import EvaluationOptions, PluginOptions
from skillevaluator.evaluation.service import EvaluationService
from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3 import commands, plugin_evidence, plugin_execution
from skillevaluator.tier3.harbor import runner, runtime_preflight
from skillevaluator.tier3.plugin_comparisons import ComparisonOptions
from skillevaluator.tier3.plugin_prepare import SourceIdentity, prepare_runtime_package
from skillevaluator.validators.private_plugin_schema import validate_inventory_schema
from tier3.test_private_comparison_engine_acceptance import _harbor_result

SNAPSHOT = "plugin_runtime_report.json"


def _prepared(root: Path):
    files = {
        ".claude-plugin/plugin.json": {
            "name": "acceptance",
            "description": "Offline acceptance fixture.",
            "mcpServers": {"fixture": {"url": "https://fixture.example.invalid/mcp"}},
        },
        "skills/member/SKILL.md": "---\nname: member\ndescription: Fixture member.\n---\nRead fixture inputs.\n",
        "skills/support/SKILL.md": "---\nname: support\ndescription: Fixture support.\n---\nSupport the member task.\n",
        "skills/member/evals/evals.json": [
            {
                "id": f"case-{index}",
                "prompt": f"Perform fixture task {index}.",
                "cross_component": True,
                "expected_skills": ["member", "support"],
                "expected_tools": ["SKILL:member", "SKILL:support", "MCP:fixture/search"],
            }
            for index in range(1, 6)
        ],
    }
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None and inventory.complete
    validation = validate_inventory_schema(inventory)
    assert validation.passed, validation.errors
    identity = SourceIdentity.of(inventory)
    package = prepare_runtime_package(inventory, expected=identity)
    assert package.identity == identity
    return package, validation


def _trajectory(variant: str) -> dict:
    calls, observations = [], []
    if variant != "without":
        for member in ("member", "support"):
            calls.append(
                {
                    "tool_call_id": f"read-{member}",
                    "function_name": "Read",
                    "arguments": {"file_path": f"/skills/{member}/SKILL.md"},
                }
            )
            observations.append({"source_call_id": f"read-{member}", "content": f"Fixture {member} instructions."})
    if variant in {"with", "looseparts"}:
        server = "plugin_acceptance_fixture" if variant == "with" else "fixture"
        calls.append(
            {
                "tool_call_id": "search",
                "function_name": f"mcp__{server}__search",
                "arguments": {"query": "private-argument-value-is-not-display-evidence"},
            }
        )
        observations.append({"source_call_id": "search", "content": "found", "extra": {"tool_result_is_error": False}})
    return {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": f"fixture-{variant}",
        "agent": {"name": "claude-code", "version": "fixture"},
        "final_metrics": {"total_prompt_tokens": 23, "total_completion_tokens": 5, "total_cost_usd": None},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "Fixture response.",
                "metrics": {"prompt_tokens": 23, "completion_tokens": 5},
                "tool_calls": calls,
                "observation": {"results": observations},
            }
        ],
    }


def _offline_engine(monkeypatch, *, missing: str | None = None):
    # Configuration, routing, task emission, collection and report rendering stay
    # real. These substitutions stop provider and container execution only.
    for name in tuple(os.environ):
        if name.startswith(("SKILL_EVAL_", "OPENAI_", "ANTHROPIC_", "CLAUDE_", "NVIDIA_", "AWS_", "LLM_JUDGE_")):
            monkeypatch.delenv(name)
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="offline-fixture-credential",
        base_url="https://provider.example.invalid",
        litellm_model="anthropic/claude-sonnet-4-6",
    )
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: SimpleNamespace(
            ok=True,
            provider=selected.provider,
            model=selected.model,
            detail="offline fixture",
            failure_kind=None,
            http_status=None,
        ),
    )
    monkeypatch.setattr(
        runner, "render_agent_eval_html_report", partial(render_agent_eval_html_report, use_llm_judge=False)
    )

    def no_host_probe(*_args, **_kwargs):
        pytest.fail("default private evidence must not probe the declared fixture endpoint")

    monkeypatch.setattr(plugin_evidence, "probe_mcp_servers", no_host_probe)
    launches = []

    def launch(**kwargs):
        tasks = sorted(path for path in kwargs["dataset"].iterdir() if path.is_dir())
        launches.append((kwargs["job_name"].rsplit("-", 1)[-1], [path.name for path in tasks]))
        assert len(tasks) == 5 and kwargs["n_attempts"] == 2
        job = kwargs["jobs_dir"] / kwargs["job_name"]
        # This helper runs emitted setup/census scripts with writable paths in
        # tmp_path and writes real Harbor model shapes, but starts no agent.
        _harbor_result(job, tasks, kwargs["n_attempts"])
        variant = kwargs["job_name"].rsplit("-", 1)[-1]
        for trial in job.iterdir():
            if not trial.is_dir():
                continue
            if variant != "without":
                startup_path = trial / "agent/claude-code.txt"
                startup = json.loads(startup_path.read_text(encoding="utf-8"))
                startup["skills"] = (
                    ["acceptance:member", "acceptance:support"] if variant == "with" else ["member", "support"]
                )
                startup_path.write_text(json.dumps(startup) + "\n", encoding="utf-8")
            if not (missing == "trajectory" and variant == "with"):
                (trial / "agent/trajectory.json").write_text(json.dumps(_trajectory(variant)), encoding="utf-8")
            if missing == "load" and variant == "with":
                (trial / "agent/claude-code.txt").unlink()
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", launch)
    return launches


def _run_private(
    tmp_path, monkeypatch, *, missing=None, publish=None, comparisons=True, public=False, default_results=False
):
    package, validation = _prepared(tmp_path / "plugin")
    launches = _offline_engine(monkeypatch, missing=missing)
    work = tmp_path / "work"
    work.mkdir()
    seen = {}
    observer_type = plugin_evidence.RuntimeEvidenceObserver

    def observer_factory(current_package, wrapper, hooks, options):
        # Configuration belongs to this temporary wrapper; the source and its
        # analyzed digest are unchanged. Use the real configuration reader.
        (wrapper.path / "evals/config.yml").write_text(
            "schema_version: 1\nharbor:\n  base_image_mode: disabled\n", encoding="utf-8"
        )
        seen["wrapper"] = wrapper.path
        observer = observer_type(current_package, wrapper, hooks, options)
        seen["observer"] = observer
        return observer

    def published(outcome):
        run_dir = Path(outcome.result["run_dir"])
        assert (run_dir / SNAPSHOT).is_file(), "private publication must retain the report before caller publication"
        assert seen["wrapper"].exists()
        seen["payload"] = json.loads((run_dir / SNAPSHOT).read_text(encoding="utf-8"))
        if publish is not None:
            publish(outcome)

    options = EvaluationOptions(
        skill_path=Path(package.identity.root),
        agents="claude-code",
        results_dir=None if default_results else tmp_path / "results",
        n_attempts=2,
        n_concurrent=2,
        max_cases=5,
        agent_runtime_preflight=False,
    )
    if public:
        if not default_results:
            authored = Path(package.identity.root) / "evals"
            authored.mkdir()
            (authored / "config.yml").write_text("schema_version: 1\nharbor:\n  base_image_mode: disabled\n")
        monkeypatch.setattr(plugin_evidence, "RuntimeEvidenceObserver", observer_factory)
        result = EvaluationService().evaluate_plugin(
            options,
            plugin_options=PluginOptions(plugin_load="native", lift_mode="ladder" if comparisons else "effectiveness"),
        )
        provenance = result["plugin_provenance"]
        from skillevaluator.publication_identity import publication_target_from_path

        assert result.get("publication_target") == publication_target_from_path(Path(package.identity.root))
        assert result["skill_name"] == Path(package.identity.root).name
        for marker in ("Plugin runtime evidence", "Plugin static evidence"):
            assert marker in Path(result["report_path"]).read_text()
        assert "Plugin runtime evidence" in Path(result["benchmark_path"]).read_text()
        assert EvaluationService().discover_latest_results(Path(package.identity.root), options.results_dir) == Path(
            result["run_dir"]
        )
        from skillevaluator.evaluation.tier3_report import agent_eval_result_from_run

        canonical = agent_eval_result_from_run(
            Path(package.identity.root),
            results_dir=options.results_dir or Path(package.identity.root) / "results",
            engine_result=result,
            use_llm_judge=False,
        )
        assert canonical is not None and canonical.metadata["agent_eval"]["skill_name"] == "plugin"
        outcome = SimpleNamespace(
            result=result,
            status=result["execution_status"],
            provenance=provenance,
            publication_error=provenance.get("publication_error"),
            cleanup_error=provenance.get("cleanup_error"),
        )
        seen["payload"] = json.loads((Path(result["run_dir"]) / SNAPSHOT).read_text())
        assert result["plugin_runtime_report"] == seen["payload"]
    else:
        outcome = plugin_execution.evaluate_prepared_plugin(
            package,
            options,
            work_dir=work,
            load="native",
            comparison_options=ComparisonOptions(loose_parts=True) if comparisons else None,
            observer_factory=observer_factory,
            publish=published,
        )
    assert plugin_execution.active_runtime() is None
    assert not seen["wrapper"].exists() and list(work.iterdir()) == []
    return outcome, package, validation, launches, seen


def test_public_default_results_do_not_create_an_authored_evals_ancestor(tmp_path, monkeypatch):
    monkeypatch.delenv("SKILLEVALUATOR_RESULTS_DIR", raising=False)
    outcome, package, _validation, _launches, _seen = _run_private(
        tmp_path, monkeypatch, public=True, default_results=True
    )
    root = Path(package.identity.root)
    assert outcome.status == "succeeded"
    assert not (root / "evals").exists()
    assert Path(outcome.result["run_dir"]).is_relative_to(root / "results")


@pytest.mark.parametrize("public", [False, True], ids=["private", "public-service"])
def test_service_run_retains_one_snapshot_before_cleanup_and_replays_six_formats(tmp_path, monkeypatch, public):
    outcome, package, validation, launches, seen = _run_private(tmp_path, monkeypatch, public=public)
    assert outcome.status == "succeeded", outcome.result
    assert outcome.cleanup_error is None and outcome.publication_error is None
    assert {variant for variant, _tasks in launches} == {"with", "without", "sumofparts", "looseparts"}
    assert len({tuple(tasks) for _variant, tasks in launches}) == 1
    run_dir = Path(outcome.result["run_dir"])
    assert not (run_dir / "_harbor-jobs").exists() and not (run_dir / "_harbor-tasks").exists()
    ladder = outcome.result["agents"]["claude-code"]["lift_ladder"]
    for condition in outcome.result["agents"]["claude-code"]["conditions"].values():
        assert condition["expected_cases"] == 5 and condition["scored_cases"] == 5
    assert ladder["n_cases"] == 5 and ladder["excluded_cases"] == []
    assert len(seen["observer"].context.cross_component_cases) == 5
    assert ladder["by_case_type"]["cross_component"]["n_cases"] == 5
    assert ladder["glue"]["n"] == 5
    assert {name: step["estimate"] for name, step in ladder["steps"].items()} == pytest.approx(
        {"skill": 0.2, "capability": 0.2, "integration": 0.2, "plugin": 0.6}
    )
    assert seen["payload"]["execution_status"] == "succeeded"
    assert seen["payload"]["evidence_complete"] is True and seen["payload"]["status"] == "complete"
    assert seen["payload"]["source"]["manifest_sha256"] == package.identity.manifest_sha256
    assert outcome.provenance["mcp_proof"]["fixture"]["status"] == "used-successfully"
    sections = {section["key"]: section for section in seen["payload"]["sections"]}
    assert ["claude-code", "with_skill", "skill:support", "exercised", 1.0] in sections["coverage"]["rows"]
    assert ["claude-code", "with_skill", 280, None, 0, None, None] in sections["cost"]["rows"]
    shutil.rmtree(Path(package.identity.root))

    def forbidden(*_args, **_kwargs):
        pytest.fail("saved private reporting must not collect source or runtime evidence")

    monkeypatch.setattr(runner, "_run_harbor", forbidden)
    monkeypatch.setattr(runtime_preflight, "probe_model", forbidden)
    monkeypatch.setattr(plugin_evidence, "probe_mcp_servers", forbidden)
    from skillevaluator.plugin_paths import PluginRootReader
    from skillevaluator.reporting import (
        BenchmarkReporter,
        CLIReporter,
        HTMLReporter,
        JSONReporter,
        MarkdownReporter,
        SARIFReporter,
    )
    from skillevaluator.reporting.plugin_runtime_io import load_plugin_runtime_report
    from skillevaluator.reporting.plugin_runtime_render import render_plugin_runtime_reports

    monkeypatch.setattr(PluginRootReader, "read_text", forbidden)
    loaded = load_plugin_runtime_report(run_dir)
    assert loaded.payload() == seen["payload"]
    output = render_plugin_runtime_reports(
        [validation],
        report=loaded,
        reporters=[
            CLIReporter(),
            JSONReporter(include_timestamp=False),
            HTMLReporter(include_timestamp=False),
            MarkdownReporter(include_timestamp=False),
            SARIFReporter(include_timestamp=False),
            BenchmarkReporter(include_timestamp=False),
        ],
    )
    assert set(output) == {"cli", "json", "html", "markdown", "sarif", "benchmark"}
    assert json.loads(output["json"])["plugin_runtime"] == loaded.payload()
    assert json.loads(output["sarif"])["runs"][0]["properties"]["pluginRuntime"] == loaded.payload()
    for document in output.values():
        assert "acceptance" in document
        assert "used-successfully" in document
        assert "280" in document
        assert "offline-fixture-credential" not in document
        assert "private-argument-value-is-not-display-evidence" not in document


@pytest.mark.parametrize("missing", ["load", "trajectory"])
def test_successful_scoring_does_not_turn_missing_runtime_evidence_into_complete_report(tmp_path, monkeypatch, missing):
    outcome, _package, _validation, _launches, seen = _run_private(
        tmp_path, monkeypatch, missing=missing, comparisons=missing == "load"
    )
    assert outcome.status == "succeeded", outcome.result
    assert outcome.publication_error is None
    if missing == "load":
        assert outcome.provenance["native_evidence"]["claude-code"]["loaded"] == []
    else:
        summary = outcome.result["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]
        assert summary["n_missing_trajectory"] == 10
    assert seen["payload"]["execution_status"] == "succeeded"
    assert seen["payload"]["evidence_complete"] is False
    assert seen["payload"]["status"] == "incomplete"


def test_caller_publication_failure_preserves_saved_snapshot_status_and_cleanup(tmp_path, monkeypatch):
    def failed_publication(_outcome):
        raise OSError("fixture publication failed")

    outcome, _package, _validation, _launches, seen = _run_private(
        tmp_path, monkeypatch, publish=failed_publication, comparisons=False
    )
    assert outcome.status == "succeeded"
    assert outcome.publication_error == "OSError: fixture publication failed"
    assert outcome.cleanup_error is None
    assert json.loads((Path(outcome.result["run_dir"]) / SNAPSHOT).read_text(encoding="utf-8")) == seen["payload"]


def test_snapshot_storage_failure_keeps_caller_publication_and_cleanup(tmp_path, monkeypatch):
    """Isolate the publication lifecycle; engine execution is covered above."""
    from skillevaluator.reporting import plugin_runtime_io

    package, _validation = _prepared(tmp_path / "plugin")
    work, run_dir = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    run_dir.mkdir()
    observed = {}

    def no_storage(_report, _run_dir):
        raise OSError("fixture snapshot storage failure")

    def completed_engine(options):
        observed["wrapper"] = options.skill_path
        return {"execution_status": "succeeded", "run_dir": str(run_dir)}

    def published(outcome):
        observed["called"] = True
        assert observed["wrapper"].exists()
        assert "fixture snapshot storage failure" in outcome.provenance["runtime_report_error"]

    monkeypatch.setattr(plugin_runtime_io, "save_plugin_runtime_report", no_storage)
    outcome = plugin_execution.evaluate_prepared_plugin(
        package,
        EvaluationOptions(skill_path=Path(package.identity.root), agents="claude-code"),
        work_dir=work,
        evaluate=completed_engine,
        publish=published,
    )
    assert outcome.status == "succeeded" and observed["called"]
    assert "fixture snapshot storage failure" in outcome.publication_error
    assert outcome.cleanup_error is None and list(work.iterdir()) == []
    assert not observed["wrapper"].exists() and not (run_dir / SNAPSHOT).exists()
    assert plugin_execution.active_runtime() is None


def test_ordinary_service_run_does_not_retain_a_private_snapshot(tmp_path, monkeypatch):
    _package, _validation = _prepared(tmp_path / "plugin")
    launches = _offline_engine(monkeypatch)
    skill = tmp_path / "plugin/skills/member"
    (skill / "evals/config.yml").write_text(
        "schema_version: 1\nharbor:\n  base_image_mode: disabled\n", encoding="utf-8"
    )
    result = EvaluationService().evaluate(
        EvaluationOptions(
            skill_path=skill,
            agents="claude-code",
            results_dir=tmp_path / "results",
            n_attempts=2,
            n_concurrent=2,
            agent_runtime_preflight=False,
        )
    )
    assert result["execution_status"] == "succeeded", result
    assert {variant for variant, _tasks in launches} == {"with", "without"}
    assert plugin_execution.active_runtime() is None
    assert "runtime_evidence" not in result and "private_comparisons" not in result
    assert not (Path(result["run_dir"]) / SNAPSHOT).exists()
