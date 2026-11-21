# SPDX-License-Identifier: Apache-2.0
"""Public runtime options and the shared service boundary."""

import json
import os
from pathlib import Path

import pytest

from skillevaluator.evaluation import EvaluationOptions, EvaluationService
from skillevaluator.evaluation.options import PluginOptions
from skillevaluator.tier3.results_location import ENV_RESULTS_DIR


@pytest.fixture(autouse=True)
def _default_results_location(monkeypatch):
    # The results-directory helper prefers this variable over ``<source>/results``.
    monkeypatch.delenv(ENV_RESULTS_DIR, raising=False)


@pytest.mark.parametrize("value", [-1, 6, True, 1.5])
def test_public_retry_option_is_a_bounded_integer(value):
    with pytest.raises(ValueError, match="trial_retries"):
        PluginOptions(trial_retries=value)


@pytest.mark.parametrize(
    "field,value",
    [("plugin_load", "other"), ("lift_mode", "other"), ("probe_mcp", 1), ("policy", {}), ("policy", "default")],
)
def test_public_options_validate_direct_api_values(field, value):
    with pytest.raises(ValueError, match=field):
        PluginOptions(**{field: value})


def test_plugin_controls_do_not_leak_into_skill_engine_kwargs():
    assert (
        not {"plugin_load", "lift_mode", "probe_mcp", "trial_retries"}
        & EvaluationOptions(Path("skill")).engine_kwargs().keys()
    )


def _plugin(root):
    files = {
        ".claude-plugin/plugin.json": {"name": "demo"},
        "skills/a/SKILL.md": "---\nname: a\ndescription: Fixture member.\n---\nBody.\n",
        "skills/a/evals/evals.json": [{"id": "one", "prompt": "Use a."}],
    }
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return root


def evaluate_plugin(options, *, plugin_options=None):
    """Run the plugin workflow with ``EvaluationService.evaluate`` as its engine callback."""
    from skillevaluator.tier3.plugin_workflow import evaluate_plugin as run_plugin_workflow

    service = EvaluationService()
    return run_plugin_workflow(
        options,
        plugin_options=plugin_options or PluginOptions(),
        evaluate=lambda prepared: service.evaluate(prepared, progress_reporter=None),
    )


@pytest.mark.parametrize("backend", ["nemo-gym", "gym"])
def test_plugin_backend_is_rejected_before_preparation_or_execution(tmp_path, monkeypatch, backend):
    root = _plugin(tmp_path / "plugin")

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported plugin backend reached preparation or execution")

    monkeypatch.setattr("skillevaluator.tier3.plugin_prepare.prepare_runtime_package", forbidden)
    monkeypatch.setattr(EvaluationService, "evaluate", forbidden)

    with pytest.raises(ValueError, match=r"Plugin evaluation requires.*harbor"):
        evaluate_plugin(EvaluationOptions(root, agents="claude-code", backend=backend))


@pytest.mark.parametrize(
    "run_options,plugin_options,config,error",
    [
        ({"skip_with_skill": True}, {}, None, "omit --skip-with-skill"),
        ({}, {}, "schema_version: 1\nharbor:\n  task_source: native_harbor\n", "native_harbor is unsupported"),
        ({}, {"probe_mcp_env": ("FIXTURE_TOKEN",)}, None, "--probe-mcp-env requires --probe-mcp"),
    ],
    ids=["skip-with-skill", "native-harbor", "env-grants-without-probing"],
)
def test_unsupported_run_settings_are_refused_before_preparation_or_execution(
    tmp_path, monkeypatch, run_options, plugin_options, config, error
):
    root = _plugin(tmp_path / "plugin")
    if config is not None:
        (root / "evals").mkdir()
        (root / "evals/config.yml").write_text(config, encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("A refused run reached preparation or execution")

    monkeypatch.setattr("skillevaluator.tier3.plugin_prepare.prepare_runtime_package", forbidden)
    monkeypatch.setattr(EvaluationService, "evaluate", forbidden)

    with pytest.raises(ValueError, match=error):
        evaluate_plugin(
            EvaluationOptions(root, agents="claude-code", **run_options), plugin_options=PluginOptions(**plugin_options)
        )


def test_public_workflow_uses_real_preparation_and_cleans_up(tmp_path, monkeypatch):
    from skillevaluator.tier3 import plugin_execution

    root = _plugin(tmp_path / "plugin")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        seen.update(root=options.skill_path, results=options.results_dir, hooks=plugin_execution.active_runtime())
        seen["cases"] = json.loads((options.skill_path / "evals/evals.json").read_text(encoding="utf-8"))
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    result = evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert seen["hooks"] is not None
    assert seen["results"] == root / "results"
    assert seen["cases"][0]["expected_skill"] == "a"
    assert not seen["root"].exists() and plugin_execution.active_runtime() is None
    assert result["plugin_runtime_report"]["execution_status"] == "succeeded"
    assert result["plugin_provenance"]["load"] == "wrapper"
    assert result["plugin_provenance"]["lift_mode"] == "effectiveness"
    assert result["plugin_provenance"]["attempts"] == 1


def test_results_location_variable_replaces_the_source_results(tmp_path, monkeypatch):
    root = _plugin(tmp_path / "plugin")
    monkeypatch.setenv(ENV_RESULTS_DIR, str(tmp_path / "shared-results"))
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        seen["results"] = options.results_dir
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert seen["results"] == tmp_path / "shared-results"


@pytest.mark.parametrize(
    "location,allowed",
    [("out", False), ("evals/other", False), ("results/nested", True), ("evals/results", True)],
)
def test_results_inside_the_source_must_use_its_results_directories(tmp_path, monkeypatch, location, allowed):
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package

    root = _plugin(tmp_path / "plugin")
    # The publication digest of the source ignores output only under these directories.
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text("schema_version: 1\n", encoding="utf-8")
    calls = []

    def prepare(*args, **kwargs):
        calls.append("prepare")
        return prepare_runtime_package(*args, **kwargs)

    monkeypatch.setattr("skillevaluator.tier3.plugin_prepare.prepare_runtime_package", prepare)
    monkeypatch.setattr(
        EvaluationService, "evaluate", lambda *_a, **_kw: calls.append("engine") or {"execution_status": "succeeded"}
    )
    options = EvaluationOptions(root, agents="claude-code", results_dir=root / location)
    if allowed:
        evaluate_plugin(options)
        assert calls == ["prepare", "engine"]
    else:
        with pytest.raises(ValueError, match="results root inside the plugin source"):
            evaluate_plugin(options)
        assert calls == []


def test_evals_results_needs_an_existing_evals_directory(tmp_path, monkeypatch):
    root = _plugin(tmp_path / "plugin")
    monkeypatch.setattr(EvaluationService, "evaluate", lambda *_a, **_kw: pytest.fail("refused run reached the engine"))
    # Creating evals/ for the output would itself change the source's publication digest.
    with pytest.raises(ValueError, match="results root inside the plugin source"):
        evaluate_plugin(EvaluationOptions(root, agents="claude-code", results_dir=root / "evals/results"))
    assert not (root / "evals").exists()


def test_authored_settings_apply_when_run_options_are_unset(tmp_path, monkeypatch):
    from skillevaluator.tier3 import plugin_execution
    from skillevaluator.validators import private_plugin_schema

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text(
        "schema_version: 1\nharbor:\n  trial_retries: 2\nplugin:\n  load: auto\n  lift_mode: integration\n",
        encoding="utf-8",
    )
    seen = {}
    run_private_evaluation = plugin_execution.run_private_evaluation
    validate_inventory_schema = private_plugin_schema.validate_inventory_schema

    def run(*args, **kwargs):
        seen["retries"] = kwargs["max_setup_retries"]
        return run_private_evaluation(*args, **kwargs)

    def validate(inventory, **kwargs):
        seen["repo_root"] = kwargs.get("repo_root")
        return validate_inventory_schema(inventory, **kwargs)

    def engine(self, options, *, progress_reporter=None):
        seen["skip_baseline"] = options.skip_baseline
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(plugin_execution, "run_private_evaluation", run)
    monkeypatch.setattr(private_plugin_schema, "validate_inventory_schema", validate)
    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    result = evaluate_plugin(
        EvaluationOptions(root, agents="claude-code"), plugin_options=PluginOptions(repo_root=tmp_path)
    )
    assert result["plugin_provenance"]["load"] == "auto"
    assert result["plugin_provenance"]["lift_mode"] == "integration"
    # Integration mode runs no baseline arm.
    assert seen == {"retries": 2, "repo_root": tmp_path, "skip_baseline": True}


def test_root_dataset_and_authored_config_survive_staging(tmp_path, monkeypatch):
    import yaml

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    (root / "evals/evals.json").write_text('[{"id":"root-case","prompt":"Do root task."}]', encoding="utf-8")
    (root / "evals/config.yml").write_text(
        "schema_version: 1\nharbor:\n  n_attempts: 3\n  trial_retries: 1\nplugin:\n  load: auto\n"
        "  lift_mode: both\ngrading:\n  mode: custom_only\n",
        encoding="utf-8",
    )
    (root / "evals/grader.py").write_text("print('fixture')\n", encoding="utf-8")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        seen["config"] = yaml.safe_load((options.skill_path / "evals/config.yml").read_text(encoding="utf-8"))
        seen["grader"] = (options.skill_path / "evals/grader.py").read_text(encoding="utf-8")
        seen["cases"] = json.loads((options.skill_path / "evals/evals.json").read_text(encoding="utf-8"))
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    result = evaluate_plugin(
        EvaluationOptions(root, agents="claude-code"),
        plugin_options=PluginOptions(plugin_load="wrapper", lift_mode="effectiveness", trial_retries=0),
    )
    assert seen["config"]["harbor"]["n_attempts"] == 3
    assert seen["config"]["grading"]["mode"] == "custom_only"
    assert seen["grader"] == "print('fixture')\n"
    assert [case["id"] for case in seen["cases"]] == ["root-case"]
    assert result["plugin_provenance"]["load"] == "wrapper"
    assert result["plugin_provenance"]["lift_mode"] == "effectiveness"


@pytest.mark.parametrize(
    "graders,selected",
    [
        (("tests/grader.py",), "tests/grader.py"),
        (("tests/grader.sh",), "tests/grader.sh"),
        (("tests/grader.py", "tests/grader.sh"), "tests/grader.py"),
        (("grader.sh", "tests/grader.py", "tests/grader.sh"), "grader.sh"),
        (("grader.py", "grader.sh", "tests/grader.py", "tests/grader.sh"), "grader.py"),
    ],
)
def test_custom_graders_preserve_engine_selection(tmp_path, monkeypatch, graders, selected):
    from skillevaluator.tier3.harbor.adapter import _copy_custom_grader

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text("schema_version: 1\ngrading:\n  mode: custom_only\n", encoding="utf-8")
    for relative in graders:
        path = root / "evals" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative}\n", encoding="utf-8")
    task = tmp_path / "task"

    def engine(self, options, *, progress_reporter=None):
        assert _copy_custom_grader(task, options.skill_path, "custom_only")
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    staged = task / "tests" / Path(selected).name
    assert staged.read_text(encoding="utf-8") == f"fixture {selected}\n"
    if os.name == "posix" and staged.suffix == ".sh":
        assert staged.stat().st_mode & 0o111


@pytest.mark.parametrize("mode", ["custom_only", "default_plus_custom"])
def test_missing_custom_grader_preserves_engine_policy(tmp_path, monkeypatch, mode):
    from skillevaluator.tier3.harbor.adapter import _copy_custom_grader

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text(f"schema_version: 1\ngrading:\n  mode: {mode}\n", encoding="utf-8")

    def engine(self, options, *, progress_reporter=None):
        assert _copy_custom_grader(tmp_path / "task", options.skill_path, mode) is False
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    if mode == "custom_only":
        with pytest.raises(FileNotFoundError, match="requires evals/grader"):
            evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    else:
        evaluate_plugin(EvaluationOptions(root, agents="claude-code"))


@pytest.mark.parametrize("link", ["file", "directory", "broken"])
def test_nested_grader_links_are_refused(tmp_path, link):
    from skillevaluator.tier3.evals_config import EvalsConfigError
    from skillevaluator.tier3.plugin_workflow import _load_config

    root = _plugin(tmp_path / "plugin")
    nested = root / "evals/tests"
    nested.parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "grader.py").write_text("fixture\n", encoding="utf-8")
    try:
        if link == "directory":
            nested.symlink_to(outside, target_is_directory=True)
        else:
            nested.mkdir()
            target = outside / ("missing.py" if link == "broken" else "grader.py")
            (nested / "grader.py").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(EvalsConfigError, match=r"Cannot read bounded .* evals/tests/grader\.py: \S"):
        _load_config(root)


def test_nested_graders_retain_the_bounded_read_limit(tmp_path):
    from skillevaluator.tier3.evals_config import EvalsConfigError
    from skillevaluator.tier3.plugin_workflow import MAX_CONFIG_BYTES, _load_config

    root = _plugin(tmp_path / "plugin")
    nested = root / "evals/tests"
    nested.mkdir(parents=True)
    (nested / "grader.py").write_bytes(b"x" * (MAX_CONFIG_BYTES + 1))
    # The error names the file and the reason, not only the fixed prefix.
    with pytest.raises(
        EvalsConfigError,
        match=rf"Cannot read bounded .* evals/tests/grader\.py: Selected file exceeds the {MAX_CONFIG_BYTES}-byte limit",
    ):
        _load_config(root)


def test_invalid_config_yaml_reports_the_file_and_position(tmp_path):
    from skillevaluator.tier3.evals_config import EvalsConfigError
    from skillevaluator.tier3.plugin_workflow import _load_config

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text(
        "schema_version: 1\nharbor: {n_attempts: 1, n_attempts: 2}\n", encoding="utf-8"
    )
    # As the engine reports it for a skill: the file, then the line, column and problem.
    with pytest.raises(EvalsConfigError, match=r"config\.yml: invalid YAML: line 2, column 9: .*duplicate mapping key"):
        _load_config(root)


def test_unsafe_mcp_refused_before_engine_call(tmp_path, monkeypatch):
    from skillevaluator.tier3.plugin_prepare import PreparationError

    root = _plugin(tmp_path / "plugin")
    (root / ".claude-plugin/plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "mcpServers": {"unsafe": {"command": "npx", "args": ["dangerous@latest"]}},
            }
        ),
        encoding="utf-8",
    )

    def never(*args, **kwargs):
        pytest.fail("Blocked package reached the engine")

    monkeypatch.setattr(EvaluationService, "evaluate", never)
    with pytest.raises(PreparationError, match="policy blocks"):
        evaluate_plugin(EvaluationOptions(root, agents="claude-code"))


def _mcp_plugin(root, package):
    root = _plugin(root)
    manifest = {"name": "demo", "mcpServers": {"server": {"command": "npx", "args": [package]}}}
    (root / ".claude-plugin/plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _policy(tmp_path, overrides):
    import yaml

    from skillevaluator.validators.policy import resolve_policy

    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({"severity_overrides": overrides}), encoding="utf-8")
    return resolve_policy(policy_path=path)


def test_policy_that_raises_a_finding_to_blocking_stops_the_run_before_the_engine(tmp_path, monkeypatch):
    from skillevaluator.tier3.plugin_prepare import PreparationError

    # An unpinned package is a MEDIUM finding by default, so only the override makes it blocking.
    root = _mcp_plugin(tmp_path / "plugin", "some-package")
    policy = _policy(tmp_path, {"MCP_DECLARATION.mcp_unpinned_package": "critical"})
    calls = []
    monkeypatch.setattr(EvaluationService, "evaluate", lambda *_a, **_kw: calls.append(1) or {"execution_status": "x"})

    evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert calls == [1]
    with pytest.raises(PreparationError, match="policy blocks runtime evaluation: mcp_unpinned_package"):
        evaluate_plugin(
            EvaluationOptions(root, agents="claude-code"), plugin_options=PluginOptions(policy=policy)
        )
    assert calls == [1]


def test_policy_that_lowers_a_blocking_finding_lets_the_run_start(tmp_path, monkeypatch):
    from skillevaluator.tier3.plugin_prepare import PreparationError

    # A floating package version is a HIGH finding by default.
    root = _mcp_plugin(tmp_path / "plugin", "some-package@latest")
    policy = _policy(tmp_path, {"MCP_DECLARATION.*": "medium"})
    calls = []
    monkeypatch.setattr(
        EvaluationService, "evaluate", lambda *_a, **_kw: calls.append(1) or {"execution_status": "succeeded"}
    )

    with pytest.raises(PreparationError, match="mcp_command_floating_version"):
        evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert calls == []
    result = evaluate_plugin(
        EvaluationOptions(root, agents="claude-code"), plugin_options=PluginOptions(policy=policy)
    )
    assert calls == [1]
    assert result["plugin_runtime_report"]["execution_status"] == "succeeded"


@pytest.mark.parametrize("value", ["strict", {"MCP_DECLARATION.*": "critical"}])
def test_plugin_policy_must_be_a_validation_policy(value):
    with pytest.raises(ValueError, match="policy"):
        PluginOptions(policy=value)


@pytest.mark.parametrize("promote", [True, False], ids=["promoted-medium-blocks", "demoted-high-runs"])
def test_runtime_preparation_uses_effective_policy_severity(tmp_path, monkeypatch, promote):
    from skillevaluator.models.result import Severity
    from skillevaluator.tier3.plugin_prepare import PreparationError
    from skillevaluator.validators.policy import ValidationPolicy

    root = _plugin(tmp_path / "plugin")
    if promote:
        manifest = root / ".cursor-plugin/plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name":"other"}', encoding="utf-8")
        overrides = {"PLUGIN_SCHEMA.plugin_manifest_conflict": Severity.HIGH}
    else:
        (root / ".claude-plugin/plugin.json").write_text(
            json.dumps({"name": "demo", "mcpServers": {"server": {"command": "npx", "args": ["demo@latest"]}}}),
            encoding="utf-8",
        )
        overrides = {"MCP_DECLARATION.*": Severity.LOW}
    engine_roots = []

    def engine(self, options, *, progress_reporter=None):
        engine_roots.append(options.skill_path)
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    options = PluginOptions(policy=ValidationPolicy(severity_overrides=overrides))
    if promote:
        with pytest.raises(PreparationError, match="plugin_manifest_conflict"):
            evaluate_plugin(EvaluationOptions(root, agents="claude-code"), plugin_options=options)
        assert engine_roots == []
    else:
        result = evaluate_plugin(
            EvaluationOptions(root, agents="claude-code"), plugin_options=options
        )
        assert result["execution_status"] == "succeeded"
        assert len(engine_roots) == 1 and not engine_roots[0].exists()


def _scored_run(run_dir):
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    for arm in ("with-skill", "without-skill"):
        summary = run_dir / "opencode" / arm / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "scores": dict.fromkeys(DEFAULT_METRICS, 0.8),
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": 1,
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            ),
            encoding="utf-8",
        )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name, "agents": {}}), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "incomplete"])
def test_refreshed_run_report_is_incomplete_when_the_package_is(tmp_path, monkeypatch, complete):
    root = _plugin(tmp_path / "plugin")
    if not complete:
        # A bundle-reference plugin: its packaged skill still runs, but the rule outside the root is never staged.
        (root / ".claude-plugin/plugin.json").unlink()
        manifest = {
            "name": "demo",
            "description": "Fixture.",
            "version": "1.0.0",
            "author": {"email": "dev@example.com"},
            "rules": {"refs": ["github::other/lib::rules::style.md"]},
        }
        (root / "agent_plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    run_dir = _scored_run(tmp_path / "results" / "demo" / "20261110_010001")

    def engine(self, options, *, progress_reporter=None):
        return {
            "execution_status": "succeeded",
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "agents": {"opencode": {"with_skill": {}}},
        }

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    result = evaluate_plugin(EvaluationOptions(root, agents="opencode"))

    assert result["plugin_provenance"]["complete"] is complete
    assert result["report_status"] == "complete"
    # The run-directory report that evaluate-plugin links carries the same incomplete mark as the requested formats.
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert ("plugin-components" in report) is not complete


# What a run saves in result.json after its own report pass (Insights output included).
SAVED_FEEDBACK = {
    "schema_version": "1.0",
    "conclusions": [
        {
            "severity": "fail",
            "title": "Skill effect",
            "message": "per-case result: skill hurts (saved by the run).",
            "source": "deterministic",
        },
        {"severity": "warn", "title": "Saved conclusion", "message": "Written by the run's Insights pass"},
    ],
    "recommendations": [{"severity": "warn", "title": "Saved recommendation", "message": "Kept from the run"}],
    "suggestions": ["Saved suggestion"],
    "suggestions_v2": [],
}
SAVED_FIXES = [
    {
        "case": "one",
        "scope": "opencode",
        "kind": "skill hurts",
        "what": "passes 1/1 without the skill but 0/1 with it",
        "attempts_lost": 1,
        "action": "Saved fix from the run",
    }
]


def test_refreshed_run_report_keeps_the_feedback_the_run_saved(tmp_path, monkeypatch):
    from skillevaluator.evaluation import insights_judge
    from skillevaluator.tier3 import case_summary

    root = _plugin(tmp_path / "plugin")
    run_dir = _scored_run(tmp_path / "results" / "demo" / "20261110_010001")
    cases = {
        "with_skill": {"cases": {"one": {"attempts": [{"passed": False, "trial": "one__1", "score": 0.4}]}}},
        "without_skill": {"cases": {"one": {"attempts": [{"passed": True, "trial": "one__1", "score": 0.6}]}}},
    }
    engine_result = {
        "execution_status": "succeeded",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "agents": {"opencode": {"pass_at_k": cases}},
        "tier3_feedback": SAVED_FEEDBACK,
    }
    (run_dir / "result.json").write_text(json.dumps(engine_result), encoding="utf-8")
    (run_dir / "what_to_fix.json").write_text(json.dumps(SAVED_FIXES), encoding="utf-8")
    saved_fixes = (run_dir / "what_to_fix.json").read_bytes()
    calls = []
    monkeypatch.setattr(insights_judge, "build_insights", lambda *_a, **_kw: calls.append("insights") or {})
    monkeypatch.setattr(case_summary, "_llm_actions", lambda _issues, *_a: calls.append("what_to_fix") or [])
    monkeypatch.setattr(EvaluationService, "evaluate", lambda *_a, **_kw: json.loads(json.dumps(engine_result)))

    result = evaluate_plugin(EvaluationOptions(root, agents="opencode"))

    assert result["report_status"] == "complete"
    assert calls == []
    # The refresh reuses the saved feedback: no rule-based rebuild replaces it in the result,
    # the report or the run's what_to_fix.json.
    assert result["tier3_feedback"] == SAVED_FEEDBACK
    assert (run_dir / "what_to_fix.json").read_bytes() == saved_fixes
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    for text in ("Saved conclusion", "Saved recommendation", "Saved suggestion"):
        assert text in report
