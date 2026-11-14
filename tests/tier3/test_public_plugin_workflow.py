# SPDX-License-Identifier: Apache-2.0
"""Public runtime options and the shared service boundary."""

import json
import os
from pathlib import Path

import pytest

from skillevaluator.evaluation import EvaluationOptions, EvaluationService, PluginOptions


@pytest.mark.parametrize("value", [-1, 6, True, 1.5])
def test_public_retry_option_is_a_bounded_integer(value):
    with pytest.raises(ValueError, match="trial_retries"):
        PluginOptions(trial_retries=value)


@pytest.mark.parametrize("field,value", [("plugin_load", "other"), ("lift_mode", "other"), ("probe_mcp", 1)])
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


@pytest.mark.parametrize("backend", ["nemo-gym", "gym"])
def test_plugin_backend_is_rejected_before_preparation_or_execution(tmp_path, monkeypatch, backend):
    root = _plugin(tmp_path / "plugin")

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported plugin backend reached preparation or execution")

    monkeypatch.setattr("skillevaluator.tier3.plugin_prepare.prepare_runtime_package", forbidden)
    monkeypatch.setattr(EvaluationService, "evaluate", forbidden)

    with pytest.raises(ValueError, match=r"Plugin evaluation requires.*harbor"):
        EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code", backend=backend))


def test_public_workflow_uses_real_preparation_and_cleans_up(tmp_path, monkeypatch):
    from skillevaluator.tier3 import plugin_execution

    root = _plugin(tmp_path / "plugin")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        seen.update(root=options.skill_path, results=options.results_dir, hooks=plugin_execution.active_runtime())
        seen["cases"] = json.loads((options.skill_path / "evals/evals.json").read_text(encoding="utf-8"))
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    result = EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert seen["hooks"] is not None
    assert seen["results"] == root / "results"
    assert seen["cases"][0]["expected_skill"] == "a"
    assert not seen["root"].exists() and plugin_execution.active_runtime() is None
    assert result["plugin_runtime_report"]["execution_status"] == "succeeded"
    assert result["plugin_provenance"]["load"] == "wrapper"
    assert result["plugin_provenance"]["lift_mode"] == "effectiveness"
    assert result["plugin_provenance"]["attempts"] == 1


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
    result = EvaluationService().evaluate_plugin(
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
    EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
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
            EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    else:
        EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))


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
    with pytest.raises(EvalsConfigError, match="Cannot read bounded"):
        _load_config(root)


def test_nested_graders_retain_the_bounded_read_limit(tmp_path):
    from skillevaluator.tier3.evals_config import EvalsConfigError
    from skillevaluator.tier3.plugin_workflow import MAX_CONFIG_BYTES, _load_config

    root = _plugin(tmp_path / "plugin")
    nested = root / "evals/tests"
    nested.mkdir(parents=True)
    (nested / "grader.py").write_bytes(b"x" * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(EvalsConfigError, match="Cannot read bounded"):
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
        EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))


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
    result = EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="opencode"))

    assert result["plugin_provenance"]["complete"] is complete
    assert result["report_status"] == "complete"
    # The run-directory report that evaluate-plugin links carries the same incomplete mark as the requested formats.
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert ("plugin-components" in report) is not complete
