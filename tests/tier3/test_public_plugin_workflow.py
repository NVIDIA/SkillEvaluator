# SPDX-License-Identifier: Apache-2.0
"""Public runtime options and the shared service boundary."""

import json
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


def test_public_workflow_uses_real_preparation_and_cleans_up(tmp_path, monkeypatch):
    from skillevaluator.tier3 import plugin_execution

    root = _plugin(tmp_path / "plugin")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        seen.update(root=options.skill_path, results=options.results_dir, hooks=plugin_execution.active_runtime())
        seen["cases"] = json.loads((options.skill_path / "evals/evals.json").read_text())
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
    (root / "evals/evals.json").write_text('[{"id":"root-case","prompt":"Do root task."}]')
    (root / "evals/config.yml").write_text(
        "schema_version: 1\nharbor:\n  n_attempts: 3\n  trial_retries: 1\nplugin:\n  load: auto\n"
        "  lift_mode: both\ngrading:\n  mode: custom_only\n"
    )
    (root / "evals/grader.py").write_text("print('fixture')\n")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        seen["config"] = yaml.safe_load((options.skill_path / "evals/config.yml").read_text())
        seen["grader"] = (options.skill_path / "evals/grader.py").read_text()
        seen["cases"] = json.loads((options.skill_path / "evals/evals.json").read_text())
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


def test_unsafe_mcp_refused_before_engine_call(tmp_path, monkeypatch):
    from skillevaluator.tier3.plugin_prepare import PreparationError

    root = _plugin(tmp_path / "plugin")
    (root / ".claude-plugin/plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "mcpServers": {"unsafe": {"command": "npx", "args": ["dangerous@latest"]}},
            }
        )
    )

    def never(*args, **kwargs):
        pytest.fail("Blocked package reached the engine")

    monkeypatch.setattr(EvaluationService, "evaluate", never)
    with pytest.raises(PreparationError, match="policy blocks"):
        EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
