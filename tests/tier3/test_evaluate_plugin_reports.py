# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tier3 evaluate-plugin`` writes the report formats ``validate`` writes, with the same options.

``-r/--report`` and ``-o/--output-dir`` come from ``validate``'s option set and
reports go through the same ``emit_reports``. Without either option the command
behaves as before: it prints the run summary and writes no report files. The
provider, Harbor and the result builder are mocked; the reporters are real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _f90_lift_fixtures import collect, metrics, payload, result_for

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService

FILE_FORMATS = {"json": ".json", "html": ".html", "markdown": ".md", "sarif": ".sarif.json"}


def _prepared(tmp_path: Path) -> SimpleNamespace:
    (tmp_path / "package").mkdir(exist_ok=True)
    return SimpleNamespace(
        skipped=False,
        skip_reason=None,
        incomplete_skip=False,
        package_path=tmp_path / "package",
        include_skills=(),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        native_source="snapshot",
        mcp_probe_targets=(),
        integration_evidence_error=lambda: None,
        loose_parts_error=lambda _agent="codex": "not measured",
        provenance=lambda: {"plugin_name": "demo", "partial": False},
    )


def _agent_eval_result(tmp_path: Path) -> Any:
    """A real Tier 3 result: a collected two-arm run turned into the report payload."""
    collect(
        tmp_path / "run", {"codex": {"with": {"a": [metrics(0.9)]}, "without": {"a": [metrics(0.4)]}}}, n_attempts=1
    )
    return result_for(payload(tmp_path / "run"))


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    *,
    failure: str | None = None,
    broken_builder: bool = False,
) -> tuple[Any, dict[str, Any]]:
    plugin = tmp_path / "plugin"
    plugin.mkdir(exist_ok=True)
    run_dir = tmp_path / "results-run"
    run_dir.mkdir(exist_ok=True)
    captured: dict[str, Any] = {"built": [], "refreshed": []}
    result = _agent_eval_result(tmp_path)

    def build(*_args: Any, **kwargs: Any) -> Any:
        captured["built"].append(kwargs)
        if broken_builder:
            raise ValueError("run artifact is malformed")
        return result

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: _prepared(tmp_path)
    )
    monkeypatch.setattr(
        EvaluationService, "evaluate", lambda *_a, **_k: {"run_dir": str(run_dir), "agents": {"codex": {}}}
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: failure))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "skillevaluator.evaluation.tier3_report.refresh_plugin_run_report",
        lambda *_a, **kwargs: captured["refreshed"].append(kwargs),
    )
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.write_plugin_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", build)
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_directory", build)
    monkeypatch.chdir(tmp_path)
    invoked = CliRunner().invoke(
        cli_module.cli, ["tier3", "evaluate-plugin", str(plugin), "-a", "codex", "--progress", "off", *arguments]
    )
    return invoked, captured


@pytest.mark.parametrize("command", [["tier3", "evaluate-plugin"], ["evaluate-plugin"]])
def test_help_lists_the_report_options_of_validate(command: list[str]) -> None:
    result = CliRunner().invoke(cli_module.cli, [*command, "--help"], terminal_width=200)

    assert result.exit_code == 0, result.output
    # The same two options validate has, with the same choices and defaults.
    for text in ("-r, --report", "[cli|json|html|markdown|sarif]", "[default: cli]", "-o, --output-dir DIRECTORY"):
        assert text in result.output, text
    assert "Directory for generated reports.  [default: reports]" in result.output
    assert "Without -r or -o only the run summary prints" in result.output
    validate_help = CliRunner().invoke(cli_module.cli, ["validate", "--help"], terminal_width=200).output
    assert "-r, --report" in validate_help and "-o, --output-dir DIRECTORY" in validate_help


@pytest.mark.parametrize("fmt", sorted(FILE_FORMATS))
def test_each_file_format_is_written_to_the_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fmt: str
) -> None:
    result, captured = _invoke(monkeypatch, tmp_path, ["-r", fmt, "-o", "out"])

    assert result.exit_code == 0, result.output
    written = sorted((tmp_path / "out").iterdir())
    assert len(written) == 1, written
    assert written[0].name.startswith("skillevaluator-output-") and written[0].name.endswith(FILE_FORMATS[fmt])
    assert f"{fmt} report:" in result.output
    # The result is built once, as validate builds it, and report.html is refreshed from that same result.
    assert len(captured["built"]) == 1
    assert captured["built"][0]["dataset_source"] == tmp_path / "package"
    assert [set(call) for call in captured["refreshed"]] == [{"result"}]
    if fmt == "json":
        report = json.loads(written[0].read_text(encoding="utf-8"))
        assert [entry["validator"] for entry in report["results"]] == ["AGENT_EVAL"]


def test_every_format_at_once_and_the_cli_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, _captured = _invoke(monkeypatch, tmp_path, ["-r", "cli,json,html,markdown,sarif", "-o", "out"])

    assert result.exit_code == 0, result.output
    suffixes = sorted(path.name.split(".", 1)[1] for path in (tmp_path / "out").iterdir())
    assert suffixes == ["html", "json", "md", "sarif.json"]
    # -r cli prints the full CLI report, as validate does.
    assert "SkillEvaluator Validation Results" in result.output
    assert "[AGENT_EVAL] Tier 3 plugin evaluation" in result.output


def test_output_dir_alone_uses_the_validate_default_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _invoke(monkeypatch, tmp_path, ["-o", "out"])

    assert result.exit_code == 0, result.output
    # validate's default is -r cli: the CLI report prints and no file is written.
    assert len(captured["built"]) == 1
    assert "[AGENT_EVAL] Tier 3 plugin evaluation" in result.output
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_without_report_options_nothing_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _invoke(monkeypatch, tmp_path, [])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "reports").exists()
    assert "report:" not in result.output
    assert "SkillEvaluator Validation Results" not in result.output
    # The report-ready result is not built; report.html is refreshed from the run as before.
    assert captured["built"] == []
    assert [sorted(call) for call in captured["refreshed"]] == [["engine_result", "env_mode", "plugin_provenance"]]


def test_a_bad_report_format_is_a_usage_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, captured = _invoke(monkeypatch, tmp_path, ["-r", "pdf"])

    assert result.exit_code == 2
    assert "Invalid value for '-r' / '--report'" in result.output
    assert "'pdf' is not one of" in result.output
    assert captured["built"] == [] and captured["refreshed"] == []


def test_a_run_that_did_not_complete_still_writes_its_reports_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The run directory is this run's latest results, so the partial result is built from it (as in validate).
    monkeypatch.setattr(
        "skillevaluator.tier3.results_location.resolve_latest_results", lambda *_a, **_k: tmp_path / "results-run"
    )
    result, _captured = _invoke(monkeypatch, tmp_path, ["-r", "json", "-o", "out"], failure="codex timed out")

    assert result.exit_code == 1
    assert "did not complete" in result.output
    (written,) = (tmp_path / "out").iterdir()
    report = json.loads(written.read_text(encoding="utf-8"))
    (tier3,) = report["results"]
    # Marked as validate marks a plugin run that did not complete.
    assert (tier3["validator"], tier3["passed"], tier3["status"]) == ("AGENT_EVAL", False, "incomplete")


class _BrokenSarif:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def get_file_extension(self) -> str:
        return ".sarif.json"

    def save(self, *_args: Any) -> None:
        raise RuntimeError("sarif writer broke")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (None, "the sarif report was not written"),
        ("codex timed out", "did not complete: codex timed out, and the sarif report was not written"),
    ],
    ids=["completed-run", "incomplete-run"],
)
def test_a_report_that_fails_to_render_fails_the_command_after_the_others_are_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str | None, message: str
) -> None:
    monkeypatch.setitem(cli_module.emit_reports.__globals__["REPORTERS"], "sarif", _BrokenSarif)

    result, _captured = _invoke(monkeypatch, tmp_path, ["-r", "json,sarif", "-o", "out"], failure=failure)

    assert result.exit_code == 1
    assert message in " ".join(result.output.split())
    assert [path.suffix for path in (tmp_path / "out").iterdir()] == [".json"]


def test_a_result_that_cannot_be_built_never_fails_a_completed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _captured = _invoke(monkeypatch, tmp_path, ["-r", "json", "-o", "out"], broken_builder=True)

    # As in validate: an advisory skip result says why, the reports are written, and the run still exits 0.
    assert result.exit_code == 0, result.output
    (written,) = (tmp_path / "out").iterdir()
    (tier3,) = json.loads(written.read_text(encoding="utf-8"))["results"]
    assert tier3["validator"] == "AGENT_EVAL" and tier3["passed"] is False
    assert "run artifact is malformed" in written.read_text(encoding="utf-8")
