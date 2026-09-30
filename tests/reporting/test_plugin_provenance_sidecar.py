# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin provenance survives re-rendering a run, and the delivered report shows it."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _plugin_fixtures import element_text, provenance, write_run_dir
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService, tier3_report
from skillevaluator.evaluation.tier3_report import (
    _MAX_PLUGIN_PROVENANCE_BYTES,
    _read_plugin_provenance,
    agent_eval_result_from_directory,
    refresh_plugin_run_report,
    render_agent_eval_html_report,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX link semantics")


@pytest.fixture(autouse=True)
def _no_llm_insights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "skillevaluator.evaluation.insights_judge.build_insights",
        lambda *_args, **_kwargs: {"conclusions": [], "recommendations": []},
    )


def _run(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir(exist_ok=True)
    return plugin_dir, write_run_dir(tmp_path / "results" / "20260101_000000", **kwargs)


def test_rerender_reads_sidecar_and_keeps_incomplete_status(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))

    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    payload = result.metadata["agent_eval"]
    assert payload["plugin_provenance"]["partial"] is True
    assert payload["plugin_provenance"]["component_coverage"]["not_evaluated"] == 2
    assert result.passed is False
    assert result.metadata["skip_reason"].startswith("INCOMPLETE:")
    assert payload["conclusions"][0]["title"] == "Evaluation INCOMPLETE - unresolved dependencies"


def test_standalone_render_round_trip_preserves_incomplete_and_coverage(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))

    report = render_agent_eval_html_report(plugin_dir, run_dir, use_llm_judge=False)
    html = report.read_text(encoding="utf-8")

    assert "Plugin under validation" in html
    assert element_text(html, "tier3-plugin-incomplete") is not None
    assert "INCOMPLETE" in (element_text(html, "tier3-plugin-incomplete") or "")
    assert '<span class="tier-card-verdict">INCOMPLETE</span>' in html
    coverage = element_text(html, "tier3-plugin-coverage") or ""
    assert "2 components not staged" in coverage
    completeness = element_text(html, "tier3-plugin-completeness") or ""
    assert "github::org/repo::skills::remote" in completeness


def test_explicit_provenance_wins_over_sidecar(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))
    explicit = provenance(partial=False)

    result = agent_eval_result_from_directory(plugin_dir, run_dir, plugin_provenance=explicit, use_llm_judge=False)

    assert result is not None
    assert result.metadata["agent_eval"]["plugin_provenance"]["partial"] is False
    assert result.passed is True


def test_run_without_sidecar_renders_as_before(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, signals=False)

    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    assert "plugin_provenance" not in result.metadata["agent_eval"]
    assert _read_plugin_provenance(run_dir) == {}


def _is_unusable(loaded: dict, code: str | None = None) -> bool:
    """Whether *loaded* is the fail-closed record for a present but unusable sidecar."""
    return loaded["partial"] is True and bool(loaded["sidecar_error"]) and code in (None, loaded["sidecar_error"])


@posix_only
def test_symlinked_sidecar_fails_closed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    plugin_dir, run_dir = _run(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(provenance(partial=False)), encoding="utf-8")
    (run_dir / "plugin_provenance.json").symlink_to(outside)

    with caplog.at_level(logging.WARNING, logger=tier3_report.__name__):
        assert _is_unusable(_read_plugin_provenance(run_dir), "symlink_or_reparse_point")
        result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    # The linked record is never followed, and the run is never reported complete.
    assert result.metadata["agent_eval"]["plugin_provenance"] == {
        "partial": True,
        "sidecar_error": "symlink_or_reparse_point",
    }
    assert result.passed is False
    assert result.metadata["execution_status"] == "skipped"
    assert "symlink" in caplog.text


@posix_only
def test_hardlinked_sidecar_fails_closed(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(provenance(partial=False)), encoding="utf-8")
    os.link(outside, run_dir / "plugin_provenance.json")

    assert _is_unusable(_read_plugin_provenance(run_dir))


@posix_only
def test_symlinked_run_directory_fails_closed(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=False))
    alias = tmp_path / "alias-run"
    alias.symlink_to(run_dir, target_is_directory=True)

    assert _is_unusable(_read_plugin_provenance(alias))


def test_directory_sidecar_fails_closed(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    (run_dir / "plugin_provenance.json").mkdir()

    assert _is_unusable(_read_plugin_provenance(run_dir))


def test_oversize_sidecar_fails_closed(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    record = provenance(partial=False)
    record["padding"] = "x" * (_MAX_PLUGIN_PROVENANCE_BYTES + 1)
    (run_dir / "plugin_provenance.json").write_text(json.dumps(record), encoding="utf-8")

    assert _read_plugin_provenance(run_dir) == {"partial": True, "sidecar_error": "file_size_limit"}


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"{not json", "invalid_json"),
        (b"", "invalid_json"),
        (b"[1, 2, 3]", "not_a_json_object"),
        (b'"a string"', "not_a_json_object"),
        (b"\xff\xfe\x00 not utf-8", "invalid_text_encoding"),
        # Either a RecursionError or a (non-object) list, depending on the interpreter.
        (b"[" * 5000 + b"]" * 5000, None),
    ],
    ids=["not-json", "empty", "list", "string", "not-utf8", "deep-nesting"],
)
def test_malformed_sidecar_fails_closed(tmp_path: Path, raw: bytes, code: str | None) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    (run_dir / "plugin_provenance.json").write_bytes(raw)

    loaded = _read_plugin_provenance(run_dir)

    assert set(loaded) == {"partial", "sidecar_error"}
    assert _is_unusable(loaded, code)


@pytest.mark.parametrize("damage", ["oversize", "truncated"])
def test_unreadable_sidecar_keeps_the_rerendered_run_incomplete(tmp_path: Path, damage: str) -> None:
    """A partial run whose sidecar was cut off or grew past the bound never re-renders as PASS."""
    plugin_dir, run_dir = _run(tmp_path)
    record = provenance(partial=True)
    if damage == "oversize":
        raw = json.dumps({**record, "padding": "x" * (_MAX_PLUGIN_PROVENANCE_BYTES + 1)})
    else:
        raw = json.dumps(record)[:-20]
    (run_dir / "plugin_provenance.json").write_text(raw, encoding="utf-8")

    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    assert result.passed is False
    assert result.metadata["execution_status"] == "skipped"
    assert result.metadata["skip_reason"].startswith("INCOMPLETE: plugin provenance sidecar unreadable (")
    assert result.metadata["agent_eval"]["conclusions"][0]["title"] == (
        "Evaluation INCOMPLETE - plugin provenance unreadable"
    )
    html = render_agent_eval_html_report(plugin_dir, run_dir, use_llm_judge=False).read_text(encoding="utf-8")
    assert '<span class="tier-card-verdict">INCOMPLETE</span>' in html
    callout = element_text(html, "tier3-plugin-incomplete") or ""
    assert "INCOMPLETE" in callout
    assert "plugin provenance sidecar could not be read" in callout
    assert "skills resolved" not in callout
    completeness = element_text(html, "tier3-plugin-completeness") or ""
    assert "INCOMPLETE" in completeness
    assert "Resolved" not in completeness


def test_unreadable_sidecar_reason_reaches_every_report_view() -> None:
    from skillevaluator.evaluation.tier3_report import _incomplete_skip_reason
    from skillevaluator.reporting.plugin_sections import tier3_plugin_view

    record = {"partial": True, "sidecar_error": "file_size_limit"}
    view = tier3_plugin_view({"plugin_provenance": record})

    assert view is not None
    assert view["partial"] is True
    assert view["incomplete_reason"] == (
        "plugin provenance sidecar unreadable (file_size_limit), so the components evaluated at Tier 3 are unknown"
    )
    assert view["excluded"] == [
        "Plugin provenance sidecar unreadable (file_size_limit), so the components evaluated at Tier 3 are unknown"
    ]
    assert _incomplete_skip_reason(record) == f"INCOMPLETE: {view['incomplete_reason']}"


def test_recorded_sidecar_error_stays_partial(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path, sidecar={"partial": False, "sidecar_error": "invalid_json"})

    assert _read_plugin_provenance(run_dir)["partial"] is True


def test_mistyped_sidecar_fields_fail_closed(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    (run_dir / "plugin_provenance.json").write_text(
        json.dumps(
            {
                "plugin_name": ["not", "a", "string"],
                "evaluated_member_skills": ["loader", 7],
                "unresolved_skill_refs": "github::org/repo::skills::remote",
                "dataset_case_count": -3,
                "partial": "no",
                "component_coverage": ["not", "a", "mapping"],
            }
        ),
        encoding="utf-8",
    )

    loaded = _read_plugin_provenance(run_dir)

    assert loaded["partial"] is True
    assert loaded["evaluated_member_skills"] == ["loader"]
    assert "plugin_name" not in loaded
    assert "unresolved_skill_refs" not in loaded
    assert "dataset_case_count" not in loaded
    assert "component_coverage" not in loaded


def test_sidecar_partial_flag_cannot_hide_deferred_components(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    record = provenance(partial=True)
    record["partial"] = False
    (run_dir / "plugin_provenance.json").write_text(json.dumps(record), encoding="utf-8")

    assert _read_plugin_provenance(run_dir)["partial"] is True


def test_refresh_replaces_the_runner_report_with_provenance(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))
    (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")

    refreshed = refresh_plugin_run_report(plugin_dir, run_dir, use_llm_judge=False)

    assert refreshed == run_dir.resolve() / "report.html"
    html = refreshed.read_text(encoding="utf-8")
    assert element_text(html, "tier3-plugin-incomplete") is not None
    assert element_text(html, "tier3-plugin-coverage") is not None


def test_refresh_prefers_the_callers_provenance_over_a_missing_sidecar(tmp_path: Path) -> None:
    """A sidecar write that failed cannot drop the INCOMPLETE status from the delivered report."""
    plugin_dir, run_dir = _run(tmp_path)

    refreshed = refresh_plugin_run_report(
        plugin_dir, run_dir, plugin_provenance=provenance(partial=True), use_llm_judge=False
    )

    assert refreshed is not None
    assert element_text(refreshed.read_text(encoding="utf-8"), "tier3-plugin-incomplete") is not None


def test_refresh_is_best_effort(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "missing-run"

    with caplog.at_level(logging.WARNING, logger=tier3_report.__name__):
        assert refresh_plugin_run_report(tmp_path, missing, use_llm_judge=False) is None


def _prepared_package(tmp_path: Path, record: dict) -> SimpleNamespace:
    package_path = tmp_path / "package"
    package_path.mkdir(exist_ok=True)
    return SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(),
        unresolved_skill_refs=tuple(record["unresolved_skill_refs"]),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=tuple(record["provider_only_mcp_servers"]),
        integration_evidence_error=lambda: None,
        provenance=lambda: dict(record),
    )


def test_evaluate_plugin_delivers_report_with_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir()
    run_dir = tmp_path / "results" / "demo-plugin" / "20260101_000000"
    record = provenance(partial=True)

    def evaluate(_self, _options, **_kwargs) -> dict:
        # The runner writes its report before the CLI persists the sidecar.
        write_run_dir(run_dir)
        (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")
        return {"run_dir": str(run_dir), "execution_status": "succeeded"}

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package",
        lambda *_a, **_k: _prepared_package(tmp_path, record),
    )
    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)

    outcome = CliRunner().invoke(
        cli_module.cli,
        ["tier3", "evaluate-plugin", str(plugin_dir), "--lift-mode", "effectiveness", "--progress", "off"],
    )

    # A partial run is still reported as INCOMPLETE on the command line ...
    assert outcome.exit_code != 0
    assert "INCOMPLETE" in outcome.output
    # ... and the delivered report now carries the provenance, coverage and status.
    assert json.loads((run_dir / "plugin_provenance.json").read_text(encoding="utf-8"))["partial"] is True
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "runner report without provenance" not in html
    assert element_text(html, "tier3-plugin-incomplete") is not None
    assert "2 components not staged" in (element_text(html, "tier3-plugin-coverage") or "")


def test_evaluate_plugin_keeps_incomplete_when_the_sidecar_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """evaluate-plugin re-renders from its in-memory provenance, not only from the sidecar on disk."""
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir()
    run_dir = tmp_path / "results" / "demo-plugin" / "20260101_000000"
    record = provenance(partial=True)

    def evaluate(_self, _options, **_kwargs) -> dict:
        write_run_dir(run_dir)
        (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")
        return {"run_dir": str(run_dir), "execution_status": "succeeded"}

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package",
        lambda *_a, **_k: _prepared_package(tmp_path, record),
    )
    # The best-effort sidecar write fails (for example, a full disk) and leaves no file.
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.write_plugin_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)

    outcome = CliRunner().invoke(
        cli_module.cli,
        ["tier3", "evaluate-plugin", str(plugin_dir), "--lift-mode", "effectiveness", "--progress", "off"],
    )

    assert outcome.exit_code != 0
    assert not (run_dir / "plugin_provenance.json").exists()
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "runner report without provenance" not in html
    assert element_text(html, "tier3-plugin-incomplete") is not None


def test_validate_plugin_path_refreshes_the_run_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir()
    run_dir = tmp_path / "results" / "20260101_000000"
    record = provenance(partial=True)

    def evaluate(_self, _options, **_kwargs) -> dict:
        write_run_dir(run_dir)
        (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")
        return {"run_dir": str(run_dir), "execution_status": "succeeded"}

    def from_run(skill_path, **kwargs):
        return agent_eval_result_from_directory(
            skill_path,
            run_dir,
            plugin_provenance=kwargs.get("plugin_provenance"),
            use_llm_judge=False,
        )

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package",
        lambda *_a, **_k: _prepared_package(tmp_path, record),
    )
    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", from_run)

    result = cli_module._run_agent_eval_or_skip(
        plugin_dir,
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
        lift_mode="effectiveness",
    )

    assert result.metadata["skip_reason"].startswith("INCOMPLETE:")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert element_text(html, "tier3-plugin-incomplete") is not None
