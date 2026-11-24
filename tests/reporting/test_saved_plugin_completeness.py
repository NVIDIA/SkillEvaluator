# SPDX-License-Identifier: Apache-2.0
"""Saved reports retain preparation limitations without live engine state."""

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting.plugin_reports import attach_saved_plugin_evidence
from skillevaluator.reporting.plugin_runtime import project_plugin_runtime
from skillevaluator.reporting.plugin_runtime_io import save_plugin_runtime_report


@pytest.mark.parametrize("engine_result", [None, {}, {"plugin_provenance": {"complete": True}}])
@pytest.mark.parametrize("prepared_complete", [True, False])
def test_saved_preparation_evidence_preserves_incomplete_mark(tmp_path, engine_result, prepared_complete):
    report = project_plugin_runtime(
        {
            "run_dir": str(tmp_path),
            "execution_status": "completed",
            "runtime_evidence": {"evidence_complete": False},
        },
        provenance={"plugin": "sample", "complete": prepared_complete},
    )
    save_plugin_runtime_report(report, tmp_path)
    result = ValidationResult(validator_name="AGENT_EVAL", metadata={"agent_eval": {"quality": 0.7}})

    attach_saved_plugin_evidence(result, tmp_path, engine_result)

    assert ("plugin-components" in result.incomplete_scans) is not prepared_complete
    assert result.metadata["plugin_runtime_report"] == report.payload()
    assert result.metadata["agent_eval"] == {"quality": 0.7}


def test_live_preparation_snapshot_preserves_incomplete_mark_when_saved_file_absent(tmp_path):
    report = project_plugin_runtime(
        {"run_dir": str(tmp_path), "runtime_evidence": {"evidence_complete": True}},
        provenance={"plugin": "sample", "complete": False},
    )
    result = ValidationResult(validator_name="AGENT_EVAL")

    attach_saved_plugin_evidence(result, tmp_path, {"plugin_runtime_report": report.payload()})

    assert "plugin-components" in result.incomplete_scans
