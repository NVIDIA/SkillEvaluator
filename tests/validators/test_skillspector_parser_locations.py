# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incomplete SkillSpector scans name parser_diagnostic spans, causes, and checks.

SkillSpector 2.13.4 records ``static_parse_limit`` on some JavaScript helpers
with ``start_line`` empty and the span, cause and check only in
``parser_diagnostic``. The security check stays ``INCOMPLETE``; these tests
cover the labels, the runtime-dependent hint, and the JSON projection.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.validators.test_security import (
    _REGULAR_ANALYZERS,
    _ledger_exception,
    _skillspector_partial_report,
    _validate_skillspector_payload,
)

from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.validators.security import (
    _SKILLSPECTOR_PARSE_LIMIT_ADVICE,
    _SKILLSPECTOR_REFERENCE_REJECTION_HINT,
    _skillspector_incomplete_analysis_error,
    _skillspector_ledger_exception_label,
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "skillspector-2.13.4-mjs-runtime-parse-limit-no-llm.json"
_RUNTIME_DIAGNOSTIC = {
    "cause": "runtime_dependent",
    "check": "command_string_runtime",
    "start_line": 74,
    "end_line": 170,
    "unchecked_scope": "remaining_input",
    "full_input_checked": False,
    "source_start_offset": 3958,
    "source_end_offset": 8650,
    "location_precision": "source_span",
}
_BOUNDARY_DIAGNOSTIC = {
    "cause": "boundary_unresolved",
    "check": "command_word_boundary",
    "start_line": 11,
    "end_line": 11,
    "unchecked_scope": "command_string",
    "full_input_checked": False,
}


def _parse_limit_row(**overrides) -> dict:
    row = {
        **_ledger_exception("scripts/helper.mjs", "static_parse_limit"),
        "parser_diagnostic": dict(_RUNTIME_DIAGNOSTIC),
    }
    row.update(overrides)
    return row


def test_captured_mjs_rows_use_the_parser_span_and_runtime_dependent_context() -> None:
    report = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    rows = report["analysis_completeness"]["ledger_exceptions"]
    labels = [_skillspector_ledger_exception_label(row) for row in rows]
    assert labels == [
        "SKILL.md:150 [reference_resolution; reference_missing]",
        "scripts/admission.mjs:74-170 [static; static_parse_limit; runtime_dependent command_string_runtime; "
        "analyzers=static_patterns_tool_misuse]",
        "scripts/machine-reading.mjs:104-236 [static; static_parse_limit; runtime_dependent command_string_runtime; "
        "analyzers=static_patterns_tool_misuse]",
    ]
    error = _skillspector_incomplete_analysis_error("partial", rows)
    assert error.startswith(
        "skillspector JSON field 'analysis_completeness' reports incomplete analysis (status 'partial'): "
    )
    assert labels[0] in error and labels[1] in error and labels[2] in error
    assert (
        "(for example a process spawn whose arguments come from variables) in "
        "scripts/admission.mjs:74-170 and scripts/machine-reading.mjs:104-236. "
    ) in error
    assert error.endswith(_SKILLSPECTOR_PARSE_LIMIT_ADVICE)
    assert _SKILLSPECTOR_REFERENCE_REJECTION_HINT not in error


def test_parser_span_uses_a_range_or_a_single_line() -> None:
    range_label = _skillspector_ledger_exception_label(_parse_limit_row())
    assert range_label.startswith("scripts/helper.mjs:74-170 [")
    single = _parse_limit_row(parser_diagnostic=dict(_BOUNDARY_DIAGNOSTIC))
    assert _skillspector_ledger_exception_label(single).startswith("scripts/helper.mjs:11 [")
    assert "11-11" not in _skillspector_ledger_exception_label(single)


def test_row_start_line_is_unchanged_when_parser_diagnostic_also_has_a_span() -> None:
    row = _parse_limit_row(start_line=12)
    assert _skillspector_ledger_exception_label(row) == (
        f"scripts/helper.mjs:12 [static; static_parse_limit; runtime_dependent command_string_runtime; "
        f"analyzers={','.join(_REGULAR_ANALYZERS)}]"
    )


@pytest.mark.parametrize(
    "diagnostic",
    [
        "not-a-dict",
        ["runtime_dependent"],
        None,
        {"cause": "runtime_dependent", "check": "command_string_runtime", "start_line": "74", "end_line": "170"},
        {"cause": "runtime_dependent", "check": "command_string_runtime", "start_line": True, "end_line": True},
        {"cause": "runtime_dependent", "check": "command_string_runtime", "start_line": -1, "end_line": 10},
        {"cause": "runtime_dependent", "check": "command_string_runtime", "start_line": 170, "end_line": 74},
        {"cause": "runtime_dependent", "check": "command_string_runtime", "start_line": 74, "end_line": False},
    ],
    ids=[
        "string",
        "list",
        "none",
        "string-lines",
        "bool-lines",
        "negative",
        "end-before-start",
        "bool-end",
    ],
)
def test_malformed_parser_diagnostic_does_not_supply_a_location(diagnostic: object) -> None:
    row = _parse_limit_row(parser_diagnostic=diagnostic)
    label = _skillspector_ledger_exception_label(row)
    assert label.startswith("scripts/helper.mjs [static; static_parse_limit")
    assert ":74" not in label
    assert ":170" not in label


def test_cause_and_check_are_sanitized_and_truncated() -> None:
    row = _parse_limit_row(
        parser_diagnostic={
            "cause": "runtime\ndependent-" + "x" * 400,
            "check": "command\tstring-" + "y" * 400,
            "start_line": 74,
            "end_line": 170,
        }
    )
    label = _skillspector_ledger_exception_label(row)
    assert "\n" not in label and "\t" not in label
    assert "runtime dependent-" + "x" * 179 + "..." in label
    assert "x" * 180 not in label
    assert "command string-" + "y" * 182 + "..." in label
    assert "y" * 183 not in label
    other_reason = _ledger_exception("scripts/helper.mjs", "opaque_content")
    other_reason["parser_diagnostic"] = dict(_RUNTIME_DIAGNOSTIC)
    assert "runtime_dependent" not in _skillspector_ledger_exception_label(other_reason)


def test_runtime_dependent_hint_requires_every_parse_limit_row() -> None:
    runtime = _parse_limit_row()
    other = _parse_limit_row(path="scripts/other.mjs", parser_diagnostic=dict(_BOUNDARY_DIAGNOSTIC))
    missing = _ledger_exception("scripts/plain.mjs", "static_parse_limit")
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE in _skillspector_incomplete_analysis_error("partial", [runtime])
    mixed = _skillspector_incomplete_analysis_error("partial", [runtime, other])
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE not in mixed
    assert "scripts/other.mjs:11 [" in mixed
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE not in _skillspector_incomplete_analysis_error("partial", [missing])
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE not in _skillspector_incomplete_analysis_error(
        "partial", [_ledger_exception("scripts/render.py", "static_parse_limit") | {"start_line": 12}]
    )


def test_parse_limit_hint_combines_with_runtime_limit_and_not_with_reference_rejection() -> None:
    runtime_limit = _ledger_exception("references/sect-1.html")
    parse_limit = _parse_limit_row()
    reference = {
        **_ledger_exception("SKILL.md", "reference_missing", analyzers=(), phase="reference_resolution"),
        "start_line": 150,
    }
    combined = _skillspector_incomplete_analysis_error("partial", [runtime_limit, parse_limit])
    assert "SkillSpector reached a runtime limit before it finished 1 file(s)." in combined
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE in combined
    assert _SKILLSPECTOR_REFERENCE_REJECTION_HINT not in combined
    with_reference = _skillspector_incomplete_analysis_error("partial", [reference, parse_limit])
    assert "SKILL.md:150 [reference_resolution; reference_missing]" in with_reference
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE in with_reference
    assert _SKILLSPECTOR_REFERENCE_REJECTION_HINT not in with_reference
    only_reference = _skillspector_incomplete_analysis_error("partial", [reference])
    assert only_reference.endswith(_SKILLSPECTOR_REFERENCE_REJECTION_HINT)
    assert _SKILLSPECTOR_PARSE_LIMIT_ADVICE not in only_reference


def _reference_row(line: int) -> dict:
    return {
        **_ledger_exception("SKILL.md", "reference_missing", analyzers=(), phase="reference_resolution"),
        "start_line": line,
    }


def test_parse_limit_hint_names_locations_hidden_behind_earlier_rows() -> None:
    rows = [_reference_row(10), _reference_row(20), _reference_row(30), _parse_limit_row()]
    error = _skillspector_incomplete_analysis_error("partial", rows)
    # The shown examples are the three reference rows; the helper's span is in "and 1 more".
    assert "scripts/helper.mjs:74-170 [" not in error
    assert "; and 1 more " in error
    assert (
        "(for example a process spawn whose arguments come from variables) in scripts/helper.mjs:74-170. "
        f"{_SKILLSPECTOR_PARSE_LIMIT_ADVICE}"
    ) in error


def test_parse_limit_hint_caps_named_locations_and_points_to_the_json_report() -> None:
    rows = [
        _parse_limit_row(path=f"scripts/helper{index}.mjs", parser_diagnostic=dict(_RUNTIME_DIAGNOSTIC))
        for index in range(5)
    ]
    rows.append(_parse_limit_row(path="scripts/helper0.mjs"))
    error = _skillspector_incomplete_analysis_error("partial", rows)
    assert (
        "in scripts/helper0.mjs:74-170, scripts/helper1.mjs:74-170 and scripts/helper2.mjs:74-170, "
        "and 2 more listed in parser_diagnostic on the skillspector_completeness rows of the JSON report. "
    ) in error


def test_parse_limit_hint_without_a_span_names_the_path_or_the_json_report() -> None:
    no_span = {"cause": "runtime_dependent", "check": "command_string_runtime"}
    path_only = _skillspector_incomplete_analysis_error("partial", [_parse_limit_row(parser_diagnostic=no_span)])
    assert "come from variables) in scripts/helper.mjs. " in path_only
    nothing = _parse_limit_row(parser_diagnostic=no_span)
    del nothing["path"]
    unknown = _skillspector_incomplete_analysis_error("partial", [nothing])
    assert (
        "come from variables) at the locations in parser_diagnostic on the skillspector_completeness rows "
        "of the JSON report. "
    ) in unknown
    assert unknown.endswith(_SKILLSPECTOR_PARSE_LIMIT_ADVICE)


@patch("skillevaluator.validators.security.Tools")
def test_partial_scan_stays_incomplete_and_projects_parser_diagnostic(mock_tools, sample_skill_dir: Path) -> None:
    payload = _skillspector_partial_report([_parse_limit_row(analyzers=["static_patterns_tool_misuse"])])

    result = _validate_skillspector_payload(mock_tools, sample_skill_dir, payload)

    assert result.status == "incomplete"
    assert not result.passed
    assert result.incomplete_scans == ["skillspector"]
    assert (
        "scripts/helper.mjs:74-170 [static; static_parse_limit; runtime_dependent command_string_runtime; "
        "analyzers=static_patterns_tool_misuse]"
    ) in result.errors[0]
    assert result.errors[0].endswith(_SKILLSPECTOR_PARSE_LIMIT_ADVICE)
    record = json.loads(JSONReporter().render(result))["skillspector_completeness"]["records"][0]
    diagnostic = record["ledger_exceptions"][0]["parser_diagnostic"]
    assert diagnostic == {
        "cause": "runtime_dependent",
        "check": "command_string_runtime",
        "start_line": 74,
        "end_line": 170,
        "unchecked_scope": "remaining_input",
        "full_input_checked": False,
    }
    encoded = json.dumps(record)
    assert "source_start_offset" not in encoded
    assert "source_end_offset" not in encoded
    assert "location_precision" not in encoded
