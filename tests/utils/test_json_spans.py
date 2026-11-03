"""Tests for the shared bounded JSON span helpers."""

from __future__ import annotations

import json

import pytest

from skillevaluator.inference import client as client_mod
from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.utils import json_spans


def test_tier3_judge_and_inference_client_share_the_helpers() -> None:
    assert llm_judge._balanced_json_container_end is json_spans.balanced_json_container_end
    assert llm_judge._json_nesting_within_limit is json_spans.json_nesting_within_limit
    assert llm_judge._reject_duplicate_object_pairs is json_spans.reject_duplicate_object_pairs
    assert llm_judge._reject_nonstandard_json_constant is json_spans.reject_nonstandard_json_constant
    assert llm_judge._parse_finite_json_float is json_spans.parse_finite_json_float
    assert llm_judge._MAX_JSON_NESTING == json_spans.MAX_JSON_NESTING
    assert client_mod.balanced_json_container_end is json_spans.balanced_json_container_end
    assert client_mod.loads_strict_json is json_spans.loads_strict_json


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        pytest.param('x {"a": "}"} y', 2, 12, id="brace-inside-string"),
        pytest.param('[1, {"a": [2]}] tail', 0, 15, id="mixed-containers"),
        pytest.param('{"a": "\\"}"} tail', 0, 12, id="escaped-quote"),
    ],
)
def test_balanced_container_end(text: str, start: int, end: int) -> None:
    assert json_spans.balanced_json_container_end(text, start) == end


@pytest.mark.parametrize(
    ("text", "start"),
    [
        pytest.param('{"a": 1', 0, id="never-closes"),
        pytest.param('{"a": [1}', 0, id="mismatched-bracket"),
        pytest.param("plain", 0, id="not-a-container"),
        pytest.param("{}", 2, id="start-past-end"),
        pytest.param(
            "[" * (json_spans.MAX_JSON_NESTING + 1) + "]" * (json_spans.MAX_JSON_NESTING + 1), 0, id="too-deep"
        ),
    ],
)
def test_balanced_container_end_rejects(text: str, start: int) -> None:
    assert json_spans.balanced_json_container_end(text, start) is None


def test_nesting_limit_is_inclusive() -> None:
    limit = json_spans.MAX_JSON_NESTING

    assert json_spans.json_nesting_within_limit("[" * limit + "]" * limit)
    assert not json_spans.json_nesting_within_limit("[" * (limit + 1) + "]" * (limit + 1))
    assert json_spans.json_nesting_within_limit('"' + "[" * (limit + 1) + '"')


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        pytest.param('{"k": 1, "k": 2}', "Duplicate JSON object member", id="duplicate-key"),
        pytest.param("[NaN]", "Non-standard JSON constant", id="nan"),
        pytest.param("Infinity", "Non-standard JSON constant", id="infinity"),
        pytest.param("1e400", "non-finite", id="float-overflow"),
    ],
)
def test_loads_strict_json_rejects_ambiguous_values(text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        json_spans.loads_strict_json(text)


def test_loads_strict_json_matches_json_loads_for_ordinary_input() -> None:
    text = '{"verdict": "DUPLICATE", "confidence": 0.9, "tags": [1, -2.5e3, null, true]}'

    assert json_spans.loads_strict_json(text) == json.loads(text)
