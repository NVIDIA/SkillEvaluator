# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skillevaluator.tier3.dataset_utils import load_dataset_entries_with_format
from skillevaluator.tier3.eval_core.optional_reference import (
    normalize_expected_behaviors,
    normalize_optional_reference,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        (" \t\n ", None),
        ("None", None),
        (" none ", None),
        ("NULL", None),
        (" N/A ", None),
        ("n/a", None),
        ("-", None),
        (0, None),
        ([], None),
        ("none.sh", "none.sh"),
        (" None.sh ", "None.sh"),
        ("scripts/main.py", "scripts/main.py"),
    ],
)
def test_normalize_optional_reference(value, expected):
    assert normalize_optional_reference(value) == expected


def test_expected_behavior_normalization_preserves_unicode_whitespace():
    assert normalize_expected_behaviors(["  Reject \u202f  ", " None ", " \t "]) == ["Reject \u202f"]


@pytest.mark.parametrize("dataset_format", ["agentskills", "legacy"])
def test_dataset_loaders_normalize_optional_references(tmp_path, dataset_format):
    entry = {
        "id": "case-1",
        "expected_script": " nOnE ",
        "expected_behavior": ["None", "  ", "n/A", None, "Keep this behavior"],
    }
    if dataset_format == "agentskills":
        entry.update(
            prompt="Use the skill.",
            expected_output="None",
            assertions=entry.pop("expected_behavior"),
        )
        payload = {"skill_name": "demo", "evals": [entry]}
    else:
        entry.update(question="Use the skill.", ground_truth="None")
        payload = [entry]

    dataset_path = tmp_path / "evals.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    (normalized,), actual_format = load_dataset_entries_with_format(dataset_path)

    assert actual_format == dataset_format
    assert normalized["expected_script"] is None
    assert normalized["expected_behavior"] == ["Keep this behavior"]
    assert normalized["ground_truth"] == "None"


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        (["calc.py"], "list"),
        (7, "int"),
        (True, "bool"),
    ],
)
def test_dataset_loader_rejects_truthy_non_string_expected_script_with_entry_index(tmp_path, value, type_name):
    dataset_path = tmp_path / "evals.json"
    dataset_path.write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {"id": "case-0", "prompt": "First", "expected_output": "A"},
                    {"id": "case-1", "prompt": "Second", "expected_output": "B", "expected_script": value},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"invalid eval dataset entry 1: expected_script must be a string or null, got {type_name}",
    ):
        load_dataset_entries_with_format(dataset_path)


@pytest.mark.parametrize("value", [0, []])
def test_dataset_loader_normalizes_falsy_non_string_expected_script(tmp_path, value):
    dataset_path = tmp_path / "evals.json"
    dataset_path.write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [{"id": "case-0", "prompt": "First", "expected_output": "A", "expected_script": value}],
            }
        ),
        encoding="utf-8",
    )

    (normalized,), _ = load_dataset_entries_with_format(dataset_path)

    assert normalized["expected_script"] is None
