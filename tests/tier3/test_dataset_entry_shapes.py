# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed authored cases must not disappear from the evaluated dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from skillevaluator.tier3.dataset_utils import load_dataset_entries, normalize_dataset_entries
from skillevaluator.tier3.evals_spec import _validate_dataset_contents


@pytest.mark.parametrize("invalid", [None, False, 7, "private authored text", []])
@pytest.mark.parametrize("wrapped", [False, True])
def test_dataset_rejects_nonobject_cases_instead_of_dropping_them(invalid: object, wrapped: bool) -> None:
    cases = [{"id": "first", "question": "Do the task."}, invalid, {"id": "last", "question": "Continue."}]
    payload = {"cases": cases} if wrapped else cases

    with pytest.raises(ValueError, match=r"entry 1 must be an object") as caught:
        normalize_dataset_entries(payload)

    assert "private authored text" not in str(caught.value)


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".yaml", ".yml"])
def test_loaded_dataset_and_source_validation_reject_a_nonobject_case(tmp_path: Path, suffix: str) -> None:
    cases = [{"id": "first", "question": "Do the task."}, None]
    path = tmp_path / f"evals{suffix}"
    if suffix == ".jsonl":
        content = "\n".join(json.dumps(case) for case in cases)
    elif suffix in {".yaml", ".yml"}:
        content = yaml.safe_dump(cases)
    else:
        content = json.dumps(cases)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"entry 1 must be an object"):
        load_dataset_entries(path)

    checks = []
    _validate_dataset_contents(path, checks, skill_path=tmp_path)
    assert len(checks) == 1
    assert checks[0].status == "error"
    assert "entry 1 must be an object" in checks[0].message


@pytest.mark.parametrize("wrapped", [False, True])
def test_valid_legacy_dataset_preserves_case_order_and_input(wrapped: bool) -> None:
    cases = [{"id": "second", "prompt": "Second."}, {"id": "first", "prompt": "First."}]
    payload = {"cases": cases} if wrapped else cases

    result = normalize_dataset_entries(payload)

    assert [case["id"] for case in result] == ["second", "first"]
    assert [case["question"] for case in result] == ["Second.", "First."]
    assert all("question" not in case for case in cases)
