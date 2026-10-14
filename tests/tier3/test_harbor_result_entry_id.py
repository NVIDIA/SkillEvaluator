# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from skillevaluator.tier3.harbor.collector import _entry_id, _entry_id_from_harbor_result


def test_entry_id_prefers_task_id_path_over_prefixed_task_name() -> None:
    result = {
        "task_name": "nvidia/skillevaluator-apt-pinning",
        "task_id": {"path": "evals/cases/case-014-apt-pinning"},
    }
    assert _entry_id_from_harbor_result(result) == "case-014-apt-pinning"


@pytest.mark.parametrize(
    ("truncated", "canonical"),
    [
        (
            "huggingface-factory-peft-sft-loc",
            "huggingface-factory-peft-sft-local-return",
        ),
        (
            "huggingface-global-negative-serv",
            "huggingface-global-negative-serving",
        ),
    ],
)
def test_canonical_case_id_resolves_unique_truncated_dataset_prefix(
    truncated: str,
    canonical: str,
) -> None:
    assert _entry_id({"entry_id": truncated}, {canonical}) == canonical


def test_canonical_case_id_prefers_exact_match_over_longer_prefix() -> None:
    assert _entry_id({"entry_id": "case-prefix"}, {"case-prefix", "case-prefix-long"}) == "case-prefix"


def test_canonical_case_id_does_not_guess_ambiguous_prefix() -> None:
    assert _entry_id({"entry_id": "case-prefix"}, {"case-prefix-long", "case-prefix-other"}) == "case-prefix"
