# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from skillevaluator.utils.structured_data import (
    StructuredDataLimitError,
    StructuredDataSyntaxError,
    load_bounded_json,
    load_bounded_yaml,
    require_bounded_string,
)


def _alias_dag(levels: int) -> str:
    lines = ["seed: &a0 [safe, safe]"]
    lines.extend(f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, levels + 1))
    lines.append(f"value: *a{levels}")
    return "\n".join(lines)


def test_bounded_yaml_rejects_deep_nesting_without_recursion_error() -> None:
    raw = "value: " + ("[" * 1_500) + "safe" + ("]" * 1_500)

    with pytest.raises(StructuredDataLimitError, match=r"depth|complex|limit"):
        load_bounded_yaml(raw)


def test_bounded_yaml_counts_alias_graph_occurrences() -> None:
    with pytest.raises(StructuredDataLimitError, match=r"node|edge|alias|complex|limit"):
        load_bounded_yaml(_alias_dag(20))


def test_bounded_yaml_rejects_duplicate_mapping_keys() -> None:
    with pytest.raises(StructuredDataSyntaxError, match=r"YAML|valid|syntax"):
        load_bounded_yaml("name: first\nname: second\n")


def test_bounded_json_rejects_deep_nesting_and_non_json_syntax() -> None:
    raw = '{"value":' + ("[" * 1_500) + "0" + ("]" * 1_500) + "}"
    with pytest.raises(StructuredDataLimitError, match=r"depth|complex|limit"):
        load_bounded_json(raw)

    with pytest.raises(StructuredDataSyntaxError, match=r"JSON|syntax|valid"):
        load_bounded_json("name: yaml-only")


@pytest.mark.parametrize("value", [["unsafe"], {"unsafe": True}, 3, True])
def test_require_bounded_string_never_stringifies_containers(value: object) -> None:
    with pytest.raises(ValueError, match=r"field.*string"):
        require_bounded_string(value, "field", max_chars=32)


def test_require_bounded_string_enforces_length_and_nonempty() -> None:
    with pytest.raises(ValueError, match=r"field.*limit|too long|32"):
        require_bounded_string("x" * 33, "field", max_chars=32)
    with pytest.raises(ValueError, match=r"field.*non-empty"):
        require_bounded_string("   ", "field", max_chars=32)
    assert require_bounded_string(" safe ", "field", max_chars=32) == " safe "
