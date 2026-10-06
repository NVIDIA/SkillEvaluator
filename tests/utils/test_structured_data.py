# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tracemalloc
from collections.abc import Callable

import pytest

from skillevaluator.utils.structured_data import (
    StructuredDataLimitError,
    StructuredDataSyntaxError,
    load_bounded_json,
    load_bounded_yaml,
    preflight_json_structure,
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


@pytest.mark.parametrize(
    "escape",
    [r"\u0041", r"\n", r"\\", r"\"", r"\ud83d\ude00"],
    ids=["unicode", "newline", "backslash", "quote", "surrogate-pair"],
)
def test_json_preflight_counts_each_escape_as_one_character(escape: str) -> None:
    preflight_json_structure('["' + escape * 4 + '"]', max_string_chars=4)

    with pytest.raises(StructuredDataLimitError, match=r"string length exceeds 4"):
        preflight_json_structure('["' + escape * 5 + '"]', max_string_chars=4)


def test_json_preflight_does_not_treat_an_escaped_backslash_as_an_escape_prefix() -> None:
    # ``\\u0041`` decodes to six characters: a backslash and "u0041".
    preflight_json_structure(r'["\\u0041"]', max_string_chars=6)

    with pytest.raises(StructuredDataLimitError, match=r"string length exceeds 5"):
        preflight_json_structure(r'["\\u0041"]', max_string_chars=5)


def test_json_preflight_bounds_an_unterminated_string() -> None:
    with pytest.raises(StructuredDataLimitError, match=r"string length exceeds 3"):
        preflight_json_structure('["abcd', max_string_chars=3)


def _traced_peak_bytes(callback: Callable[[], object]) -> int:
    """Return the most memory *callback* held at once, as tracemalloc sees it."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before, _peak = tracemalloc.get_traced_memory()
        callback()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return peak - before


def test_json_preflight_scans_a_string_of_escapes_in_memory_linear_in_its_size() -> None:
    # A 256 KiB string value made only of escapes. A backtracking repeat kept regex
    # state for every escape (about 80 bytes per input byte, so an 8 MiB lockfile
    # peaked at 655 MiB); the scan may hold only a few copies of the input.
    raw = '{"a": "' + '\\"' * 131_072 + '"}'

    def scan() -> None:
        with pytest.raises(StructuredDataLimitError, match=r"string length exceeds 65536"):
            preflight_json_structure(raw)

    assert _traced_peak_bytes(scan) < 16 * len(raw)


def test_json_preflight_refuses_a_string_far_past_the_limit_without_decoding_its_escapes() -> None:
    # One decoded character takes at most twelve input characters (an escaped
    # surrogate pair), so a 1 MiB body cannot fit 65,536 characters; decoding it
    # would hold bookkeeping for each of its 524,288 escapes.
    raw = '{"a": "' + '\\"' * 524_288 + '"}'

    def scan() -> None:
        with pytest.raises(StructuredDataLimitError, match=r"string length exceeds 65536"):
            preflight_json_structure(raw)

    assert _traced_peak_bytes(scan) < 2 * len(raw)


@pytest.mark.parametrize(
    ("body", "max_string_chars"),
    [
        # Twelve input characters decode to one: the bound is exact.
        ("\\ud83d\\ude00" * 4, 4),
        ("\\u0041" * 4, 4),
        ("a" * 4, 4),
    ],
)
def test_json_preflight_decodes_a_body_within_twelve_characters_per_allowed_character(
    body: str, max_string_chars: int
) -> None:
    preflight_json_structure(f'["{body}"]', max_string_chars=max_string_chars)

    with pytest.raises(StructuredDataLimitError, match=rf"string length exceeds {max_string_chars - 1}"):
        preflight_json_structure(f'["{body}"]', max_string_chars=max_string_chars - 1)


@pytest.mark.parametrize(
    ("raw", "max_string_chars", "max_tokens"),
    [
        ('["a\\"b"]', 3, 2),
        ('["a\\\\", "b"]', 2, 4),
        ('["\\"\\"\\""]', 3, 2),
        ('["x", "y\\\\", {"k": "v\\u00e9"}]', 2, 8),
    ],
)
def test_json_preflight_reads_strings_with_escapes_as_whole_tokens(
    raw: str, max_string_chars: int, max_tokens: int
) -> None:
    preflight_json_structure(raw, max_string_chars=max_string_chars, max_tokens=max_tokens)

    with pytest.raises(StructuredDataLimitError, match=r"string length exceeds"):
        preflight_json_structure(raw, max_string_chars=max_string_chars - 1, max_tokens=max_tokens)
    with pytest.raises(StructuredDataLimitError, match=r"token count exceeds"):
        preflight_json_structure(raw, max_string_chars=max_string_chars, max_tokens=max_tokens - 1)


def test_json_preflight_leaves_a_trailing_lone_backslash_outside_an_unterminated_string() -> None:
    # The string body stops before a backslash that has no character to escape.
    preflight_json_structure('["ab\\', max_string_chars=2)

    with pytest.raises(StructuredDataLimitError, match=r"string length exceeds 1"):
        preflight_json_structure('["ab\\', max_string_chars=1)


@pytest.mark.parametrize(
    ("raw", "limits", "message"),
    [
        ("[1, 2, 3]", {"max_collection_items": 2}, r"collection size exceeds 2"),
        ("[[1, 2, 3]]", {"max_collection_items": 2}, r"collection size exceeds 2"),
        ("[1, 2, ]", {"max_collection_items": 2}, r"collection size exceeds 2"),
        ('{"a": 1, "b": 2}', {"max_mapping_items": 1}, r"collection size exceeds 1"),
        ("[[[0]]]", {"max_depth": 2}, r"nesting depth exceeds 2"),
        ('["a", "b"]', {"max_tokens": 3}, r"token count exceeds 3"),
        ("[1, 2, 3]", {"max_tokens": 2}, r"token count exceeds 2"),
        # On the same separator the collection size is reported before the token count.
        ("[1, 2, 3]", {"max_tokens": 2, "max_collection_items": 2}, r"collection size exceeds 2"),
    ],
)
def test_json_preflight_enforces_each_limit(raw: str, limits: dict[str, int], message: str) -> None:
    with pytest.raises(StructuredDataLimitError, match=message):
        preflight_json_structure(raw, **limits)


@pytest.mark.parametrize(
    ("raw", "limits"),
    [
        ("[1, 2]", {"max_collection_items": 2}),
        ("[1, 2 ]", {"max_collection_items": 2}),
        ('{"a": [1, 2, 3]}', {"max_mapping_items": 1, "max_collection_items": 3}),
        ("[[0]]", {"max_depth": 2}),
        ('["a", "b"]', {"max_tokens": 4}),
        # Separators outside any array or object are not tokens.
        ("1, 2, 3", {"max_tokens": 0}),
        ('"[{,}]"', {"max_tokens": 1, "max_depth": 1}),
    ],
)
def test_json_preflight_accepts_input_at_each_limit(raw: str, limits: dict[str, int]) -> None:
    preflight_json_structure(raw, **limits)


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


def test_bounded_json_syntax_errors_name_the_parser_reason_and_the_constant() -> None:
    """The shared loader words syntax errors as the npm lockfile loader it replaced did."""
    with pytest.raises(StructuredDataSyntaxError, match=r"^Input is not valid JSON: Expecting value: line 1 column 1"):
        load_bounded_json("name: yaml-only")
    with pytest.raises(StructuredDataSyntaxError, match=r"^Input is not strict JSON \(NaN\)$"):
        load_bounded_json("[NaN]")


def test_bounded_json_limit_overrides_apply_to_the_preflight_and_the_parsed_graph() -> None:
    wide = "[" + ",".join(["1"] * 2_000) + "]"
    with pytest.raises(StructuredDataLimitError, match=r"collection size exceeds 1024"):
        load_bounded_json(wide)
    assert len(load_bounded_json(wide, max_collection_items=2_000)) == 2_000

    with pytest.raises(StructuredDataLimitError, match=r"token count exceeds 1000"):
        load_bounded_json(wide, max_tokens=1_000, max_collection_items=2_000)

    # 600 one-item arrays: 1,200 preflight tokens (opening brackets and separators), 1,201 parsed values.
    nested = "[" + ",".join(["[1]"] * 600) + "]"
    assert len(load_bounded_json(nested, max_nodes=1_201)) == 600
    with pytest.raises(StructuredDataLimitError, match=r"node or edge count exceeds 1200"):
        load_bounded_json(nested, max_nodes=1_200)
