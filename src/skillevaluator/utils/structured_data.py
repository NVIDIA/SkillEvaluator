# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complexity-bounded parsing and scalar validation for untrusted manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    ScalarEvent,
)
from yaml.nodes import MappingNode

MAX_STRUCTURED_DEPTH = 100
MAX_STRUCTURED_NODES = 20_000
MAX_STRUCTURED_COLLECTION_ITEMS = 1_024
MAX_YAML_ALIAS_REFERENCES = 1_024
MAX_STRUCTURED_SCALAR_CHARS = 65_536


class StructuredDataError(ValueError):
    """Base class for normalized structured-data parse failures."""


class StructuredDataSyntaxError(StructuredDataError):
    """The input does not conform to the requested serialization syntax."""


class StructuredDataLimitError(StructuredDataError):
    """The input exceeds a parser or object-graph complexity ceiling."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects ambiguous last-key-wins mappings."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, "expected a mapping node", node.start_mark)
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark, "unhashable key", key_node.start_mark
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "duplicate mapping key",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _limit(message: str) -> StructuredDataLimitError:
    return StructuredDataLimitError(f"Structured data complexity limit exceeded: {message}")


def _preflight_yaml(raw: str) -> None:
    depth = 0
    nodes = 0
    aliases = 0
    try:
        for event in yaml.parse(raw, Loader=_UniqueKeySafeLoader):
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > MAX_STRUCTURED_DEPTH:
                    raise _limit(f"nesting depth exceeds {MAX_STRUCTURED_DEPTH}")
                nodes += 1
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
            elif isinstance(event, ScalarEvent):
                nodes += 1
                if len(event.value) > MAX_STRUCTURED_SCALAR_CHARS:
                    raise _limit(f"scalar length exceeds {MAX_STRUCTURED_SCALAR_CHARS}")
            elif isinstance(event, AliasEvent):
                nodes += 1
                aliases += 1
                if aliases > MAX_YAML_ALIAS_REFERENCES:
                    raise _limit(f"alias reference count exceeds {MAX_YAML_ALIAS_REFERENCES}")
            if nodes > MAX_STRUCTURED_NODES:
                raise _limit(f"parsed node count exceeds {MAX_STRUCTURED_NODES}")
    except StructuredDataLimitError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise _limit("parser recursion or numeric range") from exc
    except yaml.YAMLError as exc:
        raise StructuredDataSyntaxError("Input is not valid YAML") from exc


def _validate_graph(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    visits = 0
    while stack:
        current, depth = stack.pop()
        visits += 1
        if visits > MAX_STRUCTURED_NODES:
            raise _limit(f"expanded node or edge count exceeds {MAX_STRUCTURED_NODES}")
        if depth > MAX_STRUCTURED_DEPTH:
            raise _limit(f"expanded nesting depth exceeds {MAX_STRUCTURED_DEPTH}")

        if isinstance(current, Mapping):
            if len(current) > MAX_STRUCTURED_COLLECTION_ITEMS:
                raise _limit(f"mapping size exceeds {MAX_STRUCTURED_COLLECTION_ITEMS}")
            for key, item in current.items():
                stack.append((item, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if len(current) > MAX_STRUCTURED_COLLECTION_ITEMS:
                raise _limit(f"sequence size exceeds {MAX_STRUCTURED_COLLECTION_ITEMS}")
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, (str, bytes, bytearray)) and len(current) > MAX_STRUCTURED_SCALAR_CHARS:
            raise _limit(f"scalar length exceeds {MAX_STRUCTURED_SCALAR_CHARS}")


def load_bounded_yaml(raw: str) -> Any:
    """Parse one YAML document after bounded event and graph validation."""
    _preflight_yaml(raw)
    try:
        value = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    except (RecursionError, OverflowError) as exc:
        raise _limit("constructor recursion or numeric range") from exc
    except (yaml.YAMLError, ValueError) as exc:
        raise StructuredDataSyntaxError("Input is not valid YAML") from exc
    _validate_graph(value)
    return value


def _reject_json_constant(_value: str) -> object:
    raise StructuredDataSyntaxError("Input is not strict JSON")


def preflight_json_structure(
    raw: str,
    *,
    max_depth: int = MAX_STRUCTURED_DEPTH,
    max_tokens: int = MAX_STRUCTURED_NODES,
    max_collection_items: int = MAX_STRUCTURED_COLLECTION_ITEMS,
    max_mapping_items: int | None = None,
    max_string_chars: int = MAX_STRUCTURED_SCALAR_CHARS,
) -> None:
    """Lexically bound JSON before ``json.loads`` materializes nested pairs."""
    stack: list[dict[str, int | bool | str]] = []
    in_string = False
    escaped = False
    string_chars = 0
    tokens = 0
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                tokens += 1
                if tokens > max_tokens:
                    raise _limit(f"JSON token count exceeds {max_tokens}")
            else:
                string_chars += 1
                if string_chars > max_string_chars:
                    raise _limit(f"JSON string length exceeds {max_string_chars}")
            continue

        if char == '"':
            in_string = True
            escaped = False
            string_chars = 0
            if stack:
                stack[-1]["has_item"] = True
        elif char in "[{":
            if stack:
                stack[-1]["has_item"] = True
            stack.append({"opening": char, "completed": 0, "has_item": False})
            tokens += 1
            if len(stack) > max_depth:
                raise _limit(f"JSON nesting depth exceeds {max_depth}")
        elif char in "]}":
            if stack:
                state = stack.pop()
                item_count = int(state["completed"]) + (1 if state["has_item"] else 0)
                item_limit = (
                    max_mapping_items
                    if state["opening"] == "{" and max_mapping_items is not None
                    else max_collection_items
                )
                if item_count > item_limit:
                    raise _limit(f"JSON collection size exceeds {item_limit}")
        elif char == "," and stack:
            state = stack[-1]
            state["completed"] = int(state["completed"]) + 1
            state["has_item"] = False
            tokens += 1
            item_limit = (
                max_mapping_items if state["opening"] == "{" and max_mapping_items is not None else max_collection_items
            )
            if int(state["completed"]) >= item_limit:
                raise _limit(f"JSON collection size exceeds {item_limit}")
        elif not char.isspace() and char != ":" and stack:
            stack[-1]["has_item"] = True
        if tokens > max_tokens:
            raise _limit(f"JSON token count exceeds {max_tokens}")


def load_bounded_json(raw: str) -> Any:
    """Parse strict JSON and validate its expanded object graph iteratively."""
    preflight_json_structure(raw)
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except StructuredDataSyntaxError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise _limit("JSON parser recursion or numeric range") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredDataSyntaxError("Input is not valid JSON") from exc
    _validate_graph(value)
    return value


def require_bounded_string(
    value: object,
    field: str,
    *,
    max_chars: int,
    allow_empty: bool = False,
) -> str:
    """Return a real bounded string without coercing attacker-controlled objects."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > max_chars:
        raise ValueError(f"{field} exceeds the {max_chars}-character limit")
    return value
