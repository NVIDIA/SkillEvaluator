# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complexity-bounded parsing and scalar validation for untrusted manifests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def _validate_graph(
    value: object,
    *,
    max_nodes: int = MAX_STRUCTURED_NODES,
    max_collection_items: int = MAX_STRUCTURED_COLLECTION_ITEMS,
) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    visits = 0
    while stack:
        current, depth = stack.pop()
        visits += 1
        if visits > max_nodes:
            raise _limit(f"expanded node or edge count exceeds {max_nodes}")
        if depth > MAX_STRUCTURED_DEPTH:
            raise _limit(f"expanded nesting depth exceeds {MAX_STRUCTURED_DEPTH}")

        if isinstance(current, Mapping):
            if len(current) > max_collection_items:
                raise _limit(f"mapping size exceeds {max_collection_items}")
            for key, item in current.items():
                stack.append((item, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if len(current) > max_collection_items:
                raise _limit(f"sequence size exceeds {max_collection_items}")
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, (str, bytes, bytearray)) and len(current) > MAX_STRUCTURED_SCALAR_CHARS:
            raise _limit(f"scalar length exceeds {MAX_STRUCTURED_SCALAR_CHARS}")


def load_bounded_yaml(raw: str, *, last_key_wins: bool = False) -> Any:
    """Parse one YAML document after bounded event and graph validation.

    A duplicate mapping key is a syntax error, unless ``last_key_wins`` keeps
    the last value, as agent clients do when they read Markdown frontmatter.
    """
    _preflight_yaml(raw)
    try:
        value = yaml.load(raw, Loader=yaml.SafeLoader if last_key_wins else _UniqueKeySafeLoader)
    except (RecursionError, OverflowError) as exc:
        raise _limit("constructor recursion or numeric range") from exc
    except (yaml.YAMLError, ValueError) as exc:
        raise StructuredDataSyntaxError("Input is not valid YAML") from exc
    _validate_graph(value)
    return value


def _reject_json_constant(value: str) -> object:
    raise StructuredDataSyntaxError(f"Input is not strict JSON ({value})")


# One preflight step: a whole string (an unterminated one runs to the end of
# the input) or one bracket. Everything between two steps is numbers,
# literals, colons, commas, and whitespace, handled as one run. The string
# repeats are possessive: nothing after them can fail, so they never need to
# backtrack, and a greedy repeat would keep backtracking state for every
# escape (about 80 bytes per input byte, far more than the input).
_JSON_STEP = re.compile(r'"(?P<body>[^"\\]*+(?:\\.[^"\\]*+)*+)(?P<closed>")?|[\[\]{}]', re.DOTALL)
# An escaped UTF-16 surrogate pair decodes to one character, like any other escape.
_JSON_ESCAPE = re.compile(
    r"\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\.", re.DOTALL
)
# The most input characters one decoded character takes: an escaped surrogate pair.
_MAX_INPUT_CHARS_PER_CHARACTER = 12
_JSON_ITEM_CHARACTER = re.compile(r"[^\s:]")


def _json_string_length(body: str) -> int:
    """Return the decoded length of a JSON string body: one character per escape sequence."""
    if "\\" not in body:
        return len(body)
    unescaped, escapes = _JSON_ESCAPE.subn("", body)
    return len(unescaped) + escapes


@dataclass
class _JsonCollection:
    """Item bookkeeping for one open JSON array or object."""

    item_limit: int
    separators: int = 0
    has_trailing_item: bool = False

    @property
    def items(self) -> int:
        return self.separators + int(self.has_trailing_item)


class _JsonPreflight:
    """Lexical JSON bounds; strings, collections, and item separators are tokens."""

    def __init__(
        self,
        *,
        max_depth: int,
        max_tokens: int,
        max_collection_items: int,
        max_mapping_items: int | None,
        max_string_chars: int,
    ) -> None:
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.max_collection_items = max_collection_items
        self.max_mapping_items = max_collection_items if max_mapping_items is None else max_mapping_items
        self.max_string_chars = max_string_chars
        self.open_collections: list[_JsonCollection] = []
        self.tokens = 0

    def _count_tokens(self, count: int) -> None:
        self.tokens += count
        if self.tokens > self.max_tokens:
            raise _limit(f"JSON token count exceeds {self.max_tokens}")

    def _mark_item(self) -> None:
        if self.open_collections:
            self.open_collections[-1].has_trailing_item = True

    def run(self, raw: str, start: int, end: int) -> None:
        """Account for the numbers, literals, colons, and commas in ``raw[start:end]``."""
        if not self.open_collections or start == end:
            return
        collection = self.open_collections[-1]
        separators = raw.count(",", start, end)
        if separators:
            # Report the limit that the separators reach first, in input order;
            # on the same separator the collection size is reported first.
            to_item_limit = collection.item_limit - collection.separators
            to_token_limit = self.max_tokens - self.tokens + 1
            if to_item_limit <= separators and to_item_limit <= to_token_limit:
                raise _limit(f"JSON collection size exceeds {collection.item_limit}")
            self._count_tokens(separators)
            collection.separators += separators
            collection.has_trailing_item = False
            start = raw.rindex(",", start, end) + 1
        if _JSON_ITEM_CHARACTER.search(raw, start, end):
            collection.has_trailing_item = True

    def _exceeds_string_limit(self, body: str) -> bool:
        """Return whether *body* decodes to more than ``max_string_chars`` characters.

        A decoded character takes one to twelve input characters, so only a
        body between those bounds is decoded; a longer one is refused without
        holding bookkeeping for each of its escapes.
        """
        if len(body) <= self.max_string_chars:
            return False
        if len(body) > _MAX_INPUT_CHARS_PER_CHARACTER * self.max_string_chars:
            return True
        return _json_string_length(body) > self.max_string_chars

    def string(self, body: str, *, closed: bool) -> None:
        self._mark_item()
        if self._exceeds_string_limit(body):
            raise _limit(f"JSON string length exceeds {self.max_string_chars}")
        if closed:
            self._count_tokens(1)

    def open(self, bracket: str) -> None:
        self._mark_item()
        limit = self.max_mapping_items if bracket == "{" else self.max_collection_items
        self.open_collections.append(_JsonCollection(limit))
        if len(self.open_collections) > self.max_depth:
            raise _limit(f"JSON nesting depth exceeds {self.max_depth}")
        self._count_tokens(1)

    def close(self) -> None:
        if not self.open_collections:
            return
        collection = self.open_collections.pop()
        if collection.items > collection.item_limit:
            raise _limit(f"JSON collection size exceeds {collection.item_limit}")


def preflight_json_structure(
    raw: str,
    *,
    max_depth: int = MAX_STRUCTURED_DEPTH,
    max_tokens: int = MAX_STRUCTURED_NODES,
    max_collection_items: int = MAX_STRUCTURED_COLLECTION_ITEMS,
    max_mapping_items: int | None = None,
    max_string_chars: int = MAX_STRUCTURED_SCALAR_CHARS,
) -> None:
    """Lexically bound JSON before ``json.loads`` materializes nested pairs.

    Tokens are strings, opened arrays and objects, and item separators. A
    string's length counts each escape sequence as the one character it
    decodes to. ``max_mapping_items`` bounds objects (``max_collection_items``
    when unset). The scan takes one step per string, bracket, or run of other
    input, never one step per character.
    """
    preflight = _JsonPreflight(
        max_depth=max_depth,
        max_tokens=max_tokens,
        max_collection_items=max_collection_items,
        max_mapping_items=max_mapping_items,
        max_string_chars=max_string_chars,
    )
    position = 0
    for step in _JSON_STEP.finditer(raw):
        preflight.run(raw, position, step.start())
        position = step.end()
        body = step.group("body")
        if body is not None:
            preflight.string(body, closed=step.group("closed") is not None)
        elif step.group() in "[{":
            preflight.open(step.group())
        else:
            preflight.close()
    preflight.run(raw, position, len(raw))


def load_bounded_json(
    raw: str,
    *,
    max_tokens: int = MAX_STRUCTURED_NODES,
    max_collection_items: int = MAX_STRUCTURED_COLLECTION_ITEMS,
    max_nodes: int = MAX_STRUCTURED_NODES,
) -> Any:
    """Parse strict JSON and validate its expanded object graph iteratively.

    The lexical preflight (:func:`preflight_json_structure`) allows
    ``max_tokens`` tokens; the parsed graph allows ``max_nodes`` values and
    keys; both allow ``max_collection_items`` items in one array or object. A
    caller with larger documents (npm lockfiles) raises them. A syntax error
    names the parser's reason and position, or the rejected constant (``NaN``).
    """
    preflight_json_structure(raw, max_tokens=max_tokens, max_collection_items=max_collection_items)
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except StructuredDataSyntaxError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise _limit("JSON parser recursion or numeric range") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredDataSyntaxError(f"Input is not valid JSON: {exc}") from exc
    _validate_graph(value, max_nodes=max_nodes, max_collection_items=max_collection_items)
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
