# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate explicit ownership contacts without granting them an exemption."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, ScalarNode

from skillevaluator.constants import SKILL_MANIFEST_VARIANTS
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_yaml
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

_EMAIL = r"[^<>@\s]+@[^<>@\s]+\.[A-Za-z]{2,}"
_CONTACT = re.compile(rf"(?P<email>{_EMAIL})|[^<>\n]+ <(?P<named_email>{_EMAIL})>", re.ASCII)
_FIELDS = frozenset({"pic", "email"})


def ownership_email_spans(file_path: Path, content: str) -> dict[int, list[tuple[int, int]]]:
    """Return source spans for single-line metadata.pic/email contact values.

    Only literal, unambiguous YAML scalars qualify. Invalid or oversized YAML,
    aliases, merged keys, block scalars and prose retain ordinary email policy.
    Spans, rather than whole lines or values, keep adjacent fields independent.
    """
    if file_path.name not in SKILL_MANIFEST_VARIANTS:
        return {}
    match = FRONTMATTER_PATTERN.match(content)
    if match is None or len(match[1]) > 65_536:
        return {}
    raw = match[1]
    try:
        load_bounded_yaml(raw)
        if any(isinstance(event, AliasEvent) or getattr(event, "anchor", None) for event in yaml.parse(raw)):
            return {}
        root = yaml.compose(raw, Loader=yaml.SafeLoader)
    except (StructuredDataError, yaml.YAMLError, RecursionError):
        return {}
    if not isinstance(root, MappingNode):
        return {}
    spans: dict[int, list[tuple[int, int]]] = {}
    for key, metadata in root.value:
        if not isinstance(key, ScalarNode) or key.value != "metadata" or not isinstance(metadata, MappingNode):
            continue
        for field, value in metadata.value:
            if (
                not isinstance(field, ScalarNode)
                or field.value not in _FIELDS
                or not isinstance(value, ScalarNode)
                or value.tag != "tag:yaml.org,2002:str"
                or value.style in {"|", ">"}
                or value.start_mark.line != value.end_mark.line
                or not _CONTACT.fullmatch(value.value)
            ):
                continue
            # YAML source begins on file line 2. Escaped/multiline contacts
            # remain ordinary PII unless the literal email is present here.
            contact = _CONTACT.fullmatch(value.value)
            assert contact is not None
            source = raw[value.start_mark.index : value.end_mark.index]
            email = contact["email"] or contact["named_email"]
            offset = source.find(email)
            if offset >= 0:
                start = value.start_mark.column + offset
                spans.setdefault(value.start_mark.line + 2, []).append((start, start + len(email)))
    return spans
