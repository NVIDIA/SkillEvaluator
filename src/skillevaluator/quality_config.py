# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for deterministic quality scoring."""
from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files
from math import isfinite
from pathlib import Path
from typing import Literal, TypedDict

import yaml

DEFAULT_QUALITY_CONFIG = files("skillevaluator.config").joinpath("quality-default.yaml")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_TOP_LEVEL_KEYS = frozenset({"expected_headers"})
_EXPECTED_HEADER_KEYS = frozenset({"heading", "alternatives", "points", "message", "suggestion"})


class QualityConfigError(ValueError):
    """Raised when a quality configuration is invalid."""


class ExpectedHeader(TypedDict):
    """A Markdown heading expected by the quality validator."""

    heading: str
    alternatives: list[str]
    points: int | float
    message: str
    suggestion: str


class QualityConfig(TypedDict):
    """Validated deterministic quality configuration."""

    expected_headers: list[ExpectedHeader]


class QualityConfigProvenance(TypedDict):
    """Which quality configuration produced a score, as recorded in reports."""

    source: Literal["packaged", "custom"]
    file: str | None
    digest: str


def _normalize_heading(value: object, source: str, location: str) -> str:
    heading_match = _ATX_HEADING_RE.fullmatch(value) if isinstance(value, str) else None
    heading_title = heading_match.group(2).strip() if heading_match else ""
    if heading_match is None or not heading_title:
        raise QualityConfigError(f"{source}: {location} must be a single ATX Markdown heading")
    return f"{heading_match.group(1)} {heading_title}"


def _reject_unknown_keys(mapping: dict, allowed: frozenset[str], source: str, location: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise QualityConfigError(
            f"{source}: unknown key(s) in {location}: {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})"
        )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects repeated mapping keys instead of keeping the last value."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # unhashable keys are rejected by the base constructor
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None)
    if mark is None or problem is None:
        return str(exc)
    return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"


def load_quality_config(path: Path | None = None) -> QualityConfig:
    source = str(path) if path else "packaged quality-default.yaml"
    try:
        text = path.read_text(encoding="utf-8") if path else DEFAULT_QUALITY_CONFIG.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise QualityConfigError(f"{source}: file is not valid UTF-8") from exc
    except OSError as exc:
        raise QualityConfigError(f"{source}: cannot read file: {exc.strerror or exc}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)  # nosec B506
    except yaml.YAMLError as exc:
        raise QualityConfigError(f"{source}: invalid YAML: {_yaml_error_detail(exc)}") from exc
    except RecursionError as exc:
        raise QualityConfigError(f"{source}: invalid YAML: nesting is too deep") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise QualityConfigError(f"{source}: invalid YAML: cannot parse a value: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("expected_headers"), list):
        raise QualityConfigError(f"{source}: expected_headers must be a list")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, source, "top level")

    expected_headers: list[ExpectedHeader] = []
    seen_headings: dict[str, str] = {}
    for index, rule in enumerate(raw["expected_headers"]):
        location = f"expected_headers[{index}]"
        if not isinstance(rule, dict):
            raise QualityConfigError(f"{source}: {location} must be a mapping")
        _reject_unknown_keys(rule, _EXPECTED_HEADER_KEYS, source, location)

        heading = _normalize_heading(rule.get("heading"), source, f"{location}.heading")
        raw_alternatives = rule.get("alternatives", [])
        if not isinstance(raw_alternatives, list):
            raise QualityConfigError(f"{source}: {location}.alternatives must be a list")
        alternatives = [
            _normalize_heading(value, source, f"{location}.alternatives[{alternative_index}]")
            for alternative_index, value in enumerate(raw_alternatives)
        ]
        points = rule.get("points", 5)
        try:
            valid_points = (
                not isinstance(points, bool) and isinstance(points, (int, float)) and isfinite(points) and points >= 0
            )
        except OverflowError:  # isfinite() on an int too large for a float
            valid_points = False
        if not valid_points:
            raise QualityConfigError(f"{source}: {location}.points must be a non-negative number")
        # message and suggestion are optional; omitted ones get generic text naming the headings.
        accepted_display = " or ".join(f"'{accepted}'" for accepted in (heading, *alternatives))
        message = rule.get("message", f"Missing expected heading: {accepted_display}")
        suggestion = rule.get(
            "suggestion",
            f"Add a {accepted_display} section" if not alternatives else f"Add one of these sections: {accepted_display}",
        )
        for field, value in (("message", message), ("suggestion", suggestion)):
            if not isinstance(value, str) or not value.strip():
                raise QualityConfigError(f"{source}: {location}.{field} must be a non-empty string when set")

        for accepted in (heading, *alternatives):
            if accepted in seen_headings:
                raise QualityConfigError(
                    f"{source}: duplicate heading '{accepted}' in {location} (already listed in {seen_headings[accepted]})"
                )
            seen_headings[accepted] = location

        expected_headers.append(
            {
                "heading": heading,
                "alternatives": alternatives,
                "points": points,
                "message": message,
                "suggestion": suggestion,
            }
        )

    return {"expected_headers": expected_headers}


def quality_config_digest(config: QualityConfig) -> str:
    """Return a stable digest of the normalized rules, independent of file formatting."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def quality_config_provenance(path: Path | None, config: QualityConfig) -> QualityConfigProvenance:
    """Describe the effective config. Only the file name is kept, so reports do not leak local paths."""
    return {
        "source": "custom" if path else "packaged",
        "file": path.name if path else None,
        "digest": quality_config_digest(config),
    }
