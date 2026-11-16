# SPDX-License-Identifier: Apache-2.0
"""Tool versions identify the brand and the tier that recorded their evidence."""

from copy import deepcopy

import pytest
from browser_navigation_fixtures import _agent_result
from test_report_navigation import Document

from skillevaluator import __version__
from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter


class _VersionDocument(Document):
    def handle_data(self, data):
        for element in self.stack:
            element.setdefault("text", []).append(data)

    def with_class(self, class_name):
        return [element for element in self.elements if class_name in element.get("class", "").split()]


def _text(element):
    return " ".join("".join(element.get("text", [])).split())


def _static(version=None):
    result = ValidationResult(validator_name="Schema Check")
    result.add_success("schema", "valid")
    if version is not None:
        result.metadata["skillspector_version"] = version
    return result


def _render(results):
    original = deepcopy(results)
    document = _VersionDocument(HTMLReporter(include_timestamp=False).render_all(results))
    assert results == original
    return document


def test_renderer_version_is_read_with_the_brand_instead_of_the_controls():
    document = _render([_static("2.13.4")])

    brands = document.with_class("header-brand-row")
    assert len(brands) == 1
    assert _text(brands[0]) == f"SkillEvaluator v{__version__}"
    versions = document.with_class("header-version")
    assert len(versions) == 1
    assert any(parent is brands[0] for parent in versions[0]["parents"])
    assert not any("header-meta" in parent.get("class", "").split() for parent in versions[0]["parents"])
    assert "SkillSpector" not in _text(document.with_class("main-header")[0])


@pytest.mark.parametrize("include_tier3", [False, True])
@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        (["2.13.4"], "SkillSpector v2.13.4"),
        (["2.13.4", "2.12.0", "2.13.4"], "SkillSpector v2.13.4, v2.12.0"),
    ],
)
def test_recorded_scanner_versions_belong_to_the_tier1_tile(include_tier3, versions, expected):
    results = [_static(version) for version in versions]
    if include_tier3:
        results.append(_agent_result("demo", ["agent"]))
    document = _render(results)

    tiles = [element for element in document.with_class("tier-card") if element.get("data-target-tab") == "tier1"]
    assert len(tiles) == 1
    assert _text(tiles[0]).count(expected) == 1
    scanner_labels = [
        element
        for element in document.elements
        if element.get("title") == ("SkillSpector version that ran the Tier 1 security scan")
    ]
    assert len(scanner_labels) == 1
    assert _text(scanner_labels[0]) == expected
    assert any(parent is tiles[0] for parent in scanner_labels[0]["parents"])
    assert any("tier-card-stats" in parent.get("class", "").split() for parent in scanner_labels[0]["parents"])
    assert "SkillSpector" not in _text(document.with_class("main-header")[0])


@pytest.mark.parametrize("standalone_tier3", [False, True])
def test_unrecorded_scanner_version_is_not_invented(standalone_tier3):
    result = _agent_result("demo", ["agent"]) if standalone_tier3 else _static()
    if standalone_tier3:
        # An unrelated Tier 3 metadata value is not a recorded Tier 1 scan.
        result.metadata["skillspector_version"] = "9.9.9"
    document = _render([result])

    assert all("SkillSpector" not in _text(element) for element in document.with_class("tier-card"))
    assert "SkillSpector" not in _text(document.with_class("main-header")[0])
    if standalone_tier3:
        assert not any(element.get("data-target-tab") == "tier1" for element in document.with_class("tier-card"))
