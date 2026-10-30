# SPDX-License-Identifier: Apache-2.0
"""The inline SkillEvaluator logo used by the HTML report and the CI pages."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from skillevaluator.reporting.html import render_logo

REPO_ROOT = Path(__file__).resolve().parents[2]
SVG = "{http://www.w3.org/2000/svg}"
GEOMETRY = ("x", "y", "width", "height", "rx", "cx", "cy", "r", "d", "stroke-width")


def _shapes(svg: str) -> list[tuple]:
    root = ET.fromstring(svg)
    return [
        (element.tag.removeprefix(SVG), tuple((key, element.get(key)) for key in GEOMETRY if element.get(key)))
        for element in root.iter()
        if element.tag.removeprefix(SVG) in {"rect", "circle", "path"}
    ]


def test_logo_matches_the_published_asset_geometry() -> None:
    asset = (REPO_ROOT / "docs" / "assets" / "skillevaluator-logo.svg").read_text(encoding="utf-8")
    assert _shapes(render_logo("se-logo-test")) == _shapes(asset)


def test_labelled_logo_has_unique_prefixed_ids() -> None:
    logo = render_logo("se-logo-rail", 28, "SkillEvaluator logo")
    root = ET.fromstring(logo)
    assert root.get("role") == "img"
    assert root.get("aria-labelledby") == "se-logo-rail-title se-logo-rail-desc"
    assert re.findall(r'\bid="([^"]+)"', logo) == ["se-logo-rail-title", "se-logo-rail-desc"]
    assert root.get("width") == root.get("height") == "28"


def test_decorative_logo_is_hidden_and_has_no_ids() -> None:
    logo = render_logo("se-logo-side", 28, "")
    root = ET.fromstring(logo)
    assert root.get("aria-hidden") == "true"
    assert root.get("role") is None
    assert " id=" not in logo


def test_logo_colors_follow_theme_variables_not_the_os_setting() -> None:
    logo = render_logo("se-logo-theme")
    assert "prefers-color-scheme" not in logo
    assert "<style" not in logo
    colors = re.findall(r"(?:fill|stroke):([^;\"]+)", logo)
    assert colors
    assert all(color.startswith("var(--se-logo-") for color in colors)


def test_logo_escapes_its_label_and_id_prefix() -> None:
    logo = render_logo('x"><script>', 28, "<b>name</b>")
    assert "<script>" not in logo
    assert "<b>" not in logo
    ET.fromstring(logo)
