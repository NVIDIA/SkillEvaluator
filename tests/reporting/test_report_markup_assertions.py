# SPDX-License-Identifier: Apache-2.0
"""Parser-based assertions keep visible markup and evidence attributes in scope."""

import pytest
from report_markup_assertions import text_content, without_scripts

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter


def test_script_extraction_preserves_unsafe_nonscript_attributes_and_original_markup():
    before = "<!--outside comment--><style>.sample{opacity:0.99}</style>\n"
    script = '<ScRiPt type="application/json" data-note="x > y">{"probe":"script-only"}</sCrIpT  >'
    after = '<a href="javascript:alert(1)" data-artifact="file:///tmp/unsafe">A &amp; B</a>'
    outside = without_scripts(before + script + after)
    assert outside == before + after
    assert 'href="javascript:alert(1)"' in outside
    assert 'data-artifact="file:///tmp/unsafe"' in outside
    assert "0.99" in outside
    assert "script-only" not in outside


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_multiple_script_ranges_preserve_line_endings_and_nested_looking_strings(newline):
    outside = f'<p title="x > y">before</p>{newline}'
    middle = f'<span data-copy-text="0.99">middle</span>{newline}'
    end = "<p>after</p>"
    document = outside + '<script>const value = "<b>not visible</b>";</script>' + middle
    document += '<SCRIPT type="application/json">{"score":0.99}</SCRIPT>' + end
    assert without_scripts(document) == outside + middle + end


@pytest.mark.parametrize(
    "fragment",
    ["<script>payload", '<script>payload</script invalid><a href="javascript:alert(1)">tail</a>', "<script/>"],
)
def test_unsupported_script_structure_fails_instead_of_hiding_following_evidence(fragment):
    with pytest.raises(AssertionError, match="script"):
        without_scripts(fragment)


def test_text_content_decodes_rendered_entities_and_ignores_quoted_attribute_angles():
    fragment = '<button title="score > baseline" data-key="attribute-only">A &amp; B &#x3C;recorded&#x3E;</button>'
    assert text_content(fragment) == "A & B <recorded>"


def test_native_comparison_text_uses_a_parser_without_changing_visible_reason_text():
    from native_comparison_assertions import _text

    assert (
        _text('<span title="score > baseline">  no baseline <b>lift</b> &amp; evidence </span>')
        == "no baseline lift & evidence"
    )


def test_real_renderer_scripts_are_excluded_while_header_markup_and_text_remain():
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_error("Synthetic recorded failure")
    result.metadata["agent_eval"] = {
        "verdict": "fail",
        "execution_status": "failed",
        "canonical_probe": "embedded-only-marker",
        "execution_errors": ["Synthetic recorded failure"],
    }
    report = HTMLReporter(include_timestamp=False, explicit_skill_name="Synthetic & encoded").render_all([result])
    assert "embedded-only-marker" in report
    outside = without_scripts(report)
    assert '<header class="main-header">' in outside
    assert 'id="report-title"' in outside
    assert "embedded-only-marker" not in outside
    assert "Synthetic & encoded" in text_content(outside)
    assert "Synthetic recorded failure" in outside
