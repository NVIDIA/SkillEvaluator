# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-visible regressions for untrusted CommonMark destinations."""

import re
from functools import partial
from pathlib import Path

import pytest
from rich.console import Console

from skillevaluator.reporting import CLIReporter, MarkdownReporter
from skillevaluator.reporting import cli as cli_reporting
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators import markdown as markdown_validator
from skillevaluator.validators.base import ValidationResult
from skillevaluator.validators.hygiene import HygieneValidator, _link_display
from skillevaluator.validators.markdown import markdown_link_targets, normalized_local_path


def _validate_document(tmp_path: Path, content: str):
    """Scan a supporting document through the public Tier 1 orchestration."""
    (tmp_path / "SKILL.md").write_text("---\nname: sample\ndescription: Example skill\n---\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text(content, encoding="utf-8")
    results = run_validation(tmp_path, checks="code-integrity", content_type="skill")
    return next(result for result in results if result.validator_name == HygieneValidator().name)


def test_encoded_colon_stays_a_local_destination(tmp_path: Path) -> None:
    result = _validate_document(tmp_path, "[guide][target]\n\n[target]: missing%3Aguide.md\n")
    assert result.errors == ["Dead link in guide.md: missing%3Aguide.md"]
    assert normalized_local_path("missing%3Aguide.md") == "missing:guide.md"


@pytest.mark.parametrize("href", ["x/../C:/outside.md", "x/../C:outside.md", "C%3A/outside.md", "%5C%5Chost/share"])
def test_normalized_windows_anchors_are_not_local(href: str) -> None:
    assert normalized_local_path(href, allow_directory=True) is None


@pytest.mark.parametrize(
    "href",
    ["x/../C:/outside.md", "x/../C:outside.md", "C%3A/outside.md", "%5C%5Chost/share/outside.md", "%2Foutside.md"],
)
def test_unsafe_relative_anchors_produce_findings_without_lookup(
    tmp_path: Path, href: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert normalized_local_path(href, allow_directory=True) is None
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        assert path.name not in {"outside.md", "C:outside.md"}
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    result = _validate_document(tmp_path, f"[bad][target]\n\n[target]: {href}\n\n[other](missing.md)")
    assert len(result.errors) == 2
    assert result.errors[0] == f"Invalid local link in guide.md: {href} (absolute or drive-relative path)"
    assert result.errors[1] == "Dead link in guide.md: missing.md"


def test_repeated_unsafe_anchor_is_reported_once_per_document(tmp_path: Path) -> None:
    content = (
        '[first](C%3A/outside.md) <a href="C%3A/outside.md">second</a> [third][target]\n\n'
        "[target]: C%3A/outside.md\n\n[other](missing.md)"
    )
    (tmp_path / "another.md").write_text(content, encoding="utf-8")
    result = _validate_document(tmp_path, content)
    assert sorted(result.errors) == sorted(
        f"{message} in {name}: {target}"
        for name in ("another.md", "guide.md")
        for message, target in (
            ("Invalid local link", "C%3A/outside.md (absolute or drive-relative path)"),
            ("Dead link", "missing.md"),
        )
    )


def test_invalid_utf8_does_not_alias_an_existing_replacement_character(tmp_path: Path) -> None:
    (tmp_path / "\ufffd.md").write_text("Existing unrelated document", encoding="utf-8")
    result = _validate_document(tmp_path, '<a href="%FF.md">one</a> <a href="%FE.md">two</a>')
    assert result.errors == ["Dead link in guide.md: %FF.md", "Dead link in guide.md: %FE.md"]
    assert normalized_local_path("%FF.md") != normalized_local_path("%FE.md")


@pytest.mark.parametrize("value", ["9" * 5000, "[" * 1500 + "0" + "]" * 1500], ids=["huge-integer", "deep-sequence"])
def test_frontmatter_limits_do_not_abort_validation(tmp_path: Path, value: str) -> None:
    result = _validate_document(tmp_path, f"---\nvalue: {value}\n---\n[guide](missing.md)\n")
    assert result.errors == ["Dead link in guide.md: missing.md"]


def test_reference_lookup_failure_does_not_abort_other_links(tmp_path: Path) -> None:
    target = "a" * 300 + ".md"
    result = _validate_document(tmp_path, f"[long][target]\n\n[target]: {target}\n\n[other](missing.md)\n")
    assert len(result.errors) == 2
    assert result.errors[-1] == "Dead link in guide.md: missing.md"
    assert len(result.errors[0]) < 220


def test_lookup_oserror_is_contained_per_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        if path.name == "unreadable.md":
            raise PermissionError("simulated inaccessible target")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    result = _validate_document(tmp_path, "[guide][target]\n\n[target]: unreadable.md\n\n[other](missing.md)")
    assert result.errors == ["Dead link in guide.md: unreadable.md", "Dead link in guide.md: missing.md"]


def test_lookup_programming_errors_are_not_silenced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        if path.name == "broken.md":
            raise RuntimeError("unexpected lookup bug")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    (tmp_path / "guide.md").write_text("[bad](broken.md)", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected lookup bug"):
        HygieneValidator()._check_dead_links(tmp_path)


@pytest.mark.parametrize("prefix_length", range(155, 161))
@pytest.mark.parametrize("control", ["\n", "\r", "\x1b", "\x00", "\u2028", "\U0001f600"])
@pytest.mark.parametrize("reporter_type", [CLIReporter, MarkdownReporter])
def test_diagnostic_truncation_cannot_restore_control_characters(
    prefix_length: int, control: str, reporter_type: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Test injected line breaks separately from the CLI's intentional reflow.
    monkeypatch.setattr(cli_reporting, "Console", partial(Console, width=1000, color_system="standard"))
    display = _link_display("a" * prefix_length + control + "::error file=SKILL.md::forged")
    assert len(display) <= 163
    assert all(32 <= ord(character) < 127 for character in display)
    result = ValidationResult()
    result.add_error(f"Dead link: {display}")
    output = reporter_type().render_all([result])
    # The CLI deliberately emits SGR colors; do not mistake its styling for
    # an injected escape. Other escape sequences must remain absent.
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    assert not any(line.startswith("::") for line in output.splitlines())
    assert "\x1b" not in output


@pytest.mark.parametrize("reporter_type", [CLIReporter, MarkdownReporter])
def test_link_diagnostics_cannot_inject_report_lines(tmp_path: Path, reporter_type: type) -> None:
    result = _validate_document(tmp_path, '<a href="missing/&#10;::error file=SKILL.md,line=1::forged&#10;.md">bad</a>')
    assert len(result.errors) == 1
    assert "\\n::error" in result.errors[0]
    assert len(result.errors[0].splitlines()) == 1
    output = reporter_type().render_all([result])
    assert not any(line.startswith("::") for line in output.splitlines())


@pytest.mark.parametrize("kib", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "opener", ["<!--", "<?", "<![CDATA[", "<!a"], ids=["comment", "processing", "cdata", "declaration"]
)
def test_unclosed_html_scanning_is_linear(kib: int, opener: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure regex input work for the four guarded forms, not all Markdown."""
    regex_input = 0
    original_rule = markdown_validator.html_inline

    def counted_rule(state, silent):
        nonlocal regex_input
        if state.src[state.pos] == "<":
            regex_input += len(state.src) - state.pos
        return original_rule(state, silent)

    monkeypatch.setattr(markdown_validator, "html_inline", counted_rule)
    content = '<a href="before.md">before</a>' + opener * (kib * 1024 // len(opener))
    assert markdown_link_targets(content) == ["before.md"]
    assert regex_input <= 2 * len(content)


@pytest.mark.parametrize(
    "fragment",
    [
        '<!-- <a href="hidden.md">hidden</a> -->',
        '<!-- <!-- <a href="hidden.md">hidden</a> -->',
        "<!-->",
        "<!--->",
        '<?<a href="hidden.md">hidden</a>?>',
        '<![CDATA[<a href="hidden.md">hidden</a>]]>',
        '<!a <a href="hidden.md">',
    ],
)
def test_html_guard_preserves_closed_fragments_and_later_links(fragment: str) -> None:
    content = f"[first](first.md) {fragment} [last](last.md)\n\n" + "<!--" * 2
    assert markdown_link_targets(content) == ["first.md", "last.md"]


@pytest.mark.parametrize(
    "opener,closed",
    [
        ("<!--", '<!-- <a href="hidden.md">hidden</a> -->'),
        ("<?", '<?<a href="hidden.md">hidden</a>?>'),
        ("<![CDATA[", '<![CDATA[<a href="hidden.md">hidden</a>]]>'),
        ("<!a", '<!a <a href="hidden.md">'),
    ],
)
def test_html_guard_uses_each_inline_sources_offsets(opener: str, closed: str) -> None:
    content = f"prefix {opener}\n\nnext [visible](visible.md) {closed}"
    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize("reporter_type", [CLIReporter, MarkdownReporter])
def test_link_diagnostics_bound_escaped_unicode(tmp_path: Path, reporter_type: type) -> None:
    result = _validate_document(tmp_path, '<a href="missing/' + "\U0001f600" * 200 + '">bad</a>')
    assert len(result.errors) == 1
    assert result.errors[0].endswith("...")
    assert len(result.errors[0]) < 220
    assert "\\U0001f600" in result.errors[0]
    assert len(reporter_type().render_all([result])) < 3000
