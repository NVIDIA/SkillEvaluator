# SPDX-License-Identifier: Apache-2.0
"""Parse generated report markup for tests; this is not an HTML sanitizer."""

from html.parser import HTMLParser


class _ScriptRanges(HTMLParser):
    def __init__(self, markup: str) -> None:
        super().__init__(convert_charrefs=False)
        self.markup = markup
        self.line_starts = [0, *(index + 1 for index, char in enumerate(markup) if char == "\n")]
        self.script_start: int | None = None
        self.ranges: list[tuple[int, int]] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag == "script":
            assert self.script_start is None, "Generated report contains nested script elements"
            self.script_start = self._offset()

    def handle_startendtag(self, tag: str, _attrs) -> None:
        assert tag != "script", "Generated report script elements require an explicit closing tag"

    def handle_endtag(self, tag: str) -> None:
        if tag != "script" or self.script_start is None:
            return
        start = self._offset()
        end = self.markup.find(">", start)
        assert end >= 0, "Generated report contains an incomplete script closing tag"
        assert self.markup[start + 2 : end].strip().casefold() == "script", (
            "Generated report contains an unsupported script closing tag"
        )
        self.ranges.append((self.script_start, end + 1))
        self.script_start = None


def without_scripts(markup: str) -> str:
    """Remove complete parsed script nodes, preserving all other markup and attributes."""
    parser = _ScriptRanges(markup)
    parser.feed(markup)
    parser.close()
    # HTMLParser's raw-text handling differs by Python patch version for malformed
    # closing tags. Fail rather than silently discarding a possibly visible tail.
    assert parser.script_start is None, "Generated report contains an unterminated or unsupported script element"
    parts = []
    previous = 0
    for start, end in parser.ranges:
        parts.append(markup[previous:start])
        previous = end
    parts.append(markup[previous:])
    return "".join(parts)


class _TextContent(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def text_content(fragment: str) -> str:
    """Read rendered fragment text without treating quoted attribute angles as tags."""
    parser = _TextContent()
    parser.feed(fragment)
    parser.close()
    return "".join(parser.parts)
