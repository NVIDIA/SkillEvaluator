# SPDX-License-Identifier: Apache-2.0
"""Read agent associations from the consolidated native comparison table."""

import re

from report_markup_assertions import text_content


def _text(fragment):
    return re.sub(r"\s+", " ", text_content(fragment)).strip()


def dimension_lift_labels(html, agents, best_agent):
    table = re.search(r'<table[^>]*id="tier3-dimension-table"[^>]*>(.*?)</table>', html, re.DOTALL)
    assert table is not None
    head = re.search(r"<thead>(.*?)</thead>", table[1], re.DOTALL)
    headings = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", head[1], re.DOTALL)
    others = [name for name in agents if name != best_agent]
    assert [_text(heading) for heading in headings[5:-2]] == others
    labels = {name: [] for name in agents}
    body = re.search(r"<tbody>(.*?)</tbody>", table[1], re.DOTALL)
    for row in re.findall(r"<tr(?:\s[^>]*)?>(.*?)</tr>", body[1], re.DOTALL):
        cells = re.findall(r"<td(?:\s[^>]*)?>(.*?)</td>", row, re.DOTALL)
        primary = _text(cells[3])
        if re.fullmatch(r"[+-]\d+\.\d{2}", primary):
            primary += " lift"
        elif "inconclusive: skill never activated" in primary:
            primary = "inconclusive: skill never activated"
        elif "with-skill unavailable" in primary:
            primary = "with-skill unavailable"
        elif "no baseline" in primary:
            primary = "no baseline lift"
        else:
            assert "comparison unavailable" in primary
            primary = "comparison unavailable"
        labels[best_agent].append(primary)
        for index, name in enumerate(others, 5):
            match = re.search(r'<span class="t3-mad-lift"[^>]*>(.*?)</span>', cells[index], re.DOTALL)
            assert match is not None
            labels[name].append(_text(match[1]))
    return labels
