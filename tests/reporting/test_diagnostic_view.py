# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped diagnostic explanations retain every recorded requirement association."""

from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser

from jinja2 import Environment, PackageLoader


def _judgment(expectation_id, **overrides):
    return {
        "expectation_id": expectation_id,
        "status": "not_checked",
        "reason": "No retained trial result could be matched to this case",
        "next_step": "Inspect execution and judge artifacts; then rerun this check.",
        "evidence": [],
        "unmet_checks": [],
        "inapplicable_checks": [],
        **overrides,
    }


def _diagnostics(judgments):
    return [
        {
            "entry_id": "case-a",
            "requirements": [
                {"expectation_id": "assertions/0", "requirement": "Read the input\nwithout changing it"},
                {"expectation_id": "assertions/1", "requirement": "Write the output"},
            ],
            "attempts": [
                {
                    "agent": "agent-a",
                    "condition": "with_skill",
                    "trial_id": None,
                    "errors": [],
                    "judge_note": "",
                    "judgments": judgments,
                }
            ],
        }
    ]


def _project(diagnostics):
    from skillevaluator.reporting.diagnostic_view import build_dataset_diagnostics_view

    return build_dataset_diagnostics_view(diagnostics)


def _render(view, shared_errors=(), case_index=0):
    environment = Environment(loader=PackageLoader("skillevaluator.reporting", "templates"), autoescape=True)
    template = environment.from_string(
        '{% from "dataset_diagnostics.html.j2" import render_case_diagnostics with context %}'
        "{{ render_case_diagnostics(view, 'case-a', shared_errors, case_index) }}"
    )
    return template.render(view=view, shared_errors=shared_errors, case_index=case_index)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        if tag == "a" and attributes.get("href", "").startswith("#"):
            self.targets.append(attributes["href"][1:])


def test_same_attempt_status_and_cause_share_one_explanation_with_all_requirement_references():
    diagnostics = _diagnostics([_judgment("assertions/0"), _judgment("assertions/1")])
    view = _project(diagnostics)
    html = _render(view)

    assert html.count("No retained trial result could be matched to this case") == 1
    assert html.count("Inspect execution and judge artifacts; then rerun this check.") == 1
    assert "Not Checked (2)" in html
    assert "assertions/0" in html and "assertions/1" in html
    assert "Read the input\nwithout changing it" in html and "Write the output" in html
    links = _Links()
    links.feed(html)
    explanation_links = [target for target in links.targets if target.startswith("tier3-diagnostic-")]
    assert len(explanation_links) == 2
    assert len(set(explanation_links)) == 1
    assert explanation_links[0] in links.ids


def test_distinct_reasons_actions_statuses_attempts_and_error_associations_stay_separate():
    judgments = [
        _judgment("assertions/0"),
        _judgment("assertions/1", reason="Judge failed; keep this entire reason"),
        _judgment("assertions/0", next_step="Inspect a different artifact"),
        _judgment("assertions/1", status="unscored"),
        _judgment("assertions/0", error_ref="judge-a"),
        _judgment("assertions/1", error_ref="judge-b"),
    ]
    diagnostics = _diagnostics(judgments)
    diagnostics[0]["attempts"].append(
        {**deepcopy(diagnostics[0]["attempts"][0]), "agent": "agent-b", "condition": "without_skill"}
    )
    view = _project(diagnostics)

    for attempt in view[0]["attempts"]:
        assert [(group["status"], group["count"]) for group in attempt["status_groups"]] == [
            ("not_checked", 5),
            ("unscored", 1),
        ]
        assert len(attempt["status_groups"][0]["groups"]) == 5
    html = _render(view)
    assert html.count("Judge failed; keep this entire reason") == 2
    assert "agent-a" in html and "agent-b" in html and "Without skill" in html
    links = _Links()
    links.feed(html)
    assert len(links.ids) == len(set(links.ids))


def test_same_explanation_preserves_each_evidence_and_rubric_association_without_mutating_canonical_data():
    judgments = [
        _judgment("assertions/0", evidence=["trace-a"], unmet_checks=["INPUT"], inapplicable_checks=["SCRIPT"]),
        _judgment("assertions/1", evidence=["trace-b"], unmet_checks=["OUTPUT"]),
    ]
    diagnostics = _diagnostics(judgments)
    before = deepcopy(diagnostics)
    view = _project(diagnostics)
    group = view[0]["attempts"][0]["status_groups"][0]["groups"][0]

    assert group["judgments"] == judgments
    html = _render(view)
    first, second = html.split('<ul class="t3-ds-diagnostics">')[1].split("</ul>")[0].split("</li>")[:2]
    assert "assertions/0" in first and "trace-a" in first and "INPUT" in first and "SCRIPT" in first
    assert "trace-b" not in first
    assert "assertions/1" in second and "trace-b" in second and "OUTPUT" in second
    assert "trace-a" not in second
    group["judgments"][0]["evidence"].append("view-only")
    view[0]["requirements"][0]["requirement"] = "view-only"
    assert diagnostics == before


def test_projection_keeps_omission_counts_and_all_retained_judgments_including_unknown_statuses():
    diagnostics = _diagnostics([_judgment(f"retained/{index}") for index in range(129)])
    diagnostics[0]["omissions"] = {"expectations": 7, "expectation_attempts": 3}
    diagnostics[0]["attempts"][0]["judgments"].append(_judgment("unrecognized", status="custom_status"))
    diagnostics[0]["attempts"][0]["judgments_omitted"] = 9
    view = _project(diagnostics)

    assert view[0]["omissions"] == {"expectations": 7, "expectation_attempts": 3}
    assert view[0]["attempts"][0]["judgments_omitted"] == 9
    assert len(view[0]["attempts"][0]["judgments"]) == 130
    groups = view[0]["attempts"][0]["status_groups"]
    assert [(group["status"], group["count"]) for group in groups] == [("not_checked", 129), ("custom_status", 1)]
    html = _render(view)
    assert "retained/128" in html and "unrecognized" in html and "Custom Status (1)" in html


def test_template_escapes_each_recorded_field_and_keeps_shared_execution_errors_accessible():
    diagnostics = _diagnostics(
        [
            _judgment(
                "assertions/0",
                reason="reason <script>alert(1)</script>",
                next_step="step <img src=x onerror=alert(1)>",
                evidence=["evidence <svg onload=alert(1)>"],
                unmet_checks=["<check>"],
            ),
            _judgment("missing<id>", status="met"),
        ]
    )
    diagnostics[0]["requirements"][0]["requirement"] = "requirement <button>"
    diagnostics[0]["attempts"][0].update(
        agent="<agent>",
        trial_id="<trial>",
        judge_note="note <strong>",
        errors=["shared run error <private>", "local error <detail>"],
    )
    html = _render(_project(diagnostics), shared_errors=["shared run error <private>"], case_index=4)

    for text in ("<script>", "<img ", "<svg ", "<button>", "<agent>", "<trial>", "<private>", "<detail>"):
        assert text not in html
    for text in ("&lt;script&gt;", "&lt;img ", "&lt;svg ", "&lt;button&gt;", "&lt;agent&gt;", "&lt;trial&gt;"):
        assert text in html
    assert "missing&lt;id&gt;" in html
    assert "local error &lt;detail&gt;" in html
    assert "shared run error" not in html
    assert 'href="#tier3-execution-errors"' in html
