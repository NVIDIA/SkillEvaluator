# SPDX-License-Identifier: Apache-2.0
"""Sanitized synthetic native fields; these are not live evaluation results."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


def api():
    module = importlib.import_module("skillevaluator.reporting.native_evidence")
    return module.build_native_evidence


def fixture():
    return json.loads((Path(__file__).parent / "fixtures" / "native-optional-evidence.json").read_text())


def fragment(view, slot):
    from skillevaluator.reporting import HTMLReporter

    return (
        HTMLReporter(include_timestamp=False)
        ._env.from_string(
            "{% from 'native_evidence.html.j2' import render_native_warnings, render_native_evidence %}"
            "{{ render_native_warnings(view) if slot == 'overview' else render_native_evidence(view, slot) }}"
        )
        .render(view=view, slot=slot)
    )


def test_native_groups_display_recorded_evidence_without_changing_input():
    raw = fixture()
    original = deepcopy(raw)
    view = api()(raw)
    overview = fragment(view, "overview")
    agents = fragment(view, "agents")
    trials = fragment(view, "trials")
    diagnostics = fragment(view, "provenance")
    assert "Harbor delivery incomplete" in overview
    assert "Provider safety notice" in overview
    assert "Plugin dependency coverage" in overview
    assert "uploads/demo.tar.gz" in diagnostics
    assert "synthetic upload timeout" in diagnostics
    assert "resolved-skill" in agents and "unresolved-rule" in agents
    assert "synthetic-tool" in agents and "connected, no recorded calls" in agents
    assert "Recorded refusal comparison basis" in agents and "refused-case" in agents
    assert "0.8" in agents and "0.6" in agents
    assert "Plugin composition" in trials and "0.71" in trials and "0.54" in trials
    assert "0.17" in trials and "0.04" in trials and "0.3" in trials
    assert "Synthetic recorded interpretation" in trials and "component-alpha" in trials
    assert "missing-part-case" in trials and "unused-rule" in trials
    assert "Synthetic recovery note" in diagnostics
    assert '<details class="t3-details"' in agents
    assert " open" not in agents + trials + diagnostics
    assert "Synthetic recovery note" not in overview
    assert raw == original


@pytest.mark.parametrize("value", [None, {}, [], "unexpected", False])
def test_absent_null_empty_and_wrong_shapes_do_not_invent_status(value):
    view = api()(
        {
            "plugin_provenance": value,
            "integration": value,
            "harbor_viewer_delivery": value,
            "agents": {"demo": {"safety_refusals": value, "recovery_notes": value}},
        }
    )
    assert not view.warnings and not view.agents and not view.trials and not view.provenance
    assert not fragment(view, "overview").strip()


def test_partial_composition_shows_only_supplied_fields_and_reason():
    view = api()({"integration": {"reason": "Synthetic missing arm", "requested_lift_mode": "both"}})
    html = fragment(view, "trials")
    assert "Synthetic missing arm" in html and "both" in html
    assert "inconclusive" not in html.lower()
    assert "0.0" not in html and "not exercised" not in html
    assert not view.warnings


@pytest.mark.parametrize("usage", [None, {}, {"calls": None}, {"calls": False}, {"calls": "0"}])
def test_mcp_missing_or_malformed_call_measurements_are_not_zero(usage):
    view = api()(
        {
            "plugin_provenance": {
                "mcp_resolutions": [
                    {
                        "name": "synthetic-mcp",
                        "status": "reachable-in-agent",
                        "tool_usage": usage,
                    }
                ]
            }
        }
    )
    html = fragment(view, "agents")
    assert "synthetic-mcp" in html
    assert "connected, no recorded calls" not in html and "tool calls failed" not in html


@pytest.mark.parametrize("status", [None, "not-evaluated", "unreachable"])
def test_zero_calls_without_recorded_connection_do_not_claim_connected(status):
    view = api()(
        {
            "plugin_provenance": {
                "mcp_resolutions": [
                    {
                        "name": "synthetic-mcp",
                        "status": status,
                        "tool_usage": {"calls": 0},
                    }
                ]
            }
        }
    )
    assert "connected, no recorded calls" not in fragment(view, "agents")


def test_explicit_mcp_failures_are_advisory_and_do_not_change_payload():
    raw = {
        "plugin_provenance": {
            "mcp_resolutions": [
                {
                    "name": "synthetic-mcp",
                    "status": "reachable-host",
                    "tool_usage": {"calls": 2, "failed": 1},
                }
            ]
        }
    }
    original = deepcopy(raw)
    html = fragment(api()(raw), "agents")
    assert "tool calls failed" in html and "advisory" in html.lower()
    assert "connected, no recorded calls" not in html
    assert raw == original


def test_presentation_escapes_redacts_and_links_only_exact_safe_archive_urls():
    raw = fixture()
    agent = raw["agents"]["demo-agent"]
    agent["recovery_notes"] = ["<script>injected()</script> password=syntheticsecret\ud800\u202e\x1b[31m"]
    raw["harbor_viewer_delivery"]["uploads"] = [
        {"status": "failed", "diagnostic_archive": "https://example.test/retained/demo.tar.gz"},
        {"status": "failed", "diagnostic_archive": "javascript:alert(1)"},
    ]
    html = fragment(api()(raw), "provenance")
    assert "<script>injected()" not in html and "&lt;script&gt;" in html
    assert "syntheticsecret" not in html and "redacted" in html
    assert "\ud800" not in html and "\x1b" not in html and "\u202e" not in html
    assert 'href="https://example.test/retained/demo.tar.gz"' in html
    assert 'href="javascript:' not in html
    assert 'href="uploads/' not in html


def test_rows_cells_and_notes_are_bounded_with_local_disclosures():
    raw = {
        "plugin_provenance": {"evaluated_member_skills": [f"skill-{n}" for n in range(90)]},
        "agents": {"demo": {"recovery_notes": ["Recorded " + "x" * 500] * 20}},
    }
    original = deepcopy(raw)
    view = api()(raw)
    html = fragment(view, "agents") + fragment(view, "provenance")
    assert "rows omitted" in html and "notes omitted" in html and "text shortened" in html
    assert "skill-89" not in html and "x" * 500 not in html
    assert len(html) < 50000
    assert raw == original


def test_large_nested_or_cyclic_unknown_fields_are_not_traversed():
    raw = {"integration": {"reason": "Bounded retained reason"}}
    raw["integration"]["unknown"] = raw
    assert "Bounded retained reason" in fragment(api()(raw), "trials")


def test_successful_delivery_and_zero_refusal_counts_do_not_emit_active_warnings():
    view = api()(
        {
            "harbor_viewer_delivery": {"viewer_status": "succeeded", "uploads": []},
            "agents": {"demo": {"safety_refusals": {"attempts": 0, "model_mixed_attempts": 0}}},
        }
    )
    assert not view.warnings


def test_native_pass_rate_basis_comes_from_pass_at_k_lift():
    raw = {
        "agents": {
            "demo": {
                "pass_at_k": {
                    "lift": {
                        "safety_refusal_cases": ["native-excluded-case"],
                        "with_skill": 0.75,
                        "without_skill": 0.25,
                    }
                }
            }
        }
    }
    html = fragment(api()(raw), "agents")
    assert "native-excluded-case" in html and "0.75" in html and "0.25" in html


def test_native_mcp_preserves_recorded_resolution_detail_and_ignores_shaped_status():
    raw = {
        "plugin_provenance": {
            "mcp_resolutions": [
                {
                    "name": "detailed",
                    "status": "not-evaluated",
                    "url": "https://example.test/mcp",
                    "tool_count": 3,
                    "reason": "Synthetic resolution reason",
                    "remediation": "Synthetic next step",
                },
                {"name": "malformed", "status": {}, "tool_usage": {"calls": 0}},
            ]
        }
    }
    html = fragment(api()(raw), "agents")
    assert (
        "https://example.test/mcp" in html and "Synthetic resolution reason" in html and "Synthetic next step" in html
    )
    assert "connected, no recorded calls" not in html


def test_case_only_refusal_record_keeps_a_short_active_notice():
    view = api()({"agents": {"demo": {"safety_refusals": {"cases": ["recorded-refusal-case"]}}}})
    assert "Provider safety notice" in fragment(view, "overview")
    assert "recorded-refusal-case" in fragment(view, "agents")


def test_native_html_places_each_group_once_and_preserves_canonical_exports():
    from test_native_report_parity import _embedded_payload

    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting import HTMLReporter

    payload = build_agent_eval_payload(
        "synthetic", {"agent": {"overall_with_skill": 0.8, "overall_without_skill": 0.4}}, use_llm_judge=False
    )
    optional = fixture()
    payload["agents"]["agent"].update(optional.pop("agents")["demo-agent"])
    payload.update(optional)
    # A scored, succeeded synthetic run qualifies for the existing safe display
    # boundary. Uncertified arm data is deliberately scrubbed before this adapter.
    payload.update(execution_status="succeeded", expected_attempts=2, scored_attempts=2)
    payload["agents"]["agent"].update(execution_status="succeeded", expected_attempts=2, scored_attempts=2)
    original = deepcopy(payload)
    result = ValidationResult(validator_name="AGENT_EVAL", metadata={"agent_eval": payload})
    html = HTMLReporter(include_timestamp=False).render_all([result])
    overview = html.split('id="tier3-page-overview"', 1)[1].split('id="tier3-page-trials"', 1)[0]
    trials = html.split('id="tier3-page-trials"', 1)[1].split('id="tier3-page-agents"', 1)[0]
    agents = html.split('id="tier3-page-agents"', 1)[1].split('id="tier3-page-dataset"', 1)[0]
    diagnostics = html.split('id="tier3-page-provenance"', 1)[1].split("</section>", 1)[0]
    assert overview.count('id="tier3-native-notices"') == 1
    for key, page in (
        ("plugin-completeness", agents),
        ("refusal-basis-0", agents),
        ("integration", trials),
        ("viewer-delivery", diagnostics),
        ("recovery-notes-0", diagnostics),
    ):
        assert html.count(f'id="native-{key}"') == 1
        assert f'id="native-{key}"' in page
    assert _embedded_payload(html) == original == payload


def test_numeric_native_values_reject_strings_booleans_and_nonfinite_numbers():
    raw = {
        "integration": {
            "with_plugin": True,
            "sum_of_parts": "not-a-score",
            "integration_lift": float("inf"),
            "lift_uncertainty": {"ci_low": "not-an-interval", "n_cases": False},
        },
        "plugin_provenance": {"partial": "not-a-boolean"},
    }
    view = api()(raw)
    assert not view.agents and not view.trials and not view.warnings


def test_explicit_numeric_zero_is_retained_without_becoming_a_missing_value():
    view = api()(
        {
            "integration": {
                "with_plugin": 0,
                "sum_of_parts": 0,
                "integration_lift": 0,
                "lift_uncertainty": {"ci_low": 0, "ci_high": 0, "n_cases": 0},
            }
        }
    )
    html = fragment(view, "trials")
    assert html.count("<td>0</td>") == 6


def test_native_completeness_retains_recorded_ignored_mcp_config_details():
    view = api()(
        {
            "plugin_provenance": {
                "mcp_untrusted_configs": [
                    {
                        "path": "evals/environment/synthetic-mcp.json",
                        "entries": 2,
                    }
                ]
            }
        }
    )
    html = fragment(view, "agents")
    assert "evals/environment/synthetic-mcp.json" in html and "<td>2</td>" in html
    assert not view.warnings


def test_nested_tool_list_omissions_are_disclosed_even_when_cell_is_shortened():
    raw = {
        "plugin_provenance": {
            "mcp_resolutions": [
                {
                    "name": "synthetic-mcp",
                    "tool_usage": {"tools": ["t" * 200] * 20},
                }
            ]
        }
    }
    html = fragment(api()(raw), "agents")
    assert "4 tool entries omitted" in html and "text shortened" in html


def test_agent_identity_survives_an_exhausted_shared_native_text_budget():
    mcp = {
        "name": "n" * 300,
        "status": "s" * 300,
        "tool_count": 1,
        "tool_usage": {"calls": 1, "failed": 0, "tools": ["t" * 300]},
        "url": "u" * 300,
        "reason": "r" * 300,
        "remediation": "m" * 300,
    }
    raw = {
        "plugin_provenance": {"mcp_resolutions": [mcp] * 64},
        "agents": {
            name: {"safety_refusals": {"attempts": 1}, "recovery_notes": ["Recorded recovery"]}
            for name in ("first-agent", "second-agent")
        },
    }
    original = deepcopy(raw)
    view = api()(raw)
    basis = [section for section in view.agents if section.key.startswith("refusal-basis-")]
    recovery = [section for section in view.provenance if section.key.startswith("recovery-notes-")]
    assert all(not section.rows and section.omitted_rows for section in basis)
    for sections in (basis, recovery):
        assert len(sections) == 2
        assert "first-agent" in sections[0].title
        assert "second-agent" in sections[1].title
        assert all(all(note.strip() for note in section.notes) for section in sections)
        assert all(section.text_shortened for section in sections)
    html = fragment(view, "agents") + fragment(view, "provenance")
    assert len(html.encode()) < 160_000
    assert raw == original


def test_unused_native_components_keep_their_kind_and_disclose_omitted_groups():
    unused = {kind: ["shared-component"] for kind in ("skill", "rule", "mcp")}
    unused.update({f"extra-{index}": ["another-component"] for index in range(16)})
    raw = {"integration": {"component_signals": {"unused_declared": unused}}}
    original = deepcopy(raw)
    view = api()(raw)
    html = fragment(view, "trials")
    assert "Declared but unused · skill" in html
    assert "Declared but unused · rule" in html
    assert "Declared but unused · mcp" in html
    assert "3 component groups omitted" in html
    assert "extra-15" not in html
    assert raw == original


def test_reserved_native_agent_labels_stay_sanitized_bounded_and_distinct():
    names = [f"{'界' * 200}-{index}" for index in range(40)]
    raw = {"agents": {name: {"safety_refusals": {"attempts": 1}, "recovery_notes": ["Recorded"]} for name in names}}
    view = api()(raw)
    assert len(view.agents) == len(view.provenance) == 32
    assert view.omitted_agents == 8
    assert len({section.title for section in view.agents}) == 32
    assert all(len(section.title.encode()) < 180 for section in (*view.agents, *view.provenance))
    assert all(section.text_shortened for section in (*view.agents, *view.provenance))
    html = fragment(view, "agents") + fragment(view, "provenance")
    assert len(html.encode()) < 160_000
