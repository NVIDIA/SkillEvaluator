# SPDX-License-Identifier: Apache-2.0
"""Private static views share facts without enabling public plugin workflows."""

from __future__ import annotations

import html
import importlib
import json
from copy import deepcopy

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting import (
    BenchmarkReporter,
    CLIReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    SARIFReporter,
)


def api():
    return importlib.import_module("skillevaluator.reporting.plugin_static")


def renderers():
    return importlib.import_module("skillevaluator.reporting.plugin_static_render")


def reporters():
    return [
        CLIReporter(),
        JSONReporter(include_timestamp=False),
        MarkdownReporter(include_timestamp=False),
        HTMLReporter(include_timestamp=False),
        SARIFReporter(include_timestamp=False),
        BenchmarkReporter(include_timestamp=False),
    ]


def bundle():
    schema = ValidationResult(validator_name="Private Manifest and Component Checks")
    schema.metadata["plugin"] = {
        "root": "/work/bundle",
        "manifest_sha256": "a" * 64,
        "manifest_declarations": [
            {
                "selected": True,
                "manifest_filename": ".claude-plugin/plugin.json",
                "manifest_type": "claude_plugin_json",
                "name": "report-demo",
                "version": "1.0",
            }
        ],
        "component_inventory": {
            "complete": True,
            "components": [
                {
                    "type": "skill",
                    "name": "loader",
                    "path": "skills/loader",
                    "origin": "packaged",
                    "support": "evaluated",
                    "findings": 1,
                },
                {
                    "type": "app",
                    "name": "calendar-app",
                    "path": "apps.json",
                    "support": "unsupported",
                    "loaded_by": "Codex",
                    "effective": False,
                },
            ],
            "counts": {"skill": 1, "app": 1},
        },
        "dependency_resolution": {
            "skills": [
                {
                    "ref": "external-ref",
                    "state": "external",
                    "declared_by": "agent_plugin.yaml",
                    "reason": "not fetched",
                }
            ]
        },
        "mcp": {
            "servers": [
                {
                    "name": "search-server",
                    "pinned": False,
                    "kind": "command",
                    "transport": "stdio",
                    "pin_detail": "floating runner",
                }
            ],
            "pinning": {"total": 1, "pinned": 0, "unpinned": 1},
        },
        "context_cost": {
            "method": "static_estimate",
            "estimator": "chars_div_4_cjk",
            "harness": "claude-code",
            "load_mode": "native",
            "always_on_tokens": -20,
            "on_demand_tokens": 250,
            "lower_bound": True,
            "not_counted": ["MCP schemas"],
            "by_component": [
                {
                    "type": "skill",
                    "name": "loader",
                    "always_on_tokens": 20,
                    "on_demand_tokens": 250,
                    "basis": "frontmatter",
                }
            ],
            "by_harness": [
                {"harness": "codex", "load_mode": "wrapper", "always_on_tokens": 0, "on_demand_tokens": 270}
            ],
        },
        "hook_risk": {
            "hooks": [
                {
                    "id": "startup-hook",
                    "event": "SessionStart",
                    "target": "echo harmless",
                    "risk_flags": ["context_injection"],
                }
            ]
        },
        "privileges": {
            "components": [
                {
                    "type": "agent",
                    "name": "reviewer-agent",
                    "path": "agents/reviewer.md",
                    "inherits_all_tools": True,
                    "flags": ["inherits_all_tools"],
                }
            ]
        },
    }
    schema.add_finding(Finding("TEST", Severity.MEDIUM, "example", "Example", "/work/bundle/skills/loader/SKILL.md"))
    audit = ValidationResult(validator_name="Private audit")
    audit.metadata["plugin"] = {
        "cve_summary": {
            "ecosystems": {
                "python": {
                    "status": "incomplete",
                    "sources": 2,
                    "audited": 1,
                    "vulnerabilities": 3,
                    "findings_truncated": True,
                    "errors": ["audit capped"],
                }
            }
        },
        "validator_parity": {
            "status": "compared",
            "claude_verdict": "passed",
            "strict_verdict": "failed",
            "skillevaluator_verdict": "passed",
            "agree": True,
            "warnings": ["parity-warning"],
        },
    }
    audit.mark_scan_incomplete("pip-audit")
    audit.add_finding(
        Finding(
            "CVE",
            Severity.HIGH,
            "python-vulnerability",
            "CVE-fixture",
            "requirements.txt",
            metadata={"ecosystem": "python"},
        )
    )
    endpoint = ValidationResult(validator_name="Private Endpoint Checks")
    endpoint.metadata["endpoint_resolution"] = {
        "endpoints": [
            {
                "kind": "mcp",
                "name": "remote-endpoint",
                "url": "https://service.example/path?token=hidden-value",
                "status": "private",
                "head": {"skipped": "non-public"},
                "redirect": {"classification": "metadata"},
            }
        ]
    }
    catalog = ValidationResult(validator_name="Inter-Plugin Deduplication")
    catalog.metadata.update(
        advisory_tier2=True,
        plugin={
            "catalog_skill_similarity": {
                "status": "compared",
                "catalog_entries": 4,
                "matches": [{"skill": "loader", "match": "catalog-loader", "similarity": 0.95}],
            },
            "inter_plugin_similarity": {"status": "skipped", "reason": "catalog unavailable", "matches": []},
        },
    )
    return [schema, audit, endpoint, catalog]


def test_projection_preserves_real_private_shapes_and_does_not_mutate():
    results = bundle()
    before = deepcopy(results)
    view = api().project_plugin_static(results)
    assert view is not None
    payload = view.payload()
    assert payload["name"] == "report-demo" and payload["status"] == "failed"
    sections = {section["key"]: section for section in payload["sections"]}
    assert {
        "manifests",
        "inventory",
        "dependencies",
        "mcp",
        "context",
        "context_harnesses",
        "hooks",
        "privileges",
        "audits",
        "parity",
        "endpoints",
        "catalog_skills",
        "catalog_plugins",
    } <= sections.keys()
    assert "3" in json.dumps(sections["audits"])
    assert "unknown" in json.dumps(sections["audits"]).lower() or "partial" in json.dumps(sections["audits"]).lower()
    assert "-20" in json.dumps(sections["context"])
    assert "hidden-value" not in json.dumps(payload)
    assert results == before


def test_all_six_private_surfaces_carry_the_same_static_facts():
    output = renderers().render_plugin_static_reports(bundle(), reporters=reporters())
    assert set(output) == {"cli", "json", "markdown", "html", "sarif", "benchmark"}
    for name, document in output.items():
        for marker in (
            "report-demo",
            "calendar-app",
            "external-ref",
            "search-server",
            "startup-hook",
            "reviewer-agent",
            "remote-endpoint",
            "catalog-loader",
            "catalog unavailable",
            "chars_div_4_cjk",
            "audit capped",
            "parity-warning",
        ):
            assert marker in html.unescape(document).replace("\\_", "_"), (name, marker)
        assert "hidden-value" not in document
    expected = json.loads(output["json"])["plugin_static"]
    assert json.loads(output["sarif"])["runs"][0]["properties"]["pluginStatic"] == expected
    baseline_html = HTMLReporter(include_timestamp=False).render_all(bundle())
    assert output["html"].split("</head>", 1)[0] == baseline_html.split("</head>", 1)[0]
    assert 'id="report-data"' in output["html"]
    assert 'id="plugin-static-evidence"' in output["html"]
    assert "Recommended for publication" not in output["benchmark"]


def test_absent_empty_and_malformed_blocks_leave_generic_reports_unchanged():
    for metadata in ({}, {"plugin": {}}, {"plugin": "wrong"}):
        result = ValidationResult(validator_name="SCHEMA", metadata=metadata)
        report_list = reporters()
        before = {reporter.name: reporter.render_all([result]) for reporter in report_list}
        assert renderers().render_plugin_static_reports([result], reporters=report_list) == before


def test_projection_reduction_happens_once_for_six_adapters(monkeypatch):
    module = renderers()
    project = module.project_plugin_static
    calls = []

    def counting(results):
        calls.append(len(results))
        return project(results)

    monkeypatch.setattr(module, "project_plugin_static", counting)
    module.render_plugin_static_reports(bundle(), reporters=reporters())
    assert calls == [4]


@pytest.mark.parametrize("force_color", [False, True])
def test_huge_cyclic_and_hostile_static_metadata_is_bounded_and_sanitized(monkeypatch, force_color):
    monkeypatch.setenv("TERM", "xterm" if force_color else "dumb")
    monkeypatch.delenv("NO_COLOR", raising=False)
    if force_color:
        monkeypatch.setenv("FORCE_COLOR", "1")
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    results = bundle()
    hostile = (
        '<img src=x onerror=alert(1)> [bold]oops[/bold] API_KEY="'
        "private-test-secret"
        '"\x1b]8;;https://evil.example\x07click\x1b]8;;\x07\u202e\ud800'
    )
    components = results[0].metadata["plugin"]["component_inventory"]["components"]
    components[:] = [
        {"type": "skill", "name": hostile, "path": f"skills/{index}", "support": "evaluated"} for index in range(800)
    ]
    results[0].metadata["plugin"]["context_cost"]["notes"] = ["x" * 2_000_000]
    results[0].metadata["plugin"]["cycle"] = results[0].metadata
    projection = api().project_plugin_static(results)
    payload = projection.payload()
    assert payload["truncated"] is True
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "\\u001b" not in serialized and "private-test-secret" not in serialized and "\ud800" not in serialized
    assert len(serialized.encode("utf-8")) <= api().MAX_OUTPUT_BYTES
    document = renderers().render_plugin_static_reports(results, reporters=reporters())
    for name, text in document.items():
        assert "private-test-secret" not in text and "\ud800" not in text, name
        if name == "cli":
            original, fragment = text.split("\n\nPlugin static evidence\n", 1)
            assert ("\x1b[" in original) is force_color
        else:
            fragment = text
        assert "\x1b" not in fragment, name
        if name in {"html", "markdown", "benchmark"}:
            assert "<img src=x onerror=alert(1)>" not in text


def test_sarif_component_attribution_is_lexical_and_unambiguous():
    results = bundle()
    view = api().project_plugin_static(results)
    assert view.component_for("[loader] SKILL.md")["path"] == "skills/loader"
    assert view.component_for("/outside/skills/loader/SKILL.md") is None
    assert view.component_for("skills/loader/../../escape") is None
    result = renderers().render_plugin_static_reports(results, reporters=[SARIFReporter(include_timestamp=False)])
    finding = json.loads(result["sarif"])["runs"][0]["results"][0]
    assert finding["properties"]["pluginComponent"]["name"] == "loader"


def test_private_schema_bridges_existing_risk_records(tmp_path):
    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.validators.private_plugin_schema import validate_inventory_schema

    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "name": "real-private",
                "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo fixture"}]}]},
            }
        )
    )
    inventory = plugin_inventory_for_root(tmp_path)
    result = validate_inventory_schema(inventory)
    assert result.metadata["plugin"]["hook_risk"]["hooks"]
    assert api().project_plugin_static([result]).payload()["name"] == "real-private"


def test_report_bounds_reserve_identity_and_later_risk_evidence():
    results = bundle()
    results[0].metadata["plugin"]["component_inventory"]["components"] = [
        {"type": "skill", "name": "long-name-" + "x" * 200, "path": f"skills/{index}", "support": "evaluated"}
        for index in range(800)
    ]
    payload = api().project_plugin_static(results).payload()
    assert payload["name"] == "report-demo"
    hooks = next(section for section in payload["sections"] if section["key"] == "hooks")
    assert "startup-hook" in json.dumps(hooks)
    assert "context_injection" in json.dumps(hooks)


@pytest.mark.parametrize("long_cells", [False, True], ids=["row-limit", "text-limit"])
def test_crowded_reports_retain_risk_evidence_across_all_six_formats(long_cells):
    results = bundle()
    block = results[0].metadata["plugin"]
    count = 64 if long_cells else 100
    text = "x" * 300 if long_cells else "fixture"
    block["component_inventory"]["components"] = [
        dict.fromkeys(("type", "name", "origin", "path", "support", "problem", "declared_by", "loaded_by"), text)
        for _ in range(count)
    ]
    block["dependency_resolution"] = {
        "skills": [dict.fromkeys(("ref", "state", "path", "declared_by", "reason"), text) for _ in range(count)]
    }
    other_count = 1 if long_cells else 100
    block["mcp"]["servers"] = [{"name": f"server-{index}"} for index in range(other_count)]
    block["context_cost"]["by_component"] = [{"name": f"context-{index}"} for index in range(other_count)]
    block["hook_risk"]["hooks"] = [
        {"id": f"flagged-hook-{index}", "event": "SessionStart", "risk_flags": ["context_injection"]}
        for index in range(5)
    ]
    block["privileges"]["components"][0].update(permission_mode="bypass_permissions", flags=["bypass_permissions"])
    before = deepcopy(results)

    output = renderers().render_plugin_static_reports(results, reporters=reporters())

    payload = json.loads(output["json"])["plugin_static"]
    sections = {section["key"]: section for section in payload["sections"]}
    assert {key: len(sections[key]["rows"]) for key in ("hooks", "privileges", "audits", "endpoints")} == {
        "hooks": 5,
        "privileges": 1,
        "audits": 1,
        "endpoints": 1,
    }
    assert sum(len(section["rows"]) for section in sections.values()) <= 256
    assert all(len(section["rows"]) <= 64 for section in sections.values())
    assert len(json.dumps(payload, ensure_ascii=False).encode()) <= 256 * 1024
    text_cells = [payload["name"]]
    for section in sections.values():
        text_cells.extend(section["notes"])
        text_cells.extend(cell for row in section["rows"] for cell in row)
    assert sum(len(cell.encode()) for cell in text_cells if isinstance(cell, str)) <= 48 * 1024
    assert payload["truncated"] is True
    if long_cells:
        assert sections["inventory"]["omitted"] > 0
    for section in sections.values():
        assert section["total"] == len(section["rows"]) + section["omitted"]
        assert all(any(cell not in ("", None) for cell in row) for row in section["rows"])
    for name, document in output.items():
        text = html.unescape(document).replace("\\_", "_")
        for marker in ("flagged-hook-4", "bypass_permissions", "audit capped", "remote-endpoint"):
            assert marker in text, (name, marker)
        assert "hidden-value" not in document
    assert json.loads(output["sarif"])["runs"][0]["properties"]["pluginStatic"] == payload
    assert results == before


def test_text_limit_omits_unrepresentable_rows_explicitly_in_all_formats(monkeypatch):
    monkeypatch.setattr(api(), "MAX_TEXT_BYTES", 100)
    result = ValidationResult(
        validator_name="Private static fixture",
        metadata={
            "plugin": {
                "name": "bounded",
                "component_inventory": {"components": [{"type": "skill", "name": "x" * 300}]},
            }
        },
    )

    output = renderers().render_plugin_static_reports([result], reporters=reporters())

    payload = json.loads(output["json"])["plugin_static"]
    inventory = next(section for section in payload["sections"] if section["key"] == "inventory")
    assert inventory["rows"] == []
    assert inventory["omitted"] == inventory["total"] == 1
    assert payload["truncated"] is True
    for name in ("cli", "markdown", "html", "benchmark"):
        assert "1 more rows omitted (1 total)." in output[name]
    assert json.loads(output["sarif"])["runs"][0]["properties"]["pluginStatic"] == payload


def test_endpoint_rows_use_only_the_authoritative_top_level_result_metadata():
    results = bundle()
    results[2].metadata["plugin"] = {"endpoint_resolution": results[2].metadata["endpoint_resolution"]}

    sections = api().project_plugin_static(results).payload()["sections"]

    endpoints = next(section for section in sections if section["key"] == "endpoints")
    assert endpoints["total"] == 1
    assert len(endpoints["rows"]) == 1
    assert endpoints["rows"][0][1] == "remote-endpoint"


def test_conflicting_source_identity_is_explicit_and_never_a_static_pass():
    results = bundle()[:1]
    other = ValidationResult(validator_name="Private Manifest and Component Checks")
    other.metadata["plugin"] = {"root": "/work/other", "manifest_sha256": "b" * 64}
    payload = api().project_plugin_static([*results, other]).payload()
    assert payload["status"] == "incomplete"
    assert "conflicting source" in json.dumps(payload).lower()


@pytest.mark.parametrize("block", [None, "high", "critical", "medium-policy", "minimum-score"])
def test_incomplete_only_gates_share_the_canonical_status_without_hiding_real_failures(block):
    schema = ValidationResult(validator_name="Plugin Schema", metadata={"plugin": {"name": "partial-demo"}})
    security = ValidationResult(validator_name="Security Scan")
    security.mark_scan_incomplete("skillspector")
    security.fail_gate("skillspector reports incomplete analysis: reference_unresolved at SKILL.md:4")
    security.add_finding(Finding("SECURITY", Severity.MEDIUM, "rug-pull", "Retained advisory", "SKILL.md"))
    results = [schema, security]
    if block == "minimum-score":
        quality = ValidationResult(validator_name="Quality Score")
        quality.fail_gate("Quality score 50.0/100 is below the required 70/100 in SKILL.md")
        results.append(quality)
    elif block is not None:
        security.add_structured_finding(
            Finding("SECURITY", "medium" if block == "medium-policy" else block, "known-risk", "Known risk", "run.py"),
            is_error=True,
        )

    output = renderers().render_plugin_static_reports(results, reporters=reporters())
    report = json.loads(output["json"])

    expected = "incomplete" if block is None else "failed"
    assert report["overall_status"] == expected
    assert report["plugin_static"]["status"] == expected
    assert report["results"][1]["findings"][0]["message"] == "Retained advisory"
    if block == "minimum-score":
        assert "Quality score 50.0/100 is below the required 70/100" in output["html"]
    elif block is not None:
        assert "Known risk" in output["html"]


def test_private_metadata_is_read_once_per_result_for_the_six_outputs():
    class CountedMetadata(dict):
        reads = 0

        def get(self, key, *default):
            if key == "plugin":
                self.reads += 1
            return super().get(key, *default)

    results = bundle()
    for result in results:
        result.metadata = CountedMetadata(result.metadata)
    renderers().render_plugin_static_reports(results, reporters=reporters())
    assert [result.metadata.reads for result in results] == [1, 1, 1, 1]


def test_nonfinite_extreme_and_shaped_values_stay_unknown_without_stringifying_objects():
    class Unprintable:
        def __str__(self):
            raise AssertionError("Untrusted objects must not be formatted")

    results = bundle()
    results[0].metadata["plugin"]["component_inventory"]["complete"] = Unprintable()
    costs = results[0].metadata["plugin"]["context_cost"]
    costs["always_on_tokens"] = float("nan")
    costs["by_component"][0]["on_demand_tokens"] = 10**10000
    costs["by_component"][0]["basis"] = Unprintable()
    payload = api().project_plugin_static(results).payload()
    context = next(section for section in payload["sections"] if section["key"] == "context")
    assert context["rows"][0][3] is None
    assert context["rows"][0][4] == "unavailable"
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_component_attribution_does_not_guess_shared_names_or_read_source_paths(monkeypatch):
    import os.path

    results = bundle()
    results[0].metadata["plugin"]["component_inventory"]["components"].append(
        {"type": "skill", "name": "loader", "path": "skills/other-loader", "support": "evaluated"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Retained static evidence must not inspect source paths")

    monkeypatch.setattr(os.path, "realpath", forbidden)
    projection = api().project_plugin_static(results)
    assert projection.component_for("[loader] SKILL.md") is None
    assert projection.component_for("skills/other-loader/SKILL.md")["path"] == "skills/other-loader"


def test_static_counts_and_support_are_consistent_with_recorded_checks():
    results = bundle()
    block = results[0].metadata["plugin"]
    block["component_inventory"]["components"].append(
        {"type": "hook", "name": "startup", "path": "hooks/hooks.json", "support": "unsupported"}
    )
    block["dependency_status_counts"] = {"provided": 2, "external": 1}
    payload = api().project_plugin_static(results).payload()
    sections = {section["key"]: section for section in payload["sections"]}
    assert sections["dependency_counts"]["rows"] == [["provided", 2], ["external", 1]]
    assert sections["mcp_pinning"]["rows"][0][:3] == [1, 0, 1]
    support = {row[1]: row[4] for row in sections["inventory"]["rows"]}
    assert support["startup"] == "statically checked"
    assert support["calendar-app"] == "inventoried only"


def test_legacy_severity_maps_preserve_scanners_and_disabled_audits_stay_unknown():
    results = bundle()
    ecosystems = results[1].metadata["plugin"]["cve_summary"]["ecosystems"]
    ecosystems["npm"] = {
        "status": "audited",
        "declarations": 5,
        "audited": 5,
        "scanners": ["scanner-fixture"],
        "vulnerabilities": {"high": 2, "low": 1},
    }
    ecosystems["container"] = {"status": "disabled", "sources": 0, "audited": 0, "vulnerabilities": 0}
    sections = api().project_plugin_static(results).payload()["sections"]
    audit = next(section for section in sections if section["key"] == "audits")
    assert "scanner-fixture" in json.dumps(audit)
    total_column = audit["columns"].index("Vulnerabilities")
    assert next(row for row in audit["rows"] if row[0] == "npm")[total_column] == 3
    assert next(row for row in audit["rows"] if row[0] == "container")[total_column] is None


def test_boolean_counts_are_not_reported_as_measurements():
    results = bundle()
    block = results[0].metadata["plugin"]
    block["context_cost"]["by_component"][0]["always_on_tokens"] = True
    block["mcp"]["pinning"]["total"] = True
    payload = api().project_plugin_static(results).payload()
    sections = {section["key"]: section for section in payload["sections"]}
    assert sections["context"]["rows"][0][2] is None
    assert sections["mcp_pinning"]["rows"][0][0] is None


def test_unrecognized_metadata_cannot_hide_known_blocks_after_a_visit_cap():
    result = bundle()[0]
    existing = result.metadata["plugin"]
    result.metadata["plugin"] = {**{f"unknown_{index}": "ignored" for index in range(500)}, **existing}
    payload = api().project_plugin_static([result]).payload()
    assert payload["name"] == "report-demo"
    assert any(section["key"] == "inventory" for section in payload["sections"])


def test_static_projection_and_adapters_add_no_optional_provider_imports():
    import subprocess
    import sys

    script = """
import json, sys
import skillevaluator.reporting
before = set(sys.modules)
import skillevaluator.reporting.plugin_static
import skillevaluator.reporting.plugin_static_render
new = set(sys.modules) - before
forbidden = ('litellm', 'anthropic', 'openai', 'harbor', 'skillevaluator.inference', 'skillevaluator.tier3', 'skillevaluator.validators')
assert not sorted(name for name in new if name.startswith(forbidden)), sorted(new)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_static_report_rendering_never_invokes_source_or_network_collectors(monkeypatch):
    import socket
    import subprocess

    from skillevaluator.plugin_paths import PluginRootReader

    def forbidden(*args, **kwargs):
        raise AssertionError("Report rendering cannot collect new evidence")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(PluginRootReader, "read_text", forbidden)
    rendered = renderers().render_plugin_static_reports(bundle(), reporters=reporters())
    assert len(rendered) == 6


def test_component_tags_preserve_ref_monitor_and_same_name_mcp_attribution():
    result = bundle()[0]
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    rows.extend(
        [
            {"type": "agent", "name": "review", "path": "agents/reviewer.md"},
            {"type": "monitor", "name": "watch", "path": "monitors/watch.json"},
            {"type": "mcp", "name": "shared", "path": "config/one.json"},
            {"type": "mcp", "name": "shared", "path": "config/two.json"},
        ]
    )
    view = api().project_plugin_static([result])
    assert (
        view.component_for(".claude-plugin/plugin.json", {"plugin_component_ref": "./agents/reviewer.md"})["name"]
        == "review"
    )
    assert (
        view.component_for("monitors/watch.json", {"plugin_component": {"type": "hook", "name": "monitor:watch"}})[
            "type"
        ]
        == "monitor"
    )
    assert view.component_for("config/two.json", {"mcp_server": "shared"})["path"] == "config/two.json"


def test_component_properties_are_sanitized_even_when_matching_uses_full_paths():
    result = bundle()[0]
    path = 'skills/API_KEY="private-path-secret"'
    result.metadata["plugin"]["component_inventory"]["components"] = [
        {"type": "skill", "name": "safe-name", "path": path}
    ]
    component = api().project_plugin_static([result]).component_for(path + "/SKILL.md")
    assert component["name"] == "safe-name"
    assert "private-path-secret" not in json.dumps(component)


def test_terminal_control_removal_cannot_reassemble_a_credential_in_static_views():
    results = bundle()
    secret = "ghp_" + "B" * 36
    disguised = "ghp_" + "B" * 18 + "\x1b[31m" + "B" * 18
    results[0].metadata["plugin"]["manifest_declarations"][0]["name"] = disguised
    assert secret not in json.dumps(api().project_plugin_static(results).payload())
    for document in renderers().render_plugin_static_reports(results, reporters=reporters()).values():
        assert secret not in document


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://user:password@example.com:8443/check?token=secret#fragment", "https://example.com:8443/check"),
        ("http://example.com:8080/check", "http://example.com:8080/check"),
        ("https://[2606:4700::1111]:8443/check?token=secret", "https://[2606:4700::1111]:8443/check"),
        ("https://user:password@[2606:4700::1111]/check#fragment", "https://[2606:4700::1111]/check"),
        ("https://example.com:wrong/check", "unavailable"),
        ("https://example.com:65536/check", "unavailable"),
        ("https:///check", "unavailable"),
    ],
)
def test_endpoint_and_redirect_urls_preserve_valid_authorities_without_credentials(url, expected):
    results = bundle()
    endpoint = results[2].metadata["endpoint_resolution"]["endpoints"][0]
    endpoint["url"] = url
    endpoint["redirect"]["url"] = url
    payload = api().project_plugin_static(results).payload()
    section = next(row for row in payload["sections"] if row["key"] == "endpoints")
    assert section["rows"][0][2] == expected
    assert section["rows"][0][5] == expected
