# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic plugin report payloads shaped like the shared plugin contracts (test helpers).

Tier 1 metadata (dependencies, components, MCP, context cost), Tier 2
local-catalog similarity, Tier 3 provenance with component coverage, per-arm
plugin signals, statistics, and the Integration block. Every producer lands in
a separate change, so these fixtures are the renderer's reference shapes.
"""

from __future__ import annotations

import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

HOSTILE = "<script>alert('x')</script>"

TIER1_PLUGIN_METADATA: dict[str, Any] = {
    "manifest_type": "agent_plugin",
    "plugin_mode": "bundle",
    "plugin": {
        "manifest_filename": "agent_plugin.yaml",
        "root": "/work/demo-plugin",
        "name": "demo-plugin",
        "declared_dependencies": {"skills": 2, "rules": 1, "mcp": 2},
        "dependency_resolution": {
            "skills": [
                {"ref": "skills/loader", "state": "provided", "path": "skills/loader", "reason": "bundled"},
                {"ref": "github::org/repo::skills::remote", "state": "external", "path": None, "reason": "cross-repo"},
            ],
            "rules": [{"ref": "rules/missing.md", "state": "missing", "path": None, "reason": f"not found {HOSTILE}"}],
        },
        "dependency_status_counts": {"provided": 1, "referenced": 0, "missing": 1, "external": 1, "unresolved": 0},
        "bundled_skills": ["skills/loader"],
        "in_plugin_skills": 1,
        "component_inventory": {
            "components": [
                {
                    "type": "skill",
                    "name": "loader",
                    "origin": "declared+packaged",
                    "path": "skills/loader",
                    "support": "evaluated",
                    "findings": 1,
                },
                {
                    "type": "mcp",
                    "name": "search",
                    "origin": "declared",
                    "path": None,
                    "support": "static_only",
                    "findings": 0,
                },
                {
                    "type": "hook",
                    "name": "pre-commit",
                    "origin": "packaged",
                    "path": "hooks/pre.sh",
                    "support": "unsupported",
                    "findings": 0,
                },
            ],
            "counts": {"skill": 1, "mcp": 1, "hook": 1},
            "unsupported_types_present": ["hook"],
        },
        "mcp": {
            "servers": [
                {
                    "name": "search",
                    "source": "inline",
                    "kind": "command",
                    "transport": "stdio",
                    "pinned": False,
                    "pin_detail": "npx search-server@latest",
                },
                {
                    "name": "docs",
                    "source": "mcp_json",
                    "kind": "url",
                    "transport": "http",
                    "pinned": True,
                    "pin_detail": "versioned URL",
                },
            ],
            "pinning": {"total": 2, "pinned": 1, "unpinned": 1, "not_applicable": 0, "ratio": 0.5},
        },
        "context_cost": {
            "method": "static_estimate",
            "estimator": "chars_div_4",
            "always_on_tokens": 1200,
            "on_demand_tokens": 3400,
            "by_component": [
                {
                    "type": "skill",
                    "name": "loader",
                    "always_on_tokens": 200,
                    "on_demand_tokens": 3000,
                    "basis": "SKILL.md frontmatter and body",
                }
            ],
            "notes": ["Rules load in every session."],
        },
    },
}

TIER2_PLUGIN_METADATA: dict[str, Any] = {
    "catalog_skill_similarity": {
        "status": "compared",
        "catalog_entries": 12,
        "matches": [{"skill": "loader", "match": "catalog-loader", "similarity": 0.91}],
        "reason": None,
    },
    "inter_plugin_similarity": {
        "status": "compared",
        "catalog_entries": 3,
        "matches": [{"name": "other-plugin", "similarity": 0.84, "member_overlap": 0.5, "verdict": "overlapping"}],
        "reason": None,
    },
}

COMPONENT_COVERAGE: dict[str, Any] = {
    "components": [
        {
            "type": "skill",
            "name": "loader",
            "origin": "declared+packaged",
            "path": "skills/loader",
            "state": "staged",
            "reason": "",
        },
        {"type": "mcp", "name": "search", "origin": "declared", "path": None, "state": "staged", "reason": ""},
        {
            "type": "mcp",
            "name": "docs",
            "origin": "declared",
            "path": None,
            "state": "unavailable",
            "reason": "provider-only MCP server",
        },
        {
            "type": "hook",
            "name": "pre-commit",
            "origin": "packaged",
            "path": "hooks/pre.sh",
            "state": "unsupported",
            "reason": "hooks are not evaluated",
        },
    ],
    "counts": {"staged": 2, "unavailable": 1, "unsupported": 1},
    "not_evaluated": 2,
}

SIGNALS_SUMMARY: dict[str, Any] = {
    "n_trials": 3,
    "n_missing_trajectory": 1,
    "activations": {"total": 7, "mean_per_trial": 2.33, "by_type": {"skill": 4, "mcp": 3}},
    "tool_selection": {
        "n_scored": 3,
        "precision": 0.8,
        "recall": 0.67,
        "f1": 0.73,
        "decoy_calls": 1,
        "decoy_call_rate": 0.1,
        "status": "scored",
    },
    "arguments": {"n_scored": 2, "checked": 4, "passed": 3, "pass_rate": 0.75, "status": "scored"},
    "handoff": {"n_scored": 0, "checked": 0, "passed": 0, "pass_rate": None, "status": "not_applicable"},
    "conflict": {"n_scored": 1, "checked": 1, "passed": 1, "pass_rate": 1.0, "status": "scored"},
    "mcp_calls": {
        "total": 5,
        "succeeded": 4,
        "failed": 1,
        "unknown": 0,
        "success_rate": 0.8,
        "by_server": {
            "search": {
                "total": 5,
                "succeeded": 4,
                "failed": 1,
                "unknown": 0,
                "tools": ["mcp__search__query"],
                "success_rate": 0.8,
            }
        },
    },
    "order": {"n_scored": 2, "edges": 2, "satisfied": 1, "satisfaction_rate": 0.5, "status": "scored"},
    "activation_coverage": {
        "declared": ["skill:loader", "mcp:search", "hook:pre-commit"],
        "exercised": ["skill:loader", "mcp:search"],
        "unverified": [],
        "unavailable": ["hook:pre-commit"],
        "exercise_rate": 0.67,
    },
}

STATISTICS: dict[str, Any] = {
    "lift_uncertainty": {
        "effectiveness": {
            "estimate": 0.3,
            "ci_low": -0.02,
            "ci_high": 0.55,
            "confidence": 0.95,
            "method": "paired_case_bootstrap",
            "resamples": 2000,
            "seed": 0,
            "n_cases": 4,
            "precision": "low",
            "ci_includes_zero": True,
        }
    },
    "reliability": {
        "with_skill": {"pass_at_k": 0.75, "pass_hat_k": 0.5, "k": 3, "n_cases": 4},
        "without_skill": {"pass_at_k": 0.5, "pass_hat_k": 0.25, "k": 3, "n_cases": 4},
    },
    "cost": {
        "with_skill": {
            "tokens_per_success": 12000.0,
            "usd_per_success": None,
            "total_tokens": 36000,
            "total_usd": None,
            "successes": 3,
        }
    },
    "token_efficiency": {"with_skill": 0.62, "without_skill": 0.41},
    "context_cost_measured": {
        "method": "paired_first_turn_prompt_tokens",
        "delta_tokens_mean": 1450.0,
        "n_pairs": 4,
        "status": "measured",
        "reason": "",
    },
}


def provenance(*, partial: bool = True, coverage: bool = True) -> dict[str, Any]:
    """Return a Tier 3 plugin provenance record (C2 shape)."""
    record: dict[str, Any] = {
        "plugin_name": "demo-plugin",
        "evaluated_member_skills": ["loader"],
        "staged_rules": [],
        "runnable_mcp_servers": ["search"],
        "unresolved_skill_refs": ["github::org/repo::skills::remote"] if partial else [],
        "unresolved_rule_refs": [],
        "provider_only_mcp_servers": ["docs"] if partial else [],
        "mcp_unsupported_config": [],
        "dataset_case_count": 4,
        "cross_component_case_count": 2,
        "integration_evidence_ready": True,
        "partial": partial,
        "context_cost": deepcopy(TIER1_PLUGIN_METADATA["plugin"]["context_cost"]),
        "mcp_pinning": {"total": 2, "pinned": 1, "unpinned": 1, "not_applicable": 0, "ratio": 0.5},
        "dependency_status_counts": {"provided": 1, "referenced": 0, "missing": 0, "external": 1, "unresolved": 0},
    }
    if coverage:
        record["component_coverage"] = deepcopy(COMPONENT_COVERAGE)
        if not partial:
            for component in record["component_coverage"]["components"]:
                component["state"] = "staged"
                component["reason"] = ""
            record["component_coverage"]["counts"] = {"staged": 4}
            record["component_coverage"]["not_evaluated"] = 0
    return record


def write_run_dir(
    run_dir: Path,
    *,
    signals: bool = True,
    sidecar: dict[str, Any] | None = None,
    with_score: float = 0.8,
    baseline_score: float = 0.5,
) -> Path:
    """Write a minimal plugin Harbor run: two arms, one trial, optional sidecar."""
    for variant, score in (("with-skill", with_score), ("without-skill", baseline_score)):
        summary = run_dir / "codex" / variant / "summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "scores": dict.fromkeys(DEFAULT_METRICS, score),
            "metrics": list(DEFAULT_METRICS),
            "num_trials": 1,
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 1,
            "scored_attempts": 1,
        }
        if signals and variant == "with-skill":
            data["plugin_signals_summary"] = deepcopy(SIGNALS_SUMMARY)
        summary.write_text(json.dumps(data), encoding="utf-8")
    trial = run_dir / "codex" / "with-skill" / "trials" / "case-1__attempt1"
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "reward.json").write_text(
        json.dumps(
            {
                "entry_id": "case-1",
                "overall": with_score,
                "plugin_signals": {
                    "arguments": {
                        "failures": [
                            {"tool": "mcp__search__query", "arg": "q", "rule": "required", "detail": "missing q"},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"eval_target": {"kind": "plugin"}, "agents": {"codex": {"model": "gpt-test"}}}),
        encoding="utf-8",
    )
    if sidecar is not None:
        (run_dir / "plugin_provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return run_dir


def tier1_plugin_result(*, finding_path: str = "/work/demo-plugin/skills/loader/SKILL.md") -> ValidationResult:
    result = ValidationResult(validator_name="Plugin Schema", validator_description="Tier 1 plugin validation")
    result.add_success("plugin_manifest", "Plugin manifest 'demo-plugin' is valid")
    result.add_finding(
        Finding(
            category="PLUGIN_SCHEMA",
            severity=Severity.MEDIUM,
            check_name="description_short",
            message="Loader description is short",
            file_path=finding_path,
            line_number=2,
        )
    )
    result.metadata.update(deepcopy(TIER1_PLUGIN_METADATA))
    return result


def tier2_plugin_result() -> ValidationResult:
    result = ValidationResult(validator_name="Plugin Dependency Deduplication", validator_description="Tier 2")
    result.add_success("plugin_dep_dedup", "No duplicate dependency references")
    result.metadata["plugin"] = deepcopy(TIER2_PLUGIN_METADATA)
    return result


class _ElementText(HTMLParser):
    _VOID = frozenset({"br", "img", "input", "meta", "link", "hr", "area", "base", "col", "embed", "source", "wbr"})

    def __init__(self, element_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.element_id = element_id
        self.depth = 0
        self.found = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._VOID:
            return
        if self.depth:
            self.depth += 1
        elif dict(attrs).get("id") == self.element_id and not self.found:
            self.found = True
            self.depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self.depth and tag not in self._VOID:
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth:
            self.parts.append(data)


def element_text(html: str, element_id: str) -> str | None:
    """Return the whitespace-collapsed text of the element with *element_id*, if present."""
    parser = _ElementText(element_id)
    parser.feed(html)
    return " ".join(" ".join(parser.parts).split()) if parser.found else None
