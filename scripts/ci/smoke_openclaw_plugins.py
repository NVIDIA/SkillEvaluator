# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat real OpenClaw manifest/profile/portable-staging checks without executing plugins.

Use local, pristine source trees (for example git archives). This is a pipeline
compatibility smoke, not a substitute for full static scanners, embeddings,
agent evaluation, or native OpenClaw runtime tests. Reports contain relative
identities only, so they can be attached to a public PR without host paths.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

from skillevaluator.constants import PLUGIN_OPENCLAW_MANIFEST_TYPE
from skillevaluator.deduplication.plugin.profile import load_plugin_profile
from skillevaluator.plugin_formats import parse_manifest_text
from skillevaluator.plugin_manifest import locate_plugin_manifest
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.utils.secure_fs import secure_atomic_write_text
from skillevaluator.validators.plugin_schema import PluginSchemaValidator


def inspect_plugin(root: Path) -> dict[str, Any]:
    """Check agreement across all three layers; scanner findings are not hidden."""
    location = locate_plugin_manifest(root)
    if location is None or location.manifest_type != PLUGIN_OPENCLAW_MANIFEST_TYPE:
        raise ValueError("Expected a selected native OpenClaw manifest")
    data = parse_manifest_text(location.manifest_type, location.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected a manifest object")
    schema = PluginSchemaValidator().validate(root)
    metadata = schema.metadata["plugin"]
    if metadata.get("name") != data.get("id"):
        raise AssertionError("Tier 1 identity differs from native id")
    invalid_contract = [
        finding.check_name
        for finding in schema.findings
        if finding.check_name.startswith(("schema:id:", "schema:configSchema:"))
    ]
    if invalid_contract:
        raise AssertionError(f"Invalid native contract: {invalid_contract}")
    profile = load_plugin_profile(root)
    if profile.name != data["id"]:
        raise AssertionError("Tier 2 identity differs from native id")
    expected = sorted(skill.root_relative for skill in profile.bundled_skills)
    components = metadata["component_inventory"]["components"]
    inventoried = sorted(row["path"] for row in components if row["type"] == "skill" and row["support"] == "evaluated")
    if inventoried != expected:
        raise AssertionError("Tier 1 and Tier 2 disagree about declared skill membership")
    with tempfile.TemporaryDirectory(prefix="openclaw-portable-smoke-") as temp:
        temporary = Path(temp)
        dataset = temporary / "evals.json"
        dataset.write_text(
            json.dumps(
                {
                    "skill_name": profile.name,
                    "evals": [
                        {
                            "id": "portable-stage",
                            "prompt": "Read the available guidance; do not call external services.",
                            "expected_output": "Describe the available portable guidance without running plugin code.",
                            "files": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        package = prepare_plugin_eval_package(root, stage_root=temporary / "stage", evals_source=dataset)
        staged = sorted(path.relative_to(root.resolve()).as_posix() for path in package.include_skills)
        if staged != expected:
            raise AssertionError("Tier 3 staged membership differs from the native declared skills")
        coverage = package.provenance()["component_coverage"]["components"]
        runtime = [row for row in coverage if row["type"] == "extension"]
        if not runtime or any(row["state"] != "unsupported" for row in runtime):
            raise AssertionError("Portable staging must never claim native OpenClaw runtime execution")
        if expected and package.skipped:
            raise AssertionError("A plugin with portable skills was unexpectedly skipped")
        if not expected and not data.get("mcpServers") and not package.skipped:
            raise AssertionError("A runtime-only plugin must not silently pass as evaluated")
        staged_skills = {row["path"] for row in coverage if row["type"] == "skill" and row["state"] == "staged"}
        if not set(expected).issubset(staged_skills):
            raise AssertionError("Portable member coverage is missing staged skills")
        skipped = package.skipped
    return {
        "plugin_id": profile.name,
        "manifest_type": location.manifest_type,
        "declared_skill_paths": expected,
        "tier1_schema_passed": schema.passed,
        "tier1_findings": dict(sorted(Counter(f.check_name for f in schema.findings).items())),
        "tier2_profile_members": list(profile.members),
        "portable_tier3_skipped": skipped,
        "runtime_components_unsupported": len(runtime),
        "native_runtime_executed": False,
    }


def run_smoke(plugins: dict[str, Path], *, iterations: int, trace_memory: bool = False) -> dict[str, Any]:
    """Require repeatable identities, findings, membership and coverage."""
    if not 1 <= iterations <= 200:
        raise ValueError("iterations must be between 1 and 200")
    results: dict[str, Any] = {}
    if trace_memory:
        tracemalloc.start()
    peak = None
    try:
        for label, root in plugins.items():
            durations = []
            reference = None
            for _attempt in range(iterations):
                started = time.monotonic()
                observed = inspect_plugin(root)
                durations.append(time.monotonic() - started)
                if reference is not None and observed != reference:
                    raise AssertionError(f"Non-deterministic plugin result: {label}")
                reference = observed
            ordered = sorted(durations)
            results[label] = {
                **reference,
                "iterations_passed": len(durations),
                "seconds_median": round(statistics.median(durations), 4),
                "seconds_p95": round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 4),
                "seconds_max": round(max(durations), 4),
            }
        if trace_memory:
            _current, peak = tracemalloc.get_traced_memory()
    finally:
        if trace_memory:
            tracemalloc.stop()
    return {
        "scope": "native metadata, schema, local profile and portable staging only",
        "live_agent_executed": False,
        "native_runtime_executed": False,
        "observations_passed": len(plugins) * iterations,
        "python_peak_traced_bytes": peak,
        "plugins": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace-memory", action="store_true", help="Enable allocation profiling (slower)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plugins = {}
    for value in args.plugin:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in plugins:
            parser.error("Each --plugin must have a unique LABEL=PATH")
        plugins[label] = Path(path).resolve()
    result = run_smoke(plugins, iterations=args.iterations, trace_memory=args.trace_memory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.parent.resolve() / args.output.name
    secure_atomic_write_text(output, json.dumps(result, indent=2) + "\n", 4 * 1024 * 1024)
    print(f"Passed {result['observations_passed']} pipeline observations across {len(plugins)} plugins")


if __name__ == "__main__":
    main()
