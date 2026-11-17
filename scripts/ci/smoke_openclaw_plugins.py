#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat real OpenClaw manifest, profile and portable-staging checks without executing plugins.

Use local, pristine source trees (for example ``git archive`` exports without
dependencies or build artifacts). This is a pipeline compatibility smoke over
the private plugin helpers, not a substitute for full static scanners,
embeddings, agent evaluation, or native OpenClaw runtime tests. The JSON
report holds relative identities only, so it can be attached to a public PR
without host paths.

    uv run --extra tier3 python scripts/ci/smoke_openclaw_plugins.py \\
        --plugin kitchen-sink=/path/to/pristine/kitchen-sink --iterations 50 --output ./openclaw-smoke.json
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
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.plugin_manifest import locate_plugin_manifest
from skillevaluator.tier3.plugin_native import OPENCLAW_RUNTIME_NOT_LOADED
from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill
from skillevaluator.validators.private_plugin_schema import validate_inventory_schema

MAX_ITERATIONS = 200
MAX_REPORT_BYTES = 4 * 1024 * 1024


def inspect_plugin(root: Path) -> dict[str, Any]:
    """Check that Tier 1, Tier 2 and Tier 3 agree; scanner findings are reported, never hidden."""
    location = locate_plugin_manifest(root)
    if location is None or location.manifest_type != PLUGIN_OPENCLAW_MANIFEST_TYPE:
        raise ValueError("Expected a selected native OpenClaw manifest")
    data = parse_manifest_text(location.manifest_type, location.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected a manifest object")
    inventory = plugin_inventory_for_root(root)
    if inventory is None:
        raise ValueError("Expected a plugin inventory")
    schema = validate_inventory_schema(inventory)
    metadata = schema.metadata["plugin"]
    [declared] = [row for row in metadata["manifest_declarations"] if row["selected"]]
    if declared["name"] != data.get("id"):
        raise AssertionError("Tier 1 identity differs from the native id")
    invalid_contract = [
        finding.check_name
        for finding in schema.findings
        if finding.check_name.startswith(("schema:id:", "schema:configSchema:"))
    ]
    if invalid_contract:
        raise AssertionError(f"Invalid native contract: {invalid_contract}")
    profile = load_plugin_profile(root)
    if profile.name != data["id"]:
        raise AssertionError("Tier 2 identity differs from the native id")
    expected = sorted(skill.root_relative for skill in profile.bundled_skills)
    inventoried = sorted(
        row["path"]
        for row in metadata["component_inventory"]["components"]
        if row["type"] == "skill"
        and row["support"] == "evaluated"
        and "declared_by" not in row
        and "problem" not in row
    )
    if inventoried != expected:
        raise AssertionError("Tier 1 and Tier 2 disagree about declared skill membership")
    # A root SKILL.md remains inventoried, but copying the whole plugin as a
    # portable member would expose runtime metadata and evaluator data.
    portable_expected = [path for path in expected if path != "."]
    package = prepare_runtime_package(inventory)
    staged = sorted(str(member.path) for member in package.skills)
    if staged != portable_expected:
        raise AssertionError("Tier 3 staged membership differs from the native declared skills")
    native_members = sorted(path.relative_to(inventory.root).as_posix() for path in package.native_source.member_skills)
    if native_members != portable_expected:
        raise AssertionError("Native adapters would stage a different member set")
    runtime = [row for row in package.audit if row.type == "extension"]
    if not runtime or any(row.reason != OPENCLAW_RUNTIME_NOT_LOADED for row in runtime):
        raise AssertionError("Portable staging must never claim native OpenClaw runtime execution")
    if not package.runnable and package.complete:
        raise AssertionError("A native OpenClaw package with nothing to evaluate must be INCOMPLETE")
    if portable_expected and not package.runnable:
        raise AssertionError("A plugin with portable skills was unexpectedly not runnable")
    if not portable_expected and not data.get("mcpServers") and package.runnable:
        raise AssertionError("A runtime-only plugin must not silently pass as evaluated")
    wrapper_members: list[str] = []
    if package.runnable and package.skills:
        with tempfile.TemporaryDirectory(prefix="openclaw-portable-smoke-") as temp:
            temporary = Path(temp)
            dataset = temporary / "evals.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "portable-stage",
                            "prompt": "Read the available guidance; do not call external services.",
                            "expected_output": "Describe the portable guidance without running plugin code.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            wrapper = stage_wrapper_skill(package, temporary / "wrapper", evals_source=dataset)
            wrapper_members = sorted(path.name for path in (wrapper.path / "skills").iterdir())
        if wrapper_members != sorted(Path(path).name for path in portable_expected):
            raise AssertionError("The wrapper staged a different member set")
    return {
        "plugin_id": profile.name,
        "manifest_type": location.manifest_type,
        "declared_skill_paths": expected,
        "portable_tier3_skill_paths": staged,
        "tier1_schema_passed": schema.passed,
        "tier1_findings": dict(sorted(Counter(finding.check_name for finding in schema.findings).items())),
        "tier2_profile_members": list(profile.members),
        "portable_tier3_runnable": package.runnable,
        "portable_tier3_incomplete_reasons": list(package.incomplete_reasons),
        "runtime_components_unsupported": len(runtime),
        "native_runtime_executed": False,
    }


def run_smoke(plugins: dict[str, Path], *, iterations: int, trace_memory: bool = False) -> dict[str, Any]:
    """Require repeatable identities, findings, membership and coverage."""
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
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
                **(reference or {}),
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
        "scope": "native metadata, private schema, local profile and portable staging only",
        "live_agent_executed": False,
        "native_runtime_executed": False,
        "observations_passed": len(plugins) * iterations,
        "python_peak_traced_bytes": peak,
        "plugins": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace-memory", action="store_true", help="Enable allocation profiling (slower)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plugins: dict[str, Path] = {}
    for value in args.plugin:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in plugins:
            parser.error("Each --plugin must have a unique LABEL=PATH")
        plugins[label] = Path(path).resolve()
    result = run_smoke(plugins, iterations=args.iterations, trace_memory=args.trace_memory)
    text = json.dumps(result, indent=2) + "\n"
    if len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise SystemExit(f"Smoke report exceeds {MAX_REPORT_BYTES} bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Passed {result['observations_passed']} pipeline observations across {len(plugins)} plugins")


if __name__ == "__main__":
    main()
