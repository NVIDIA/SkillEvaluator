# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- trusted CI reporting helper.
"""Compose source-bound cards from retained JSON, without opening skill sources."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from pathlib import Path, PurePosixPath

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, JSONReporter
from skillevaluator.reporting.retained import retained_results
from skillevaluator.reporting.run_status import resolve_run_status
from skillevaluator.source_identity import (
    EVALUATED_SOURCE_FIELDS,
    EvaluatedSourceConflict,
    evaluated_source_carriers,
    merge_evaluated_sources,
    normalized_evaluated_source,
)

try:
    from github_api import strict_json
except ModuleNotFoundError:
    from ci.github_api import strict_json

_MAX_REPORT_BYTES = 32 * 1024 * 1024


def _report_path(reports: Path, relative: str) -> Path:
    """Accept only canonical relative paths through regular retained directories."""
    if (
        not isinstance(relative, str)
        or not relative
        or any(character in relative for character in "\\:%?#")
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
        or PurePosixPath(relative).as_posix() != relative
    ):
        raise ValueError("Unsafe retained report path")
    path = reports
    for part in relative.split("/"):
        path /= part
        if path.is_symlink():
            raise ValueError("Linked retained report path")
    return path


def _load_payload(reports: Path, relative: str) -> dict:
    path = _report_path(reports, relative)
    if not path.is_file() or path.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError("Missing or oversized retained report")
    return strict_json(path.read_bytes(), ValueError, "Invalid retained JSON")


def _incomplete(tier: int) -> ValidationResult:
    """Record transport/execution gaps without certifying a native check."""
    name = {1: "CI Tier 1 Evidence", 2: "CI Similarity Evidence", 3: "CI Tier 3 Evidence"}[tier]
    metadata = {
        "incomplete_scans": [f"Tier {tier} retained evidence is missing, invalid or incomplete."],
        "gating": {"tier": tier, "blocking": tier == 1},
    }
    if tier == 3:
        metadata["agent_eval"] = {}
    return ValidationResult(validator_name=name, passed=False, metadata=metadata)


def _bind_identity(results: list[ValidationResult], skill: str) -> str | None:
    """Preserve claims exactly and fail closed even on case-only source conflicts."""
    targets = []
    for result in results:
        metadata = result.metadata
        containers = [metadata]
        agent = metadata.get("agent_eval")
        if isinstance(agent, dict):
            containers += [agent, agent.get("summary", {})]
        targets.extend(container["publication_target"] for container in containers if "publication_target" in container)
    names = [target.get("skill_name") for target in targets if isinstance(target, dict)]
    expected = unicodedata.normalize("NFC", PurePosixPath(skill).name) if skill != "." else next(iter(names), None)
    if targets and (any(target != targets[0] for target in targets) or any(name != expected for name in names)):
        for result in results:
            result.metadata["publication_target_conflict"] = "Retained tiers identify conflicting source snapshots."
    return expected


def _bind_evaluated_source(results: list[ValidationResult], manifest: dict) -> list[dict]:
    """Bind trusted run context; retain conflicting claims separately and fail closed.

    Reporters reject contradictory source carriers. The working render copy
    therefore moves those carriers into an explicit conflict record instead of
    silently replacing them. Canonical native inputs are never mutated.
    """
    source = {"repository": manifest.get("repository"), "commit": manifest.get("commit")}
    runtime = manifest.get("runtime")
    revision = runtime.get("tools_ref") if isinstance(runtime, dict) else None
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        source["evaluator_container_revision"] = revision
    trusted = normalized_evaluated_source(source)
    claims = [
        carrier for result in results for carrier in evaluated_source_carriers(result.metadata) if carrier is not None
    ]
    try:
        for claim in claims:
            normalized = normalized_evaluated_source(claim) or {}
            if not isinstance(claim, dict) or any(
                key in claim and key not in normalized for key, _predicate in EVALUATED_SOURCE_FIELDS
            ):
                raise EvaluatedSourceConflict("Malformed retained evaluated-source claim")
        bound = merge_evaluated_sources([trusted, *claims])
    except EvaluatedSourceConflict:
        conflicts = copy.deepcopy(claims)
        for result in results:
            metadata = result.metadata
            containers = [metadata]
            agent = metadata.get("agent_eval")
            if isinstance(agent, dict):
                containers += [agent, agent.get("summary", {})]
            for container in containers:
                container.pop("evaluated_source", None)
            metadata["publication_target_conflict"] = "Retained evaluated-source claims contradict the trusted run."
            if trusted:
                metadata["evaluated_source"] = dict(trusted)
        return conflicts
    if bound:
        for result in results:
            # This merge adds only compatible fields; original native values
            # and nested payload claims remain unchanged when they agree.
            result.metadata["evaluated_source"] = dict(bound)
    return []


def _native_status(results: list[ValidationResult], tier: int) -> str:
    status = resolve_run_status(results).tiers[tier].status.value
    if tier == 3 and status == "passed":
        for result in results:
            agent = result.metadata.get("agent_eval", {})
            summary = agent.get("summary", {})
            if agent.get("verdict", summary.get("verdict")) == "neutral":
                return "neutral"
    return status


def benchmark_documents(manifest: dict, native_reports: dict[str, dict]) -> list[dict]:
    """Compose trusted native JSON in memory; never trust artifact Markdown.

    The caller validates the CI manifest and bounds the native JSON mapping.
    Rows contain link metadata plus ``combined_content`` and
    ``benchmark_content`` strings. Policy and provenance logic is shared with
    the filesystem writer; CI gate flags never grant publication waivers.
    """
    grouped: dict[str, dict[int, dict]] = {}
    for tier in manifest["tiers"]:
        number = tier["tier"]
        for skill in tier["skills"]:
            grouped.setdefault(skill["path"], {})[number] = skill
        for selected in tier.get("selected", []):
            grouped.setdefault(selected, {}).setdefault(number, {"status": "incomplete", "report_json": None})
    documents = []
    for index, (skill, tiers) in enumerate(sorted(grouped.items()), 1):
        results = []
        for number in (1, 2, 3):
            row = tiers.get(number)
            if row is None:
                if number == 1:
                    results.append(_incomplete(number))
                continue
            retained = []
            try:
                retained = retained_results(native_reports[row.get("report_json")], allowed_tiers={number})
            except (ValueError, TypeError, KeyError):
                # Worker-controlled errors may contain secrets. Keep fixed diagnostics.
                if row.get("status") != "skipped" or row.get("report_json") is not None:
                    retained.append(_incomplete(number))
            contradictory_status = bool(retained) and row.get("status") != _native_status(retained, number)
            missing_lineup = number == 1 and any(
                result.metadata.get("publication_evidence", {}).get("tier") == 1
                and "tier1_required_checks" not in result.metadata
                for result in retained
            )
            if (
                retained
                and (
                    row.get("status") not in {"passed", "failed", "neutral", "skipped"}
                    or row.get("error")
                    or contradictory_status
                    or missing_lineup
                )
                and not any(result.is_incomplete for result in retained)
            ):
                retained.append(_incomplete(number))
            # Process gating is independent of the native publication policy.
            for result in retained:
                result.metadata["gating"] = {"tier": number, "blocking": number == 1}
            results.extend(retained)
        expected = _bind_identity(results, skill)
        conflicts = _bind_evaluated_source(results, manifest)
        combined = JSONReporter(include_timestamp=False, expected_skill_name=expected).render_all(results)
        if conflicts:
            payload = json.loads(combined)
            payload["evaluated_source_conflicts"] = conflicts
            combined = json.dumps(payload, indent=2, ensure_ascii=False)
        card = BenchmarkReporter(include_timestamp=False, skill_name=expected).render_all(results)
        directory = f"benchmarks/skill-{index:03}"
        paths = {"combined_json": f"{directory}/skillevaluator.json", "benchmark_md": f"{directory}/BENCHMARK.md"}
        publication = json.loads(combined)["publication"]
        documents.append(
            {
                "path": skill,
                **paths,
                "publication_status": publication["status"],
                "eligible": publication["eligible"],
                "combined_content": combined,
                "benchmark_content": card,
            }
        )
    return documents


def write_benchmark_reports(reports: Path, manifest: dict) -> list[dict]:
    """Write deterministic per-skill reports and return only their link metadata."""
    native_reports = {}
    for tier in manifest["tiers"]:
        for row in tier["skills"]:
            relative = row.get("report_json")
            if relative is not None and relative not in native_reports:
                try:
                    native_reports[relative] = _load_payload(reports, relative)
                except (OSError, ValueError, TypeError, KeyError):
                    # Absence is turned into fixed incomplete evidence by the composer.
                    continue
    links = []
    for document in benchmark_documents(manifest, native_reports):
        metadata = dict(document)
        for field, content_key in (("combined_json", "combined_content"), ("benchmark_md", "benchmark_content")):
            destination = _report_path(reports, metadata[field])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(metadata.pop(content_key), encoding="utf-8")
        links.append(metadata)
    return links
