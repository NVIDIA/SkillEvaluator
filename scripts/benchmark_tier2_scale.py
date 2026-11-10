#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible synthetic Tier 2 scale/limit measurements; never calls a provider.

Run with the project's Python environment. Results measure the real discovery,
registry, all-pairs, serialization and saved-candidate paths, but synthetic
one-hot vectors do not establish production embedding latency or match quality.
Each stage records Python allocation peaks (tracemalloc slows computation).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from skillevaluator.constants import SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS
from skillevaluator.embedding.extractor import (
    MAX_COLLECTION_BYTES,
    MAX_DISCOVERED_PATHS,
    MAX_MANIFEST_BYTES,
    discover_and_extract_detailed,
    extract_from_skill,
)
from skillevaluator.embedding.registry import (
    MAX_CATALOG_BYTES,
    MAX_SIMILARITY_MATCHES,
    MAX_VECTOR_DIMENSION,
    EmbeddingRegistry,
)


class SyntheticClient:
    """Deterministic sparse vectors for exercising limits, with zero paid usage."""

    model = "synthetic-one-hot-v1"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.embedding_calls = 0
        self.embedding_inputs = 0
        self.input_utf8_bytes = 0

    def _resolved_config(self) -> SimpleNamespace:
        return SimpleNamespace(provider="synthetic", base_url="https://synthetic.invalid/v1")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedding_calls += 1
        self.embedding_inputs += len(texts)
        self.input_utf8_bytes += sum(len(text.encode("utf-8")) for text in texts)
        vectors = []
        for text in texts:
            match = re.search(r"skill-(\d+)", text)
            index = int(match[1]) if match else 0
            vector = [0.0] * self.dimension
            vector[index % self.dimension] = 1.0
            vectors.append(vector)
        return vectors

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_chunked(self, text: str) -> list[float]:
        # Deliberately one synthetic vector; does not simulate production token
        # chunking, pooling, HTTP calls, token usage or provider batch latency.
        return self.embed_single(text)


def measure(stages: dict[str, Any], name: str, action):
    gc.collect()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        value = action()
    except (ValueError, OSError) as exc:
        stages[name] = {"status": "rejected", "reason": str(exc)}
        value = None
    else:
        stages[name] = {"status": "ok"}
    stages[name].update(seconds=time.perf_counter() - started, peak_python_bytes=tracemalloc.get_traced_memory()[1])
    return value


def populate(root: Path, size: int, layout: str, supporting: int, malformed: int) -> None:
    for index in range(size):
        parent = root if layout == "flat" else root / f"group-{index // 100:03d}" / "skills"
        skill = parent / f"skill-{index:05d}"
        skill.mkdir(parents=True)
        text = (
            "---\nname: [invalid\n---\n"
            if index >= size - malformed
            else f"---\nname: skill-{index:05d}\ndescription: Synthetic task {index}\n---\n\n# Procedure\nSynthetic benchmark instructions.\n"
        )
        (skill / "SKILL.md").write_text(text, encoding="utf-8")
        for number in range(supporting):
            (skill / f"reference-{number}.txt").write_text("Synthetic supporting file.\n", encoding="utf-8")


def run_case(args: argparse.Namespace, size: int, layout: str, mode: str, malformed: int) -> dict[str, Any]:
    client = SyntheticClient(args.dimensions)
    stages: dict[str, Any] = {}
    result: dict[str, Any] = {
        "embedding_model": client.model,
        "vector_source": "synthetic one-hot; no production quality or latency model",
        "size": size,
        "layout": layout,
        "mode": mode,
        "supporting_files_per_skill": args.supporting_files,
        "malformed_manifests": malformed,
        "dimensions": args.dimensions,
        "budgets": {"max_entries": size, "max_scalar_comparisons": args.max_scalar_comparisons},
        "fixed_limits": {
            "discovered_paths": MAX_DISCOVERED_PATHS,
            "manifest_bytes": MAX_MANIFEST_BYTES,
            "collection_bytes": MAX_COLLECTION_BYTES,
            "catalog_bytes": MAX_CATALOG_BYTES,
            "vector_dimensions": MAX_VECTOR_DIMENSION,
            "result_count": MAX_SIMILARITY_MATCHES,
        },
        "stages": stages,
    }
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="tier2-scale-") as temporary:
        root = Path(temporary) / "collection"
        root.mkdir()
        measure(stages, "fixture_generation", lambda: populate(root, size, layout, args.supporting_files, malformed))
        coverage = measure(stages, "extraction", lambda: discover_and_extract_detailed(root, "skill", max_entries=size))
        if coverage is not None:
            result["coverage"] = coverage.to_metadata()
        registry = EmbeddingRegistry(
            client, full_body=mode == "full-body", max_entries=size, max_scalar_comparisons=args.max_scalar_comparisons
        )
        # Index-only build permits the saved-candidate workflow even when a
        # collection's all-pairs workload exceeds the independent work budget.
        built = measure(
            stages, "index_build_including_extraction", lambda: registry.build_from_directory(root, "skill")
        )
        if built is not None:
            result["pairwise_scalar_work"] = built * (built - 1) // 2 * args.dimensions
            result["candidate_scalar_work"] = built * args.dimensions
            matches = measure(stages, "all_pairs", lambda: registry.find_duplicates(0.75))
            if matches is not None:
                result["all_pairs_matches"] = len(matches)
            catalog = Path(temporary) / "catalog.json"
            measure(stages, "catalog_save", lambda: registry.save_catalog(catalog))
            result["index_bytes"] = catalog.stat().st_size if catalog.exists() else None
            if catalog.exists():
                loaded = EmbeddingRegistry(
                    client,
                    full_body=mode == "full-body",
                    max_entries=size,
                    max_scalar_comparisons=args.max_scalar_comparisons,
                )
                measure(stages, "catalog_load", lambda: loaded.load_catalog(catalog))
                target_path = (
                    root / "skill-00000" if layout == "flat" else root / "group-000" / "skills" / "skill-00000"
                )
                target = extract_from_skill(target_path)
                own = loaded.catalog_self_entry(target)
                matches = measure(
                    stages,
                    "saved_candidate_query",
                    lambda: loaded.query_entry(target, 0.75, exclude_entry_id=own.entry_id if own else None),
                )
                if matches is not None:
                    result["candidate_matches"] = len(matches)
                    result["self_entries_excluded"] = int(own is not None)
        # Separately exercise fresh-scan preflight, including its early stop
        # after the first batch when all-pairs work is over budget.
        preflight = EmbeddingRegistry(
            client, full_body=mode == "full-body", max_entries=size, max_scalar_comparisons=args.max_scalar_comparisons
        )
        measure(
            stages,
            "fresh_pairwise_build",
            lambda: preflight.build_from_directory(root, "skill", for_pairwise_scan=True),
        )
    result["synthetic_usage"] = {
        "embedding_calls": client.embedding_calls,
        "embedding_inputs": client.embedding_inputs,
        "input_utf8_bytes": client.input_utf8_bytes,
    }
    result["provider_usage"] = {
        "provider": "none",
        "calls": 0,
        "tokens": None,
        "note": "Synthetic embed_chunked does not measure production tokenization or requests",
    }
    tracemalloc.stop()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2500, 5000])
    parser.add_argument("--layouts", choices=["flat", "nested"], nargs="+", default=["flat", "nested"])
    parser.add_argument(
        "--modes", choices=["description", "full-body"], nargs="+", default=["description", "full-body"]
    )
    parser.add_argument("--malformed-counts", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--supporting-files", type=int, default=1)
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument("--max-scalar-comparisons", type=int, default=SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not 2 <= size <= 5000 for size in args.sizes) or not 1 <= args.dimensions <= 65536:
        parser.error("sizes must be 2..5000 and dimensions 1..65536")
    if args.max_scalar_comparisons < 1:
        parser.error("max-scalar-comparisons must be positive")
    if args.supporting_files < 0 or any(count < 0 or count > min(args.sizes) for count in args.malformed_counts):
        parser.error("supporting files and malformed counts must be nonnegative; malformed counts must fit every size")
    output = {
        "kind": "synthetic-limit-benchmark",
        "production_performance_claim": False,
        "python": sys.version,
        "platform": platform.platform(),
        "command": sys.argv,
        "memory_measurement": "tracemalloc stage peak absolute Python allocations; excludes native allocations; tracing adds runtime overhead",
        "cases": [],
    }
    for size in args.sizes:
        for layout in args.layouts:
            for mode in args.modes:
                for malformed in args.malformed_counts:
                    output["cases"].append(run_case(args, size, layout, mode, malformed))
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                    print(f"Recorded {size} {layout} {mode} malformed={malformed}", flush=True)


if __name__ == "__main__":
    main()
