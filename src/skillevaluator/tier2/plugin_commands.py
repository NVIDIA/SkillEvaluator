# SPDX-License-Identifier: Apache-2.0
"""Public plugin deduplication and versioned catalogs with lazy providers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from skillevaluator.cli_core import detect_content_type, resolve_plugin_path
from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier2.results import guarded_results


def plugin_target(path: Path, content_type: str = "auto") -> Path | None:
    if content_type == CONTENT_TYPE_PLUGIN or (
        content_type == "auto" and detect_content_type(path) == CONTENT_TYPE_PLUGIN
    ):
        return path.absolute()
    return None


def plugin_dedup_results(path: Path, **options) -> list[ValidationResult]:
    from skillevaluator.deduplication.plugin.commands import run_plugin_dedup_scan

    return run_plugin_dedup_scan(path, **options)


def run_public_plugin_dedup(path: Path, **options) -> list[ValidationResult]:
    root = resolve_plugin_path(path)
    return guarded_results(
        "Plugin Deduplication", root, lambda: plugin_dedup_results(path, **options), check_id="context-optimization"
    )


class _IndexedQueries:
    """Reuse vectors just created from the same source descriptions."""

    def __init__(self, catalog):
        self.client = catalog.client
        self.vectors = {
            f"{entry.name}: {entry.description}": entry.embedding for entry in (*catalog.skills, *catalog.plugins)
        }

    def __getattr__(self, name):
        return getattr(self.client, name)

    def embed(self, texts):
        if all(text in self.vectors for text in texts):
            return [self.vectors[text] for text in texts]
        return self.client.embed(texts)


def plugin_similarity_results(
    path: Path,
    *,
    threshold,
    full_body,
    model,
    catalog,
    save_catalog,
    max_entries,
    max_scalar_comparisons,
    llm_verdict=False,
    llm_model=None,
) -> list[ValidationResult]:
    from skillevaluator.deduplication.plugin.catalog import LocalCatalog
    from skillevaluator.deduplication.plugin.catalog_checks import check_catalog_plugins, check_catalog_skills
    from skillevaluator.deduplication.plugin.commands import _security_result
    from skillevaluator.deduplication.plugin.profile import load_plugin_profile
    from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
    from skillevaluator.embedding.registry import _validate_scalar_work, _validate_threshold
    from skillevaluator.inference.diagnostics import llm_failure_diagnostic
    from skillevaluator.utils.secure_fs import SecurePathError
    from skillevaluator.utils.tier2_paths import sanitize_path_text

    summary = ValidationResult(validator_name="Plugin Catalog", metadata={"advisory_tier2": True})
    results = [summary]
    root = resolve_plugin_path(path)
    try:
        _validate_threshold(threshold)
        if full_body:
            raise ValueError("Plugin catalogs compare descriptions; --full-body is not supported for plugins")
        if catalog is not None and save_catalog is not None:
            raise ValueError("--catalog and --save-catalog cannot be used together")
        client = EmbeddingClient(model=model)
        if catalog is not None:
            local = LocalCatalog.load(catalog, client=client, max_scalar_comparisons=max_scalar_comparisons)
            profiles = [load_plugin_profile(path)]
            summary.add_success("catalog_loaded", f"Loaded {len(local.skills) + len(local.plugins)} catalog entries")
        else:
            local = LocalCatalog.build(
                path, client=client, max_entries=max_entries, max_scalar_comparisons=max_scalar_comparisons
            )
            prefixes = dict.fromkeys(
                [entry.path for entry in local.plugins] + [prefix for prefix, _ in local.skipped_plugins]
            )
            profiles = []
            for prefix in prefixes:
                try:
                    profiles.append(load_plugin_profile(path if prefix == "." else root / prefix))
                except SecurePathError:
                    raise
                except ValueError:
                    summary.mark_scan_incomplete("plugin-catalog")
                    summary.add_warning(f"Plugin {prefix} has no complete comparison profile")
            summary.add_success(
                "index_built",
                f"Indexed {len(local.skills)} skills and {len(local.plugins)} plugins",
                content_type="plugin",
                entry_count=len(local.skills) + len(local.plugins),
            )
            for prefix, reason in local.skipped_plugins:
                summary.add_warning(f"Plugin {prefix}: {reason}")
            local = replace(local, client=_IndexedQueries(local))
        comparisons = len(profiles) * len(local.plugins) + sum(
            sum(skill.entry is not None for skill in profile.bundled_skills) for profile in profiles
        ) * len(local.skills)
        _validate_scalar_work(comparisons, local.vector_dimension, max_scalar_comparisons)
        for profile in profiles:
            compared = [
                check_catalog_skills(profile, local, threshold=threshold),
                check_catalog_plugins(
                    profile, local, threshold=threshold, llm_verdict=llm_verdict, llm_model=llm_model
                ),
            ]
            prefix = profile.root.relative_to(root)
            for result in compared:
                result.metadata["plugin_catalog_target"] = {"name": profile.name, "path": prefix.as_posix()}
                if prefix != Path():
                    for finding in result.findings:
                        if finding.file_path and not Path(finding.file_path).is_absolute():
                            finding.file_path = (prefix / finding.file_path).as_posix()
            results.extend(compared)
        if save_catalog is not None:
            local.save(save_catalog)
            summary.add_success("catalog_saved", f"Saved local catalog to {save_catalog.name}")
    except SecurePathError as exc:
        results.append(_security_result(exc))
    except SimilarityConfigError as exc:
        summary.mark_scan_incomplete("embedding-provider")
        summary.add_error(f"Embedding provider error: {llm_failure_diagnostic(exc)}")
    except (ValueError, OSError) as exc:
        summary.add_error(sanitize_path_text(str(exc), (path, catalog, save_catalog)))
    return results
