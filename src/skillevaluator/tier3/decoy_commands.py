# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit authoring command for frozen group-workspace selections."""

from __future__ import annotations

import json
from pathlib import Path

import click

from skillevaluator.cli_help import RichCommand


@click.command("select-decoys", cls=RichCommand, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("skill_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--catalog",
    "catalog_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Description-mode embedding catalog created with similarity-check --save-catalog.",
)
@click.option(
    "--catalog-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local collection whose relative skill paths the catalog describes.",
)
@click.option("--count", type=click.IntRange(min=1, max=1000), default=4, show_default=True)
@click.option("--exclude", multiple=True, help="Catalog-relative skill path to exclude (repeatable).")
@click.option("--model", help="Embedding model; must match the catalog. Defaults to the configured provider's model.")
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the frozen selection manifest here, for use with skill_workspace.decoy_manifest.",
)
def select_decoys_command(
    skill_path: Path,
    catalog_path: Path,
    catalog_root: Path,
    count: int,
    exclude: tuple[str, ...],
    model: str | None,
    output_path: Path,
) -> None:
    """Select neighboring skills for a paired group evaluation.

    Rank a local catalog by description similarity and save a reviewable
    manifest. This authoring step may call the configured embedding provider;
    evaluation reuses the manifest without querying that provider again.
    """
    from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
    from skillevaluator.tier3.decoys import select_decoys, write_decoy_manifest

    try:
        _validate_output_path(output_path, skill_path, catalog_path, catalog_root)
        manifest = select_decoys(
            skill_path,
            catalog_path=catalog_path,
            catalog_root=catalog_root,
            count=count,
            exclude=exclude,
            client=EmbeddingClient(model=model),
        )
        write_decoy_manifest(manifest, output_path)
    except (OSError, ValueError, SimilarityConfigError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved {len(manifest['selected'])} decoys to {output_path}")
    click.echo("Review the selection, then set skill_workspace.mode: group and skill_workspace.decoy_manifest.")


def _validate_output_path(output: Path, target: Path, catalog: Path, catalog_root: Path) -> None:
    """Keep manifest output from replacing inputs or changing selected sources."""
    destination = output.resolve()
    target_root = target.resolve()
    if destination == catalog.resolve() or _same_existing_path(destination, catalog):
        raise ValueError("Decoy manifest output must not replace the embedding catalog")
    # The target's evaluator-only directory is a natural place to version this
    # file. Other skill contents and catalog members are author-owned inputs.
    relative = _relative_to_directory(destination, target_root)
    in_target_evals = relative is not None and len(relative.parts) > 1 and relative.parts[0].casefold() == "evals"
    if in_target_evals:
        reserved = {"config.yml", "config.yaml"} | {
            f"{stem}.{extension}" for stem in ("evals", "dataset") for extension in ("json", "jsonl", "yaml", "yml")
        }
        if destination.name.casefold() in reserved:
            raise ValueError("Decoy manifest output must not replace an evaluation dataset or run policy")
    elif relative is not None or _relative_to_directory(destination, catalog_root.resolve()) is not None:
        raise ValueError("Save the decoy manifest outside the skill collection or inside the target's evals directory")
    _validate_existing_output(output)


def _same_existing_path(first: Path, second: Path) -> bool:
    """Recognize filesystem aliases without equating distinct case-sensitive names."""
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def _relative_to_directory(path: Path, directory: Path) -> Path | None:
    """Find containment through existing ancestors, including case aliases on macOS."""
    for ancestor in (path, *path.parents):
        if ancestor == directory or _same_existing_path(ancestor, directory):
            return path.relative_to(ancestor)
    return None


def _validate_existing_output(output: Path) -> None:
    """Allow regenerating saved manifests, including ones with stale source hashes."""
    from skillevaluator.embedding.registry import _catalog_object, _read_catalog_text, _reject_json_constant
    from skillevaluator.tier3.decoys import (
        _CATALOG_FIELDS,
        _MANIFEST_FIELDS,
        _SELECTED_FIELDS,
        _TARGET_FIELDS,
        DECOY_MANIFEST_VERSION,
        DECOY_SELECTION_STRATEGY,
        _count,
        _hash,
        _object,
    )

    if not output.exists() and not output.is_symlink():
        return
    try:
        data = _object(
            json.loads(
                _read_catalog_text(output), object_pairs_hook=_catalog_object, parse_constant=_reject_json_constant
            ),
            _MANIFEST_FIELDS,
            "root",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != DECOY_MANIFEST_VERSION:
            raise ValueError("Unsupported decoy manifest schema version")
        if data["strategy"] != DECOY_SELECTION_STRATEGY:
            raise ValueError("Unsupported decoy selection strategy")
        catalog = _object(data["catalog"], _CATALOG_FIELDS, "catalog")
        target = _object(data["target"], _TARGET_FIELDS, "target")
        _hash(catalog["sha256"], "catalog")
        _hash(target["manifest_sha256"], "target")
        if not isinstance(data["selected"], list) or len(data["selected"]) != _count(data["count"]):
            raise ValueError("Invalid decoy manifest selected count")
        for item in data["selected"]:
            _hash(_object(item, _SELECTED_FIELDS, "selected skill")["source_digest"], "source")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Refusing to replace an existing file that is not a decoy manifest: {output}") from exc
