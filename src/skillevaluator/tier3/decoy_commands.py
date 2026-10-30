# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit authoring command for frozen group-workspace selections."""

from __future__ import annotations

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
    if destination == catalog.resolve():
        raise ValueError("Decoy manifest output must not replace the embedding catalog")
    # The target's evaluator-only directory is a natural place to version this
    # file. Other skill contents and catalog members are author-owned inputs.
    if destination.is_relative_to(target_root / "evals"):
        return
    if destination.is_relative_to(target_root) or destination.is_relative_to(catalog_root.resolve()):
        raise ValueError("Save the decoy manifest outside the skill collection or inside the target's evals directory")
