# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator CLI for replaying a local registry and finding overlap candidates."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import click

from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
from skillevaluator.embedding.extractor import MAX_MANIFEST_BYTES
from skillevaluator.embedding.registry import _client_endpoint_fingerprint, _client_provider
from skillevaluator.registry_overlap.events import embedding_input, read_batch
from skillevaluator.utils.secure_fs import SecureRoot


@dataclass(frozen=True)
class _Options:
    dimension: int
    revision: str

    def embedding(self):
        from skillevaluator.registry_overlap.postgres import EmbeddingProfile

        client = EmbeddingClient(timeout=10.0, max_retries=0)
        profile = EmbeddingProfile(
            _client_provider(client),
            client.model,
            _client_endpoint_fingerprint(client),
            self.dimension,
            self.revision,
        )
        return profile, client


@contextmanager
def _database():
    dsn = os.getenv("SKILL_EVAL_REGISTRY_DSN")
    if not dsn:
        raise click.ClickException("Set SKILL_EVAL_REGISTRY_DSN to the authorized PostgreSQL connection string.")
    try:
        from psycopg import Error
        from psycopg.errors import LockNotAvailable

        from skillevaluator.registry_overlap.postgres import RegistryStore
    except ImportError:
        raise click.ClickException("Install skillevaluator[registry] to use the registry pilot.") from None
    try:
        with RegistryStore(dsn) as store:
            yield store
    except (ValueError, SimilarityConfigError) as exc:
        raise click.ClickException(str(exc)) from None
    except LockNotAvailable:
        raise click.ClickException("Registry database is busy. Retry the operation.") from None
    except Error:
        raise click.ClickException(
            "Registry database operation failed. Check the database and pgvector setup."
        ) from None


def _output(value):
    click.echo(json.dumps(value, default=str, sort_keys=True, ensure_ascii=True))


@click.group()
@click.option("--dimension", default=2048, type=click.IntRange(1, 4000), show_default=True)
@click.option("--profile-revision", default="1", show_default=True)
@click.pass_context
def cli(ctx, dimension, profile_revision):
    """Replay verified registry batches and search the active vector index."""
    ctx.obj = _Options(dimension, profile_revision)


@cli.command()
@click.option("--source", required=True)
@click.pass_obj
def init(options, source):
    """Create the registry and its first active embedding profile."""
    with _database() as store:
        profile, _ = options.embedding()
        _output({"active_generation": store.initialize(source, profile)})


@cli.command()
@click.option("--batch", type=click.Path(path_type=Path, dir_okay=False, exists=True))
@click.option("--sha256")
@click.option("--generation", type=click.IntRange(min=1))
@click.option("--accept-only", is_flag=True, help="Commit the batch without calling the embedding provider.")
@click.pass_obj
def sync(options, batch, sha256, generation, accept_only):
    """Accept a checksum-verified batch and drain up to 250 pending events."""
    if (batch is None) != (sha256 is None):
        raise click.UsageError("--batch and --sha256 must be provided together.")
    if accept_only and batch is None:
        raise click.UsageError("--accept-only requires a batch.")
    with _database() as store:
        accepted = store.accept(read_batch(batch, sha256)) if batch is not None else 0
        applied = 0
        if not accept_only:
            profile, client = options.embedding()
            applied = store.drain(profile, client.embed, generation=generation)
        _output(
            {
                "accepted_events": accepted,
                "applied_events": applied,
                "accepted_cursor": store.status()["accepted_cursor"],
            }
        )


@cli.command()
@click.argument("manifest", type=click.Path(path_type=Path, dir_okay=False, exists=True))
@click.option("--top-k", type=click.IntRange(1, 50), default=10, show_default=True)
@click.option("--exclude", help="Stable ID of the query skill to exclude.")
@click.pass_obj
def search(options, manifest, top_k, exclude):
    """Return bounded overlap candidates and their provenance as JSON."""
    with _database() as store:
        try:
            with SecureRoot(manifest.parent) as root:
                raw, _ = root.read_bytes(Path(manifest.name), MAX_MANIFEST_BYTES)
            text = embedding_input(raw.decode("utf-8"))
        except (OSError, ValueError):
            raise click.ClickException("Cannot read a valid bounded skill manifest.") from None
        profile, client = options.embedding()
        _output(store.search(profile, lambda: client.embed_single(text), top_k=top_k, exclude=exclude))


@cli.command()
def status():
    """Show source checkpoint, generation progress, and index readiness."""
    with _database() as store:
        _output(store.status())


@cli.command()
@click.pass_obj
def reindex(options):
    """Allocate an inactive generation from the current verified snapshot."""
    with _database() as store:
        profile, _ = options.embedding()
        _output({"generation": store.reindex(profile)})


@cli.command()
@click.argument("generation", type=click.IntRange(min=1))
def activate(generation):
    """Switch to a generation caught up with the registry."""
    with _database() as store:
        store.activate(generation)
        _output({"active_generation": generation})


@cli.command()
@click.argument("generation", type=click.IntRange(min=1))
def retire(generation):
    """Remove an inactive generation and its vectors."""
    with _database() as store:
        store.retire(generation)
        _output({"retired_generation": generation})


if __name__ == "__main__":
    cli()
