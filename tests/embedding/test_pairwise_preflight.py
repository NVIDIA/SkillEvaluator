# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh pairwise scans reject unaffordable work before embedding later batches."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry, RegistryEntry
from skillevaluator.tier2.commands import run_similarity_check

ENTRY_COUNT = 65
DIMENSION = 65
PAIRWISE_WORK = ENTRY_COUNT * (ENTRY_COUNT - 1) // 2 * DIMENSION


def _collection(root: Path) -> Path:
    root.mkdir()
    for index in range(ENTRY_COUNT):
        directory = root / f"skill-{index:04}"
        directory.mkdir()
        (directory / "SKILL.md").write_text(
            f"---\nname: skill-{index:04}\ndescription: Description {index}\n---\nBody\n"
        )
    return root


def _client() -> MagicMock:
    client = MagicMock(spec=EmbeddingClient)
    client.model = "test-model"
    client._resolved_config.return_value = SimpleNamespace(provider="openai", base_url="https://api.openai.com/v1")

    def vector(text: str) -> list[float]:
        name = text.split("name: ", 1)[1].splitlines()[0] if text.startswith("---") else text.split(":", 1)[0]
        index = int(name.removeprefix("skill-"))
        embedding = [0.0] * DIMENSION
        embedding[0 if index == ENTRY_COUNT - 1 else index] = 1.0
        return embedding

    client.embed.side_effect = lambda texts: [vector(text) for text in texts]
    client.embed_chunked.side_effect = vector
    client.embed_single.return_value = vector("skill-0000: query")
    return client


def _assert_requests(client: MagicMock, *, full_body: bool, completes: bool) -> None:
    if full_body:
        assert client.embed_chunked.call_count == (ENTRY_COUNT if completes else 1)
        client.embed.assert_not_called()
    else:
        assert [len(call.args[0]) for call in client.embed.call_args_list] == ([64, 1] if completes else [64])
        client.embed_chunked.assert_not_called()


@pytest.mark.parametrize("full_body", [False, True])
@pytest.mark.parametrize("preexisting", [False, True])
def test_preflight_stops_after_first_response_without_mutating_registry(tmp_path: Path, full_body, preexisting) -> None:
    collection = _collection(tmp_path / "skills")
    client = _client()
    registry = EmbeddingRegistry(client, full_body=full_body, max_scalar_comparisons=PAIRWISE_WORK - 1)
    if preexisting:
        registry._entries["prior"] = RegistryEntry("prior", "Prior", "prior", "skill", [1.0] + [0.0] * 64)
        registry._vector_dimension = DIMENSION
    before = dict(registry._entries)
    before_dimension = registry._vector_dimension

    with pytest.raises(ValueError, match="--max-scalar-comparisons"):
        registry.build_from_directory(collection, "skill", for_pairwise_scan=True)

    assert registry._entries == before
    assert registry._vector_dimension == before_dimension
    _assert_requests(client, full_body=full_body, completes=False)


@pytest.mark.parametrize("full_body", [False, True])
@pytest.mark.parametrize("existing_output", [False, True])
def test_failed_scan_does_not_create_or_replace_saved_index(tmp_path: Path, full_body, existing_output) -> None:
    collection = _collection(tmp_path / "skills")
    client = _client()
    output = tmp_path / "catalog.json"
    if existing_output:
        output.write_text("existing index")
    with patch("skillevaluator.validators.similarity.EmbeddingClient", return_value=client):
        (result,) = run_similarity_check(
            collection,
            content_type="skill",
            full_body=full_body,
            max_scalar_comparisons=PAIRWISE_WORK - 1,
            save_catalog=output,
        )

    assert any("--max-scalar-comparisons" in error for error in result.errors)
    assert not result.findings
    assert not any(success.check_name == "index_built" for success in result.success_details)
    assert output.read_text() == "existing index" if existing_output else not output.exists()
    _assert_requests(client, full_body=full_body, completes=False)


@pytest.mark.parametrize("full_body", [False, True])
@pytest.mark.parametrize("budget,completes", [(PAIRWISE_WORK - 1, False), (PAIRWISE_WORK, True)])
def test_exact_pairwise_budget_allows_cross_batch_duplicate(tmp_path: Path, full_body, budget, completes) -> None:
    collection = _collection(tmp_path / "skills")
    client = _client()
    with patch("skillevaluator.validators.similarity.EmbeddingClient", return_value=client):
        (result,) = run_similarity_check(
            collection, content_type="skill", full_body=full_body, max_scalar_comparisons=budget
        )
    _assert_requests(client, full_body=full_body, completes=completes)
    if completes:
        assert len(result.findings) == 1
        assert {result.findings[0].metadata[key] for key in ("entry_a", "entry_b")} == {"skill-0000", "skill-0064"}
    else:
        assert any("--max-scalar-comparisons" in error for error in result.errors)
        assert not result.findings


def test_index_only_build_and_saved_query_do_not_require_pairwise_budget(tmp_path: Path) -> None:
    collection = _collection(tmp_path / "skills")
    client = _client()
    query_work = ENTRY_COUNT * DIMENSION
    registry = EmbeddingRegistry(client, max_scalar_comparisons=query_work)
    assert registry.build_from_directory(collection, "skill") == ENTRY_COUNT
    _assert_requests(client, full_body=False, completes=True)
    with pytest.raises(ValueError, match="--max-scalar-comparisons"):
        registry.find_duplicates(0.75)
    assert len(registry.query("skill-0000: query", 0.75)) == 2
    output = tmp_path / "catalog.json"
    registry.save_catalog(output)
    loaded = EmbeddingRegistry(client, max_entries=1, max_scalar_comparisons=query_work)
    loaded.load_catalog(output)
    assert len(loaded.query("skill-0000: query", 0.75)) == 2


@pytest.mark.parametrize("failure", ["dimension", "count", "exception"])
def test_later_invalid_batch_leaves_existing_registry_unchanged(tmp_path: Path, failure) -> None:
    collection = _collection(tmp_path / "skills")
    client = _client()
    embed = client.embed.side_effect

    def faulty_embed(texts):
        if len(texts) == 64:
            return embed(texts)
        if failure == "exception":
            raise ValueError("Provider failure")
        return [] if failure == "count" else [[1.0, 0.0]]

    client.embed.side_effect = faulty_embed
    registry = EmbeddingRegistry(client)
    prior = RegistryEntry("prior", "Prior", "prior", "skill", [1.0] + [0.0] * 64)
    registry._entries["prior"] = prior
    registry._vector_dimension = DIMENSION
    before_dimension = registry._vector_dimension
    with pytest.raises(ValueError):
        registry.build_from_directory(collection, "skill", for_pairwise_scan=True)
    assert registry._entries == {"prior": prior}
    assert registry._vector_dimension == before_dimension
    _assert_requests(client, full_body=False, completes=True)
