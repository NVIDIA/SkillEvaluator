# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security and behavior tests for the Tier 2 local embedding catalog."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import skillevaluator.embedding.registry as registry_module
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.extractor import ContentEntry
from skillevaluator.embedding.registry import EmbeddingRegistry, RegistryEntry


def _endpoint_fingerprint(url: str) -> str:
    return f"sha256:{hashlib.sha256(url.encode('utf-8')).hexdigest()}"


def _client(vectors: list[list[float]] | None = None) -> EmbeddingClient:
    client = MagicMock(spec=EmbeddingClient)
    client.embed.return_value = vectors or []
    client.embed_single.return_value = (vectors or [[1.0, 0.0]])[0]
    client.embed_chunked.return_value = (vectors or [[1.0, 0.0]])[0]
    client.model = "test-model"
    client._resolved_config.return_value = SimpleNamespace(
        provider="nv_build",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    return client


def _catalog_data() -> dict:
    return {
        "schema_version": 1,
        "provider": "nv_build",
        "model": "test-model",
        "mode": "description",
        "endpoint_fingerprint": _endpoint_fingerprint("https://integrate.api.nvidia.com/v1"),
        "vector_dimension": 2,
        "created_at": "2026-07-06T00:00:00+00:00",
        "entries": [
            {
                "id": "skill:team-a",
                "name": "shared-name",
                "description": "First catalog skill",
                "path": "team-a",
                "content_type": "skill",
                "content_fingerprint": "a" * 64,
                "embedding": [1.0, 0.0],
            }
        ],
    }


def _write_catalog(path: Path, data: dict | None = None) -> Path:
    path.write_text(json.dumps(data or _catalog_data()), encoding="utf-8")
    return path


class TestCatalogPersistence:
    def test_save_catalog_is_versioned_relative_and_preserves_duplicate_names(self, tmp_path: Path) -> None:
        for directory, description in (("team-a", "First"), ("team-b", "Second")):
            skill_dir = tmp_path / "skills" / directory
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: shared-name\ndescription: {description} catalog skill\n---\n"
            )

        registry = EmbeddingRegistry(_client([[1.0, 0.0], [0.0, 1.0]]))
        assert registry.build_from_directory(tmp_path / "skills", "skill") == 2
        catalog_path = tmp_path / "catalog.json"

        registry.save_catalog(catalog_path)

        data = json.loads(catalog_path.read_text())
        assert data["schema_version"] == 1
        assert data["provider"] == "nv_build"
        assert data["model"] == "test-model"
        assert data["mode"] == "description"
        assert data["endpoint_fingerprint"].startswith("sha256:")
        assert data["vector_dimension"] == 2
        assert isinstance(data["entries"], list)
        assert [item["name"] for item in data["entries"]] == ["shared-name", "shared-name"]
        assert {item["id"] for item in data["entries"]} == {"skill:team-a", "skill:team-b"}
        assert {item["path"] for item in data["entries"]} == {"team-a", "team-b"}
        assert all(not Path(item["path"]).is_absolute() for item in data["entries"])
        assert all(len(item["content_fingerprint"]) == 64 for item in data["entries"])

    def test_roundtrip_and_target_query_name_both_sides(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path / "catalog.json")
        client = _client([[1.0, 0.0]])
        registry = EmbeddingRegistry(client)
        registry.load_catalog(catalog_path)
        target = ContentEntry(
            name="candidate-skill",
            description="Candidate catalog skill",
            path=str(tmp_path / "candidate-skill"),
            content_type="skill",
        )

        matches = registry.query_entry(target, threshold=0.75)

        assert len(matches) == 1
        assert matches[0].entry_a == "candidate-skill"
        assert matches[0].entry_b == "shared-name"
        assert matches[0].path_a == "candidate-skill"
        assert matches[0].path_b == "team-a"

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("schema_version", 99, "schema"),
            ("provider", "other", "provider"),
            ("model", "other-model", "model"),
            ("mode", "full-body", "mode"),
            ("vector_dimension", 3, "dimension"),
            ("created_at", "not-a-timestamp", "created_at"),
        ],
    )
    def test_load_rejects_incompatible_metadata(self, tmp_path: Path, field: str, value: object, message: str) -> None:
        data = _catalog_data()
        data[field] = value
        catalog_path = _write_catalog(tmp_path / "bad.json", data)

        with pytest.raises(ValueError, match=message):
            EmbeddingRegistry(_client()).load_catalog(catalog_path)

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda data: data["entries"][0].update(embedding=[math.nan, 0.0]), "finite"),
            (lambda data: data["entries"][0].update(embedding=[0.0, 0.0]), "zero"),
            (lambda data: data["entries"][0].update(embedding=[1.0]), "dimension"),
            (lambda data: data["entries"][0].update(path="/private/skill"), "relative"),
            (lambda data: data["entries"][0].update(path="../escape"), "relative"),
            (lambda data: data["entries"][0].update(content_fingerprint="not-a-hash"), "fingerprint"),
            (
                lambda data: data["entries"][0].update(id="unknown:team-a", content_type="unknown"),
                "content type",
            ),
        ],
    )
    def test_load_rejects_malformed_entries(self, tmp_path: Path, mutation, message: str) -> None:
        data = copy.deepcopy(_catalog_data())
        mutation(data)
        catalog_path = _write_catalog(tmp_path / "bad-entry.json", data)

        with pytest.raises(ValueError, match=message):
            EmbeddingRegistry(_client()).load_catalog(catalog_path)

    def test_load_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        data = _catalog_data()
        data["entries"].append(copy.deepcopy(data["entries"][0]))

        with pytest.raises(ValueError, match=r"duplicate.*id"):
            EmbeddingRegistry(_client()).load_catalog(_write_catalog(tmp_path / "duplicate.json", data))

    def test_load_rejects_duplicate_json_keys(self, tmp_path: Path) -> None:
        serialized = json.dumps(_catalog_data())
        serialized = serialized.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
        path = tmp_path / "duplicate-key.json"
        path.write_text(serialized, encoding="utf-8")

        with pytest.raises(ValueError, match=r"duplicate.*schema_version"):
            EmbeddingRegistry(_client()).load_catalog(path)

    @pytest.mark.parametrize("scope", ["root", "entry"])
    def test_load_rejects_unknown_schema_fields(self, tmp_path: Path, scope: str) -> None:
        data = _catalog_data()
        if scope == "root":
            data["secret"] = "unexpected"
        else:
            data["entries"][0]["secret"] = "unexpected"

        with pytest.raises(ValueError, match=r"unknown.*secret"):
            EmbeddingRegistry(_client()).load_catalog(_write_catalog(tmp_path / f"unknown-{scope}.json", data))

    @pytest.mark.parametrize("unsafe", ["bad\ud800value", "bad\u0000value"])
    def test_load_rejects_surrogate_and_control_strings(self, tmp_path: Path, unsafe: str) -> None:
        data = _catalog_data()
        data["entries"][0]["name"] = unsafe

        with pytest.raises(ValueError, match=r"unsafe|control|surrogate"):
            EmbeddingRegistry(_client()).load_catalog(_write_catalog(tmp_path / "unsafe-text.json", data))

    def test_load_accepts_bracketed_catalog_text(self, tmp_path: Path) -> None:
        data = _catalog_data()
        data["entries"][0]["name"] = "[tool]"
        data["entries"][0]["description"] = "Use [tool] for this workflow"
        registry = EmbeddingRegistry(_client())

        registry.load_catalog(_write_catalog(tmp_path / "bracketed-text.json", data))

        assert registry._entries["skill:team-a"].name == "[tool]"
        assert registry._entries["skill:team-a"].description == "Use [tool] for this workflow"

    def test_load_enforces_entry_and_file_size_limits(self, tmp_path: Path, monkeypatch) -> None:
        data = _catalog_data()
        data["entries"].append(
            {
                **copy.deepcopy(data["entries"][0]),
                "id": "skill:team-b",
                "path": "team-b",
            }
        )
        path = _write_catalog(tmp_path / "large.json", data)

        monkeypatch.setattr(registry_module, "MAX_CATALOG_ENTRIES", 1)
        with pytest.raises(ValueError, match="entry limit"):
            EmbeddingRegistry(_client()).load_catalog(path)

        monkeypatch.setattr(registry_module, "MAX_CATALOG_ENTRIES", 10)
        monkeypatch.setattr(registry_module, "MAX_CATALOG_BYTES", 10)
        with pytest.raises(ValueError, match="size limit"):
            EmbeddingRegistry(_client()).load_catalog(path)

    def test_save_rejects_nonfinite_vectors(self, tmp_path: Path) -> None:
        registry = EmbeddingRegistry(_client())
        registry._entries["skill:bad"] = RegistryEntry(
            name="bad",
            description="Bad vector",
            path="bad",
            content_type="skill",
            embedding=[math.nan, 0.0],
        )

        with pytest.raises(ValueError, match="finite"):
            registry.save_catalog(tmp_path / "bad.json")

    def test_load_rejects_overflow_magnitude_vector(self, tmp_path: Path) -> None:
        data = _catalog_data()
        data["entries"][0]["embedding"] = [1e308, 1e308]

        with pytest.raises(ValueError, match=r"magnitude|stable|overflow"):
            EmbeddingRegistry(_client()).load_catalog(_write_catalog(tmp_path / "overflow.json", data))

    def test_save_does_not_follow_predictable_temporary_symlink(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim.txt"
        victim.write_text("DO NOT OVERWRITE", encoding="utf-8")
        catalog = tmp_path / "catalog.json"
        predictable = catalog.with_name(f".{catalog.name}.{os.getpid()}.tmp")
        try:
            predictable.symlink_to(victim)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        registry = EmbeddingRegistry(_client())
        registry._entries["skill:team-a"] = RegistryEntry(
            name="safe",
            description="Safe catalog entry",
            path="team-a",
            content_type="skill",
            embedding=[1.0, 0.0],
            entry_id="skill:team-a",
            content_fingerprint="a" * 64,
        )

        registry.save_catalog(catalog)

        assert victim.read_text(encoding="utf-8") == "DO NOT OVERWRITE"
        assert catalog.is_file() and not catalog.is_symlink()
        assert predictable.is_symlink()

    @pytest.mark.skipif(os.name == "nt", reason="Windows denies replacement while the catalog handle is open")
    def test_load_fails_closed_if_path_changes_after_descriptor_open(self, tmp_path: Path, monkeypatch) -> None:
        catalog = _write_catalog(tmp_path / "catalog.json")
        outside_data = _catalog_data()
        outside_data["entries"][0].update(
            id="skill:outside",
            name="outside",
            path="outside",
            content_fingerprint="b" * 64,
        )
        outside = _write_catalog(tmp_path / "outside.json", outside_data)
        original_read = os.read
        swapped = False

        def swap_after_open(fd: int, size: int) -> bytes:
            nonlocal swapped
            if not swapped:
                swapped = True
                backup = tmp_path / "catalog-original.json"
                catalog.rename(backup)
                catalog.symlink_to(outside)
            return original_read(fd, size)

        monkeypatch.setattr(registry_module.os, "read", swap_after_open)

        with pytest.raises(ValueError, match=r"changed|symlink|reparse"):
            EmbeddingRegistry(_client()).load_catalog(catalog)

        assert swapped is True

    @pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing semantics")
    def test_windows_open_catalog_handle_blocks_path_replacement(self, tmp_path: Path, monkeypatch) -> None:
        catalog = _write_catalog(tmp_path / "catalog.json")
        original_read = os.read
        replacement_denied = False

        def try_replace_after_open(fd: int, size: int) -> bytes:
            nonlocal replacement_denied
            if not replacement_denied:
                try:
                    catalog.rename(tmp_path / "catalog-original.json")
                except PermissionError:
                    replacement_denied = True
                else:
                    pytest.fail("Windows allowed an open catalog path to be replaced")
            return original_read(fd, size)

        monkeypatch.setattr(registry_module.os, "read", try_replace_after_open)

        registry = EmbeddingRegistry(_client())
        registry.load_catalog(catalog)

        assert replacement_denied is True
        assert registry.size == 1

    def test_openai_compatible_catalog_fingerprints_sanitized_endpoint(self, tmp_path: Path) -> None:
        client = _client()
        client._resolved_config.return_value = SimpleNamespace(
            provider="openai-compatible",
            base_url="https://user:password@example.test/v1?api_key=secret#fragment",
        )
        registry = EmbeddingRegistry(client)
        registry._entries["skill:team-a"] = RegistryEntry(
            name="safe",
            description="Safe catalog entry",
            path="team-a",
            content_type="skill",
            embedding=[1.0, 0.0],
            entry_id="skill:team-a",
            content_fingerprint="a" * 64,
        )
        catalog = tmp_path / "catalog.json"

        registry.save_catalog(catalog)

        serialized = catalog.read_text(encoding="utf-8")
        data = json.loads(serialized)
        assert data["endpoint_fingerprint"].startswith("sha256:")
        assert "user" not in serialized
        assert "password" not in serialized
        assert "api_key" not in serialized
        assert "secret" not in serialized
        other_client = _client()
        other_client._resolved_config.return_value = SimpleNamespace(
            provider="openai-compatible",
            base_url="https://other.example.test/v1",
        )
        with pytest.raises(ValueError, match="endpoint"):
            EmbeddingRegistry(other_client).load_catalog(catalog)

    def test_rejects_symlinked_catalog_file(self, tmp_path: Path) -> None:
        real = _write_catalog(tmp_path / "real.json")
        linked = tmp_path / "linked.json"
        try:
            linked.symlink_to(real)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse"):
            EmbeddingRegistry(_client()).load_catalog(linked)

    def test_load_rejects_symlinked_intermediate_directory(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        nested = outside / "nested"
        nested.mkdir()
        _write_catalog(nested / "catalog.json")
        safe = tmp_path / "safe"
        safe.mkdir()
        linked = safe / "linked"
        try:
            linked.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse|component"):
            EmbeddingRegistry(_client()).load_catalog(linked / "nested" / "catalog.json")

    def test_save_rejects_symlinked_intermediate_directory(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        safe = tmp_path / "safe"
        safe.mkdir()
        linked = safe / "linked"
        try:
            linked.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        registry = EmbeddingRegistry(_client())
        registry._entries["skill:team-a"] = RegistryEntry(
            name="safe",
            description="Safe catalog entry",
            path="team-a",
            content_type="skill",
            embedding=[1.0, 0.0],
            entry_id="skill:team-a",
            content_fingerprint="a" * 64,
        )

        with pytest.raises(ValueError, match=r"symlink|reparse|component"):
            registry.save_catalog(linked / "nested" / "catalog.json")

        assert not (outside / "nested" / "catalog.json").exists()

    @pytest.mark.skipif(
        os.name != "posix" or not Path("/tmp").is_symlink(),
        reason="macOS-style root-owned /tmp alias is unavailable",
    )
    def test_catalog_io_supports_root_owned_tmp_alias(self) -> None:
        registry = EmbeddingRegistry(_client())
        registry._entries["skill:team-a"] = RegistryEntry(
            name="safe",
            description="Safe catalog entry",
            path="team-a",
            content_type="skill",
            embedding=[1.0, 0.0],
            entry_id="skill:team-a",
            content_fingerprint="a" * 64,
        )

        with tempfile.TemporaryDirectory(prefix="se-catalog-alias-", dir="/tmp") as directory:
            catalog = Path(directory) / "catalog.json"
            registry.save_catalog(catalog)

            loaded = EmbeddingRegistry(_client())
            loaded.load_catalog(catalog)

        assert loaded.size == 1

    @pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO semantics")
    def test_load_rejects_fifo_without_blocking(self, tmp_path: Path) -> None:
        fifo = tmp_path / "catalog.fifo"
        os.mkfifo(fifo)
        script = (
            "from pathlib import Path\n"
            "from skillevaluator.embedding.registry import _read_catalog_text\n"
            "try:\n"
            "    _read_catalog_text(Path(__import__('sys').argv[1]))\n"
            "except ValueError as exc:\n"
            "    assert 'regular file' in str(exc), str(exc)\n"
            "else:\n"
            "    raise AssertionError('FIFO catalog was accepted')\n"
        )

        # The timeout guards against a regression where opening the FIFO
        # blocks forever. It must absorb interpreter startup plus the package
        # import, which can take several seconds when pytest-xdist saturates
        # the CPU — 2s flaked under -n auto while always passing standalone.
        completed = subprocess.run(
            [sys.executable, "-c", script, str(fifo)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert completed.returncode == 0, completed.stderr

    @pytest.mark.skipif(os.name != "posix", reason="POSIX special-file semantics")
    def test_load_rejects_device_catalog(self) -> None:
        with pytest.raises(ValueError, match="regular file"):
            registry_module._read_catalog_text(Path(os.devnull))

    @pytest.mark.skipif(os.name != "posix" or not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
    def test_load_rejects_socket_catalog(self) -> None:
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="se-catalog-", dir=temporary_root) as directory:
            socket_path = Path(directory) / "catalog.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(socket_path))

                with pytest.raises(ValueError, match=r"regular file|Unable to open"):
                    registry_module._read_catalog_text(socket_path)


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -0.01, 1.01])
def test_registry_rejects_invalid_thresholds(threshold: float) -> None:
    registry = EmbeddingRegistry(_client())
    with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
        registry.find_duplicates(threshold)


def test_catalog_query_match_limit_fails_closed(tmp_path: Path, monkeypatch) -> None:
    data = _catalog_data()
    data["entries"].append(
        {
            **copy.deepcopy(data["entries"][0]),
            "id": "skill:team-b",
            "path": "team-b",
        }
    )
    registry = EmbeddingRegistry(_client([[1.0, 0.0]]))
    registry.load_catalog(_write_catalog(tmp_path / "catalog.json", data))
    monkeypatch.setattr(registry_module, "MAX_SIMILARITY_MATCHES", 1, raising=False)
    target = ContentEntry(
        name="candidate",
        description="Candidate skill",
        path=str(tmp_path / "candidate"),
        content_type="skill",
    )

    with pytest.raises(ValueError, match="match limit"):
        registry.query_entry(target, 0.75)
