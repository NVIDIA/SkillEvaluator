# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security and behavior tests for the Tier 2 local embedding catalog."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
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
from skillevaluator.embedding.extractor import ContentEntry, extract_from_skill
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
    @pytest.mark.parametrize("full_body", [False, True])
    @pytest.mark.parametrize(
        ("style", "description"),
        [
            (">", "Summarize reports and extract findings.\n"),
            ("|", "Summarize reports\nand extract findings.\n"),
            (">-", "Summarize reports and extract findings."),
        ],
    )
    def test_yaml_multiline_description_roundtrip(
        self, tmp_path: Path, full_body: bool, style: str, description: str
    ) -> None:
        skill_dir = tmp_path / "skills" / "summarize-reports"
        skill_dir.mkdir(parents=True)
        manifest = (
            f"---\nname: summarize-reports\ndescription: {style}\n"
            "  Summarize reports\n  and extract findings.\nlicense: MIT\n---\n\n# Reports\n"
        )
        (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8", newline="")
        client = _client([[1.0, 0.0]])
        registry = EmbeddingRegistry(client, full_body=full_body)
        assert registry.build_from_directory(tmp_path / "skills", "skill") == 1
        original = registry._entries["skill:summarize-reports"]
        assert original.description == description
        if full_body:
            client.embed_chunked.assert_called_once_with(manifest)
        else:
            client.embed.assert_called_once_with([f"summarize-reports: {description}"])

        catalog_path = tmp_path / "catalog.json"
        registry.save_catalog(catalog_path)
        restored = EmbeddingRegistry(client, full_body=full_body)
        restored.load_catalog(catalog_path)

        assert restored._entries["skill:summarize-reports"] == original
        matches = restored.query("Summarize reports", threshold=0.9)
        assert len(matches) == 1
        assert matches[0].entry_b == "summarize-reports"
        assert matches[0].score == pytest.approx(1.0)

    @pytest.mark.parametrize("whitespace", ["\n", "\r", "\t"])
    def test_description_whitespace_survives_load_and_save(self, tmp_path: Path, whitespace: str) -> None:
        data = _catalog_data()
        description = f"First{whitespace}catalog skill"
        data["entries"][0]["description"] = description
        registry = EmbeddingRegistry(_client())
        registry.load_catalog(_write_catalog(tmp_path / "input.json", data))
        output = tmp_path / "output.json"
        registry.save_catalog(output)
        assert json.loads(output.read_text(encoding="utf-8"))["entries"][0]["description"] == description

    @pytest.mark.parametrize(
        ("field", "character"),
        [("name", char) for char in "\n\r\t"] + [("description", char) for char in "\x00\x0b\x0c\x1b\x85\u202e\ud800"],
    )
    def test_catalog_still_rejects_unsafe_text_on_load_and_save(
        self, tmp_path: Path, field: str, character: str
    ) -> None:
        data = _catalog_data()
        data["entries"][0][field] = f"First{character}skill"
        with pytest.raises(ValueError, match="unsafe control or surrogate"):
            EmbeddingRegistry(_client()).load_catalog(_write_catalog(tmp_path / "unsafe.json", data))

        registry = EmbeddingRegistry(_client())
        registry.load_catalog(_write_catalog(tmp_path / "safe.json"))
        setattr(registry._entries["skill:team-a"], field, data["entries"][0][field])
        with pytest.raises(ValueError, match="unsafe control or surrogate"):
            registry.save_catalog(tmp_path / "output.json")
        assert not (tmp_path / "output.json").exists()

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


_SKILL_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon")


def _keyed_client() -> EmbeddingClient:
    """Embed each known skill name as its own unit vector, whatever else its text says."""

    def embed_one(text: str) -> list[float]:
        words = set(re.findall(r"[a-z]+", text))
        key = next((name for name in _SKILL_NAMES[1:] if name in words), _SKILL_NAMES[0])
        return [1.0 if name == key else 0.0 for name in _SKILL_NAMES]

    client = _client()
    client.embed.side_effect = lambda texts: [embed_one(text) for text in texts]
    client.embed_single.side_effect = embed_one
    client.embed_chunked.side_effect = embed_one
    return client


def _saved_catalog(tmp_path: Path, scan_root: Path, *, full_body: bool = False) -> EmbeddingRegistry:
    builder = EmbeddingRegistry(_keyed_client(), full_body=full_body)
    builder.build_from_directory(scan_root, "skill")
    catalog_path = tmp_path / "catalog.json"
    builder.save_catalog(catalog_path)
    registry = EmbeddingRegistry(_keyed_client(), full_body=full_body)
    registry.load_catalog(catalog_path)
    return registry


def _write_skills(write_skill, root: Path, *names: str) -> Path:
    for name in names:
        write_skill(root, name, f"{name.title()} catalog skill")
    return root


def _target(skill_dir: Path) -> ContentEntry:
    target = extract_from_skill(skill_dir)
    assert target is not None
    return target


def _copy_alpha_to(relative: str):
    def build(tmp_path: Path, skills: Path) -> Path:
        return Path(shutil.copytree(skills / "alpha", tmp_path / relative))

    return build


def _new_nested_alpha(tmp_path: Path, skills: Path) -> Path:
    # A new skill in the catalog tree with the same name and description but its own body.
    nested = skills / "team" / "alpha"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: alpha\ndescription: Alpha catalog skill\n---\n\n# team alpha\n")
    return nested


def _filesystem_is_case_insensitive(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.write_text("")
    try:
        return (directory / "CASE-PROBE").exists()
    finally:
        probe.unlink()


class TestCatalogSelfEntry:
    def test_existing_skill_skips_only_its_own_entry(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", "alpha", "beta")
        registry = _saved_catalog(tmp_path, skills)
        target = _target(skills / "alpha")

        self_entry = registry.catalog_self_entry(target)

        assert self_entry is not None
        assert self_entry.path == "alpha"
        assert [match.classification for match in registry.query_entry(target, 0.75)] == ["EXACT_DUPLICATE"]
        assert registry.query_entry(target, 0.75, exclude_entry_id=self_entry.entry_id) == []

    @pytest.mark.parametrize("full_body", [False, True])
    def test_another_checkout_of_the_catalog_tree_skips_its_own_entry(
        self, tmp_path: Path, write_skill, full_body: bool
    ) -> None:
        build_skills = _write_skills(write_skill, tmp_path / "build" / "skills", "alpha", "beta", "gamma")
        registry = _saved_catalog(tmp_path, build_skills, full_body=full_body)
        shutil.copytree(tmp_path / "build", tmp_path / "ci")

        self_entry = registry.catalog_self_entry(_target(tmp_path / "ci" / "skills" / "alpha"))

        assert self_entry is not None
        assert self_entry.path == "alpha"

    @pytest.mark.parametrize("full_body", [False, True])
    @pytest.mark.parametrize(
        "make_target",
        [
            _copy_alpha_to("candidates/alpha"),
            _copy_alpha_to("other-repo/skills/alpha"),
            _copy_alpha_to("repo/skills/alpha-copy"),
            _new_nested_alpha,
        ],
        ids=["candidates-copy", "other-repo-copy", "renamed-copy", "new-nested-same-name"],
    )
    def test_copies_and_same_name_skills_outside_the_catalog_tree_are_still_reported(
        self, tmp_path: Path, write_skill, full_body: bool, make_target
    ) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", "alpha", "beta")
        registry = _saved_catalog(tmp_path, skills, full_body=full_body)
        target = _target(make_target(tmp_path, skills))
        assert target.name == "alpha"

        assert registry.catalog_self_entry(target) is None
        matches = registry.query_entry(target, 0.75)
        assert [(match.path_b, match.classification) for match in matches] == [("alpha", "EXACT_DUPLICATE")]

    def test_copies_of_a_few_catalog_skills_are_not_a_catalog_tree(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", *_SKILL_NAMES)
        registry = _saved_catalog(tmp_path, skills)
        for name in ("alpha", "beta"):
            shutil.copytree(skills / name, tmp_path / "candidates" / name)

        assert registry.catalog_self_entry(_target(tmp_path / "candidates" / "alpha")) is None

    def test_catalog_tree_with_a_minority_of_edited_skills_still_skips(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", *_SKILL_NAMES)
        registry = _saved_catalog(tmp_path, skills)
        # Three of the four other catalog skills are still present unchanged.
        manifest = skills / "beta" / "SKILL.md"
        manifest.write_text(manifest.read_text().replace("catalog skill", "catalog skill, revised"))

        self_entry = registry.catalog_self_entry(_target(skills / "alpha"))

        assert self_entry is not None
        assert self_entry.path == "alpha"

    def test_catalog_without_another_skill_entry_never_skips(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", "alpha")
        registry = _saved_catalog(tmp_path, skills)

        assert registry.catalog_self_entry(_target(skills / "alpha")) is None

    def test_skills_nested_in_the_target_are_not_catalog_tree_evidence(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", "alpha", "beta")
        _write_skills(write_skill, skills / "alpha", "gamma", "delta")
        registry = _saved_catalog(tmp_path, skills)
        candidate = Path(shutil.copytree(skills / "alpha", tmp_path / "candidates" / "alpha"))

        assert registry.catalog_self_entry(_target(candidate)) is None
        assert registry.catalog_self_entry(_target(skills / "alpha")) is not None

    def test_linked_catalog_skill_is_not_catalog_tree_evidence(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", "alpha", "beta")
        registry = _saved_catalog(tmp_path, skills)
        candidate = Path(shutil.copytree(skills / "alpha", tmp_path / "candidates" / "alpha"))
        try:
            (tmp_path / "candidates" / "beta").symlink_to(skills / "beta", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        assert registry.catalog_self_entry(_target(candidate)) is None

    def test_catalog_tree_evidence_reads_a_bounded_sample(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        others = [f"skill-{index:02d}" for index in range(12)]
        skills = _write_skills(write_skill, tmp_path / "repo" / "skills", "alpha", *others)
        registry = _saved_catalog(tmp_path, skills)
        target = _target(skills / "alpha")
        reads: list[str] = []

        def counting_extract(skill_dir: Path) -> ContentEntry | None:
            reads.append(skill_dir.name)
            return extract_from_skill(skill_dir)

        monkeypatch.setattr(registry_module, "extract_from_skill", counting_extract)

        assert registry.catalog_self_entry(target) is not None
        assert reads == others[: registry_module.SELF_ENTRY_SAMPLE_SIZE]

    def test_longest_catalog_path_identifies_the_target(self, tmp_path: Path, write_skill) -> None:
        repo = tmp_path / "repo"
        _write_skills(write_skill, repo / "skills", "alpha")
        _write_skills(write_skill, repo / "vendor" / "skills", "alpha")
        registry = _saved_catalog(tmp_path, repo)
        target = _target(repo / "vendor" / "skills" / "alpha")

        self_entry = registry.catalog_self_entry(target)

        assert self_entry is not None
        assert self_entry.path == "vendor/skills/alpha"
        matches = registry.query_entry(target, 0.75, exclude_entry_id=self_entry.entry_id)
        assert [match.path_b for match in matches] == ["skills/alpha"]

    @pytest.mark.parametrize(
        ("full_body", "old_text", "new_text", "is_self"),
        [
            (False, "# alpha", "# alpha, revised body", True),
            (False, "Alpha catalog skill", "Alpha catalog skill, revised", False),
            (True, "# alpha", "# alpha, revised body", False),
        ],
    )
    def test_edited_skill_matches_its_entry_only_while_compared_content_is_unchanged(
        self,
        tmp_path: Path,
        write_skill,
        full_body: bool,
        old_text: str,
        new_text: str,
        is_self: bool,
    ) -> None:
        skills = _write_skills(write_skill, tmp_path / "skills", "alpha", "beta")
        registry = _saved_catalog(tmp_path, skills, full_body=full_body)
        manifest = skills / "alpha" / "SKILL.md"
        manifest.write_text(manifest.read_text().replace(old_text, new_text))

        assert (registry.catalog_self_entry(_target(skills / "alpha")) is not None) is is_self

    def test_scan_root_entry_never_identifies_a_target(self, tmp_path: Path, write_skill) -> None:
        # A catalog built from one skill root records it as ".", which carries no path evidence.
        alpha = write_skill(tmp_path / "skills", "alpha", "Alpha catalog skill")
        registry = _saved_catalog(tmp_path, alpha)

        assert registry.catalog_self_entry(_target(alpha)) is None

    def test_differently_cased_target_path(self, tmp_path: Path, write_skill) -> None:
        skills = _write_skills(write_skill, tmp_path / "skills", "alpha", "beta")
        registry = _saved_catalog(tmp_path, skills)
        if _filesystem_is_case_insensitive(tmp_path):
            # The same directory, spelled differently, is still the catalog's own entry.
            self_entry = registry.catalog_self_entry(_target(skills / "ALPHA"))
            assert self_entry is not None
            assert self_entry.path == "alpha"
        else:
            # A separate directory that differs only in case is a copy.
            shutil.copytree(skills / "alpha", skills / "ALPHA")
            assert registry.catalog_self_entry(_target(skills / "ALPHA")) is None
