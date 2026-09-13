# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned regression for public SkillEvaluator issue #32's OpenClaw layout."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.embedding.client import EmbeddingClient

_FIXTURE = Path(__file__).parent / "fixtures" / "openclaw-autoreview"
_OPENCLAW_COMMIT = "2a409d348a4bcf6f15e41e9a20efd0b298a32528"


def _git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _verify_pinned_fixture(source: dict[str, object]) -> None:
    assert source["repository"] == "https://github.com/openclaw/agent-skills"
    assert source["commit"] == _OPENCLAW_COMMIT
    assert source["path"] == "skills/autoreview"
    assert source["fixture_kind"] == "exact-pinned-files"

    files = source["files"]
    assert isinstance(files, dict)
    for name in ("SKILL.md", "AGENTS.md"):
        metadata = files[name]
        assert isinstance(metadata, dict)
        raw = (_FIXTURE / name).read_bytes()
        assert metadata["mode"] == "100644"
        assert len(raw) == metadata["bytes"]
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
        assert _git_blob_sha(raw) == metadata["blob"]

    alias = source["alias"]
    assert isinstance(alias, dict)
    target = alias["target"]
    assert isinstance(target, str)
    raw_target = target.encode()
    assert alias["mode"] == "120000"
    assert len(raw_target) == alias["bytes"]
    assert hashlib.sha256(raw_target).hexdigest() == alias["sha256"]
    assert _git_blob_sha(raw_target) == alias["blob"]


def _materialize_pinned_layout(tmp_path: Path) -> Path:
    source = json.loads((_FIXTURE / "SOURCE.json").read_text(encoding="utf-8"))
    _verify_pinned_fixture(source)
    skill = tmp_path / "skills" / "autoreview"
    shutil.copytree(_FIXTURE, skill)
    try:
        (skill / source["alias"]["name"]).symlink_to(source["alias"]["target"])
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    return skill


class _FakeEmbeddingsEndpoint:
    """OpenAI-compatible embeddings boundary with deterministic orthogonal vectors."""

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.requests: list[tuple[str, str, int]] = []

    def create(self, *, model: str, input: list[str], encoding_format: str) -> SimpleNamespace:  # noqa: A002
        self.requests.append((model, encoding_format, len(input)))
        response_data: list[SimpleNamespace] = []
        for response_index, text in enumerate(input):
            vector_index = len(self.inputs)
            assert vector_index < 64, "Pinned OpenClaw regression unexpectedly exceeded its vector fixture"
            self.inputs.append(text)
            vector = [0.0] * 64
            vector[vector_index] = 1.0
            response_data.append(SimpleNamespace(index=response_index, embedding=vector))
        return SimpleNamespace(data=response_data)


def test_pinned_openclaw_alias_succeeds_through_similarity_and_context_clis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _materialize_pinned_layout(tmp_path)
    peer = skill.parent / "peer-skill"
    peer.mkdir()
    (peer / "SKILL.md").write_text(
        "---\n"
        "name: peer-skill\n"
        "description: A second catalog entry for public collection similarity.\n"
        "metadata:\n"
        "  author: Test Author <test@example.com>\n"
        "---\n\n"
        "# Peer Skill\n",
        encoding="utf-8",
    )
    endpoint = _FakeEmbeddingsEndpoint()
    fake_client = SimpleNamespace(embeddings=endpoint)
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(EmbeddingClient, "_get_client", lambda _self: fake_client)

    similarity = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(skill.parent),
            "--type",
            "skill",
            "--threshold",
            "1.0",
            "--report",
            "cli",
        ],
    )
    similarity_request_count = len(endpoint.inputs)
    context = CliRunner().invoke(
        cli,
        [
            "context-optimization-check",
            str(skill),
            "--threshold",
            "1.0",
            "--report",
            "cli",
        ],
    )
    similarity_output = unstyle(similarity.output)
    context_output = unstyle(context.output)

    assert similarity.exit_code == 0, similarity.output
    assert "Similarity Check" in similarity_output
    assert "[PASS] All validations passed" in similarity_output
    assert "symlink or reparse point" not in similarity_output
    assert similarity_request_count == 2
    assert context.exit_code == 0, context.output
    assert "Context Deduplication" in context_output
    assert "[PASS] All validations passed" in context_output
    assert "symlink or reparse point" not in context_output
    assert len(endpoint.inputs) == 40
    agents_text = (_FIXTURE / "AGENTS.md").read_text(encoding="utf-8").strip()
    assert sum(text.strip() == agents_text for text in endpoint.inputs) == 1
    assert endpoint.requests
    assert all(encoding_format == "float" and count > 0 for _model, encoding_format, count in endpoint.requests)
