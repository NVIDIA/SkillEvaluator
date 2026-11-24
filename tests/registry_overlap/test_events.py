# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from skillevaluator.registry_overlap.events import embedding_input, parse_batch, read_batch

MANIFEST = "---\nname: pdf-summary\ndescription: Summarize a PDF.\n---\n"


def _source_event(**overrides):
    return {
        "id": "public/pdf",
        "sequence": 4,
        "revision": "1.0",
        "kind": "upsert",
        "manifest": MANIFEST,
        "artifact": "https://example.com/public/pdf/1.0",
        **overrides,
    }


def _seal(*events):
    raw = json.dumps({"schema_version": 1, "source": "registry", "cursor": 1, "events": events}).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def test_verified_batch_preserves_identity_and_fingerprints():
    raw, digest = _seal(_source_event())
    batch = parse_batch(raw, digest)
    event = batch.events[0]
    assert (batch.source, batch.cursor, batch.digest) == ("registry", 1, digest)
    assert (event.skill_id, event.sequence, event.revision) == ("public/pdf", 4, "1.0")
    assert event.text == "pdf-summary: Summarize a PDF."
    assert event.manifest_sha256 == hashlib.sha256(MANIFEST.encode()).hexdigest()
    assert event.input_sha256 == hashlib.sha256(event.text.encode()).hexdigest()
    same_input = parse_batch(*_seal(_source_event(manifest=MANIFEST + "\nNew body.\n"))).events[0]
    assert same_input.input_sha256 == event.input_sha256
    assert same_input.manifest_sha256 != event.manifest_sha256
    assert same_input.digest != event.digest


def test_removal_has_no_embedding_input():
    raw, digest = _seal({"id": "public/pdf", "sequence": 5, "revision": "1.0", "kind": "remove"})
    event = parse_batch(raw, digest).events[0]
    assert event.text is None
    assert event.input_sha256 is None


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "remove"},
        {"sequence": True},
        {"sequence": 2**63},
        {"manifest": "---\nname: pdf\ndescription: [invalid]\n---\n"},
        {"artifact": "url\ninjected-log"},
        {"id": ""},
    ],
)
def test_reject_invalid_event(changes):
    with pytest.raises(ValueError, match="Registry event 1"):
        parse_batch(*_seal(_source_event(**changes)))


@pytest.mark.parametrize("escape", [r"\0", r"\x1B", r"\u0085", r"\u202E", r"\uD800"])
def test_rejects_controls_after_yaml_decoding_for_ingest_and_query(escape):
    manifest = f'---\nname: pdf-summary\ndescription: "PDF{escape}summary"\n---\n'

    with pytest.raises(ValueError, match="control or surrogate"):
        embedding_input(manifest)
    with pytest.raises(ValueError, match="Registry event 1"):
        parse_batch(*_seal(_source_event(manifest=manifest)))


def test_preserves_multiline_unicode_embedding_input():
    manifest = '---\nname: résumé\ndescription: "中文\\nPDF\\ttable"\n---\n'
    batch = parse_batch(*_seal(_source_event(manifest=manifest)))

    assert batch.events[0].text == "résumé: 中文\nPDF\ttable"


@pytest.mark.parametrize(
    "artifact",
    [
        "javascript:alert(1)",
        "http://example.com/releases/1",
        "//example.com/releases/1",
        "https://user:password@example.com/releases/1",
        "https://@example.com/releases/1",
        "https://example.com/releases/1?token=secret",
        "https://example.com/releases/1?",
        "https://example.com/releases/1#secret",
        "https://example.com/releases/1#",
        "https:///releases/1",
        "https://example.com:65536/releases/1",
        "https://example.com\\releases\\1",
        " https://example.com/releases/1",
        "https://example.com/release card",
        "ref:",
        "ref:https://example.com/releases/1",
        "ref:release/id",
    ],
)
def test_rejects_unsafe_artifact_references(artifact):
    with pytest.raises(ValueError, match="Registry event 1") as exc:
        parse_batch(*_seal(_source_event(artifact=artifact)))

    assert artifact not in str(exc.value)


@pytest.mark.parametrize("artifact", ["https://example.com:8443/releases/1%20beta", "ref:pdf-v1.0+build_8"])
def test_preserves_safe_artifact_references(artifact):
    batch = parse_batch(*_seal(_source_event(artifact=artifact)))

    assert batch.events[0].artifact == artifact


def test_rejects_oversized_json_integer_before_event_validation():
    raw, _ = _seal(_source_event())
    raw = raw.replace(b'"sequence": 4', b'"sequence": ' + b"9" * 1000)

    with pytest.raises(ValueError, match="oversized registry batch JSON"):
        parse_batch(raw, hashlib.sha256(raw).hexdigest())


def test_accepts_largest_supported_sequence():
    batch = parse_batch(*_seal(_source_event(sequence=2**63 - 1)))

    assert batch.events[0].sequence == 2**63 - 1


def test_reject_manifest_above_preflight_size_limit():
    with pytest.raises(ValueError, match="oversized registry batch JSON"):
        parse_batch(*_seal(_source_event(manifest=MANIFEST + "x" * (1024 * 1024))))


def test_accepts_uppercase_checksum_and_preserves_lowercase_digest():
    raw, digest = _seal(_source_event())

    assert parse_batch(raw, digest.upper()).digest == digest


@pytest.mark.parametrize("checksum", ["0" * 63, "0" * 65, "g" * 64, " " + "0" * 63])
def test_rejects_invalid_checksum_format(checksum):
    raw, _ = _seal(_source_event())

    with pytest.raises(ValueError, match="checksum must be 64 hexadecimal characters"):
        parse_batch(raw, checksum)


def test_reject_checksum_and_duplicate_json():
    raw, digest = _seal(_source_event())
    with pytest.raises(ValueError, match="checksum does not match"):
        parse_batch(raw + b" ", digest.upper())
    duplicate = raw.replace(b'"cursor": 1', b'"cursor": 1, "cursor": 2')
    with pytest.raises(ValueError, match="JSON"):
        parse_batch(duplicate, hashlib.sha256(duplicate).hexdigest())


def test_read_batch_rejects_symlink(tmp_path):
    target, link = tmp_path / "target.json", tmp_path / "link.json"
    raw, digest = _seal(_source_event())
    target.write_bytes(raw)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert read_batch(target, digest) == parse_batch(raw, digest)
    with pytest.raises(ValueError, match="securely read"):
        read_batch(link, digest)
