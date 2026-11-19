# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from skillevaluator.validators.hook_waivers import HookWaiver, load_hook_waivers


def _entry(**overrides: object) -> dict:
    return {
        "id": "HOOK-01",
        "manifest": "hooks/hooks.json",
        "manifest_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "payload": [{"path": "hooks/run.sh", "sha256": "c" * 64}],
        "reason": "Reviewed the local build hook and its packaged files.",
        "owner": "plugin-reviewers",
        "created": date(2025, 10, 19),
        "expires": date(2027, 1, 19),
        **overrides,
    }


def _write_waivers(tmp_path: Path, *entries: dict) -> Path:
    path = tmp_path.resolve() / "hook-waivers.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "waivers": list(entries)}), encoding="utf-8")
    return path


def test_loads_immutable_review_record(tmp_path: Path) -> None:
    path = _write_waivers(tmp_path, _entry(id=" HOOK-01 "))

    waivers = load_hook_waivers(path)

    assert isinstance(waivers, tuple)
    [waiver] = waivers
    assert isinstance(waiver, HookWaiver)
    assert waiver.id == "HOOK-01"
    assert waiver.manifest == "hooks/hooks.json"
    assert waiver.manifest_sha256 == "a" * 64
    assert waiver.artifact_sha256 == "b" * 64
    assert waiver.payload == {"hooks/run.sh": "c" * 64}
    assert waiver.reason == "Reviewed the local build hook and its packaged files."
    assert waiver.owner == "plugin-reviewers"
    assert waiver.created == date(2025, 10, 19)
    assert waiver.expires == date(2027, 1, 19)
    with pytest.raises(FrozenInstanceError):
        waiver.owner = "someone-else"
    with pytest.raises(TypeError):
        waiver.payload["hooks/run.sh"] = "d" * 64


def test_loads_expired_records_and_expiry_date(tmp_path: Path) -> None:
    today = datetime.now(UTC).date()
    yesterday = today - timedelta(days=1)
    path = _write_waivers(
        tmp_path,
        _entry(created=yesterday, expires=yesterday, payload=[]),
        _entry(id="HOOK-02", created=today.isoformat(), expires=today.isoformat()),
    )

    expired, expires_today = load_hook_waivers(path)

    assert expired.expires == yesterday
    assert not expired.payload
    assert expires_today.expires == today


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("version: true\nwaivers: []\n", id="boolean-version"),
        pytest.param("version: 2\nwaivers: []\n", id="unsupported-version"),
        pytest.param("version: 1\nwaivers: {}\n", id="waivers-not-list"),
        pytest.param("version: 1\nwaivers: []\nextra: false\n", id="extra-root-key"),
        pytest.param("version: 1\nversion: 1\nwaivers: []\n", id="duplicate-root-key"),
        pytest.param("version: 1\nwaivers: [\n", id="malformed-yaml"),
    ],
)
def test_rejects_invalid_document(tmp_path: Path, document: str) -> None:
    path = tmp_path.resolve() / "hook-waivers.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError):
        load_hook_waivers(path)


@pytest.mark.parametrize(
    "invalid_fields",
    [
        pytest.param({"extra": True}, id="extra-field"),
        pytest.param({"id": " "}, id="blank-id"),
        pytest.param({"reason": "brief"}, id="short-reason"),
        pytest.param({"owner": 123}, id="owner-not-string"),
        pytest.param({"manifest_sha256": "A" * 64}, id="uppercase-manifest-hash"),
        pytest.param({"artifact_sha256": "b" * 63}, id="short-artifact-hash"),
        pytest.param({"payload": [{"path": "hooks/run.sh", "sha256": "z" * 64}]}, id="non-hex-payload-hash"),
        pytest.param({"payload": [{"path": "hooks/run.sh", "sha256": "c" * 64, "line": 1}]}, id="extra-payload-key"),
        pytest.param({"payload": [{"path": "hooks/run.sh", "sha256": "c" * 64}] * 2}, id="duplicate-payload-path"),
        pytest.param({"payload": [{"path": "hooks/hooks.json", "sha256": "a" * 64}]}, id="manifest-in-payload"),
        pytest.param({"created": "2025-02-30"}, id="invalid-date"),
        pytest.param({"created": "2025-10-19T12:00:00Z"}, id="datetime-instead-of-date"),
        pytest.param({"expires": "2025-10-18"}, id="expires-before-created"),
    ],
)
def test_rejects_invalid_entry_even_after_a_valid_one(tmp_path: Path, invalid_fields: dict) -> None:
    path = _write_waivers(tmp_path, _entry(), _entry(**{"id": "HOOK-02", **invalid_fields}))

    with pytest.raises(ValueError):
        load_hook_waivers(path)


def test_rejects_missing_artifact_hash(tmp_path: Path) -> None:
    path = _write_waivers(tmp_path, {key: value for key, value in _entry().items() if key != "artifact_sha256"})

    with pytest.raises(ValueError, match="artifact_sha256"):
        load_hook_waivers(path)


def test_rejects_duplicate_ids_after_trimming(tmp_path: Path) -> None:
    path = _write_waivers(tmp_path, _entry(), _entry(id=" HOOK-01 "))

    with pytest.raises(ValueError, match="duplicate"):
        load_hook_waivers(path)


@pytest.mark.parametrize(
    "path",
    [
        "/hooks/run.sh",
        "../run.sh",
        "hooks/../run.sh",
        "./hooks/run.sh",
        "hooks//run.sh",
        "hooks/run.sh/",
        "hooks\\run.sh",
        "C:hooks/run.sh",
        "hooks/file:stream",
        "hooks/a.zip!/run.sh",
        "hooks/\x1frun.sh",
    ],
)
def test_rejects_noncanonical_paths(tmp_path: Path, path: str) -> None:
    for entry in (_entry(manifest=path), _entry(payload=[{"path": path, "sha256": "c" * 64}])):
        with pytest.raises(ValueError, match="path"):
            load_hook_waivers(_write_waivers(tmp_path, entry))


def test_rejects_review_created_in_the_future(tmp_path: Path) -> None:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    path = _write_waivers(tmp_path, _entry(created=tomorrow, expires=tomorrow))

    with pytest.raises(ValueError, match="created"):
        load_hook_waivers(path)


def test_rejects_excess_waivers(tmp_path: Path) -> None:
    path = _write_waivers(tmp_path, *(_entry(id=f"HOOK-{i}") for i in range(33)))

    with pytest.raises(ValueError, match="32"):
        load_hook_waivers(path)


def test_rejects_excess_payload_pins(tmp_path: Path) -> None:
    payload = [{"path": f"hooks/run-{i}.sh", "sha256": "c" * 64} for i in range(513)]

    with pytest.raises(ValueError, match="512"):
        load_hook_waivers(_write_waivers(tmp_path, _entry(payload=payload)))


@pytest.mark.parametrize("contents", [b"\xff", b"# " + b"x" * 1024 * 1024], ids=["non-utf8", "oversized"])
def test_rejects_unreadable_approval_bytes(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path.resolve() / "hook-waivers.yaml"
    path.write_bytes(contents)

    with pytest.raises(ValueError):
        load_hook_waivers(path)


def test_rejects_symlinked_approval_file(tmp_path: Path) -> None:
    approval = _write_waivers(tmp_path, _entry())
    link = approval.with_name("link.yaml")
    try:
        link.symlink_to(approval)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(ValueError):
        load_hook_waivers(link)
