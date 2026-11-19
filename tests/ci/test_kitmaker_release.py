# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ci import kitmaker_release as kitmaker

VERSION = "2.3.4"
SOURCE = "a" * 40
MANIFEST = "b" * 64
WHEEL = "skillevaluator-2.3.4-py3-none-any.whl"
SDIST = "skillevaluator-2.3.4.tar.gz"
UUID = "71234567-1234-4567-89ab-123456789abc"
BASE = "https://github.com/NVIDIA/SkillEvaluator/releases/download/v2.3.4/"


class MemoryReceipts:
    def __init__(self):
        self.data = None
        self.revision = 0

    def read(self):
        return deepcopy(self.data), self.revision

    def write(self, data, revision):
        assert revision == self.revision
        self.data = deepcopy(data)
        self.revision += 1
        return self.revision


class Portal:
    project_id = 123

    def __init__(self, fail_transport=False, status="completed"):
        self.submitted = []
        self.fail_transport = fail_transport
        self.status = status

    def submit(self, urls, *, upload):
        self.submitted.append((urls, upload))
        if self.fail_transport:
            raise kitmaker.ReleaseError("Unknown submission result")
        return UUID

    def wait(self, receipt):
        assert receipt["uuid"] == UUID
        return self.status


def release():
    return kitmaker.Release(
        VERSION,
        SOURCE,
        MANIFEST,
        {WHEEL: kitmaker.Artifact("c" * 64, 20), SDIST: kitmaker.Artifact("d" * 64, 40)},
    )


def check_absent(_release):
    return set()


def dry_run(receipts):
    portal = Portal()
    kitmaker.release_to_kitmaker("dry-run", release(), portal, receipts, check_absent)
    assert portal.submitted == [((BASE + WHEEL, BASE + SDIST), False)]


def test_publication_requires_a_completed_dry_run():
    portal = Portal()
    with pytest.raises(kitmaker.ReleaseError, match="completed dry run"):
        kitmaker.release_to_kitmaker("publish", release(), portal, MemoryReceipts(), check_absent)
    assert portal.submitted == []


def test_approval_binds_to_manifest_source_and_project():
    receipts = MemoryReceipts()
    dry_run(receipts)
    portal = Portal()
    with pytest.raises(kitmaker.ReleaseError, match="different release"):
        changed = kitmaker.Release(VERSION, "f" * 40, MANIFEST, release().artifacts)
        kitmaker.release_to_kitmaker("publish", changed, portal, receipts, check_absent)
    assert portal.submitted == []


def test_unknown_post_result_is_durable_and_never_posts_twice():
    receipts = MemoryReceipts()
    dry_run(receipts)
    portal = Portal(fail_transport=True)
    with pytest.raises(kitmaker.ReleaseError, match="Unknown submission"):
        kitmaker.release_to_kitmaker("publish", release(), portal, receipts, check_absent)
    retry = Portal()
    with pytest.raises(kitmaker.ReleaseError, match="ambiguous"):
        kitmaker.release_to_kitmaker("publish", release(), retry, receipts, check_absent)
    assert portal.submitted == [((BASE + WHEEL, BASE + SDIST), True)]
    assert retry.submitted == []


def test_failed_batch_retries_only_asset_missing_from_both_indexes():
    receipts = MemoryReceipts()
    dry_run(receipts)
    portal = Portal(status="failed")
    with pytest.raises(kitmaker.ReleaseError, match="failed in Kitmaker"):
        kitmaker.release_to_kitmaker("publish", release(), portal, receipts, check_absent)
    retry = Portal()
    kitmaker.release_to_kitmaker("publish", release(), retry, receipts, lambda _: {WHEEL})
    assert retry.submitted == [((BASE + SDIST,), True)]


def test_completed_submission_with_index_delay_cannot_resubmit():
    receipts = MemoryReceipts()
    dry_run(receipts)
    kitmaker.release_to_kitmaker("publish", release(), Portal(), receipts, check_absent)
    retry = Portal()
    kitmaker.release_to_kitmaker("publish", release(), retry, receipts, check_absent)
    assert retry.submitted == []


def test_matching_public_indexes_accept_a_completed_external_upload():
    receipts = MemoryReceipts()
    dry_run(receipts)
    portal = Portal()
    kitmaker.release_to_kitmaker("publish", release(), portal, receipts, lambda _: {WHEEL, SDIST})
    assert portal.submitted == []


def set_index_links(monkeypatch, pypi, devzone):
    monkeypatch.setattr(kitmaker.product_release, "pypi_entries", lambda *_: pypi)
    monkeypatch.setattr(kitmaker.product_release, "devzone_entries", lambda *_: devzone)


def test_partial_index_replication_cannot_be_retried_as_a_missing_file(monkeypatch):
    raw = b"wheel"
    manifest = kitmaker.Release(
        VERSION, SOURCE, MANIFEST, {WHEEL: kitmaker.Artifact(hashlib.sha256(raw).hexdigest(), 5)}
    )
    index_entry = (WHEEL, "https://files.pythonhosted.org/" + WHEEL, hashlib.sha256(raw).hexdigest(), False, 5)
    set_index_links(monkeypatch, [index_entry], [])
    monkeypatch.setattr(kitmaker.product_release, "get_url", lambda *_: raw)
    with pytest.raises(kitmaker.ReleaseError, match="only one index"):
        kitmaker.check_indexes(manifest)


def test_false_index_metadata_cannot_hide_a_conflicting_download(monkeypatch):
    raw = b"wheel"
    manifest = kitmaker.Release(
        VERSION, SOURCE, MANIFEST, {WHEEL: kitmaker.Artifact(hashlib.sha256(raw).hexdigest(), 5)}
    )
    index_entry = (WHEEL, "https://files.pythonhosted.org/" + WHEEL, hashlib.sha256(raw).hexdigest(), False, 5)
    set_index_links(monkeypatch, [index_entry], [index_entry])
    monkeypatch.setattr(kitmaker.product_release, "get_url", lambda *_: b"other")
    with pytest.raises(kitmaker.ReleaseError, match="could not be verified safely"):
        kitmaker.check_indexes(manifest)


@pytest.mark.parametrize("missing", [True, False])
def test_check_indexes_distinguishes_missing_pypi_version_from_outage(monkeypatch, missing):
    def pypi(*_):
        error = kitmaker.product_release.IndexMissing if missing else kitmaker.product_release.IndexPending
        raise error("cannot see version")

    monkeypatch.setattr(kitmaker.product_release, "pypi_entries", pypi)
    monkeypatch.setattr(kitmaker.product_release, "devzone_entries", lambda *_: [])
    if missing:
        assert kitmaker.check_indexes(release()) == set()
    else:
        with pytest.raises(kitmaker.ReleaseError, match="could not be verified safely"):
            kitmaker.check_indexes(release())


def test_release_manifest_cannot_choose_which_urls_receive_credentials():
    raw = json.dumps(
        {
            "version": VERSION,
            "source_commit": SOURCE,
            "artifacts": [
                {"filename": name, "size": artifact.size, "sha256": artifact.sha256, "url": "https://example.org"}
                for name, artifact in release().artifacts.items()
            ],
        }
    ).encode()
    candidate = kitmaker.parse_manifest(VERSION, SOURCE, hashlib.sha256(raw).hexdigest(), raw)
    assert candidate.urls(candidate.artifacts) == (BASE + WHEEL, BASE + SDIST)
    with pytest.raises(kitmaker.ReleaseError, match="manifest digest"):
        kitmaker.parse_manifest(VERSION, SOURCE, MANIFEST, raw)


def test_vault_write_requires_the_read_revision_and_versioned_path(monkeypatch):
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return {"data": {"version": 10}}

    monkeypatch.setattr(kitmaker, "request_json", request)
    vault = kitmaker.Vault("https://vault.example.org", "robot-token", "release", None)
    assert vault.write("kv/data/evidence/2.3.4", {"state": "submitting"}, 9) == 10
    assert calls[0][0] == "https://vault.example.org/v1/kv/data/evidence/2.3.4"
    assert calls[0][1]["data"] == {"options": {"cas": 9}, "data": {"state": "submitting"}}
    assert calls[0][1]["headers"]["X-Vault-Token"] == "robot-token"


def test_portal_submits_exact_json_and_refreshes_auth_when_polling(monkeypatch):
    calls = []
    oidc = iter(["oidc-for-post", "fresh-oidc"])

    def request(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/releases"):
            return {"release_uuid": UUID}
        return {
            "uuid": UUID,
            "project_id": 123,
            "status": "completed",
            "jobs": [{"wheel_url": BASE + WHEEL, "status": "success"}],
        }

    monkeypatch.setattr(kitmaker, "request_json", request)
    monkeypatch.setattr(kitmaker, "oidc_token", lambda _: next(oidc))
    portal = kitmaker.PortalClient(123, "owner@example.org", "machine-token")
    assert portal.submit((BASE + WHEEL,), upload=False) == UUID
    assert portal.wait({"uuid": UUID, "urls": [BASE + WHEEL]}) == "completed"
    assert calls[0][0] == "http://127.0.0.1:8888/kitmaker-portal/api/v0/projects/123/releases"
    assert calls[0][1]["data"] == {
        "project_name": "skillevaluator",
        "payload": [
            {
                "pic": "owner@example.org",
                "job_type": "wheel-release-job",
                "url": BASE + WHEEL,
                "upload": False,
                "mirroring_strategy": "upload-both",
            }
        ],
    }
    assert calls[1][0].endswith("/status/" + UUID)
    assert calls[1][1]["headers"]["X-Charon-GHA-Token"] == "fresh-oidc"


def test_portal_completion_must_cover_every_submitted_asset(monkeypatch):
    monkeypatch.setattr(kitmaker, "oidc_token", lambda _: "oidc")
    monkeypatch.setattr(
        kitmaker,
        "request_json",
        lambda *_, **__: {
            "uuid": UUID,
            "project_id": 123,
            "status": "completed",
            "jobs": [{"wheel_url": BASE + WHEEL, "status": "success"}],
        },
    )
    portal = kitmaker.PortalClient(123, "owner@example.org", "machine-token")
    with pytest.raises(kitmaker.ReleaseError, match="every submitted artifact"):
        portal.wait({"uuid": UUID, "urls": [BASE + WHEEL, BASE + SDIST]})


def test_reusable_workflow_uses_an_environment_gate_after_validation():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/release-kitmaker.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_call"}
    jobs = workflow["jobs"]
    assert jobs["dry-run"]["environment"] == "release-kitmaker-dry-run"
    assert jobs["upload"]["environment"] == "release-kitmaker"
    assert jobs["upload"]["needs"] == "dry-run"
    assert "release.yml@refs/heads/main" in jobs["dry-run"]["if"]
    for job in jobs.values():
        assert job["steps"][-1]["env"]["RELEASE_DEVZONE_INDEX_URL"] == "${{ vars.RELEASE_DEVZONE_INDEX_URL }}"


def test_vault_checkpoint_failure_does_not_send_post(monkeypatch):
    receipts = MemoryReceipts()
    dry_run(receipts)
    portal = Portal()

    def failed_checkpoint(*_args):
        raise kitmaker.ReleaseError("Compare-and-set conflict")

    monkeypatch.setattr(receipts, "write", failed_checkpoint)
    with pytest.raises(kitmaker.ReleaseError, match="Compare-and-set conflict"):
        kitmaker.release_to_kitmaker("publish", release(), portal, receipts, check_absent)
    assert portal.submitted == []


def test_status_timeout_resumes_saved_uuid_without_a_new_post():
    class TimesOut(Portal):
        def wait(self, _receipt):
            raise kitmaker.ReleaseError("Status timed out")

    receipts = MemoryReceipts()
    dry_run(receipts)
    portal = TimesOut()
    with pytest.raises(kitmaker.ReleaseError, match="Status timed out"):
        kitmaker.release_to_kitmaker("publish", release(), portal, receipts, check_absent)
    retry = Portal()
    kitmaker.release_to_kitmaker("publish", release(), retry, receipts, check_absent)
    assert portal.submitted == [((BASE + WHEEL, BASE + SDIST), True)]
    assert retry.submitted == []


def test_github_rejects_mutable_release_before_reading_assets(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "read-only-token")
    calls = []

    def github(url, **_kwargs):
        calls.append(url)
        return {
            "tag_name": "v2.3.4",
            "immutable": False,
            "draft": False,
            "prerelease": False,
            "published_at": "2026-11-17T00:00:00Z",
        }

    monkeypatch.setattr(kitmaker, "request_json", github)
    with pytest.raises(kitmaker.ReleaseError, match="immutable GitHub release"):
        kitmaker.verify_github_release(VERSION, SOURCE, MANIFEST)
    assert calls == ["https://api.github.com/repos/NVIDIA/SkillEvaluator/releases/tags/v2.3.4"]


def test_portal_failed_batch_with_active_job_remains_unresolved(monkeypatch):
    monkeypatch.setattr(kitmaker, "oidc_token", lambda _: "oidc")
    monkeypatch.setattr(
        kitmaker,
        "request_json",
        lambda *_, **__: {
            "uuid": UUID,
            "project_id": 123,
            "status": "failed",
            "jobs": [
                {"wheel_url": BASE + WHEEL, "status": "failed"},
                {"wheel_url": BASE + SDIST, "status": "in_progress"},
            ],
        },
    )
    portal = kitmaker.PortalClient(123, "owner@example.org", "machine-token")
    with pytest.raises(kitmaker.ReleaseError, match="all artifact jobs are terminal"):
        portal.wait({"uuid": UUID, "urls": [BASE + WHEEL, BASE + SDIST]})
