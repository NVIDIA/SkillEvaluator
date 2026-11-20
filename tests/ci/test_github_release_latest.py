# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest

from ci import github_release

SOURCE = "1" * 40


def published(tag, **overrides):
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "published_at": "2026-11-18T00:10:18Z",
        **overrides,
    }


def github_releases(monkeypatch, status, latest, *, pages=None):
    endpoint = github_release.API_ROOT

    def run(args, **_kwargs):
        if args == ["gh", "api", "--include", f"{endpoint}/releases/latest"]:
            body = f"HTTP/2.0 {status} Response\nContent-Type: application/json\n\n{json.dumps(latest)}"
            return CompletedProcess(args, 0 if status == 200 else 1, stdout=body)
        if args == ["gh", "api", "--paginate", "--slurp", f"{endpoint}/releases?per_page=100"]:
            return CompletedProcess(args, 0, stdout=json.dumps(pages))
        pytest.fail(f"Unexpected GitHub call: {args!r}")

    monkeypatch.setattr(github_release.subprocess, "run", run)


def stage_draft(monkeypatch, version):
    names = github_release.expected_filenames(version)
    draft = {"id": 101, "tag_name": f"v{version}", "draft": True, "assets": []}

    def api(path, method="GET", data=None):
        assert path == "releases/101"
        if method == "PATCH":
            return {"draft": False, "immutable": True}
        return draft

    gh_api = Mock(side_effect=api)
    monkeypatch.setattr(github_release, "check_tag", lambda *_: SOURCE)
    monkeypatch.setattr(github_release, "verify_local", Mock(return_value={}))
    monkeypatch.setattr(github_release, "get_release", Mock(return_value=draft))
    monkeypatch.setattr(github_release, "api", gh_api)
    monkeypatch.setattr(github_release, "release_notes", Mock(return_value="Reviewed notes"))
    monkeypatch.setattr(github_release.product_release, "inspect_public_indexes", Mock())
    monkeypatch.setattr(github_release, "verify_staged", Mock(return_value=names))
    return gh_api


@pytest.mark.parametrize(
    ("version", "current_tag", "value"),
    [
        ("1.9.9", "v1.10.0", "false"),
        ("1.10.0", "v1.9.9", "true"),
        ("2.0.0", "v1.99.99", "true"),
    ],
)
def test_publish_explicitly_updates_latest_only_for_a_higher_version(
    monkeypatch, tmp_path, version, current_tag, value
):
    api = stage_draft(monkeypatch, version)
    github_releases(monkeypatch, 200, published(current_tag))
    github_release.stage(version, SOURCE, tmp_path, "123-1")
    api.assert_called_with("releases/101", "PATCH", {"draft": False, "body": "Reviewed notes", "make_latest": value})


def test_publish_first_stable_release_when_latest_and_stable_list_are_empty(monkeypatch, tmp_path):
    api = stage_draft(monkeypatch, "1.0.0")
    github_releases(
        monkeypatch,
        404,
        {"message": "Not Found"},
        pages=[[published("v1.0.0", draft=True, published_at=None), published("v1.0.0-rc1", prerelease=True)]],
    )
    github_release.stage("1.0.0", SOURCE, tmp_path, "123-1")
    api.assert_called_with("releases/101", "PATCH", {"draft": False, "body": "Reviewed notes", "make_latest": "true"})


@pytest.mark.parametrize(
    ("status", "latest", "pages", "message"),
    [
        (403, {"message": "Forbidden"}, None, "HTTP 403"),
        (500, {"message": "Server error"}, None, "HTTP 500"),
        (200, published("stable"), None, "canonical"),
        (200, published("v1.9.9", prerelease=True), None, "published stable"),
        (404, {"message": "Not Found"}, [[published("v1.9.9")]], "published stable"),
        (404, {"message": "Not Found"}, [[], [None]], "listing"),
    ],
)
def test_ambiguous_latest_lookup_leaves_draft_unpublished(monkeypatch, tmp_path, status, latest, pages, message):
    api = stage_draft(monkeypatch, "1.10.0")
    github_releases(monkeypatch, status, latest, pages=pages)
    with pytest.raises(ValueError, match=message):
        github_release.stage("1.10.0", SOURCE, tmp_path, "123-1")
    api.assert_called_once_with("releases/101")
