# SPDX-License-Identifier: Apache-2.0
"""Default-branch admission requires the live protected branch head."""

import importlib.util
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SHA = "a" * 40
BRANCH = {"name": "main", "protected": True, "commit": {"id": SHA}}


@pytest.fixture
def gate(monkeypatch):
    spec = importlib.util.spec_from_file_location("default_gate", ROOT / "ci/eligible_gitlab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = {
        "CI_PIPELINE_SOURCE": "push",
        "CI_API_V4_URL": "https://gitlab.example.test/api/v4",
        "CI_PROJECT_ID": "42",
        "CI_COMMIT_SHA": SHA,
        "CI_COMMIT_BRANCH": "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": "true",
        "CI_JOB_TOKEN": "job-secret",
        "SKILLEVALUATOR_GITLAB_TOKEN": "publisher-secret",
    }
    monkeypatch.setattr(module.os, "environ", env)
    return module


def fake_api(monkeypatch, gate, payload):
    real_builder = urllib.request.build_opener

    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self, request):
            assert request.full_url == (
                "https://gitlab.example.test/api/v4/projects/42/repository/branches/"
                + ("release%2Fstable%20%231" if gate.os.environ["CI_DEFAULT_BRANCH"] != "main" else "main")
            )
            assert request.timeout == 30
            headers = {
                key.lower(): value for key, value in request.header_items() if key.lower() not in {"host", "user-agent"}
            }
            read_token = gate.os.environ.get("SKILLEVALUATOR_GITLAB_READ_TOKEN")
            assert headers == ({"private-token": read_token} if read_token else {"job-token": "job-secret"})
            if isinstance(payload, Exception):
                raise payload
            response = io.BytesIO(payload)
            response.code = 200
            response.msg = "OK"
            response.info = dict
            return response

    def builder(*handlers):
        assert any(
            isinstance(handler, urllib.request.HTTPRedirectHandler)
            and handler.redirect_request(None, None, 302, "", {}, "https://elsewhere.test") is None
            for handler in handlers
        )
        return real_builder(*handlers, HTTPS())

    monkeypatch.setattr(gate.urllib.request, "build_opener", builder)


@pytest.mark.parametrize("source", ["push", "web"])
@pytest.mark.parametrize("read_token", [None, "reader-secret"])
def test_current_protected_default_head_is_admitted(gate, monkeypatch, source, read_token):
    gate.os.environ["CI_PIPELINE_SOURCE"] = source
    if read_token:
        gate.os.environ["SKILLEVALUATOR_GITLAB_READ_TOKEN"] = read_token
    fake_api(monkeypatch, gate, json.dumps(BRANCH).encode())
    assert gate.main() == 0


@pytest.mark.parametrize(
    "current",
    [
        None,
        {},
        {**BRANCH, "commit": {"id": "b" * 40}},
        {**BRANCH, "name": "other"},
        {**BRANCH, "protected": False},
        {**BRANCH, "protected": "true"},
        {**BRANCH, "commit": None},
        {**BRANCH, "commit": []},
        {**BRANCH, "commit": {}},
    ],
)
def test_unverified_stale_or_forged_branch_is_rejected(gate, current):
    assert not gate.eligible(gate.os.environ, current)


@pytest.mark.parametrize(
    "payload",
    [
        {**BRANCH, "commit": {"id": "b" * 40}},
        {**BRANCH, "protected": False},
        [],
        None,
    ],
)
def test_live_branch_advance_or_unprotection_blocks_provider_admission(gate, monkeypatch, payload):
    fake_api(monkeypatch, gate, json.dumps(payload).encode())
    assert gate.main() == 2


def test_branch_name_is_encoded_as_one_path_segment(gate, monkeypatch):
    gate.os.environ.update(CI_DEFAULT_BRANCH="release/stable #1", CI_COMMIT_BRANCH="release/stable #1")
    fake_api(monkeypatch, gate, json.dumps({**BRANCH, "name": "release/stable #1"}).encode())
    assert gate.main() == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("CI_API_V4_URL", ""),
        ("CI_API_V4_URL", "http://gitlab.example.test/api/v4"),
        ("CI_API_V4_URL", "https://reader-secret@gitlab.example.test/api/v4"),
        ("CI_API_V4_URL", "https://gitlab.example.test/api/v4?token=reader-secret"),
        ("CI_API_V4_URL", "https://[bad/api/v4"),
        ("CI_PROJECT_ID", ""),
        ("CI_DEFAULT_BRANCH", ""),
        ("CI_COMMIT_SHA", ""),
    ],
)
def test_invalid_or_missing_api_identity_fails_closed(gate, monkeypatch, key, value, capsys):
    gate.os.environ[key] = value
    monkeypatch.setattr(
        gate.urllib.request, "build_opener", lambda *_args: pytest.fail("invalid API must not be called")
    )
    assert gate.main() == 2
    assert "reader-secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"not json reader-secret", id="invalid-json"),
        pytest.param(b"x" * (1024 * 1024 + 1), id="oversized"),
        urllib.error.URLError("reader-secret publisher-secret job-secret"),
        urllib.error.HTTPError("https://reader-secret", 302, "publisher-secret", {}, None),
    ],
)
def test_transport_and_malformed_responses_fail_without_credential_leaks(gate, monkeypatch, capsys, payload):
    fake_api(monkeypatch, gate, payload)
    assert gate.main() == 2
    output = capsys.readouterr().out
    assert all(secret not in output for secret in ("reader-secret", "publisher-secret", "job-secret"))


@pytest.mark.parametrize(
    "changes",
    [
        {"CI_COMMIT_BRANCH": "feature"},
        {"CI_COMMIT_REF_PROTECTED": "false"},
        {"CI_PIPELINE_SOURCE": "schedule"},
        {"CI_PIPELINE_SOURCE": "api"},
        {"CI_PIPELINE_SOURCE": "external_pull_request_event"},
    ],
)
def test_live_branch_cannot_override_pipeline_constraints(gate, monkeypatch, changes):
    gate.os.environ.update(changes)
    monkeypatch.setattr(
        gate.urllib.request, "build_opener", lambda *_args: pytest.fail("ineligible pipeline must not call API")
    )
    assert not gate.eligible(gate.os.environ, BRANCH)
    assert gate.main() == 2
