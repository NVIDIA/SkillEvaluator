# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signed execution receipts reject tampering and cross-run replay."""

import io
import json
import stat
import zipfile
from datetime import datetime
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWK = {
    **json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key())),
    "kid": "test-key",
    "alg": "RS256",
    "use": "sig",
}
JWKS = {"keys": [JWK]}


def zip_receipt(token, name="execution.jwt"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, token)
    return output.getvalue()


def signed_receipt(run, **changes):
    started = int(datetime.fromisoformat(run["run_started_at"]).timestamp())
    sha = "d" * 40 if run["event"] == "pull_request" else run["head_sha"]
    ref = "refs/pull/7/merge" if run["event"] == "pull_request" else "refs/heads/main"
    claims = {
        "iss": "https://token.actions.githubusercontent.com",
        "aud": "urn:skillevaluator:report-provenance:v1",
        "iat": started + 30,
        "nbf": started - 570,
        "exp": started + 330,
        "repository": "owner/consumer",
        "repository_id": "1234",
        "run_id": str(run["id"]),
        "run_attempt": str(run["run_attempt"]),
        "event_name": run["event"],
        "workflow_ref": f"owner/consumer/.github/workflows/skill-checks.yml@{ref}",
        "workflow_sha": sha,
        "sha": sha,
        "ref": ref,
        "job_workflow_ref": run["referenced_workflows"][0]["path"],
        "job_workflow_sha": run["referenced_workflows"][0]["sha"],
        **changes,
    }
    return zip_receipt(jwt.encode(claims, KEY, algorithm="RS256", headers={"kid": "test-key"}))


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "ci"))
    import verify_provenance

    monkeypatch.setattr(verify_provenance, "fetch_signing_keys", lambda: JWKS)
    return verify_provenance


@pytest.fixture
def record():
    return {
        "id": 123,
        "repository": {"full_name": "owner/consumer", "id": 1234},
        "run_attempt": 1,
        "event": "pull_request",
        "head_sha": "a" * 40,
        "run_started_at": "2026-10-21T12:00:00Z",
        "updated_at": "2026-10-21T12:02:00Z",
        "referenced_workflows": [
            {"path": "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@" + "c" * 40, "sha": "c" * 40}
        ],
    }


def verify(verifier, record, payload):
    return verifier.verify_receipt(
        payload,
        repository="owner/consumer",
        run=record,
        evaluator_workflow="NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml",
        evaluator_sha="c" * 40,
    )


def test_historical_expired_receipt_attests_the_executed_merge(verifier, record):
    identity = verify(verifier, record, signed_receipt(record))
    assert identity["workflow_sha"] == "d" * 40
    assert identity["workflow_sha"] != record["head_sha"]
    assert "token" not in identity


@pytest.mark.parametrize(
    "changes",
    [
        {"aud": "other-purpose"},
        {"aud": ["urn:skillevaluator:report-provenance:v1"]},
        {"iss": "https://attacker.invalid"},
        {"repository": "attacker/consumer"},
        {"repository_id": "9999"},
        {"repository_id": 1234},
        {"run_id": "124"},
        {"run_attempt": "2"},
        {"event_name": "push"},
        {"job_workflow_sha": "e" * 40},
        {"job_workflow_ref": "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@main"},
        {"workflow_ref": "owner/consumer/.github/workflows/other.yml@refs/pull/7/merge"},
        {"workflow_sha": "main"},
        {"sha": "e" * 40},
        {"ref": "refs/heads/main", "workflow_ref": "owner/consumer/.github/workflows/skill-checks.yml@refs/heads/main"},
        {"iat": True},
        {"iat": 1},
        {"iat": 9999999999},
        {"exp": 9999999999},
        {"nbf": 1},
    ],
)
def test_signed_wrong_identity_or_time_is_rejected(verifier, record, changes):
    with pytest.raises(verifier.ProvenanceError):
        verify(verifier, record, signed_receipt(record, **changes))


def test_tampered_signature_and_unsigned_token_are_rejected(verifier, record):
    token = verifier.read_receipt(signed_receipt(record))
    parts = token.split(".")
    signature = parts[2]
    parts[2] = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(verifier.ProvenanceError):
        verify(verifier, record, zip_receipt(".".join(parts)))
    unsigned = jwt.encode({"aud": verifier.AUDIENCE}, "", algorithm="none")
    with pytest.raises(verifier.ProvenanceError):
        verify(verifier, record, zip_receipt(unsigned))


def test_unknown_key_and_wrong_key_type_are_rejected(verifier, record, monkeypatch):
    for keys in ({"keys": []}, {"keys": [{**JWK, "kty": "oct"}]}, {"keys": [JWK, JWK]}):
        monkeypatch.setattr(verifier, "fetch_signing_keys", lambda keys=keys: keys)
        with pytest.raises(verifier.ProvenanceError):
            verify(verifier, record, signed_receipt(record))


@pytest.mark.parametrize("name", ["../execution.jwt", "/execution.jwt", "dir/execution.jwt", "execution.jwt/"])
def test_receipt_member_names_are_bounded(verifier, name):
    with pytest.raises(verifier.ProvenanceError):
        verifier.read_receipt(zip_receipt("a.b.c", name))


def test_receipt_rejects_extra_members_symlinks_and_oversize(verifier):
    for kind in ("extra", "symlink", "large"):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            entry = zipfile.ZipInfo("execution.jwt")
            if kind == "symlink":
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, b"a" * (verifier.MAX_TOKEN_BYTES + 1) if kind == "large" else b"a.b.c")
            if kind == "extra":
                archive.writestr("another", b"a.b.c")
        with pytest.raises(verifier.ProvenanceError):
            verifier.read_receipt(output.getvalue())
    with pytest.raises(verifier.ProvenanceError):
        verifier.read_receipt(b"a" * (verifier.MAX_RECEIPT_BYTES + 1))


def test_jwks_uses_only_fixed_host_without_credentials(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "ci"))
    import verify_provenance

    class Response(io.BytesIO):
        @property
        def headers(self):
            return {}

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://token.actions.githubusercontent.com/.well-known/jwks"
            assert request.get_header("Authorization") is None
            assert timeout == 30
            return Response(json.dumps(JWKS).encode())

    def opener(handler):
        assert handler.redirect_request(None, None, None, None, None, "https://elsewhere.invalid") is None
        return Opener()

    monkeypatch.setattr(verify_provenance.urllib.request, "build_opener", opener)
    assert verify_provenance.fetch_signing_keys() == JWKS
