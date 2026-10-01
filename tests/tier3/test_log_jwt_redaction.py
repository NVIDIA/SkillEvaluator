# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JWTs are masked in Tier 3 log lines on the host and in the Harbor verifier, in linear time."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_log_jwt", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()

REDACTORS = pytest.mark.parametrize(
    "redact",
    [redact_secrets_in_log_line, eval_template.redact_secrets_in_log_line],
    ids=["host", "template"],
)


def _fixture_secret(*parts: str) -> str:
    """Build committed fake secrets from pieces so static scanners do not flag them."""
    return "".join(parts)


# Synthetic JWT: the {"alg":"HS256","typ":"JWT"} header, a sample payload and a fake signature.
_HEADER = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
_PAYLOAD = "eyJzdWIiOiIxMjM0NTY3ODkwIn0"
_JWT = _fixture_secret(_HEADER, ".", _PAYLOAD, ".", "c3ludGhldGljLXNpZ25hdHVyZS1ub3QtcmVhbA")
_SHORT_SIGNATURE_JWT = _fixture_secret(_HEADER, ".", _PAYLOAD, ".", "c" * 19)


@REDACTORS
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(_JWT, "jwt-<redacted>", id="whole-line"),
        pytest.param(f"token {_JWT} ok", "token jwt-<redacted> ok", id="after-space"),
        pytest.param(f'"{_JWT}"', '"jwt-<redacted>"', id="quoted"),
        pytest.param(f"id_token={_JWT}", "id_token=jwt-<redacted>", id="after-equals"),
        pytest.param(f"Authorization: Bearer {_JWT}", "Authorization: Bearer jwt-<redacted>", id="after-bearer"),
        pytest.param(f"x-{_JWT}", "x-jwt-<redacted>", id="glued-to-dash"),
        pytest.param(f"abc-def-{_JWT}", "abc-def-jwt-<redacted>", id="glued-to-dashed-run"),
        pytest.param(f"é-{_JWT}", "é-jwt-<redacted>", id="dash-after-non-ascii-letter"),
        pytest.param(f"eyJ-eyJ-{_JWT}", "jwt-<redacted>", id="run-starts-with-eyJ"),
        pytest.param(f"x-eyJshort.{_JWT}", "x-eyJshort.jwt-<redacted>", id="after-short-run"),
        pytest.param(f"{_JWT}.{_JWT}", "jwt-<redacted>.jwt-<redacted>", id="two-dotted"),
        pytest.param(f"{_JWT} {_JWT}", "jwt-<redacted> jwt-<redacted>", id="two-spaced"),
        pytest.param(f"{_JWT}-", "jwt-<redacted>-", id="trailing-dash-kept"),
        pytest.param(f"a_{_JWT}", f"a_{_JWT}", id="glued-to-underscore-kept"),
        pytest.param(f"x{_JWT}", f"x{_JWT}", id="glued-to-letter-kept"),
        pytest.param(f"é{_JWT}", f"é{_JWT}", id="glued-to-non-ascii-letter-kept"),
        pytest.param(_SHORT_SIGNATURE_JWT, _SHORT_SIGNATURE_JWT, id="short-signature-kept"),
    ],
)
def test_log_jwt_is_redacted_by_position(redact, line, expected):
    assert redact(line) == expected


# Near misses for the former ``\beyJ[A-Za-z0-9_-]{20,}\.…`` pattern. It could start
# after every "-" of one long run of JWT characters, and each start rescanned the
# rest of the run, then the following segments. Each 256 KB line took 11 s to 52 s
# with that pattern and takes about 25 ms now. None of them contains a JWT.
_ADVERSARIAL_LINES = {
    "eyJ-after-every-dash": "eyJ-" * 65_536,
    "dash-before-every-eyJ": "-eyJ" * 65_536,
    "long-second-segment": "eyJ-" * 32_768 + "." + "a" * 131_072,
    "third-segment-without-boundary": "eyJ-" * 32_768 + "." + "a" * 20 + "." + "-" * 131_072,
}


@REDACTORS
@pytest.mark.parametrize("line", list(_ADVERSARIAL_LINES.values()), ids=list(_ADVERSARIAL_LINES))
def test_log_jwt_redaction_is_fast_on_adversarial_lines(redact, line):
    started = time.perf_counter()
    redacted = redact(line)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"redaction took {elapsed:.2f}s on a {len(line)}-character line"
    assert redacted == line
