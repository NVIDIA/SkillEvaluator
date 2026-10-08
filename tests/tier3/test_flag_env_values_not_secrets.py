# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-named flag variables (for example ``FOO_AUTH_ENABLED=1``) are not secrets.

Treating their values as secrets corrupted progress text ("with-skill (<redacted>/2 scored)")
and made the Docker sidecar refuse ordinary Compose values such as ``127.0.0.1``.
"""

from __future__ import annotations

from skillevaluator.tier3.harbor.progress import redact_progress_detail, secret_values_from_environment
from skillevaluator.tier3.harbor.secure_docker_environment import _sensitive_environment_values

_FLAG_ENV = {
    "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH": "1",
    "CLAUDE_CODE_CHILD_SESSION": "1",
    "CLAUDE_CODE_SESSION_ATTENDED": "1",
    "XDG_SESSION_ID": "1",
    "FOO_AUTH_ENABLED": "true",
    "BAR_TOKEN_REFRESH": "off",
}
_REAL_SECRET = "tok_" + "abcdef123456"


def test_flag_values_are_not_progress_secrets() -> None:
    secrets = secret_values_from_environment({**_FLAG_ENV, "MY_API_TOKEN": _REAL_SECRET})

    assert secrets == {_REAL_SECRET}


def test_progress_counts_survive_flag_environment() -> None:
    secrets = secret_values_from_environment(_FLAG_ENV)

    assert redact_progress_detail("with-skill (1/2 scored)", secret_values=secrets) == "with-skill (1/2 scored)"


def test_real_named_secret_is_still_redacted_in_progress() -> None:
    secrets = secret_values_from_environment({"MY_API_TOKEN": _REAL_SECRET})

    assert _REAL_SECRET not in redact_progress_detail(f"calling with {_REAL_SECRET}", secret_values=secrets)


def test_short_proxy_userinfo_is_still_redacted_in_progress() -> None:
    secrets = secret_values_from_environment({"HTTPS_PROXY": "http://u:pw@proxy.example:3128"})

    assert "pw@" not in redact_progress_detail("proxy http://u:pw@proxy.example:3128 failed", secret_values=secrets)


def test_flag_values_are_not_sensitive_docker_values() -> None:
    values = _sensitive_environment_values({**_FLAG_ENV, "MY_API_TOKEN": _REAL_SECRET})

    assert values == {_REAL_SECRET}
