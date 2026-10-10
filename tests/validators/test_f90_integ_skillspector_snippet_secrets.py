# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H9 (check-10 p03): a secret in a plugin script is never copied into a report through SkillSpector.

The MCP static package stopped the PII scan and the MCP messages from echoing a
secret. Re-running check-10 ``p03-hook-script-secret`` on the merged build still
found the token in the JSON (``line_content``), the SARIF snippet, and the HTML:
SkillSpector's own finding for the script (an external-transmission pattern)
quotes the line that assigns the token, and that code snippet was copied as is.
"""

from __future__ import annotations

import pytest

from skillevaluator.validators.security import SecurityValidator

TOKEN = "ghp_" + "A1b2" * 9


def test_skillspector_snippet_and_texts_never_carry_the_secret() -> None:
    issue = {
        "id": "E1",
        "pattern": "External Transmission",
        "category": "Data Exfiltration",
        "finding": "https://api.github.com/",
        "severity": "MEDIUM",
        "explanation": f"Data is being sent to an external URL with GITHUB_TOKEN={TOKEN}.",
        "remediation": "Ensure no secrets are transmitted.",
        "code_snippet": (
            "#!/bin/sh\n"
            f'GITHUB_TOKEN="{TOKEN}"\n'
            'curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user >/dev/null'
        ),
        "location": {"file": "hooks/scripts/notify.sh", "start_line": 4},
    }

    finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)

    for text in (finding.line_content, finding.message, finding.suggestion):
        assert TOKEN not in (text or "")
    # The rest of the line stays readable for the reviewer.
    assert "curl -s -H" in finding.line_content
    assert "<redacted>" in finding.line_content
    assert finding.file_path == "hooks/scripts/notify.sh" and finding.line_number == 4


_AWS_SECRET = "FAKEawsSecret" + "Key0123456789abcdefghijklmn+/"
_URL_PASSWORD = "C9F-FAKE-PW" + "-9"


@pytest.mark.parametrize(
    ("snippet", "secret"),
    [
        # check-10 s03-p2: a quoted value under a credential name, which has no known token shape
        (
            f'#!/bin/sh\nexport AWS_SECRET_ACCESS_KEY="{_AWS_SECRET}"\naws s3 cp ./build.log s3://example-logs/',
            _AWS_SECRET,
        ),
        (f"API_KEY = '{_AWS_SECRET}'", _AWS_SECRET),
        (f"os.environ['API_KEY']='{_AWS_SECRET}'", _AWS_SECRET),
        (f'service:\n  api_token: "{_AWS_SECRET}"', _AWS_SECRET),
        # check-09 s05-edge-2: a WHATWG client reads 'https:user:pw@host' as user information
        (f'"url": "https:bob:{_URL_PASSWORD}@mcp.example.com/mcp"', _URL_PASSWORD),
    ],
    ids=["shell-export", "python-assignment", "python-subscript", "yaml", "https-without-slashes"],
)
def test_skillspector_snippet_never_carries_a_credential_without_a_token_shape(snippet: str, secret: str) -> None:
    issue = {
        "id": "E5",
        "pattern": "Cloud Storage Exfiltration",
        "category": "Data Exfiltration",
        "finding": "aws s3 cp",
        "severity": "MEDIUM",
        "code_snippet": snippet,
        "location": {"file": "hooks/scripts/upload.sh", "start_line": 3},
    }

    finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)

    assert secret not in (finding.line_content or "")
    assert "<redacted>" in finding.line_content or "mcp.example.com" in finding.line_content
