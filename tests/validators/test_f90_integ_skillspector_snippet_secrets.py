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
