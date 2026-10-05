# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M24 for the dependency audit: packages behind launch wrappers are audited like they are pinned.

The MCP static package made the pinning check look through launch wrappers
(``cmd /c``, ``env X=1``) and read ``bun x``, ``uv run --with`` and ``pnpm
--package=... dlx`` (check-11 e02, skeptic x05 and x07). The dependency audit
reads the packages an MCP runner installs through its own helper, which still
looked at the first word only, so those packages were never checked for CVEs.
"""

from __future__ import annotations

import pytest

from skillevaluator.validators.dependency_ecosystems import mcp_runner_packages


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"command": "cmd", "args": ["/c", "npx", "-y", "@scope/mcp@1.2.3"]}, ("npm", ["@scope/mcp@1.2.3"])),
        ({"command": "env", "args": ["NODE_ENV=production", "npx", "-y", "pkg@1.0.0"]}, ("npm", ["pkg@1.0.0"])),
        ({"command": "timeout", "args": ["30", "uvx", "mcp-server-git==1.0.0"]}, ("pypi", ["mcp-server-git==1.0.0"])),
        ({"command": "bun", "args": ["x", "@scope/mcp@2.0.0"]}, ("npm", ["@scope/mcp@2.0.0"])),
        ({"command": "pnpm", "args": ["--package=@scope/mcp@1.0.0", "dlx", "mcp-bin"]}, ("npm", ["@scope/mcp@1.0.0"])),
        ({"command": "uv", "args": ["run", "--with", "requests==2.31.0", "server.py"]}, ("pypi", ["requests==2.31.0"])),
        ({"command": "sh", "args": ["-c", "npx -y pkg@3.0.0"]}, ("npm", ["pkg@3.0.0"])),
    ],
)
def test_wrapped_runner_packages_are_audited(config: dict, expected: tuple[str, list[str]]) -> None:
    assert mcp_runner_packages(config) == expected


@pytest.mark.parametrize(
    "config",
    [
        {"command": "node", "args": ["./server.js"]},
        {"command": "uv", "args": ["run", "server.py"]},
        {"command": "bun", "args": ["run", "server.ts"]},
    ],
)
def test_local_launches_install_no_package(config: dict) -> None:
    assert mcp_runner_packages(config) is None


def test_direct_runners_are_unchanged() -> None:
    assert mcp_runner_packages({"command": "npx", "args": ["-y", "pkg@1.0.0"]}) == ("npm", ["pkg@1.0.0"])
    assert mcp_runner_packages({"command": "uvx", "args": ["--from", "a==1.0", "a-cli", "--with", "b==2.0"]}) == (
        "pypi",
        ["a==1.0", "b==2.0"],
    )
    assert mcp_runner_packages({"command": "npm exec -- pkg@1.0.0"}) == ("npm", ["pkg@1.0.0"])
