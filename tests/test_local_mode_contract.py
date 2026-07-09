# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import local_sandbox

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "local_mode_contract.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", ("require", "prefer", "off"))
def test_native_windows_fails_for_every_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Windows")

    with pytest.raises(local_sandbox.SandboxUnavailable, match=r"WSL2.*docker"):
        local_sandbox.detect(mode)


@pytest.mark.parametrize("system", ("CYGWIN_NT", "MSYS_NT", "FreeBSD"))
@pytest.mark.parametrize("mode", ("require", "prefer", "off"))
def test_other_unsupported_native_platforms_fail_for_every_mode(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    mode: str,
) -> None:
    monkeypatch.setattr(local_sandbox.platform, "system", lambda: system)

    with pytest.raises(local_sandbox.SandboxUnavailable, match="unsupported"):
        local_sandbox.detect(mode)


def test_contract_matches_public_local_mode_constants(contract: dict[str, object]) -> None:
    assert contract["sandbox_modes"] == list(local_sandbox.SANDBOX_MODES)
    assert contract["supported_native_systems"] == sorted(local_sandbox.SUPPORTED_LOCAL_SYSTEMS)
    assert contract["linux_strength"] == "kernel"
    assert contract["macos_strength"] == "semi-trusted"
    assert contract["strict_reads_env"] == local_sandbox.STRICT_READS_ENV
    assert contract["network_env"] == local_sandbox.ALLOW_NET_ENV
