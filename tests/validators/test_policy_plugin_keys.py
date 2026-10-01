# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy keys for plugin hooks (hooks.allowed_urls) and endpoint resolution (endpoints.resolve)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.validators import policy as policy_module
from skillevaluator.validators.policy import ValidationPolicy, default_policy, load_policy_file

if TYPE_CHECKING:
    import pytest


def test_new_keys_default_off_and_keep_the_digest_stable() -> None:
    policy = default_policy()
    assert policy.hook_allowed_urls == ()
    assert policy.resolve_endpoints is False
    summary = policy.to_dict()
    assert "hook_allowed_urls" not in summary
    assert "resolve_endpoints" not in summary
    assert (
        policy.digest
        == ValidationPolicy(
            profile=policy.profile,
            severity_overrides=policy.severity_overrides,
            author_email_regex=policy.author_email_regex,
        ).digest
    )


def test_overlay_sets_hook_urls_and_resolution(tmp_path: Path) -> None:
    overlay = tmp_path / "policy.yaml"
    overlay.write_text(
        "hooks:\n  allowed_urls: ['https://hooks.example.com/', '*.corp.example', 42, '']\n"
        "endpoints:\n  resolve: true\n"
    )
    policy = load_policy_file(overlay)
    assert policy.hook_allowed_urls == ("https://hooks.example.com/", "*.corp.example")
    assert policy.resolve_endpoints is True
    summary = policy.to_dict()
    assert summary["hook_allowed_urls"] == ["https://hooks.example.com/", "*.corp.example"]
    assert summary["resolve_endpoints"] is True
    assert policy.digest != default_policy().digest


def test_invalid_values_are_ignored(tmp_path: Path) -> None:
    overlay = tmp_path / "policy.yaml"
    overlay.write_text("hooks:\n  allowed_urls: https://one.example.com\nendpoints:\n  resolve: 'yes'\n")
    policy = load_policy_file(overlay)
    assert policy.hook_allowed_urls == ()
    assert policy.resolve_endpoints is False


def test_hook_url_entries_with_userinfo_query_or_fragment_warn_but_stay_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(policy_module.logger, "warning", lambda message, *args: warnings.append(message % args))
    overlay = tmp_path / "policy.yaml"
    overlay.write_text(
        "hooks:\n  allowed_urls: ['https://u:secret@hooks.example.com/', 'https://hooks.example.com/?k=v']\n"
    )
    policy = load_policy_file(overlay)
    # Kept, so the allowlist stays enforced; the matcher never lets them match.
    assert policy.hook_allowed_urls == ("https://u:secret@hooks.example.com/", "https://hooks.example.com/?k=v")
    assert sum("matches no hook URL" in warning for warning in warnings) == 2
    assert not any("secret" in warning for warning in warnings)
