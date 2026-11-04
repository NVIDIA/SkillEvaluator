# SPDX-License-Identifier: Apache-2.0
"""Native Windows assertions for the path-based workflow update fallback."""

import os
import stat
import subprocess
import sys

import pytest

from skillevaluator import ci_onboarding as wizard
from test_ci_onboarding import repository as repository  # noqa: PLC0414 -- export shared pytest fixture


@pytest.mark.skipif(sys.platform != "win32", reason="Requires the native Windows filesystem fallback")
def test_native_windows_atomic_save_preserves_crlf_and_mode(repository, monkeypatch):
    assert os.open not in os.supports_dir_fd
    assert not hasattr(os, "O_NOFOLLOW")
    target = repository / wizard.SCAN
    original = "# Team workflow\r\n"
    updated = "# Team workflow\r\n# Reporting added\r\n"
    target.write_bytes(original.encode())
    original_mode = stat.S_IMODE(target.stat().st_mode)
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)

    wizard.save_files(repository, {wizard.SCAN: updated, wizard.PUBLISH: "# Publisher\r\n"}, {wizard.SCAN: original})

    assert target.read_bytes() == updated.encode()
    assert (repository / wizard.PUBLISH).read_bytes() == b"# Publisher\r\n"
    assert stat.S_IMODE(target.stat().st_mode) == original_mode
    assert not list(target.parent.glob(".skillevaluator-*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows junction semantics")
def test_native_windows_workflow_junction_cannot_redirect_writes(repository, monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "skill-checks.yml"
    external.write_bytes(b"# External workflow\r\n")
    workflows = repository / ".github" / "workflows"
    workflows.rename(repository / ".github" / "original-workflows")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(workflows), str(outside)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert workflows.is_junction()
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)

    with pytest.raises(wizard.SetupError, match=r"[Ss]ymlink|[Jj]unction|[Rr]eparse"):
        wizard.save_files(repository, {wizard.SCAN: "# Replacement\r\n"}, {wizard.SCAN: "# External workflow\r\n"})

    assert external.read_bytes() == b"# External workflow\r\n"
    assert not list(outside.glob(".skillevaluator-*.tmp"))
