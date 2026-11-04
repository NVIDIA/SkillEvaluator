# SPDX-License-Identifier: Apache-2.0
"""Real filesystem checks for atomic workflow updates and safe rollback."""

import os
import stat

import pytest

from skillevaluator import ci_onboarding as wizard
from test_ci_onboarding import repository as repository  # noqa: PLC0414 -- export shared pytest fixture

OLD = "# Existing team workflow\n"
NEW = "# Updated team workflow\n"


def existing_scan(repository, monkeypatch):
    target = repository / wizard.SCAN
    target.write_bytes(OLD.encode())
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)
    return target


def test_replacement_is_atomic_and_preserves_mode(repository, monkeypatch):
    target = existing_scan(repository, monkeypatch)
    target.chmod(0o751)
    original_mode = stat.S_IMODE(target.stat().st_mode)
    original_replace = os.replace
    replacements = []

    def observe_replace(source, destination, **kwargs):
        assert target.read_text() == OLD
        replacements.append(destination)
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(wizard.os, "replace", observe_replace)
    wizard.save_files(repository, {wizard.SCAN: NEW}, {wizard.SCAN: OLD})
    assert target.read_text() == NEW
    assert stat.S_IMODE(target.stat().st_mode) == original_mode
    assert len(replacements) == 1
    assert not list(target.parent.glob(".skillevaluator-*.tmp"))


def test_symlink_after_initial_validation_never_overwrites_referent(repository, monkeypatch, tmp_path):
    target = existing_scan(repository, monkeypatch)
    outside = tmp_path / "outside.yml"
    outside.write_text(OLD)

    def replace_with_symlink(_repo):
        target.unlink()
        target.symlink_to(outside)

    monkeypatch.setattr(wizard.git_helpers, "require_clean", replace_with_symlink)
    with pytest.raises(wizard.SetupError, match="Symlink"):
        wizard.save_files(repository, {wizard.SCAN: NEW}, {wizard.SCAN: OLD})
    assert outside.read_text() == OLD


def test_last_instant_symlink_replacement_never_writes_through_target(repository, monkeypatch, tmp_path):
    target = existing_scan(repository, monkeypatch)
    outside = tmp_path / "outside.yml"
    outside.write_text(OLD)
    original_replace = os.replace

    def swap_before_replace(source, destination, **kwargs):
        target.unlink()
        target.symlink_to(outside)
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(wizard.os, "replace", swap_before_replace)
    wizard.save_files(repository, {wizard.SCAN: NEW}, {wizard.SCAN: OLD})
    assert outside.read_text() == OLD
    assert not target.is_symlink()
    assert target.read_text() == NEW


def test_deleted_existing_workflow_is_not_recreated(repository, monkeypatch):
    target = existing_scan(repository, monkeypatch)
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: target.unlink())
    with pytest.raises(wizard.SetupError, match="disappeared"):
        wizard.save_files(repository, {wizard.SCAN: NEW}, {wizard.SCAN: OLD})
    assert not target.exists()


def test_concurrent_new_file_is_not_overwritten(repository, monkeypatch):
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)
    target = repository / wizard.SCAN
    original_link = os.link

    def collision(source, destination, **kwargs):
        target.write_text("# Concurrent workflow\n")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(wizard.os, "link", collision)
    with pytest.raises(FileExistsError):
        wizard.save_files(repository, {wizard.SCAN: NEW})
    assert target.read_text() == "# Concurrent workflow\n"
    assert not list(target.parent.glob(".skillevaluator-*.tmp"))


def test_later_write_failure_rolls_back_original_bytes_and_mode(repository, monkeypatch):
    target = existing_scan(repository, monkeypatch)
    target.chmod(0o750)
    original_mode = stat.S_IMODE(target.stat().st_mode)

    def fail_publication(*_args, **_kwargs):
        raise OSError("Simulated second-file publication failure")

    monkeypatch.setattr(wizard.os, "link", fail_publication)
    with pytest.raises(OSError, match="second-file"):
        wizard.save_files(repository, {wizard.SCAN: NEW, wizard.PUBLISH: "publisher\n"}, {wizard.SCAN: OLD})
    assert target.read_text() == OLD
    assert stat.S_IMODE(target.stat().st_mode) == original_mode
    assert not (repository / wizard.PUBLISH).exists()
    assert not list(target.parent.glob(".skillevaluator-*.tmp"))


@pytest.mark.parametrize("had_original", [True, False])
def test_rollback_leaves_concurrently_replaced_target_untouched(repository, monkeypatch, tmp_path, had_original):
    target = existing_scan(repository, monkeypatch) if had_original else repository / wizard.SCAN
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)
    outside = tmp_path / "outside.yml"
    outside.write_text("# Concurrent external file\n")
    original_link = os.link

    def fail_after_swap(source, destination, **kwargs):
        if str(destination).endswith("publish-skill-reports.yml"):
            target.unlink()
            target.symlink_to(outside)
            raise OSError("Simulated later failure")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(wizard.os, "link", fail_after_swap)
    with pytest.raises(wizard.SetupError, match="Changed targets were left untouched"):
        wizard.save_files(
            repository,
            {wizard.SCAN: NEW, wizard.PUBLISH: "publisher\n"},
            {wizard.SCAN: OLD} if had_original else None,
        )
    assert target.is_symlink()
    assert outside.read_text() == "# Concurrent external file\n"
    assert not list(target.parent.glob(".skillevaluator-*.tmp"))


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Directory descriptors are unavailable")
def test_parent_directory_swap_cannot_redirect_staging_or_write(repository, monkeypatch, tmp_path):
    target = existing_scan(repository, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / target.name).write_text(OLD)
    moved = tmp_path / "moved-original-directory"
    original_open = os.open
    swapped = False

    def swap_parent(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if not swapped and str(path).startswith(".skillevaluator-"):
            swapped = True
            target.parent.rename(moved)
            target.parent.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(wizard.os, "open", swap_parent)
    monkeypatch.setattr(wizard.os, "supports_dir_fd", {*os.supports_dir_fd, swap_parent})
    with pytest.raises(wizard.SetupError, match="Symlink"):
        wizard.save_files(repository, {wizard.SCAN: NEW}, {wizard.SCAN: OLD})
    assert (outside / target.name).read_text() == OLD
    assert (moved / target.name).read_text() == OLD
    assert not list(moved.glob(".skillevaluator-*.tmp"))
