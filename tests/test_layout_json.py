# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured JSON test data stays in the layout ``scripts/layout_json.py`` writes."""

import pytest
from scripts import layout_json


def test_managed_json_is_in_the_layout(capsys) -> None:
    unmatched = [pattern for pattern in layout_json.MANAGED if not any(layout_json.ROOT.glob(pattern))]
    assert not unmatched, f"No file matches {unmatched}; update MANAGED in scripts/layout_json.py."
    assert layout_json.main(["--check"]) == 0, f"Run `python scripts/layout_json.py`.\n{capsys.readouterr().out}"


def test_layout_check_detects_crlf_and_rewrite_uses_lf(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(layout_json, "ROOT", tmp_path)
    fixture = tmp_path / "tests" / "golden" / "cli_surface.json"
    fixture.parent.mkdir(parents=True)
    canonical = b'{"commands": []}\n'
    crlf = canonical.replace(b"\n", b"\r\n")
    fixture.write_bytes(crlf)

    assert layout_json.main(["--check"]) == 1
    assert capsys.readouterr().out == "not in layout: tests/golden/cli_surface.json\n"
    assert fixture.read_bytes() == crlf

    assert layout_json.main([]) == 0
    assert capsys.readouterr().out == "rewrote: tests/golden/cli_surface.json\n"
    assert fixture.read_bytes() == canonical
    assert layout_json.main(["--check"]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(("raw", "reason"), [(b'{"a": [1,', "Expecting value"), (b'"\xff"', "'utf-8' codec")])
def test_invalid_managed_json_reports_its_path(tmp_path, monkeypatch, raw: bytes, reason: str) -> None:
    monkeypatch.setattr(layout_json, "ROOT", tmp_path)
    fixture = tmp_path / "tests" / "golden" / "cli_surface.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(raw)

    for flags in ([], ["--check"]):
        with pytest.raises(SystemExit) as error:
            layout_json.main(flags)
        assert str(error.value).startswith("tests/golden/cli_surface.json: ")
        assert reason in str(error.value)
        assert fixture.read_bytes() == raw


@pytest.mark.parametrize("linked_parent", [False, True], ids=["linked-file", "linked-parent"])
def test_managed_json_rejects_symlink_paths(tmp_path, monkeypatch, linked_parent: bool) -> None:
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(layout_json, "ROOT", checkout)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "report.json"
    raw = b'{  "status": "ok" }'
    target.write_bytes(raw)
    managed = checkout / "tests" / "fixtures" / "skillspector-test" / "report.json"
    try:
        if linked_parent:
            managed.parent.parent.mkdir(parents=True)
            managed.parent.symlink_to(outside, target_is_directory=True)
        else:
            managed.parent.mkdir(parents=True)
            managed.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    for flags in ([], ["--check"]):
        with pytest.raises(SystemExit) as error:
            layout_json.main(flags)
        assert str(error.value) == "tests/fixtures/skillspector-test/report.json: symbolic links are not supported"
        assert target.read_bytes() == raw


def test_managed_json_rejects_hard_links(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(layout_json, "ROOT", checkout)
    target = tmp_path / "report.json"
    raw = b'{  "status": "ok" }'
    target.write_bytes(raw)
    managed = checkout / "tests" / "golden" / "cli_surface.json"
    managed.parent.mkdir(parents=True)
    try:
        managed.hardlink_to(target)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")

    for flags in ([], ["--check"]):
        with pytest.raises(SystemExit) as error:
            layout_json.main(flags)
        assert str(error.value) == "tests/golden/cli_surface.json: expected a regular file with exactly one hard link"
        assert target.read_bytes() == raw
