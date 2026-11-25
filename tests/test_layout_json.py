# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured JSON test data stays in the layout ``scripts/layout_json.py`` writes."""

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
