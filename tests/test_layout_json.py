# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured JSON test data stays in the layout ``scripts/layout_json.py`` writes."""

from scripts import layout_json


def test_managed_json_is_in_the_layout(capsys) -> None:
    unmatched = [pattern for pattern in layout_json.MANAGED if not any(layout_json.ROOT.glob(pattern))]
    assert not unmatched, f"No file matches {unmatched}; update MANAGED in scripts/layout_json.py."
    assert layout_json.main(["--check"]) == 0, f"Run `python scripts/layout_json.py`.\n{capsys.readouterr().out}"
