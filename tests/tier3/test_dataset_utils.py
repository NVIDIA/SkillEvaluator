# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset text must preserve authored Unicode independently of the host locale."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from skillevaluator.tier3.dataset_utils import load_dataset_entries


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".yaml", ".yml"])
def test_load_utf8_dataset_under_non_utf8_locale(tmp_path: Path, suffix: str) -> None:
    entry = {
        "id": "unicode-001",
        "question": "R\u00e9sumez ce document en fran\u00e7ais. \u65e5\u672c\u8a9e",
        "ground_truth": "R\u00e9sum\u00e9 fid\u00e8le \u2014 \u2713",
        "expected_behavior": ["Conserver le texte : \u00e9\u00e8\u00e0"],
    }
    dataset_path = tmp_path / f"evals{suffix}"
    content = (
        yaml.safe_dump([entry], allow_unicode=True)
        if suffix in {".yaml", ".yml"}
        else json.dumps(entry if suffix == ".jsonl" else [entry], ensure_ascii=False)
    )
    dataset_path.write_text(content + "\n", encoding="utf-8")
    environment = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    script = """
import json
import locale
import os
import sys
from pathlib import Path
from skillevaluator.tier3.dataset_utils import load_dataset_entries

if os.name == "posix":
    assert locale.getencoding().lower() in {"ascii", "us-ascii", "ansi_x3.4-1968"}
print(json.dumps(load_dataset_entries(Path(sys.argv[1])), ensure_ascii=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(dataset_path)],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [entry]


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".yaml", ".yml"])
def test_invalid_utf8_dataset_is_rejected(tmp_path: Path, suffix: str) -> None:
    dataset_path = tmp_path / f"evals{suffix}"
    dataset_path.write_bytes(b'[{"question": "\xff"}]')

    with pytest.raises(UnicodeDecodeError):
        load_dataset_entries(dataset_path)
