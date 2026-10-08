# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime security block is one copy in two files: the Harbor verifier and its host mirror.

``check_security`` in ``templates/eval.py`` and in ``eval_core/checks.py`` are
thin wrappers around ``security_scan`` in the "Runtime security" shared block,
so a fix lands in both or the drift guard below fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from skillevaluator.tier3.eval_core import checks as eval_core_checks

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)
_BEGIN = "# ── Runtime security (begin shared block)"
_END = "# ── Runtime security (end shared block)"


def _block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text[text.index(_BEGIN) : text.index(_END)]


def test_runtime_security_block_is_byte_identical_in_both_copies() -> None:
    assert _block(_TEMPLATE) == _block(Path(eval_core_checks.__file__))


def test_both_check_security_wrappers_use_the_shared_scan() -> None:
    spec = importlib.util.spec_from_file_location("harbor_template_eval_f90_sync", _TEMPLATE)
    assert spec and spec.loader
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    call = {"action": "Bash", "action_input": {"command": "cat /tmp/agent-home/.ssh/id_rsa"}, "observation": ""}
    paths = {"SKILLEVAL_AGENT_HOME": "/tmp/agent-home"}

    left = template.check_security({"steps": []}, [call], agent_paths=paths)
    right = eval_core_checks.check_security([call], "", agent_paths=paths)

    assert left == right
    assert [f["evidence"] for f in left["findings"]] == ["~/.ssh"]
