# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the command corpus that tests/tier3/test_checks_template_parity.py replays.

The corpus is plain strings only, grouped by where they came from:

* ``security_tests``: commands and paths from the existing exfiltration, rm,
  chmod, credential-read, protected-write, canary and secret tests;
* ``script_invocation_tests``: commands from the script-execution tests;
* ``other_check_tests``: commands and paths from the other Tier 3 check tests;
* ``real_trajectories``: shell commands agents issued in saved Harbor runs
  (pass their ``trajectory.json`` files with ``--trajectories``; the files
  themselves are never committed). Without ``--trajectories`` the group
  already in the committed corpus is kept as it is;
* ``paths``: file paths the tests pass to Read, Write and Edit tools.

Strings that look like a credential are dropped, so the committed file holds
no secret shape; the parity test builds its secret-shaped inputs from split
literals instead. Strings naming a home directory under ``/Users/<name>`` or
``/home/<name>`` other than the synthetic fixture users are dropped as well, and
so is any string the public-release boundary scanner
(``scripts/check_oss_boundary.py``) flags in the committed JSON.

    uv run python tests/tier3/_tools/build_checks_parity_corpus.py \\
        --trajectories 'runs/*/*/results/*/_harbor-jobs/*/*/agent/trajectory.json'
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path

from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

_REPO = Path(__file__).resolve().parents[3]
OUT = _REPO / "tests" / "fixtures" / "tier3_checks_parity_corpus.json"


def _boundary_scan_text():
    spec = importlib.util.spec_from_file_location("check_oss_boundary", _REPO / "scripts" / "check_oss_boundary.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scan_text


_SCAN_TEXT = _boundary_scan_text()

_SECURITY_FILES = re.compile(
    r"test_(canary_|f90_runtime_security_|security_|apply_patch_security|checks_secret_patterns|"
    r"harbor_template_secret_patterns|shell_expansion_bounds|harbor_security_attribution|github_token_redaction|"
    r"f90_integ_harbor_subagent_security|flag_env_values_not_secrets)"
)
_SCRIPT_FILES = re.compile(r"test_(script_invocation_evidence|negative_control_invocation_evidence)")
_WORDS = (
    r"(?:sudo|env|cat|head|tail|less|more|curl|wget|rm|rmdir|chmod|chown|echo|printf|python3?|python3?\.\d+|bash|sh|"
    r"zsh|dash|ksh|mksh|git|scp|rsync|nc|ncat|socat|cp|mv|tee|sed|awk|find|ls|cd|export|ssh|base64|tar|zip|dd|ln|"
    r"mkdir|touch|eval|source|\.|xargs|perl|node|ruby|openssl|http|https|install|truncate|shred|unlink|ftp|sftp|"
    r"telnet|aws|gsutil|gh|npm|pip|uv|jq|grep|rg|exec|command|time|nohup|setsid|timeout|nice|stdbuf|apply_patch|"
    r"read|while|for|if|until|case|\{|\(|[A-Za-z_]\w*=\S*)"
)
_COMMAND_RE = re.compile(r"^\s*" + _WORDS + r"(?:\s|$|;)")
_PATH_KEYS = {"file_path", "path", "filePath", "notebook_path"}
_COMMAND_KEYS = {"command", "cmd"}
# Synthetic users the tests already use; anything else under a home root is dropped.
_HOME_RE = re.compile(r"/(?:Users|home)/(?!(?:agent|alice|Alice|user|runner|bob|someone|me|x|u)\b)[A-Za-z]")
# Long mixed-case alphanumeric runs: what gitleaks' generic-api-key rule flags.
_TOKEN_RUN_RE = re.compile(r"(?=[A-Za-z0-9_\-]*[a-z])(?=[A-Za-z0-9_\-]*[A-Z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{20,}")


def _skipped_constant_ids(tree: ast.AST) -> set[int]:
    """Docstrings, and the literal pieces of f-strings (a fragment is not the command the test runs)."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
        elif isinstance(node, ast.JoinedStr):
            ids.update(id(part) for part in node.values if isinstance(part, ast.Constant))
    return ids


def _strings_from_test(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _skipped_constant_ids(tree)
    commands, paths = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if not (isinstance(key, ast.Constant) and isinstance(value, ast.Constant)):
                    continue
                if not isinstance(value.value, str) or not value.value.strip():
                    continue
                if key.value in _COMMAND_KEYS:
                    commands.add(value.value)
                elif key.value in _PATH_KEYS:
                    paths.add(value.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            text = node.value
            if 3 <= len(text) <= 2000 and " " in text.strip() and _COMMAND_RE.match(text):
                commands.add(text)
    return commands, paths


def _clean(text: str) -> bool:
    return (
        "\x00" not in text
        and redact_secrets_in_log_line(text) == text
        and not _TOKEN_RUN_RE.search(text)
        and not _HOME_RE.search(text)
        and not _SCAN_TEXT(OUT.name, json.dumps(text, ensure_ascii=False))
    )


def _trajectory_commands(pattern: str) -> set[str]:
    commands = set()
    root = Path(Path(pattern).anchor or ".")
    for name in root.glob(str(Path(pattern).relative_to(root)) if Path(pattern).anchor else pattern):
        try:
            data = json.loads(name.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for step in data.get("steps") or []:
            for call in step.get("tool_calls") or []:
                args = call.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        continue
                for key in _COMMAND_KEYS:
                    value = args.get(key) if isinstance(args, dict) else None
                    if isinstance(value, list):
                        value = " ".join(map(str, value))
                    if isinstance(value, str) and value.strip():
                        commands.add(value)
    return commands


def build(trajectories: str | None, committed_real: list[str] | None = None) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {
        "security_tests": set(),
        "script_invocation_tests": set(),
        "other_check_tests": set(),
        "paths": set(),
    }
    test_files = sorted((_REPO / "tests" / "tier3").glob("test_*.py"))
    test_files += [
        _REPO / "tests" / "test_checks_routing.py",
        _REPO / "tests" / "test_harbor_negative_control_evidence.py",
    ]
    for path in test_files:
        if not path.exists() or path.name == "test_checks_template_parity.py":
            continue
        commands, paths = _strings_from_test(path)
        if _SECURITY_FILES.search(path.name):
            groups["security_tests"] |= commands
        elif _SCRIPT_FILES.search(path.name):
            groups["script_invocation_tests"] |= commands
        else:
            groups["other_check_tests"] |= commands
        groups["paths"] |= paths
    groups["real_trajectories"] = _trajectory_commands(trajectories) if trajectories else set(committed_real or ())
    seen: set[str] = set()
    corpus = {}
    # Commands are deduplicated across their groups; paths only among themselves, since
    # a string can be both a command and a path and each is probed differently.
    for name in ("security_tests", "script_invocation_tests", "other_check_tests", "real_trajectories"):
        kept = sorted(text for text in groups[name] if _clean(text) and text not in seen)
        seen.update(kept)
        corpus[name] = kept
    corpus["paths"] = sorted(text for text in groups["paths"] if _clean(text))
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trajectories", help="glob of saved Harbor trajectory.json files (default: keep the committed group)"
    )
    args = parser.parse_args()
    committed = json.loads(OUT.read_text(encoding="utf-8")).get("real_trajectories", []) if OUT.exists() else []
    corpus = build(args.trajectories, committed)
    OUT.write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print({name: len(values) for name, values in corpus.items()})


if __name__ == "__main__":
    main()
