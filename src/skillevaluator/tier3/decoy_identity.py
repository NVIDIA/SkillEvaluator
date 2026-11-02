# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned identities for the runtime content of selected decoy skills.

Decoy identity follows the same projection as staging. Evaluator-only inputs
do not affect that identity; the full publication source identity retains its
separate contract, which deliberately includes authored evaluation inputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

# Bump this recipe if the runtime projection or reused canonical encoding changes.
DECOY_SOURCE_DIGEST_ALGORITHM = "skill-evaluator-decoy-runtime/1"


def decoy_source_copy_ignore(source: Path) -> Callable[[str, list[str]], Iterable[str]]:
    """Use the agent-visible skill projection for both identity and snapshots."""
    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore

    return _runtime_skill_copy_ignore(source)


def decoy_source_digest(source: Path) -> str | None:
    """Hash one stable runtime projection, rejecting unsafe or unstable trees.

    Secure staging traversal rejects links, hard-linked files, special nodes
    and mount crossings. Three captures detect incidental changes between
    scans. As with staging, this does not promise a coherent snapshot against
    another process concurrently mutating the tree with the same OS identity.
    Paths are NFC-normalized, and executable bits participate in the digest.
    """
    from skillevaluator.publication_identity import _manifest_digest
    from skillevaluator.tier3.harbor.secure_copy import _build_tree_manifest

    try:
        ignore = decoy_source_copy_ignore(source)
        captured = _build_tree_manifest(source, allowed_root=source, ignore=ignore)
        for _ in range(2):
            if _build_tree_manifest(source, allowed_root=source, ignore=ignore) != captured:
                return None
        # Reuse the publication recipe's canonical paths, node kinds, content
        # hashes and executable-bit encoding, under a distinct runtime label.
        encoded = _manifest_digest((captured.root, *captured.entries))
        return hashlib.sha256(
            DECOY_SOURCE_DIGEST_ALGORITHM.encode("ascii") + b"\0" + bytes.fromhex(encoded)
        ).hexdigest()
    except (OSError, OverflowError, RuntimeError, UnicodeError, ValueError):
        return None
