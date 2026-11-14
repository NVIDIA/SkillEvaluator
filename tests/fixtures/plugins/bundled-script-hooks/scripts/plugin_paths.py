#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the plugin install and the user's work tree in one place."""

import os
from pathlib import Path

MARKER = ".build-notes"


def plugin_root():
    env = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    return Path(__file__).resolve().parent.parent


def project_root():
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and (Path(env) / MARKER).is_dir():
        return Path(env)
    return None


def stamp_origin(path=None):
    """Record where the notes cache lives so a copied install can find it."""
    target = path or os.environ.get("BUILD_NOTES_CACHE") or str(Path("~/.cache/build-notes").expanduser())
    dest = Path(plugin_root(), "notes-cache-origin")
    dest.write_text(target + "\n", encoding="utf-8")
    return dest
