#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a note into action items."""

from __future__ import annotations

import argparse


def summarize(note: str) -> str:
    """Return an action-item summary for a note."""
    if not note.strip():
        raise ValueError("note must not be empty")
    if "follow up" in note.lower():
        return f"Action item: {note.strip()}"
    return "No action item found."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("note")
    args = parser.parse_args()
    print(summarize(args.note))


if __name__ == "__main__":
    main()
