#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text Analyzer — produce word/line/char statistics for any text input."""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def analyze(text: str) -> dict:
    lines = text.splitlines()
    words = text.split()
    chars = len(text)
    word_count = len(words)
    line_count = len(lines)

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)

    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 2)

    freq = Counter(w.lower().strip(".,;:!?\"'()[]{}") for w in words)
    top_words = freq.most_common(5)

    return {
        "word_count": word_count,
        "line_count": line_count,
        "character_count": chars,
        "sentence_count": sentence_count,
        "average_word_length": avg_word_len,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }


def format_report(stats: dict) -> str:
    lines = [
        "=== Text Analysis Report ===",
        f"  Words:              {stats['word_count']}",
        f"  Lines:              {stats['line_count']}",
        f"  Characters:         {stats['character_count']}",
        f"  Sentences:          {stats['sentence_count']}",
        f"  Avg word length:    {stats['average_word_length']}",
        "",
        "  Top words:",
    ]
    for entry in stats["top_words"]:
        lines.append(f"    {entry['word']:<20s} {entry['count']}")
    lines.append("=" * 28)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze text content")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Inline text to analyze")
    group.add_argument("--file", help="Path to a text file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = args.text

    stats = analyze(text)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_report(stats))


if __name__ == "__main__":
    main()
