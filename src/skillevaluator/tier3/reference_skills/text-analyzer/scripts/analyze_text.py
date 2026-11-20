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

# A run of .!? ends a sentence only when it is followed by whitespace or the
# end of the text. Closing quotes or brackets, Markdown emphasis markers and
# numeric citation markers directly after the terminator still count (He said
# "go." Then / (see above.) Next / **done.** Next / 1993.[1] The). Periods
# inside tokens such as input.txt, 3.14, or v1.2.3 are not sentence ends.
# The lookbehind starts a match only at the first terminator of a run, so long
# runs split in linear time.
_SENTENCE_END = re.compile("(?<![.!?])[.!?]+(?:[\"')\\]}\u201d\u2019\u00bb\u203a*_]|\\[\\d+\\])*(?:\\s+|$)")


def analyze(text: str) -> dict:
    lines = text.splitlines()
    words = text.split()
    chars = len(text)
    word_count = len(words)
    line_count = len(lines)

    sentence_count = len([part for part in _SENTENCE_END.split(text) if part.strip()])

    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 2)

    tokens = (w.lower().strip(".,;:!?\"'()[]{}") for w in words)
    freq = Counter(t for t in tokens if any(ch.isalnum() for ch in t))
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
        # The read decides: Path.exists() and is_dir() report False for a
        # path they cannot stat on Python 3.14, which hid permission errors.
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, NotADirectoryError):
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except IsADirectoryError:
            print(f"Error: not a file: {args.file}", file=sys.stderr)
            sys.exit(1)
        except OSError as exc:
            # Windows refuses to open a directory with PermissionError.
            try:
                is_dir = path.is_dir()
            except OSError:
                is_dir = False
            if is_dir:
                print(f"Error: not a file: {args.file}", file=sys.stderr)
            else:
                print(f"Error: cannot read file: {args.file} ({exc.strerror})", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    stats = analyze(text)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_report(stats))


if __name__ == "__main__":
    main()
