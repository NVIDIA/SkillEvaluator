---
name: text-analyzer
description: Analyze text content and produce statistics including word count, line count, character count, sentence count, average word length, and most frequent words. Use when asked to analyze, count, or summarize a text or a text file (inline text or a file path such as input.txt), or when a request names text-analyzer.
compatibility: Python 3.10+, no external dependencies
metadata:
  author: SkillEvaluator Maintainers <maintainers@example.com>
---

# Text Analyzer

Analyze text and produce structured statistics.

## Purpose

Provide quick text analytics — word counts, line counts, character counts,
top frequent words, average word length, and sentence count — for any
plain-text input.

## Agent Instructions

1. Read this SKILL.md to understand capabilities.
2. Run `scripts/analyze_text.py` with the appropriate arguments.
3. Return the analysis results to the user.

## Examples

### Usage

```bash
# Analyze inline text
python scripts/analyze_text.py --text "Your text content here"

# Analyze a file
python scripts/analyze_text.py --file /path/to/file.txt

# JSON output (default is human-readable)
python scripts/analyze_text.py --text "Hello world" --json
```

### Output

The script prints a structured report with:
- **Word count** — total words
- **Line count** — total lines
- **Character count** — total characters (including whitespace)
- **Sentence count** — estimated sentence count
- **Average word length** — mean characters per word
- **Top 5 words** — most frequent words and their counts

## Limitations

- Sentence detection counts a run of `.`, `!` or `?` only when it is
  followed by whitespace or the end of the text. Closing quotes or brackets,
  Markdown `*` / `_` emphasis and numeric citations such as `[1]` may sit
  between the terminator and that whitespace. Abbreviations followed by a
  space (e.g., "Dr. Smith") still split.
- Non-UTF-8 files will be read with errors replaced.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Error: file not found: <path>` | Check the `--file` path exists |
| `Error: not a file: <path>` | Pass a regular file, not a directory |
| `Error: cannot read file: <path> (...)` | Check that the file is readable |
| Empty output | Ensure `--text` or `--file` is provided |
