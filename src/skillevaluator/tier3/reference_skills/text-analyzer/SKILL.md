---
name: text-analyzer
description: Analyze text content and produce statistics including word count, line count, character count, most frequent words, and readability metrics. Works on any plain text input provided inline or from a file path.
compatibility: Python 3.10+, no external dependencies
metadata:
  author: Skill Evaluator Maintainers <maintainers@example.com>
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

- Sentence detection uses simple period/question/exclamation splitting;
  may be inaccurate for abbreviations (e.g., "U.S.A.").
- Non-UTF-8 files will be read with errors replaced.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` | Check the `--file` path exists |
| Empty output | Ensure `--text` or `--file` is provided |
