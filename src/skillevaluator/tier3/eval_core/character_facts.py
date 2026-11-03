# SPDX-License-Identifier: Apache-2.0

"""Optional Unicode metadata shared by host and standalone verifier judges."""

from __future__ import annotations

import json
import os
import unicodedata


def character_facts_enabled() -> bool:
    """Read the operator's opt-in; existing prompts remain unchanged by default."""
    return os.environ.get("SKILL_EVAL_JUDGE_CHARACTER_FACTS", "").strip().lower() in {"1", "true", "yes", "on"}


def format_character_facts(
    text: str,
    *,
    label: str,
    max_occurrences: int = 20,
    context_radius: int = 20,
) -> str:
    """Describe non-ASCII whitespace and Cf characters without changing the text.

    Positions count Unicode code points, starting at one; only LF advances the
    line. Context and labels are ASCII-escaped JSON. This is descriptive evidence,
    not a verdict or an exhaustive detector of invisible/look-alike characters.
    """
    if not isinstance(text, str) or not isinstance(label, str):
        raise TypeError("text and label must be strings")
    if not isinstance(max_occurrences, int) or not isinstance(context_radius, int):
        raise TypeError("max_occurrences and context_radius must be integers")
    if max_occurrences < 1 or context_radius < 0:
        raise ValueError("max_occurrences must be positive and context_radius nonnegative")

    entries: list[str] = []
    total = 0
    line, column = 1, 1
    for index, character in enumerate(text):
        if ord(character) > 127 and (character.isspace() or unicodedata.category(character) == "Cf"):
            total += 1
            if len(entries) < max_occurrences:
                context = text[max(0, index - context_radius) : index + context_radius + 1]
                entries.append(
                    f"- line {line}, column {column}: U+{ord(character):04X} "
                    f"{unicodedata.name(character, '<control>')}; "
                    f"escaped context: {json.dumps(context, ensure_ascii=True)}"
                )
        if character == "\n":
            line, column = line + 1, 1
        else:
            column += 1
    if not entries:
        return ""
    if total > len(entries):
        entries.append(f"Omitted {total - len(entries)} additional matching characters.")
    return "\n".join(
        [
            f"Character facts for {json.dumps(label, ensure_ascii=True)} "
            "(descriptive metadata, not a grading verdict).",
            "Positions are one-based Unicode code points; lines are separated by LF.",
            *entries,
        ]
    )


def append_character_facts(prompt: str, **evidence: object) -> str:
    """Append bounded facts for the evidence's prompt representation, when enabled."""
    if not character_facts_enabled():
        return prompt
    # Match the empty format spec used by the judges' str.format/f-string fields.
    panels = [
        facts for label, value in evidence.items() if (facts := format_character_facts(format(value), label=label))
    ]
    if not panels:
        return prompt
    return prompt + "\n\n" + "\n\n".join(panels)
