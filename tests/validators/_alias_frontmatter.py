"""Shared fixture text: frontmatter whose aliases expand far beyond its size.

Each key is a nine-item list of aliases to the previous key, so the document
is a few hundred bytes but expands to 9**8 leaves. Readers must reject it
through the bounded loader instead of walking or stringifying the result.
"""

from __future__ import annotations

NESTED_ALIAS_FRONTMATTER = "\n".join(
    [
        "name: demo-skill",
        "description: Demo skill used for testing. Use when demoing.",
        'a: &a ["x","x","x","x","x","x","x","x","x"]',
        *[
            f"{chr(ord('a') + i)}: &{chr(ord('a') + i)} [" + ",".join([f"*{chr(ord('a') + i - 1)}"] * 9) + "]"
            for i in range(1, 9)
        ],
    ]
)

NESTED_ALIAS_SKILL_MD = "---\n" + NESTED_ALIAS_FRONTMATTER + "\n---\n# Demo\n\nSee [guide](guide.md).\n"
