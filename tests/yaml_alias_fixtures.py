# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tiny SKILL.md whose nested YAML aliases expand far beyond the file's size.

``yaml.safe_load`` shares an aliased node, so such a file parses in
microseconds. Calling ``str()`` on the value, or copying or serializing it,
expands the whole graph: nine levels of nine-way fan-out is 9**8 leaves, which
exhausts memory from a file of a few hundred bytes. The bounded loader rejects
the graph before any value is read.
"""

from __future__ import annotations

_ANCHORS = "abcdefghij"

# The reported size: 9**8 leaves from a SKILL.md of a few hundred bytes, which needs
# gigabytes if any value is stringified. Only parse it with a loader that is meant to
# reject it, never through code that would expand it.
REPORTED_LEVELS = 8

# End-to-end tests run a validator over the file, so they use a graph that is
# over the loader's 20,000-node limit but costs well under a megabyte to expand
# if a reader regresses to the unbounded loader (9**5 leaves).
OVER_LIMIT_LEVELS = 5


def alias_graph_lines(levels: int) -> list[str]:
    """Return mapping entries ``a`` .. ``<levels>`` where each list repeats the previous one nine times."""
    lines = ['a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]']
    for i in range(1, levels + 1):
        lines.append(f"{_ANCHORS[i]}: &{_ANCHORS[i]} [" + ",".join([f"*{_ANCHORS[i - 1]}"] * 9) + "]")
    return lines


_DEFAULT_BODY = "# Demo\n\n## Instructions\n\n1. Step\n"


def alias_bomb_frontmatter(levels: int = OVER_LIMIT_LEVELS, *, alias_fields: tuple[str, ...] = ()) -> str:
    """Return frontmatter YAML (no ``---`` fences) with a valid name and description plus the alias graph.

    A field in ``alias_fields`` (``name`` or ``description``) is the alias itself, so a reader that
    stringifies that field expands the graph; the plain string for it is dropped.
    """
    graph = alias_graph_lines(levels)
    head = []
    if "name" not in alias_fields:
        head.append("name: demo-skill")
    if "description" not in alias_fields:
        head.append("description: Demo skill used for testing. Use when demoing.")
    # An alias must come after its anchor, so the aliased fields go last.
    tail = [f"{field}: *{_ANCHORS[levels]}" for field in alias_fields]
    return "\n".join([*head, *graph, *tail])


def alias_bomb_skill_md(
    levels: int = OVER_LIMIT_LEVELS, *, alias_fields: tuple[str, ...] = (), body: str = _DEFAULT_BODY
) -> str:
    """Return a SKILL.md whose frontmatter is :func:`alias_bomb_frontmatter`, followed by ``body``."""
    return f"---\n{alias_bomb_frontmatter(levels, alias_fields=alias_fields)}\n---\n{body}"


def merge_bomb_yaml(levels: int = 16) -> str:
    """Chained merge keys: each level merges the previous one twice.

    PyYAML flattens the merged pairs into a list while it builds each mapping, so plain
    ``yaml.safe_load`` does ``2 ** levels`` pairs of work before anything reads the value.
    """
    lines = ["seed: &a0 {name: safe}"]
    lines += [f"a{i}: &a{i} {{<<: [*a{i - 1}, *a{i - 1}]}}" for i in range(1, levels + 1)]
    return "\n".join(lines)


def scalar_alias_fan_yaml(levels: int = 3, fan: int = 6, size: int = 65_000) -> str:
    """One ``size``-character scalar reached ``fan ** (levels + 1)`` times through nested aliases.

    It stays under the node, alias and per-scalar limits, so only the expanded-character
    budget rejects it. The defaults expand to 84 MB if a reader stringifies it.
    """
    lines = [f'big: &big "{"x" * size}"', "l0: &l0 [" + ",".join(["*big"] * fan) + "]"]
    lines += [f"l{i}: &l{i} [" + ",".join([f"*l{i - 1}"] * fan) + "]" for i in range(1, levels + 1)]
    return "\n".join(lines)


def deeply_nested_flow(depth: int = 3_000) -> str:
    """A flow sequence nested ``depth`` levels: PyYAML's composer recurses once per level."""
    return "deep: " + "[" * depth + "]" * depth


# A ``%YAML`` directive with an over-long version number makes PyYAML's scanner raise a plain
# ValueError (Python's integer-string digit limit), not a YAMLError.
YAML_DIRECTIVE_VALUE_ERROR = "%YAML 1." + "1" * 4_301
