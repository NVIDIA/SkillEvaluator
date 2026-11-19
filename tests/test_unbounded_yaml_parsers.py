# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No module calls a raw YAML loader on a file a skill author or the repository under test controls.

``yaml.safe_load`` and ``yaml.load`` expand chained merge keys inside the parser and share aliased nodes
that a later ``str()`` or ``json.dumps`` expands without limit; ``skillevaluator.utils.structured_data``
reads the same documents within limits. A module that still calls a raw loader must be listed here with
the reason it is safe. The test also fails when a listed module no longer calls one, so the list cannot
go stale.
"""

import ast
from pathlib import Path

import skillevaluator

_RAW_LOADERS = frozenset({"safe_load", "load", "full_load", "unsafe_load"})

# Module path under src/skillevaluator -> why a raw loader is acceptable there.
_ALLOWED = {
    "utils/structured_data.py": "the bounded loader itself",
    "config/__init__.py": "reads configuration files that ship in the package",
    "harness/overlay.py": "reads harness profile files that ship in the package",
    "validators/security.py": "uses its own bounded SkillSpector baseline loader",
    "validators/pii_waivers.py": "re-parses with the structured_data loader for an error position after the bounded read",
    "tier3/evals_config.py": "re-parses with the structured_data loader for an error position after the bounded read",
    "tier3/dataset_migration.py": "the author runs it on their own dataset; its loader rejects repeated keys",
    "tier3/commands.py": "reads the user's own --config file",
    "tier3/harbor/kubeconfig.py": "reads the user's own kubeconfig",
    "tier3/harbor/secure_docker_environment.py": "reads the output of `docker compose config`, which has no anchors",
    "ci_onboarding.py": "reads workflow files of the repository being onboarded, in an interactive command",
    "_ci_onboarding_all_tiers.py": "reads workflow files of the repository being onboarded, in an interactive command",
    "_ci_onboarding_workflows.py": "reads workflow files of the repository being onboarded, in an interactive command",
}


def _raw_loader_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "yaml"
            and node.func.attr in _RAW_LOADERS
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
            lines.extend(node.lineno for alias in node.names if alias.name in _RAW_LOADERS)
    return lines


def test_raw_yaml_loaders_are_only_called_where_the_input_is_trusted():
    root = Path(skillevaluator.__file__).parent
    found = {
        path.relative_to(root).as_posix(): lines
        for path in sorted(root.rglob("*.py"))
        if (lines := _raw_loader_calls(path))
    }

    unexpected = {name: lines for name, lines in found.items() if name not in _ALLOWED}
    stale = sorted(set(_ALLOWED) - set(found))

    assert not unexpected, (
        "Read these with skillevaluator.utils.structured_data.load_bounded_yaml, or list the module in "
        f"_ALLOWED with the reason a raw loader is safe there: {unexpected}"
    )
    assert not stale, f"These modules no longer call a raw YAML loader; remove them from _ALLOWED: {stale}"
