"""Golden matrix: the harness pi model vs the real pi 1.6.0 loader, case by case.

Fixtures were recorded on a dev machine by the probe (CI has no node):

    node tests/golden/generate_pi_skill_matrix.mjs tests/golden/pi_skill_matrix_cases.json \
      > tests/golden/pi_skill_matrix.json

This test rebuilds each fixture tree under a .pi/skills root and compares our
model of pi (manifest selection, strict frontmatter validity, hidden flags)
against what the pinned loader actually did.
"""

import json
from pathlib import Path

import pytest

from skillevaluator.harness.overlay import (
    FRONTMATTER_PATTERN,
    _pi_manifests,
    _pi_top_level_bool,
)
from skillevaluator.utils.structured_data import load_bounded_yaml

GOLDEN_DIR = Path(__file__).resolve().parent
GOLDEN = json.loads((GOLDEN_DIR / "pi_skill_matrix.json").read_text())
CASES = {c["id"]: c for c in json.loads((GOLDEN_DIR / "pi_skill_matrix_cases.json").read_text())["cases"]}


def build_case(tmp_path: Path, case_id: str) -> Path:
    """Rebuild the fixture tree under a .pi/skills root: the probe's load target becomes the scanned root."""
    case = CASES[case_id]
    target = case["targets"][0]
    root = tmp_path / ".pi" / "skills"
    for rel, content in case["files"].items():
        if rel.startswith(target + "/"):
            rel = rel[len(target) + 1 :]
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


def strip_target(case: dict, path: str) -> str:
    """Golden paths are relative to the probe root; ours to the scanned target dir."""
    path = path.removeprefix("<root>/")
    target = CASES[case["id"]]["targets"][0]
    return path.removeprefix(target + "/") if target != "." else path


def golden_expected_files(case: dict) -> list[str]:
    """Files pi 1.6.0 selects: recorded skills plus collision losers (seen, but name-shadowed)."""
    files = {strip_target(case, s["filePath"]) for s in case["skills"]}
    for d in case["diagnostics"]:
        if d["type"] == "collision":
            files |= {strip_target(case, d["collision"][k]) for k in ("winnerPath", "loserPath")}
    return sorted(files)


def _strict_loadable(p: Path) -> bool:
    """End-to-end pi load rule: strict frontmatter parse (dup keys fail) carrying a description."""
    match = FRONTMATTER_PATTERN.match(p.read_text())
    if match is None:
        return False
    try:
        data = load_bounded_yaml(match.group(1))
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("description"), str) and data["description"].strip() != ""


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
def test_pi_discovery_matches_golden(tmp_path, case):
    """Manifests our model selects == manifests the real pi 1.6.0 loader selected."""
    root = build_case(tmp_path, case["id"])
    ours = sorted(p.relative_to(root).as_posix() for p in _pi_manifests(root) if _strict_loadable(p))
    assert ours == golden_expected_files(case)


@pytest.mark.parametrize(
    "case",
    [c for c in GOLDEN["cases"] if any(f["filePath"].endswith("SKILL.md") for f in c["skills"])],
    ids=lambda c: c["id"],
)
def test_pi_hidden_flag_matches_golden(tmp_path, case):
    """disable-model-invocation as pi resolves it == _pi_top_level_bool on the same frontmatter."""
    skill = next(s for s in case["skills"] if s["filePath"].endswith("SKILL.md"))
    fm = (build_case(tmp_path, case["id"]) / skill["filePath"]).read_text().split("---")[1]
    ours = _pi_top_level_bool(fm, "disable-model-invocation")
    assert bool(ours) == bool(skill["disableModelInvocation"])
