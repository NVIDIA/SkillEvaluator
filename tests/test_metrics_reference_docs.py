# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin ``docs/metrics-reference.mdx`` to the constants and formulas it documents.

The page states that the code is the contract and the page is the bug whenever
the two disagree. These tests are what makes that true: each one *parses* a
value out of the published page and compares it against the live value in the
source module, so retuning a constant fails here instead of rotting the page.

Nothing below restates a documented number as a literal. Where a test needs a
literal it reads it out of the ``.mdx``.

Deliberately not pinned, because there is no single source value to compare to:

- The ``0.50`` default ``--pass-threshold``. It is a repeated per-call-site
  default (``collector.py``, ``harbor/runner.py``, ``tier3_report.py``,
  ``console_ui.py``), not a named constant, so there is nothing to import.
- The ``security`` finding-type ladder and which types are critical. The types
  are ``finding_type=`` keyword arguments spread through branches of
  ``check_security``, not a table a test can read.
- The prose describing each individual Tier 1 finding. Only the deduction sizes
  are pinned, not the wording that introduces them.
- The main Tier 1 area rows are pinned by the *set* of deduction sizes they
  quote rather than one-for-one, because the page collapses repeats as
  "(N each)". The type-specific table does not collapse, so it is pinned
  one-for-one and is what catches a dropped check.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from skillevaluator.constants import (
    DEFAULT_SCORE_POLICY,
    DIMENSION_MAPPING,
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
    LIFT_BOOTSTRAP_CONFIDENCE,
    LIFT_BOOTSTRAP_RESAMPLES,
    LIFT_BOOTSTRAP_SEED,
    QUALITY_CHARS_PER_TOKEN,
    QUALITY_DEFAULT_MIN_SCORE,
    QUALITY_GRADE_THRESHOLDS,
    QUALITY_RECOMMENDED_MAX_TOKENS,
    QUALITY_SCORE_WEIGHTS,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
    TOKEN_EFFICIENCY_HALF_LIFE,
)
from skillevaluator.quality_config import load_quality_config_with_provenance
from skillevaluator.tier3 import dataset_sufficiency
from skillevaluator.tier3.eval_core.llm_judge import _ACCURACY_CRITERIA
from skillevaluator.tier3.harbor.collector import (
    _count_derived_pass_rate_delta,
    _wilson_score_interval,
)
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, LEGACY_SCORE_POLICY
from skillevaluator.tier3.harbor.stats import token_efficiency_score

REPO = Path(__file__).resolve().parents[1]
DOC_PATH = REPO / "docs/metrics-reference.mdx"
COLLECTOR_PATH = REPO / "src/skillevaluator/tier3/harbor/collector.py"
QUALITY_SCORE_PATH = REPO / "src/skillevaluator/validators/quality_score.py"
CHECKS_PATH = REPO / "src/skillevaluator/tier3/eval_core/checks.py"
GRADER_SUITE_PATH = REPO / "src/skillevaluator/tier3/grader_reference/suite.json"

DOC = DOC_PATH.read_text(encoding="utf-8")
# The page uses typographic minus (U+2212) and en dash (U+2013); fold both to
# ASCII hyphen so a signed range parses the same way everywhere.
TYPOGRAPHIC_DASHES = (chr(0x2212), chr(0x2013))


def _ascii_dashes(text: str) -> str:
    for dash in TYPOGRAPHIC_DASHES:
        text = text.replace(dash, "-")
    return text


FLAT = " ".join(_ascii_dashes(DOC).split())


def _one(pattern: str, text: str = FLAT) -> re.Match[str]:
    """Return the single match for *pattern*, failing loudly if the prose moved."""
    matches = list(re.finditer(pattern, text))
    assert matches, f"docs/metrics-reference.mdx no longer contains text matching {pattern!r}"
    assert len(matches) == 1, f"{pattern!r} matched {len(matches)} times; tighten the pattern"
    return matches[0]


def _every(pattern: str, text: str = FLAT) -> str:
    """Return the capture shared by every match, so a value quoted twice stays consistent."""
    found = re.findall(pattern, text)
    assert found, f"docs/metrics-reference.mdx no longer contains text matching {pattern!r}"
    assert len(set(found)) == 1, f"{pattern!r} is quoted inconsistently in the page: {sorted(set(found))}"
    return found[0]


def _table_rows(heading_cell: str) -> dict[str, list[str]]:
    """Return ``{first cell: [remaining cells]}`` for the table whose header starts with *heading_cell*."""
    rows: dict[str, list[str]] = {}
    in_table = False
    for line in DOC.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0] == heading_cell:
            in_table = True
            continue
        if not in_table or set(cells[0]) <= {"-", ":"}:
            continue
        rows[cells[0]] = cells[1:]
    assert rows, f"no table in docs/metrics-reference.mdx has a {heading_cell!r} header cell"
    return rows


# ---------------------------------------------------------------------------
# The six Tier 3 signals
# ---------------------------------------------------------------------------


def test_signal_sections_match_default_metrics() -> None:
    """Every ``### `signal`` heading in the signals section tracks ``DEFAULT_METRICS``."""
    section = DOC.split("## The six Tier 3 signals", 1)[1].split("\n## ", 1)[0]
    documented = tuple(re.findall(r"^### `(\w+)`", section, flags=re.MULTILINE))

    assert documented[: len(DEFAULT_METRICS)] == DEFAULT_METRICS
    # token_efficiency is documented beside the six but is deliberately not one of them.
    extra = documented[len(DEFAULT_METRICS) :]
    assert extra == ("token_efficiency",)
    assert "token_efficiency" not in DEFAULT_METRICS


def test_accuracy_criteria_match_the_judge() -> None:
    criteria = [name for name, _ in _ACCURACY_CRITERIA]
    documented = re.findall(r"`([A-Z_]+)`", _one(r"The five criteria are ([^.]+)\.").group(1))

    assert documented == criteria
    # "share of five boolean criteria" and "/ 5" must track the real criterion count.
    assert int(_one(r"accuracy = \(number of the (\d+) criteria judged true\) / \d+", DOC).group(1)) == len(criteria)
    assert int(_one(r"accuracy = \(number of the \d+ criteria judged true\) / (\d+)", DOC).group(1)) == len(criteria)


def _fault_penalty(variable: str) -> tuple[float, float]:
    """Return ``(floor, coefficient)`` for ``max(floor, 1 - coefficient * <variable>)`` in checks.py."""
    tree = ast.parse(CHECKS_PATH.read_text(encoding="utf-8"))
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "max"):
            continue
        if len(call.args) != 2 or not isinstance(call.args[0], ast.Constant):
            continue
        names = {node.id for node in ast.walk(call.args[1]) if isinstance(node, ast.Name)}
        if variable not in names:
            continue
        constants = [
            node.value
            for node in ast.walk(call.args[1])
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value != 1.0
        ]
        assert len(constants) == 1, f"{variable}: expected one penalty coefficient, found {constants}"
        return float(call.args[0].value), float(constants[0])
    raise AssertionError(f"checks.py no longer scales error_recovery by {variable}")


def test_error_recovery_penalties_match_the_check() -> None:
    """The graduated error_recovery ladder quotes real floors and coefficients."""
    for variable in ("skill_faults", "agent_faults"):
        floor, coefficient = _fault_penalty(variable)
        match = _one(rf"max\(([\d.]+), 1 - ([\d.]+) \* {variable}\)")
        assert float(match.group(1)) == floor, f"{variable}: documented floor drifted"
        assert float(match.group(2)) == coefficient, f"{variable}: documented coefficient drifted"


def test_token_efficiency_half_life_matches_constants() -> None:
    documented = int(_one(r"token_efficiency = 1 / \(1 \+ U / (\d+)\)", DOC).group(1))
    assert documented == TOKEN_EFFICIENCY_HALF_LIFE

    # The prose repeats the constant with a thousands separator, and claims the
    # curve's half-life behaviour. Pin both against the shipped scorer.
    prose = int(_one(r"A trial at ([\d,]+) uncached tokens scores ([\d.]+)").group(1).replace(",", ""))
    scored = float(_one(r"A trial at [\d,]+ uncached tokens scores ([\d.]+)").group(1))
    assert prose == TOKEN_EFFICIENCY_HALF_LIFE
    assert token_efficiency_score(prose) == pytest.approx(scored)

    conventions = int(_one(r"the ([\d,]+)-token efficiency constant").group(1).replace(",", ""))
    assert conventions == TOKEN_EFFICIENCY_HALF_LIFE


# ---------------------------------------------------------------------------
# Dimensions and the Overall Score
# ---------------------------------------------------------------------------


def test_dimension_table_matches_dimension_mapping() -> None:
    rows = _table_rows("Dimension")
    assert {name.lower() for name in rows} == set(DIMENSION_MAPPING)

    for name, (question, formula, _evidence) in rows.items():
        spec = DIMENSION_MAPPING[name.lower()]
        assert question == spec["question"], f"{name}: documented question drifted from DIMENSION_MAPPING"
        # The formula cell is backticked code: a bare signal name, or mean(a, b).
        quoted = " ".join(re.findall(r"`([^`]+)`", formula))
        evaluators = [token for token in re.findall(r"[a-z_]+", quoted) if token != "mean"]
        assert evaluators == list(spec["evaluators"]), f"{name}: documented evaluators drifted"

    # Effectiveness is the only dimension that blends two signals; its split is quoted.
    documented_weights = [float(w) for w in _one(r"\(weights ([\d.]+) / ([\d.]+)\)").groups()]
    assert documented_weights == list(DIMENSION_MAPPING["effectiveness"]["weights"])


def test_dimension_verdict_bands_match_constants() -> None:
    match = _one(
        r"\*\*PASS\*\* at (\d+\.\d+) or above, \*\*NEUTRAL\*\* from (\d+\.\d+) up to (\d+\.\d+), "
        r"and \*\*FAIL\*\* below (\d+\.\d+)"
    )
    pass_at, neutral_from, neutral_to, fail_below = (float(value) for value in match.groups())

    assert pass_at == DIMENSION_VERDICT_PASS_THRESHOLD
    assert neutral_to == DIMENSION_VERDICT_PASS_THRESHOLD
    assert neutral_from == DIMENSION_VERDICT_NEUTRAL_THRESHOLD
    assert fail_below == DIMENSION_VERDICT_NEUTRAL_THRESHOLD


def test_score_policy_identifiers_match_the_source() -> None:
    assert f"`score_policy: {DEFAULT_SCORE_POLICY}`" in FLAT
    assert f"`{LEGACY_SCORE_POLICY}`" in FLAT
    assert f"`{dataset_sufficiency.POLICY}`" in FLAT


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def _wilson_z_in_source() -> float:
    """Read the ``z`` literal out of ``_wilson_score_interval`` itself."""
    tree = ast.parse(COLLECTOR_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_wilson_score_interval":
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "z"
                    and isinstance(statement.value, ast.Constant)
                ):
                    return float(statement.value.value)
    raise AssertionError("_wilson_score_interval no longer assigns a literal z")


def test_wilson_z_matches_the_implementation() -> None:
    documented = float(_one(r"with `z = ([\d.]+)`").group(1))
    assert documented == _wilson_z_in_source()


def test_wilson_confidence_level_matches_the_interval() -> None:
    documented = int(_one(r"two-sided (\d+)% \*\*Wilson score interval\*\*").group(1))
    interval = _wilson_score_interval(1, 4)
    assert interval is not None
    assert interval["confidence_level"] == pytest.approx(documented / 100)


def test_quoted_four_case_half_widths_match_the_interval() -> None:
    """The 'a run on four cases is a smoke test' argument quotes real half-widths."""
    half_widths = []
    for successes in range(5):
        interval = _wilson_score_interval(successes, 4)
        assert interval is not None
        half_widths.append((interval["upper"] - interval["lower"]) / 2)

    widest = float(_one(r"the Wilson interval on four cases has a half-width of up to ([\d.]+)").group(1))
    narrowest = float(_one(r"The narrowest a 4-case interval ever gets is ([\d.]+)").group(1))

    assert max(half_widths) == pytest.approx(widest, abs=5e-5)
    assert min(half_widths) == pytest.approx(narrowest, abs=5e-5)
    # The page's point is that even the best 4-case interval misses the policy bar.
    assert min(half_widths) > dataset_sufficiency.MAX_PASS_RATE_INTERVAL_HALF_WIDTH


# ---------------------------------------------------------------------------
# Skill Lift and pass@k
# ---------------------------------------------------------------------------


def test_lift_verdict_bands_match_constants() -> None:
    match = _one(r"PASS at `>= \+([\d.]+)`, NEUTRAL between `-([\d.]+)` and `\+([\d.]+)`, FAIL at `<= -([\d.]+)`")
    pass_at, neutral_low, neutral_high, fail_at = (float(value) for value in match.groups())

    assert pass_at == TIER3_LIFT_PASS_THRESHOLD
    assert neutral_high == TIER3_LIFT_PASS_THRESHOLD
    assert -neutral_low == TIER3_LIFT_FAIL_THRESHOLD
    assert -fail_at == TIER3_LIFT_FAIL_THRESHOLD

    # The "thresholds are conventions" bullet repeats the same pair.
    repeated = _one(r"the \+([\d.]+)/-([\d.]+) lift bands")
    assert float(repeated.group(1)) == TIER3_LIFT_PASS_THRESHOLD
    assert -float(repeated.group(2)) == TIER3_LIFT_FAIL_THRESHOLD


def test_bootstrap_parameters_match_constants() -> None:
    match = _one(r"\(seed `(\d+)`, at least ([\d,]+) resamples, (\d+)% confidence\)")
    seed, resamples, confidence = match.groups()

    assert int(seed) == LIFT_BOOTSTRAP_SEED
    assert int(resamples.replace(",", "")) == LIFT_BOOTSTRAP_RESAMPLES
    assert int(confidence) / 100 == pytest.approx(LIFT_BOOTSTRAP_CONFIDENCE)


def test_pass_rate_difference_units_match_the_artifact() -> None:
    """The JSON row must advertise a fraction, because that is what is written."""
    rows = _table_rows("Number")
    json_row = next(cells for name, cells in rows.items() if "Pass-rate difference]" in name)
    displayed_row = next(cells for name, cells in rows.items() if "Pass-rate difference, as displayed]" in name)

    shown_in, units, _produced_by = json_row
    assert "pass_at_k_lift.json" in shown_in
    assert "percentage point" not in units.lower()
    low, high = (float(value) for value in re.findall(r"[-+]?[\d.]+", _ascii_dashes(units))[:2])

    # The documented bounds are the bounds of a difference of two rates in [0, 1].
    assert (low, high) == (-1.0, 1.0)
    assert "percentage points" in displayed_row[1]

    # And the worked example in the prose is the value the helper actually returns.
    example = _one(
        r"A with-skill arm at (\d+)/(\d+) against a baseline at (\d+)/(\d+) writes `([\d.]+)` to"
        r" `pass_at_k_lift\.json` and shows `\+(\d+) percentage points`"
    )
    with_passed, with_total, without_passed, without_total, written, displayed = example.groups()
    treatment = {"passed_cases": int(with_passed), "total_cases": int(with_total)}
    control = {"passed_cases": int(without_passed), "total_cases": int(without_total)}
    delta = _count_derived_pass_rate_delta(treatment, control)

    assert delta == pytest.approx(float(written))
    assert round(delta * 100) == int(displayed)


def test_dataset_sufficiency_thresholds_match_the_policy() -> None:
    match = _one(
        r"at least (\d+) distinct cases including at least (\d+) negative case; "
        r"a (\d+)% Wilson pass-rate interval half-width no wider than ([\d.]+); "
        r"and at least (\d+) cases scored by both arms"
    )
    distinct, negative, confidence, half_width, paired = match.groups()

    assert int(distinct) == dataset_sufficiency.MIN_DISTINCT_CASES
    assert int(negative) == dataset_sufficiency.MIN_NEGATIVE_CASES
    assert float(half_width) == dataset_sufficiency.MAX_PASS_RATE_INTERVAL_HALF_WIDTH
    assert int(paired) == dataset_sufficiency.MIN_PAIRED_CASES
    assert int(confidence) / 100 == pytest.approx(_wilson_score_interval(1, 4)["confidence_level"])  # type: ignore[index]

    repeated = int(_one(r"the (\d+)-case sufficiency minimum").group(1))
    assert repeated == dataset_sufficiency.MIN_DISTINCT_CASES


# ---------------------------------------------------------------------------
# Tier 1 Quality Score
# ---------------------------------------------------------------------------


def test_quality_weights_match_constants() -> None:
    formula = {
        area.lower(): float(weight)
        for weight, area in re.findall(r"([\d.]+) \* (Correctness|Discoverability|Reliability|Efficiency)", DOC)
    }
    assert formula == QUALITY_SCORE_WEIGHTS

    # The area table repeats the same weights as percentages.
    table = {
        area.lower(): int(cells[0].rstrip("%")) / 100
        for area, cells in _table_rows("Area").items()
        if cells and cells[0].endswith("%")
    }
    assert table == QUALITY_SCORE_WEIGHTS


def test_quality_grades_and_minimum_match_constants() -> None:
    match = _one(r"Grades are A at (\d+) or above, B at (\d+), C at (\d+), D at (\d+), F below")
    documented = list(zip("ABCD", (int(value) for value in match.groups()), strict=True))
    assert [(grade, score) for score, grade in QUALITY_GRADE_THRESHOLDS if grade != "F"] == [
        (grade, score) for grade, score in documented
    ]

    assert int(_one(r"The default `--min-score` is (\d+)").group(1)) == QUALITY_DEFAULT_MIN_SCORE


def test_token_estimate_divisor_matches_constants() -> None:
    assert int(_every(r"`len\(SKILL\.md\) // (\d+)`", DOC)) == QUALITY_CHARS_PER_TOKEN
    budget = int(_one(r"Over the ([\d,]+) estimated-token budget").group(1).replace(",", ""))
    assert budget == QUALITY_RECOMMENDED_MAX_TOKENS


def _quality_score_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(QUALITY_SCORE_PATH.read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _literal_deduct_amounts(node: ast.FunctionDef) -> list[int]:
    """Every literal ``deduct(<int>, ...)`` amount in one function, in source order."""
    found: list[tuple[int, int]] = []
    for call in ast.walk(node):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "deduct"):
            continue
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, int):
            found.append((call.lineno, call.args[0].value))
    return [amount for _lineno, amount in sorted(found)]


def _deduction_amounts_by_area() -> dict[str, set[int]]:
    """Literal ``deduct`` amounts in ``quality_score.py``, grouped by quality area.

    The area is resolved from the source rather than hardcoded: a function that
    binds ``dim = qs.<area>`` owns that area, and a helper it calls with ``dim``
    as its first argument inherits it.
    """
    functions = _quality_score_functions()
    areas = set(QUALITY_SCORE_WEIGHTS)
    owner: dict[str, str] = {}

    for name, node in functions.items():
        for statement in ast.walk(node):
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Attribute)
                and statement.value.attr in areas
                and isinstance(statement.value.value, ast.Name)
                and statement.value.value.id == "qs"
            ):
                owner[name] = statement.value.attr

    # Propagate to helpers invoked as ``self._check_x(dim, ...)`` until stable.
    for _ in range(len(functions)):
        for name, node in functions.items():
            if name not in owner:
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.args
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == "dim"
                    and call.func.attr in functions
                ):
                    owner.setdefault(call.func.attr, owner[name])

    # The type-specific checks are the ones a reviewer caught missing from the
    # page; if the call shape changes, fail rather than silently cover nothing.
    assert owner.get("_check_type_specific") == "correctness"
    assert owner.get("_check_frontmatter_correctness") == "correctness"

    amounts: dict[str, set[int]] = {area: set() for area in areas}
    for name, node in functions.items():
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "deduct"):
                continue
            if not (call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, int)):
                continue  # config-driven amount; covered by the reliability test instead
            target = call.func.value
            if isinstance(target, ast.Attribute) and target.attr in areas:
                amounts[target.attr].add(call.args[0].value)
            elif name in owner:
                amounts[owner[name]].add(call.args[0].value)
    return amounts


def _documented_amounts(cell: str) -> set[int]:
    return {int(value) for value in re.findall(r"\((\d+)(?:\s+each)?\)", cell)}


def test_tier1_deduction_amounts_are_fully_documented() -> None:
    """Every deduction size the scorer can apply appears in the page, and vice versa."""
    source = _deduction_amounts_by_area()
    rows = _table_rows("Area")
    documented = {area.lower(): _documented_amounts(cells[1]) for area, cells in rows.items() if len(cells) > 1}

    # The type-specific sub-table extends the Correctness row.
    for cells in _table_rows("Skill type").values():
        documented["correctness"] |= _documented_amounts(cells[0])

    for area in ("correctness", "discoverability", "efficiency"):
        assert documented[area] == source[area], (
            f"{area}: documented deduction sizes {sorted(documented[area])} "
            f"do not match quality_score.py {sorted(source[area])}"
        )
    # Reliability deductions all come from the quality config, so none is a literal.
    assert source["reliability"] == set()


def test_type_specific_table_lists_every_type_specific_deduction() -> None:
    """The type-specific table is pinned one-for-one, not just by the sizes it uses.

    Its rows quote one amount per check and never collapse them with "each", so
    the counts must match exactly. This is what catches a check being dropped
    from the page while another check of the same size keeps the size present.
    """
    source = Counter(_literal_deduct_amounts(_quality_score_functions()["_check_type_specific"]))

    documented: Counter[int] = Counter()
    for cells in _table_rows("Skill type").values():
        documented.update(int(value) for value in re.findall(r"\((\d+)\)", cells[0]))

    assert documented == source, (
        f"type-specific table lists {sorted(documented.elements())} deductions, "
        f"but _check_type_specific applies {sorted(source.elements())}"
    )


def test_reliability_deductions_match_the_default_quality_config() -> None:
    config, _provenance = load_quality_config_with_provenance(None)
    error_handling = config["error_handling"]
    source = {int(error_handling[rule]["points"]) for rule in ("mentioned", "scripts", "lib")}
    source |= {int(expected["points"]) for expected in config["expected_headers"]}

    documented = _documented_amounts(_table_rows("Area")["Reliability"][1])
    assert documented == source

    # The page must also name each configurable heading it promises a deduction for.
    for expected in config["expected_headers"]:
        assert f"`{expected['heading']}`" in FLAT


# ---------------------------------------------------------------------------
# Cross-references to shipped data
# ---------------------------------------------------------------------------


def test_bundled_grader_suite_size_matches_the_shipped_suite() -> None:
    cases = json.loads(GRADER_SUITE_PATH.read_text(encoding="utf-8"))["cases"]
    splits: dict[str, int] = {}
    for case in cases:
        splits[case["split"]] = splits.get(case["split"], 0) + 1

    match = _one(r"A bundled (\d+)-case reference suite \((\d+) assessment, (\d+) tuning\) ships in the wheel")
    total, assessment, tuning = (int(value) for value in match.groups())

    assert total == len(cases)
    assert assessment == splits["assessment"]
    assert tuning == splits["tuning"]
    assert assessment + tuning == total
