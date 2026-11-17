# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price-file ambiguity must fail before it can affect budget accounting."""

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.run_budget import BudgetConfigurationError, BudgetController, PriceBook

_RATE_FIELDS = ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok")


@pytest.mark.parametrize(
    ("contents", "duplicate_key"),
    [
        (
            '{"model":{"input_per_mtok":2,"output_per_mtok":4},"model":{"input_per_mtok":0,"output_per_mtok":0}}',
            "model",
        ),
        ('{"model":{"input_per_mtok":2,"input_per_mtok":0,"output_per_mtok":4}}', "input_per_mtok"),
        ('{"model":{"input_per_mtok":2,"output_per_mtok":4,"output_per_mtok":0}}', "output_per_mtok"),
        (
            '{"model":{"input_per_mtok":2,"output_per_mtok":4,"cached_input_per_mtok":1,"cached_input_per_mtok":0}}',
            "cached_input_per_mtok",
        ),
        ('{"model":{"input_per_mtok":2,"input_\\u0070er_mtok":0,"output_per_mtok":4}}', "input_per_mtok"),
    ],
    ids=["model", "input", "output", "cached", "escaped-key"],
)
def test_duplicate_price_keys_are_rejected(tmp_path: Path, contents: str, duplicate_key: str) -> None:
    path = tmp_path / "prices.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(BudgetConfigurationError, match=f"duplicate JSON object key {duplicate_key!r}"):
        PriceBook.load(path)


@pytest.mark.parametrize("field", _RATE_FIELDS)
@pytest.mark.parametrize("value", [True, False, 10**1000], ids=["true", "false", "overflowing-integer"])
def test_invalid_price_scalars_have_configuration_errors(tmp_path: Path, field: str, value: object) -> None:
    rates = {"input_per_mtok": 2, "output_per_mtok": 4, "cached_input_per_mtok": 1, field: value}
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"model": rates}), encoding="utf-8")
    message = "must not be booleans" if isinstance(value, bool) else "prices must be finite and non-negative"
    with pytest.raises(BudgetConfigurationError, match=message):
        PriceBook.load(path)


@pytest.mark.parametrize("as_strings", [False, True])
def test_valid_zero_rates_and_repeated_field_names_in_different_models_remain_usable(
    tmp_path: Path, as_strings: bool
) -> None:
    rates = {"input_per_mtok": 0, "output_per_mtok": 4, "cached_input_per_mtok": 0}
    if as_strings:
        rates = {field: str(value) for field, value in rates.items()}
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"model": rates, "other": rates}), encoding="utf-8")
    prices = PriceBook.load(path)
    assert prices.price_for("model") == prices.price_for("other")
    budget = BudgetController(price_book=prices)
    budget.record_usage({"input_tokens": 1000, "output_tokens": 500, "cached_input_tokens": 200}, model="model")
    assert budget.metadata()["used"]["cost_usd"] == pytest.approx(0.002)
