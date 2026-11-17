import hashlib
import json

import pytest
from rich.console import Console

from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.run_budget import BudgetConfigurationError, BudgetController, PriceBook, budget_line
from skillevaluator.tier3.result_display import render_evaluation_result


def load_prices(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "priced": {"input_per_mtok": 2, "output_per_mtok": 4, "cached_input_per_mtok": 1},
                "fallback": {"input_per_mtok": 1, "output_per_mtok": 2},
                "unused": {"input_per_mtok": 9, "output_per_mtok": 9},
            }
        )
    )
    return path, PriceBook.load(path)


def test_file_provenance_hashes_loaded_bytes_and_reports_only_applied_rates(tmp_path):
    path, prices = load_prices(tmp_path)
    original = path.read_bytes()
    budget = BudgetController(price_book=prices)
    assert budget.metadata()["pricing"]["rates_per_mtok"] == {}
    budget.record_usage({"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 400}, model="priced")
    budget.record_usage({"input_tokens": 1000, "output_tokens": 0, "cached_input_tokens": 500}, model="fallback")
    path.write_text("{}")
    assert budget.metadata()["pricing"] == {
        "source": "user_price_file",
        "file_name": "prices.json",
        "sha256": hashlib.sha256(original).hexdigest(),
        "rates_per_mtok": {
            "priced": {"input_per_mtok": 2, "output_per_mtok": 4, "cached_input_per_mtok": 1},
            "fallback": {"input_per_mtok": 1, "output_per_mtok": 2, "cached_input_per_mtok": 1},
        },
        "currency": "as supplied by the price file (labelled USD by the --max-cost-usd flag)",
        "verified_against_provider_billing": False,
    }
    assert budget.metadata()["used"]["cost_usd"] == pytest.approx(0.003)
    assert str(tmp_path) not in json.dumps(budget.metadata())


def test_derivation_counts_follow_pricing_precedence_fallback_and_unknown_usage(tmp_path):
    _, prices = load_prices(tmp_path)
    budget = BudgetController(price_book=prices)
    assert budget.metadata()["charge_coverage"]["attempts_by_derivation"] == {
        "price_file": 0,
        "provider_reported": 0,
        "unmeasured": 0,
    }
    budget.record_usage(
        {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 400, "cost_usd": 99}, model="priced"
    )
    budget.record_usage({"total_cost_usd": 0.5}, model="fallback")
    budget.record_usage(None, model="unused")
    metadata = budget.metadata()
    assert metadata["charge_coverage"]["attempts_by_derivation"] == {
        "price_file": 1,
        "provider_reported": 1,
        "unmeasured": 1,
    }
    assert sum(metadata["charge_coverage"]["attempts_by_derivation"].values()) == metadata["finished_attempts"] == 3
    assert metadata["used"]["cost_usd"] == pytest.approx(0.502)
    assert metadata["unmeasured_attempts"]["cost_usd"] == 1
    assert set(metadata["pricing"]["rates_per_mtok"]) == {"priced"}
    metadata["charge_coverage"]["attempts_by_derivation"]["price_file"] = 99
    assert budget.metadata()["charge_coverage"]["attempts_by_derivation"]["price_file"] == 1


@pytest.mark.parametrize("usage,derivation", [({"cost_usd": 0}, "provider_reported"), ({}, "unmeasured")])
def test_no_price_file_has_no_pricing_object(usage, derivation):
    budget = BudgetController()
    budget.record_usage(usage)
    metadata = budget.metadata()
    assert "pricing" not in metadata
    assert metadata["charge_coverage"]["attempts_by_derivation"][derivation] == 1
    assert metadata["charge_coverage"]["judge_inference"].startswith("not priced")
    for charge in ("environment_setup", "container", "backend", "infrastructure"):
        assert metadata["charge_coverage"][charge] == "not included"


@pytest.mark.parametrize("contents", [b"\xff", b"{", b"{}"])
def test_invalid_price_file_fails_without_provenance(tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_bytes(contents)
    with pytest.raises(BudgetConfigurationError):
        PriceBook.load(path)


def test_judge_token_or_cost_fields_do_not_feed_agent_pricing(tmp_path):
    _, prices = load_prices(tmp_path)
    job = tmp_path / "job"
    verifier = job / "trial" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "judge_attempts.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {"method": "llm", "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}, "cost_usd": 99}
                ]
            }
        )
    )
    usage = runner._budget_usage(job)
    assert usage == {"judge_calls": 1}
    budget = BudgetController(price_book=prices)
    budget.record_usage(usage, model="priced")
    assert budget.metadata()["used"]["cost_usd"] == "not measured"
    assert budget.metadata()["charge_coverage"]["attempts_by_derivation"]["unmeasured"] == 1


@pytest.mark.parametrize("cap", [0, 1])
def test_cost_qualifier_reaches_terminal_and_unknown_usage_stops_admission(tmp_path, cap):
    _, prices = load_prices(tmp_path)
    budget = BudgetController(max_cost_usd=cap, price_book=prices)
    budget.record_usage(None)
    assert not budget.admit(agent="a", arm="with", case_id="c", attempt=1)
    assert budget.metadata()["charge_coverage"]["attempts_by_derivation"]["unmeasured"] == 1
    qualifier = "agent inference only (user-supplied rates; provider-cost fallback), not billed spend"
    assert budget_line(budget.metadata()).endswith(qualifier)
    output = Console(record=True, force_terminal=False, width=500)
    render_evaluation_result({"execution_status": "failed", "budget": budget.metadata()}, console=output)
    assert qualifier in output.export_text()
    assert qualifier not in budget_line(BudgetController(price_book=prices).metadata())
