# SPDX-License-Identifier: Apache-2.0
"""Public runtime options and the shared service boundary."""

from pathlib import Path

import pytest

from skillevaluator.evaluation import EvaluationOptions
from skillevaluator.evaluation.options import PluginOptions


@pytest.mark.parametrize("value", [-1, 6, True, 1.5])
def test_public_retry_option_is_a_bounded_integer(value):
    with pytest.raises(ValueError, match="trial_retries"):
        PluginOptions(trial_retries=value)


@pytest.mark.parametrize("value", [["A"], ("A",) * 65, ("x" * 513,), (1,)])
def test_probe_mcp_env_is_a_bounded_tuple_of_strings(value):
    with pytest.raises(ValueError, match="probe_mcp_env"):
        PluginOptions(probe_mcp_env=value)


def test_probe_mcp_env_accepts_grants_at_the_bounds():
    grants = ("x" * 512,) * 64
    assert PluginOptions(probe_mcp_env=grants).probe_mcp_env == grants


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("allowed_private_hosts", ("169.254.169.254",), "cloud metadata"),
        ("allowed_private_hosts", (".corp.example",), r"\*\.suffix"),
        ("hook_allowed_urls", ("ftp://h.example.test",), "HTTP or HTTPS"),
    ],
)
def test_run_option_allowlists_follow_the_policy_rules(field, value, match):
    with pytest.raises(ValueError, match=match):
        PluginOptions(**{field: value})


def test_run_option_allowlists_are_stripped_and_deduplicated():
    options = PluginOptions(
        allowed_private_hosts=(" localhost ", "localhost"),
        hook_allowed_urls=(" https://h.example.test/ ", "https://h.example.test/"),
    )
    assert options.allowed_private_hosts == ("localhost",)
    assert options.hook_allowed_urls == ("https://h.example.test/",)


@pytest.mark.parametrize(
    "field,value",
    [("plugin_load", "other"), ("lift_mode", "other"), ("probe_mcp", 1), ("policy", {}), ("policy", "default")],
)
def test_public_options_validate_direct_api_values(field, value):
    with pytest.raises(ValueError, match=field):
        PluginOptions(**{field: value})


def test_plugin_controls_do_not_leak_into_skill_engine_kwargs():
    assert (
        not {"plugin_load", "lift_mode", "probe_mcp", "trial_retries"}
        & EvaluationOptions(Path("skill")).engine_kwargs().keys()
    )


@pytest.mark.parametrize("value", ["strict", {"MCP_DECLARATION.*": "critical"}])
def test_plugin_policy_must_be_a_validation_policy(value):
    with pytest.raises(ValueError, match="policy"):
        PluginOptions(policy=value)
