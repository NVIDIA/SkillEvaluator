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
