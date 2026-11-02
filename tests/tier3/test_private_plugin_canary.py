# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import importlib.util
import json

import pytest

MODULE = "skillevaluator.tier3.plugin_canary"


def test_private_canary_module_exists():
    assert importlib.util.find_spec(MODULE) is not None


@pytest.fixture
def canary():
    return importlib.import_module(MODULE)


def test_private_preparation_strips_authored_canary_when_planting_is_off(canary):
    original = {"id": "fixture", "question": "hello", "skilleval_canary": {"token": "authored"}}
    prepared = canary.prepare_canary_entry(original)
    assert "skilleval_canary" not in prepared.entry
    assert prepared.spec is None and prepared.environment == {}
    assert original["skilleval_canary"]["token"] == "authored"


def test_each_explicit_plant_has_a_fresh_verifier_only_token(canary):
    one = canary.prepare_canary_entry(
        {"question": "hello"}, plant=True, workspace_roots=("/workspace",), mcp_servers=("relay",)
    )
    two = canary.prepare_canary_entry({"question": "hello"}, plant=True)
    assert one.spec["token"] != two.spec["token"]
    assert one.entry["skilleval_canary"] == one.spec
    assert one.environment[one.spec["env_var"]] == one.spec["token"]
    assert one.spec["token"] not in one.entry["question"]


def test_private_context_staging_creates_decoy_and_gitignore(canary, tmp_path):
    prepared = canary.prepare_canary_entry({}, plant=True)
    copied = canary.stage_canary_context(tmp_path, prepared.spec)
    decoy = tmp_path / "skilleval-canary" / "canary.env"
    assert prepared.spec["token"] in decoy.read_text()
    assert (decoy.parent / ".gitignore").read_text() == "*\n"
    assert copied == ["COPY skilleval-canary/ /workspace/.skilleval/"]


def test_context_staging_refuses_existing_symlinks(canary, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "skilleval-canary").symlink_to(outside, target_is_directory=True)
    prepared = canary.prepare_canary_entry({}, plant=True)
    with pytest.raises(ValueError):
        canary.stage_canary_context(tmp_path, prepared.spec)
    assert list(outside.iterdir()) == []


def test_published_entry_redacts_token_without_changing_verifier_entry(canary):
    prepared = canary.prepare_canary_entry({}, plant=True)
    published = canary.redact_canary_entry(prepared.entry)
    assert prepared.spec["token"] not in json.dumps(published)
    assert prepared.entry["skilleval_canary"]["token"] == prepared.spec["token"]


@pytest.mark.parametrize("plant", ["true", 1, [], None])
def test_plant_opt_in_requires_a_real_boolean(canary, plant):
    with pytest.raises(ValueError):
        canary.prepare_canary_entry({}, plant=plant)


def test_canary_preparation_bounds_server_and_workspace_collections(canary):
    with pytest.raises(ValueError):
        canary.prepare_canary_entry({}, plant=True, mcp_servers=[f"server{index}" for index in range(33)])
    with pytest.raises(ValueError):
        canary.prepare_canary_entry({}, plant=True, workspace_roots=[f"/workspace/{index}" for index in range(9)])


def test_canary_context_does_not_follow_a_linked_output_parent(canary, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "linked-output"
    parent.symlink_to(outside, target_is_directory=True)
    prepared = canary.prepare_canary_entry({}, plant=True)
    with pytest.raises(ValueError):
        canary.stage_canary_context(parent, prepared.spec)
    assert list(outside.iterdir()) == []


def test_additional_workdir_keeps_the_actual_decoy_workspace_root(canary):
    prepared = canary.prepare_canary_entry({}, plant=True, workspace_roots=("/agent/work",))
    assert prepared.spec["workspace_roots"] == ["/workspace", "/agent/work"]


def test_server_enumeration_cap_cannot_be_hidden_by_duplicates(canary):
    with pytest.raises(ValueError):
        canary.prepare_canary_entry({}, plant=True, mcp_servers=["relay"] * 33)


def test_context_rejects_a_spec_that_names_another_decoy_file(canary, tmp_path):
    prepared = canary.prepare_canary_entry({}, plant=True)
    changed = {**prepared.spec, "file": "another.env"}
    with pytest.raises(ValueError):
        canary.stage_canary_context(tmp_path, changed)
