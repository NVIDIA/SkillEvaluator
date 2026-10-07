# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bundle-reference plugin manifest model (skillevaluator.models.plugin)."""

from typing import get_args

import pytest
from pydantic import ValidationError

from skillevaluator.models.plugin import PLUGIN_REF_SOURCES, PluginManifest, PluginSelector

GITLAB_SUBGROUP_REPO = "example-group/tools/agent-catalog"


def _valid_data(**overrides):
    data = {
        "name": "my-bundle",
        "author": {"email": "dev@example.com"},
        "skills": {"refs": ["github::example-org/example-repo::skills::build-infra"]},
    }
    data.update(overrides)
    return data


class TestPluginManifestValid:
    def test_minimal_valid_with_skill_ref(self):
        manifest = PluginManifest(**_valid_data())
        assert manifest.name == "my-bundle"
        assert manifest.author.email == "dev@example.com"

    def test_valid_with_selector_dict(self):
        manifest = PluginManifest(
            **_valid_data(
                skills={"refs": [{"source": "github", "repo": "example-org/example-repo", "path": "skills/foo"}]}
            )
        )
        assert isinstance(manifest.skills.refs[0], PluginSelector)

    def test_valid_with_mcp_only(self):
        manifest = PluginManifest(
            name="bundle",
            author={"email": "dev@example.com"},
            mcp=[{"name": "filesystem", "provider": "stdio"}],
        )
        assert manifest.mcp[0].provider == "stdio"

    def test_numeric_version_coerced_to_str(self):
        manifest = PluginManifest(**_valid_data(version=1.0))
        assert manifest.version == "1.0"

    def test_empty_legacy_filters_tolerated(self):
        manifest = PluginManifest(
            **_valid_data(
                skills={
                    "refs": ["github::example-org/example-repo::skills::foo"],
                    "include": [],
                    "exclude": [],
                }
            )
        )
        assert manifest.skills.refs


class TestPluginManifestInvalid:
    def test_missing_author(self):
        with pytest.raises(ValidationError):
            PluginManifest(name="bundle", skills={"refs": ["github::a/b::skills::x"]})

    def test_author_email_without_at(self):
        with pytest.raises(ValidationError, match="valid email"):
            PluginManifest(**_valid_data(author={"email": "not-an-email"}))

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError):
            PluginManifest(**_valid_data(workflows={"refs": []}))

    def test_requires_at_least_one_dependency(self):
        with pytest.raises(ValidationError, match="at least one dependency"):
            PluginManifest(name="bundle", author={"email": "dev@example.com"})

    def test_populated_legacy_include_rejected(self):
        with pytest.raises(ValidationError, match="not supported"):
            PluginManifest(**_valid_data(skills={"refs": ["github::a/b::skills::x"], "include": ["nope"]}))

    def test_canonical_ref_bad_source_rejected(self):
        with pytest.raises(ValidationError, match="canonical ref"):
            PluginManifest(**_valid_data(skills={"refs": ["badsource::a/b::skills::x"]}))

    def test_canonical_ref_repo_without_slash_rejected(self):
        with pytest.raises(ValidationError, match="canonical ref"):
            PluginManifest(**_valid_data(skills={"refs": ["github::noslash::skills::x"]}))

    def test_selector_repo_without_slash_rejected(self):
        with pytest.raises(ValidationError, match="full repository name"):
            PluginManifest(
                **_valid_data(skills={"refs": [{"source": "github", "repo": "noslash", "path": "skills/x"}]})
            )

    def test_selector_path_without_slash_rejected(self):
        with pytest.raises(ValidationError, match="repo-relative path"):
            PluginManifest(**_valid_data(skills={"refs": [{"source": "github", "repo": "a/b", "path": "foo"}]}))

    def test_duplicate_mcp_entries_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate MCP entry"):
            PluginManifest(
                name="bundle",
                author={"email": "dev@example.com"},
                mcp=[
                    {"name": "fs", "provider": "stdio"},
                    {"name": "fs", "provider": "stdio"},
                ],
            )

    def test_empty_mcp_provider_rejected(self):
        with pytest.raises(ValidationError):
            PluginManifest(
                name="bundle",
                author={"email": "dev@example.com"},
                mcp=[{"name": "fs", "provider": ""}],
            )


class TestPluginRefSources:
    """``gitlab`` is accepted wherever ``github`` and ``git`` are, in both ref forms."""

    def test_selector_and_canonical_forms_share_one_source_list(self):
        assert get_args(PluginSelector.model_fields["source"].annotation) == PLUGIN_REF_SOURCES
        assert set(PLUGIN_REF_SOURCES) == {"github", "gitlab", "git"}

    @pytest.mark.parametrize(
        "ref",
        [
            f"gitlab::{GITLAB_SUBGROUP_REPO}::skills::ticket-triage",
            "gitlab::example-group/agent-catalog::skills::ticket-triage",
            {"source": "gitlab", "repo": GITLAB_SUBGROUP_REPO, "path": "skills/ticket-triage"},
            {"source": "gitlab", "repo": "example-group/agent-catalog", "path": "team-skills/ops/ticket-triage"},
        ],
        ids=["canonical-subgroup", "canonical", "selector-subgroup", "selector"],
    )
    def test_gitlab_skill_and_rule_refs_are_accepted(self, ref):
        manifest = PluginManifest(**_valid_data(skills={"refs": [ref]}, rules={"refs": [ref]}))

        for section in (manifest.skills, manifest.rules):
            [parsed] = section.refs
            if isinstance(ref, dict):
                assert isinstance(parsed, PluginSelector)
                assert (parsed.source, parsed.repo) == ("gitlab", ref["repo"])
            else:
                assert parsed == ref

    @pytest.mark.parametrize(
        "ref",
        [
            "bitbucket::example-group/agent-catalog::skills::ticket-triage",
            {"source": "bitbucket", "repo": "example-group/agent-catalog", "path": "skills/ticket-triage"},
        ],
        ids=["canonical", "selector"],
    )
    def test_unknown_source_is_still_rejected(self, ref):
        with pytest.raises(ValidationError, match="gitlab"):
            PluginManifest(**_valid_data(skills={"refs": [ref]}))
