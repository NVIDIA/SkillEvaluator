# SPDX-License-Identifier: Apache-2.0
"""Plugin context limits bound actual aggregate work without starving large members."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator.deduplication.plugin.commands import run_plugin_skill_context_dedup
from skillevaluator.deduplication.plugin.profile import load_plugin_profile


def plugin(root, counts):
    manifest = root / ".claude-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "budget-test", "description": "Test context accounting"}))
    for index, count in enumerate(counts):
        folder = root / "skills" / f"member-{index:02}"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: member-{index:02}\ndescription: Test context accounting\n---\n"
            + "\n".join(f"## Section {section}\n" + "Distinct useful instructions. " * 8 for section in range(count))
        )
    return root


class OrthogonalEmbedding:
    def __init__(self, dimension=2048):
        self.dimension = dimension
        self.embedded = 0

    def embed(self, texts):
        vectors = []
        for _ in texts:
            vector = [0.0] * self.dimension
            vector[self.embedded % self.dimension] = 1.0 if (self.embedded // self.dimension) % 2 == 0 else -1.0
            vectors.append(vector)
            self.embedded += 1
        return vectors


@pytest.fixture
def embedding(monkeypatch):
    client = OrthogonalEmbedding()
    monkeypatch.setattr(
        "skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient", lambda **_: client
    )
    return client


def test_explicit_budget_completes_all_18_skewed_members(tmp_path, embedding):
    # Exact unchanged AEM distribution: 716 chunks and 32,652 within-member pairs.
    root = plugin(tmp_path / "plugin", [19, 16, 12, 19, 83, 15, 18, 11, 13, 9, 171, 24, 107, 5, 116, 39, 14, 25])
    results = run_plugin_skill_context_dedup(
        load_plugin_profile(root), max_chunks=1024, max_scalar_comparisons=100_000_000
    )
    assert len(results) == 18
    assert all(result.passed and not result.is_incomplete for result in results)
    assert embedding.embedded == 716
    budget = results[-1].metadata["plugin_context_budget"]
    assert budget["limits"]["chunks"] == 1024
    assert budget["limits"]["scalar_comparisons"] == 100_000_000
    assert budget["consumed"]["chunks_embedded"] == 716
    assert budget["consumed"]["scalar_comparisons"] == 66_871_296
    assert budget["vector_dimension"] == 2048


@pytest.mark.parametrize("allowance,complete,embedded,used", [(7, False, 3, 6), (8, True, 5, 8)])
def test_actual_remaining_scalar_budget_is_checked_before_next_member_provider_work(
    tmp_path, embedding, allowance, complete, embedded, used
):
    embedding.dimension = 2
    root = plugin(tmp_path / "plugin", [3, 2])
    results = run_plugin_skill_context_dedup(load_plugin_profile(root), max_scalar_comparisons=allowance)
    assert all(result.passed and not result.is_incomplete for result in results) is complete
    assert embedding.embedded == embedded
    assert results[-1].metadata["plugin_context_budget"]["consumed"]["scalar_comparisons"] == used
    if not complete:
        assert results[-1].metadata["work_limit"]["check_name"] == "scalar_comparison_limit"


def test_embedding_dimension_must_remain_consistent_across_members(tmp_path, embedding):
    embedding.dimension = 2
    original = embedding.embed

    def changing(texts):
        if embedding.embedded:
            embedding.dimension = 3
        return original(texts)

    embedding.embed = changing
    results = run_plugin_skill_context_dedup(load_plugin_profile(plugin(tmp_path / "plugin", [2, 2])))
    assert results[0].passed
    assert results[1].is_incomplete
    assert "embedding-provider" in results[1].incomplete_scans
    assert results[-1].metadata["plugin_context_budget"]["consumed"]["scalar_comparisons"] == 2


@pytest.mark.parametrize("cap,limit", [("MAX_PLUGIN_DEDUP_LLM_CALLS", 1), ("MAX_PLUGIN_DEDUP_PROMPT_CHARS", 3000)])
def test_cluster_reviews_and_prompt_characters_debit_the_shared_remaining_budget(
    tmp_path, monkeypatch, embedding, cap, limit
):
    from skillevaluator.deduplication.plugin import commands as plugin_commands

    prompts = []

    class Judge:
        def __init__(self, **_kwargs):
            pass

        def _resolved_config(self):
            return SimpleNamespace(provider="fixture", model="fixture")

        def extract_json_from_response(self, system, user, **_kwargs):
            prompts.append((system, user))
            return {"verdict": "RELATED_BUT_DISTINCT", "confidence": 0.9, "rationale": "different", "suggestion": ""}

    embedding.embed = lambda texts: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(plugin_commands, cap, limit)
    monkeypatch.setattr("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient", Judge)
    results = run_plugin_skill_context_dedup(load_plugin_profile(plugin(tmp_path / "plugin", [2, 2])))
    assert results[0].passed and not results[0].is_incomplete
    assert results[1].is_incomplete
    used = results[-1].metadata["plugin_context_budget"]["consumed"]
    assert used["llm_clusters_submitted"] == 1
    assert len(prompts) == 1
    assert used["prompt_chars_submitted"] == len(prompts[0][0]) + len(prompts[0][1])
    assert used["prompt_chars_submitted"] <= 3000


def test_context_budget_survives_json_and_retained_html(tmp_path, embedding):
    from skillevaluator.reporting.html import HTMLReporter
    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.reporting.retained import retained_results

    result = run_plugin_skill_context_dedup(load_plugin_profile(plugin(tmp_path / "plugin", [2])))[0]
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    restored = retained_results(payload)
    assert restored[0].metadata["plugin_context_budget"] == result.metadata["plugin_context_budget"]
    html = HTMLReporter(include_timestamp=False).render_all(restored)
    assert "2/512 chunks collected" in html
    assert "2048/25000000 scalar comparisons consumed" in html


@pytest.mark.parametrize(
    "options",
    [
        {"max_chunks": 0},
        {"max_chunks": True},
        {"max_chunks": 32769},
        {"max_scalar_comparisons": 0},
        {"max_scalar_comparisons": 1.5},
        {"max_scalar_comparisons": 536_346_625},
    ],
)
def test_invalid_operator_limits_are_rejected_before_provider_work(tmp_path, embedding, options):
    profile = load_plugin_profile(plugin(tmp_path / "plugin", [2]))
    with pytest.raises(ValueError, match=r"context.*(chunks|scalar)|(?:chunks|scalar).*context"):
        run_plugin_skill_context_dedup(profile, **options)
    assert embedding.embedded == 0


@pytest.mark.parametrize("command", [["tier2", "dedup-scan"], ["tier2", "context-optimization-check"], ["validate"]])
def test_cli_operator_limits_reach_the_real_context_gate(tmp_path, monkeypatch, embedding, command):
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    root = plugin(tmp_path / "plugin", [2])
    output = tmp_path / "report"
    monkeypatch.setattr(
        cli_module, "run_validation", lambda *_args, **_kwargs: [ValidationResult(validator_name="schema")]
    )
    monkeypatch.setattr("skillevaluator.provider_config.resolve_embedding_provider", object)
    flags = ["--tiers", "1,2"] if command == ["validate"] else []
    result = CliRunner().invoke(
        cli_module.cli,
        [
            *command,
            str(root),
            *flags,
            "--plugin-context-max-chunks",
            "1",
            "--plugin-context-max-scalar-comparisons",
            "100000000",
            "-r",
            "json",
            "-o",
            str(output),
        ],
    )
    # Plugin context remains advisory in combined validation; native Tier 2
    # commands report the incomplete result through their existing nonzero exit.
    assert result.exit_code == (0 if command == ["validate"] else 1), result.output
    reports = list(output.glob("*.json"))
    assert len(reports) == 1, result.output
    payload = json.loads(reports[0].read_text())
    limited = [
        row for row in payload["results"] if row.get("work_limit", {}).get("check_name") == "plugin_content_limit"
    ]
    assert len(limited) == 1, result.output
    assert limited[0]["work_limit"]["chunk_limit"] == 1
    assert limited[0]["status"] == "incomplete"
    assert limited[0]["plugin_context_budget"]["limits"]["scalar_comparisons"] == 100_000_000
    assert embedding.embedded == 0
