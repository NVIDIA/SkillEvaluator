# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM rubric evaluation validator."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.constants import (
    RUBRIC_CRITERIA,
    RUBRIC_MAX_TOKENS,
    RUBRIC_REASONING_MAX_TOKENS,
)
from skillevaluator.inference.types import LLMClientError, LLMClientRefusedError, LLMClientTruncatedError
from skillevaluator.reporting import HTMLReporter
from skillevaluator.validators import rubric_eval
from skillevaluator.validators.rubric_eval import (
    _DEFAULT_SUPPLEMENTARY_TOTAL_BYTES,
    _MAX_CONFIGURED_SUPPLEMENTARY_TOTAL_BYTES,
    _SUPPLEMENTARY_TOTAL_BYTES_ENV,
    RUBRIC_JSON_SCHEMA,
    RubricEvalValidator,
    RubricJudge,
    _collect_supplementary_content,
    _extract_json,
    _UnsafeRubricInputError,
    supplementary_total_bytes,
)


def _write_skill(tmp_path: Path, name: str = "my-skill") -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n# Test",
        encoding="utf-8",
    )
    return skill_dir


def _complete_checks(
    *,
    scores: dict[str, int | float] | None = None,
    passes: dict[str, bool] | None = None,
) -> list[dict]:
    scores = scores or {}
    passes = passes or {}
    return [
        {
            "id": criterion["id"],
            "criterion": criterion["criterion"],
            "pass": passes.get(criterion["id"], True),
            "score": scores.get(criterion["id"], 9),
            "notes": f"Evidence for {criterion['id']}",
        }
        for criterion in RUBRIC_CRITERIA
    ]


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"score": 85, "overall_pass": true}')
        assert result["score"] == 85

    def test_json_in_code_fence(self):
        result = _extract_json('```json\n{"score": 90}\n```')
        assert result["score"] == 90

    def test_json_with_preamble(self):
        result = _extract_json('Here is the result:\n{"score": 75, "summary": "ok"}')
        assert result["score"] == 75

    def test_empty_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("This is just text with no JSON")


class TestRubricJudge:
    def test_system_prompt_not_empty(self):
        judge = RubricJudge()
        assert "criterion" in judge.get_system_prompt().lower()

    def test_user_prompt_contains_skill_content(self):
        judge = RubricJudge()
        prompt = judge.create_user_prompt(
            skill_name="test-skill",
            skill_content="# Test\nThis is a test skill.",
        )
        assert "test-skill" in prompt
        assert "This is a test skill" in prompt

    @pytest.mark.parametrize(
        "content", ["# Test", "---\nname: demo\n---\n# Test\n", "---\r\nname: demo\r\n---\r\n", ""]
    )
    def test_user_prompt_preserves_source_without_added_yaml_separators(self, content):
        prompt = RubricJudge().create_user_prompt(skill_name="demo", skill_content=content)

        assert f"SKILL.md Content:\n{content}\n\nEvaluation Criteria:" in prompt

    def test_user_prompt_with_supplementary(self):
        judge = RubricJudge()
        prompt = judge.create_user_prompt(
            skill_name="test-skill",
            skill_content="# Test",
            supplementary_content={
                "scripts": {"main.py": "print('hello')"},
                "references": {},
            },
        )
        assert "main.py" in prompt
        assert "print('hello')" in prompt

    def test_fallback_response(self):
        judge = RubricJudge()
        fallback = judge.get_fallback_response()
        assert fallback["overall_pass"] is False
        assert fallback["score"] == 0
        assert fallback["checks"] == []

    def test_parse_response_valid(self):
        judge = RubricJudge()
        result = judge.parse_response('{"score": 80, "overall_pass": true, "checks": []}')
        assert result["score"] == 80


class TestRubricJudgeTokenBudget:
    """The reasoning budget follows the model each request actually targets."""

    @staticmethod
    def _gateway_env(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
        monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "test-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "https://gateway.example.com/v1")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", model)

    @staticmethod
    def _openai_request(judge: RubricJudge) -> dict:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="{}"))]
        with patch("openai.OpenAI", return_value=mock_openai):
            judge.completions("system", "user")
        return mock_openai.chat.completions.create.call_args.kwargs

    def test_reasoning_model_gets_a_larger_token_budget(self, monkeypatch: pytest.MonkeyPatch):
        """Reasoning models spend the completion budget before emitting content.

        Measured on deepseek/deepseek-v4.1-flash with a 4096-token cap:
        ``reasoning_tokens=4000, text_tokens=0`` and ``finish_reason=length``,
        so the rubric JSON was truncated mid-object and never parsed.
        """
        self._gateway_env(monkeypatch, "deepseek/deepseek-v4.1-flash")
        request = self._openai_request(RubricJudge())
        assert request["max_tokens"] == RUBRIC_REASONING_MAX_TOKENS
        assert request["max_tokens"] > RUBRIC_MAX_TOKENS

    def test_standard_model_keeps_default_token_budget(self, monkeypatch: pytest.MonkeyPatch):
        """Non-reasoning models must not pay the larger budget's latency."""
        self._gateway_env(monkeypatch, "gpt-4o")
        assert self._openai_request(RubricJudge())["max_tokens"] == RUBRIC_MAX_TOKENS

    def test_explicit_token_override_is_preserved(self, monkeypatch: pytest.MonkeyPatch):
        """An explicit max_tokens always wins over the reasoning-model default."""
        self._gateway_env(monkeypatch, "deepseek/deepseek-v4.1-flash")
        assert self._openai_request(RubricJudge(max_tokens=1234))["max_tokens"] == 1234

    def test_positional_model_overrides_environment_model(self, monkeypatch: pytest.MonkeyPatch):
        """A positional model is the target, so the environment model must not pick the budget."""
        self._gateway_env(monkeypatch, "deepseek/deepseek-v4.1-flash")
        request = self._openai_request(RubricJudge("gpt-4o"))
        assert request["model"] == "gpt-4o"
        assert request["max_tokens"] == RUBRIC_MAX_TOKENS

    def test_explicit_endpoint_uses_its_own_model(self, monkeypatch: pytest.MonkeyPatch):
        """An explicit endpoint targets the class default model, not SKILL_EVAL_LLM_MODEL."""
        self._gateway_env(monkeypatch, "o3")
        judge = RubricJudge(api_key="test-key", base_url="https://gateway.example.com/v1")
        request = self._openai_request(judge)
        assert request["model"] == judge._resolved_config().model != "o3"
        assert request["max_tokens"] == RUBRIC_MAX_TOKENS

    def test_provider_configured_after_construction(self, monkeypatch: pytest.MonkeyPatch):
        """The budget is chosen per request, after the provider is configured."""
        for name in (
            "SKILL_EVAL_LLM_PROVIDER",
            "SKILL_EVAL_LLM_MODEL",
            "OPENAI_API_KEY",
            "NVIDIA_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        judge = RubricJudge()
        self._gateway_env(monkeypatch, "deepseek/deepseek-v4.1-flash")
        assert self._openai_request(judge)["max_tokens"] == RUBRIC_REASONING_MAX_TOKENS

    def test_anthropic_reasoning_budget_is_sent_without_streaming(self, monkeypatch: pytest.MonkeyPatch):
        """The Anthropic SDK refuses non-streaming requests above 21,333 output tokens.

        A raised budget on that transport must stay under the limit, or the SDK
        raises before sending and the judge reports itself unavailable.
        """
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")
        monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "deepseek-reasoner")
        requests: list[dict] = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "deepseek-reasoner",
                    "content": [{"type": "text", "text": "{}"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            content = RubricJudge(http_client=http_client).completions("system", "user")

        assert content == "{}"
        assert len(requests) == 1
        assert RUBRIC_MAX_TOKENS <= requests[0]["max_tokens"] <= 21333


class TestCollectSupplementaryContent:
    def test_empty_dir(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")
        result = _collect_supplementary_content(skill_dir)
        assert result == {"scripts": {}, "references": {}}

    def test_collects_scripts(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "main.py").write_text("print('hello')")
        result = _collect_supplementary_content(skill_dir)
        assert "main.py" in result["scripts"]

    def test_collects_node_scripts_for_the_judge(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        for name in ("run.mjs", "run.cjs", "run.js"):
            (scripts / name).write_text("console.log('hello')")
        (scripts / "notes.txt").write_text("not a script")
        result = _collect_supplementary_content(skill_dir)
        assert set(result["scripts"]) == {"run.mjs", "run.cjs", "run.js"}

    def test_collects_references(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("# Guide")
        result = _collect_supplementary_content(skill_dir)
        assert "guide.md" in result["references"]

    def test_truncates_large_files(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "big.py").write_text("x" * 10000)
        result = _collect_supplementary_content(skill_dir, max_bytes_per_file=100)
        assert len(result["scripts"]["big.py"]) < 200


class TestLinkedRubricGuidance:
    def test_collects_transitive_guides_with_cycles_and_distinct_source_paths(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text(
            "[Guide][g]\n\n[g]: GUIDE.md#intro\n\n"
            '<a href="guides/a/README.MD">A</a>\n'
            "[Space](guides/with%20space.markdown?version=1#section)\n"
            "![Image](missing.md)\n`[Example](missing.md)`\n"
            "```md\n[Example](missing.md)\n```\n[Web](https://example.com/missing.md)\n"
            "[Asset](missing.json)\n[Anchor](#intro)"
        )
        (root / "guides/a").mkdir(parents=True)
        (root / "guides/b").mkdir()
        guide = "Required guidance. [A](guides/a/README.MD#again)"
        first = "A guidance. [B](../b/README.md?version=1)"
        second = "B guidance. [Root](../../GUIDE.md) [Skill](../../SKILL.md)"
        (root / "GUIDE.md").write_text(guide)
        (root / "guides/a/README.MD").write_text(first)
        (root / "guides/b/README.md").write_text(second)
        (root / "guides/with space.markdown").write_text("Space guidance")

        collected = _collect_supplementary_content(root)

        assert collected["linked_markdown"] == {
            "GUIDE.md": guide,
            "guides/a/README.MD": first,
            "guides/b/README.md": second,
            "guides/with space.markdown": "Space guidance",
        }

    def test_required_guidance_gets_budget_before_optional_previews(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](GUIDE.md)")
        (root / "GUIDE.md").write_text("required")
        (root / "scripts").mkdir()
        (root / "scripts/main.py").write_text("optional" * 100)

        collected = _collect_supplementary_content(root, max_bytes_per_file=100, max_total_bytes=8)

        assert collected["linked_markdown"] == {"GUIDE.md": "required"}
        assert collected["scripts"] == {}

    def test_linked_reference_is_not_duplicated_as_optional_preview(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](references/guide.md)")
        (root / "references").mkdir()
        (root / "references/guide.md").write_text("required")

        collected = _collect_supplementary_content(root, max_total_bytes=8)

        assert collected["linked_markdown"] == {"references/guide.md": "required"}
        assert collected["references"] == {}

    def test_exact_utf8_budget_preserves_complete_required_document(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](GUIDE.md)")
        (root / "GUIDE.md").write_text("π", encoding="utf-8")

        collected = _collect_supplementary_content(root, max_bytes_per_file=2, max_total_bytes=2)

        assert collected["linked_markdown"] == {"GUIDE.md": "π"}

    @pytest.mark.parametrize(
        ("earlier_guides", "total", "remaining"),
        [
            # The budget itself splits the character: nothing has been collected yet.
            pytest.param(0, 1, 1, id="budget-splits-the-first-document"),
            # The running total does: an earlier document leaves too few bytes for this one.
            pytest.param(1, 8, 0, id="earlier-document-exhausts-the-budget"),
        ],
    )
    def test_required_utf8_content_is_never_truncated(self, tmp_path, earlier_guides, total, remaining):
        # The supplement budget is the only bound on required guidance, so it must stop the
        # collection rather than cut a multi-byte character in half -- whether the budget is too
        # small from the start or an earlier document has already spent it.
        root = _write_skill(tmp_path)
        links = [f"[Earlier](earlier{index}.md)" for index in range(earlier_guides)]
        (root / "SKILL.md").write_text("\n".join([*links, "[Guide](GUIDE.md)"]))
        for index in range(earlier_guides):
            (root / f"earlier{index}.md").write_text("x" * 8)
        (root / "GUIDE.md").write_text("π", encoding="utf-8")

        with pytest.raises(_UnsafeRubricInputError, match=f"{remaining}-byte limit: GUIDE.md"):
            _collect_supplementary_content(root, max_bytes_per_file=10, max_total_bytes=total)

    def test_required_guidance_fits_when_the_budget_covers_every_document(self, tmp_path):
        # The companion of the case above: the same two documents are collected whole as soon as
        # the budget holds both, so the earlier read is spending the budget, not failing on it.
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Earlier](earlier0.md)\n[Guide](GUIDE.md)")
        (root / "earlier0.md").write_text("x" * 8)
        (root / "GUIDE.md").write_text("π", encoding="utf-8")

        collected = _collect_supplementary_content(root, max_bytes_per_file=10, max_total_bytes=10)

        assert collected["linked_markdown"] == {"earlier0.md": "x" * 8, "GUIDE.md": "π"}

    def test_required_guidance_is_not_bounded_by_the_preview_size(self, tmp_path):
        # The per-file bound summarizes optional previews; required guidance answers to the
        # supplement budget alone, so a guide larger than a preview is still collected whole.
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](GUIDE.md)")
        (root / "GUIDE.md").write_text("g" * 9000)

        collected = _collect_supplementary_content(root, max_bytes_per_file=100, max_total_bytes=10000)

        assert collected["linked_markdown"] == {"GUIDE.md": "g" * 9000}

    @pytest.mark.parametrize(
        "case",
        [
            "missing",
            "escape",
            "encoded_escape",
            "invalid_utf8",
            "oversized",
            "total_budget",
            "symlink",
            "ancestor_symlink",
            "hardlink",
            "directory",
            "fifo",
        ],
    )
    def test_incomplete_required_guidance_stops_before_judge(self, tmp_path, case, monkeypatch):
        root = _write_skill(tmp_path)
        target = root / "GUIDE.md"
        link = "GUIDE.md"
        if case == "escape":
            link = "../outside.md"
            (tmp_path / "outside.md").write_text("outside")
        elif case == "encoded_escape":
            link = "%2e%2e%5coutside.md"
            (tmp_path / "outside.md").write_text("outside")
        elif case == "invalid_utf8":
            target.write_bytes(b"\xff")
        elif case == "oversized":
            # A single document larger than the configured budget cannot be collected whole.
            monkeypatch.setenv("SKILLEVALUATOR_RUBRIC_SUPPLEMENT_BYTES", "3000")
            target.write_text("x" * 3001)
        elif case == "total_budget":
            monkeypatch.setenv("SKILLEVALUATOR_RUBRIC_SUPPLEMENT_BYTES", "30000")
            for index in range(11):
                (root / f"guide{index}.md").write_text("x" * 3000)
            (root / "SKILL.md").write_text("\n".join(f"[Guide](guide{i}.md)" for i in range(11)))
        elif case == "symlink":
            (root / "actual.md").write_text("guidance")
            target.symlink_to(root / "actual.md")
        elif case == "ancestor_symlink":
            (tmp_path / "outside").mkdir()
            (tmp_path / "outside/guide.md").write_text("outside")
            (root / "docs").symlink_to(tmp_path / "outside", target_is_directory=True)
            link = "docs/guide.md"
        elif case == "hardlink":
            (tmp_path / "outside.md").write_text("outside")
            os.link(tmp_path / "outside.md", target)
        elif case == "directory":
            target.mkdir()
        elif case == "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFO creation unavailable")
            os.mkfifo(target)
        if case != "total_budget":
            (root / "SKILL.md").write_text(f"[Guide]({link})")

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(root)

        judge.assert_not_called()
        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None
        assert any(f.check_name == "unsafe_skill_input" for f in result.findings)

    def test_empty_document_graph_is_bounded_before_judging(self, tmp_path, monkeypatch):
        from skillevaluator.validators import rubric_eval

        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[A](A.md) [B](B.md) [C](C.md)")
        for name in ("A.md", "B.md", "C.md"):
            (root / name).write_text("")
        monkeypatch.setattr(rubric_eval, "_MAX_LINKED_MARKDOWN_FILES", 2, raising=False)

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(root)

        judge.assert_not_called()
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None

    def test_manifest_case_alias_cycle_does_not_apply_supplementary_size_limit(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](guide.md)\n" + "x" * 3100)
        alias = root / "skill.md"
        if not alias.exists() or not alias.samefile(root / "SKILL.md"):
            pytest.skip("requires a case-insensitive filesystem")
        (root / "guide.md").write_text("[Root](skill.md)")

        collected = _collect_supplementary_content(root)

        assert collected["linked_markdown"] == {"guide.md": "[Root](skill.md)"}

    def test_guide_case_aliases_do_not_consume_budget_twice(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[First](GUIDE.md) [Again](guide.md)")
        (root / "GUIDE.md").write_text("required")
        alias = root / "guide.md"
        if not alias.exists() or not alias.samefile(root / "GUIDE.md"):
            pytest.skip("requires a case-insensitive filesystem")

        collected = _collect_supplementary_content(root, max_total_bytes=8)

        assert collected["linked_markdown"] == {"GUIDE.md": "required"}

    def test_case_sensitive_guides_keep_distinct_evidence(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Upper](GUIDE.md) [Lower](guide.md)")
        (root / "GUIDE.md").write_text("UPPER")
        (root / "guide.md").write_text("lower")
        if (root / "GUIDE.md").samefile(root / "guide.md"):
            pytest.skip("requires a case-sensitive filesystem")

        assert _collect_supplementary_content(root)["linked_markdown"] == {"GUIDE.md": "UPPER", "guide.md": "lower"}

    @pytest.mark.parametrize("unsafe_alias", ["ancestor_symlink", "hardlink"])
    def test_known_manifest_identity_does_not_bypass_unsafe_path_rejection(self, tmp_path, unsafe_alias):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](guide.md)")
        if unsafe_alias == "ancestor_symlink":
            (root / "guide.md").write_text("[Alias](alias/SKILL.md)")
            (root / "alias").symlink_to(root, target_is_directory=True)
        else:
            (root / "guide.md").write_text("[Alias](COPY.md)")
            os.link(root / "SKILL.md", root / "COPY.md")
        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(root)

        judge.assert_not_called()
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None

    @pytest.mark.parametrize("line_ending", [pytest.param("\n", id="lf"), pytest.param("\r\n", id="crlf")])
    def test_case_alias_with_changed_manifest_bytes_is_rejected(self, tmp_path, line_ending):
        root = _write_skill(tmp_path)
        source = "[Guide](guide.md)\nOriginal source".replace("\n", line_ending)
        manifest = root / "SKILL.md"
        manifest.write_bytes(source.replace("Original", "Modified").encode("utf-8"))
        if not (root / "skill.md").exists() or not (root / "skill.md").samefile(manifest):
            pytest.skip("requires a case-insensitive filesystem")
        (root / "guide.md").write_text("[Root](skill.md)")

        with pytest.raises(_UnsafeRubricInputError, match="changed while collecting"):
            _collect_supplementary_content(root, manifest=manifest, skill_content=source)

    @pytest.mark.parametrize("href", ["%2Fimages/logo.png", "%2Fassets/data.json", "%2e%2e%5cassets/data.json"])
    def test_non_markdown_assets_do_not_block_rubric_judging(self, tmp_path, href):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text(f"[Asset]({href})")
        response = json.dumps({"checks": _complete_checks(), "score": 90, "overall_pass": True, "summary": "ok"})
        with patch.object(RubricJudge, "completions", return_value=response) as provider:
            result = RubricEvalValidator().validate(root)

        assert result.metadata["rubric_eval"]["execution_status"] == "succeeded"
        assert result.metadata["rubric_eval"]["overall_score"] == 90
        assert provider.call_count == 1

    def test_encoded_absolute_markdown_remains_blocked_before_judging(self, tmp_path):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](%2Fdocs/GUIDE.md)")
        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(root)

        judge.assert_not_called()
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None

    @pytest.mark.parametrize("line_ending", [pytest.param("\n", id="lf"), pytest.param("\r\n", id="crlf")])
    def test_successful_validator_sends_source_and_complete_guidance_to_provider(self, tmp_path, line_ending):
        root = _write_skill(tmp_path)
        source = "---\nname: demo\ndescription: demo\n---\n[Guide](GUIDE.md)\n".replace("\n", line_ending)
        (root / "SKILL.md").write_bytes(source.encode("utf-8"))
        (root / "GUIDE.md").write_text("Required guidance")
        response = json.dumps({"checks": _complete_checks(), "score": 90, "overall_pass": True, "summary": "ok"})
        with patch.object(RubricJudge, "completions", return_value=response) as provider:
            result = RubricEvalValidator().validate(root)

        assert result.metadata["rubric_eval"]["execution_status"] == "succeeded"
        assert result.metadata["rubric_eval"]["overall_score"] == 90
        prompt = provider.call_args.args[1]
        assert f"SKILL.md Content:\n{source}\n\n" in prompt
        assert "Linked Markdown (paths relative to skill root):" in prompt
        assert "--- GUIDE.md ---\nRequired guidance" in prompt
        assert (root / "SKILL.md").read_bytes() == source.encode("utf-8")


class TestSupplementaryBudget:
    def test_default_budget_holds_a_real_linked_guide_graph(self, tmp_path):
        # The reporter's graph in #236 is about 65 KB over 13 documents; the stock budget takes it.
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("\n".join(f"[Guide](guide{index}.md)" for index in range(13)))
        for index in range(13):
            (root / f"guide{index}.md").write_text("g" * 5000)

        collected = _collect_supplementary_content(root)

        assert len(collected["linked_markdown"]) == 13
        assert all(len(text) == 5000 for text in collected["linked_markdown"].values())

    def test_budget_default_and_overrides(self, monkeypatch):
        monkeypatch.delenv(_SUPPLEMENTARY_TOTAL_BYTES_ENV, raising=False)
        assert supplementary_total_bytes() == _DEFAULT_SUPPLEMENTARY_TOTAL_BYTES
        assert supplementary_total_bytes({_SUPPLEMENTARY_TOTAL_BYTES_ENV: " 65536 "}) == 65536

    @pytest.mark.parametrize("raw", ["", "   ", "0", "-1", "1.5", "lots", "0x10"])
    def test_unusable_budget_falls_back_to_the_default(self, raw):
        assert supplementary_total_bytes({_SUPPLEMENTARY_TOTAL_BYTES_ENV: raw}) == _DEFAULT_SUPPLEMENTARY_TOTAL_BYTES

    def test_budget_is_capped(self):
        oversized = str(_MAX_CONFIGURED_SUPPLEMENTARY_TOTAL_BYTES * 4)
        budget = supplementary_total_bytes({_SUPPLEMENTARY_TOTAL_BYTES_ENV: oversized})

        assert budget == _MAX_CONFIGURED_SUPPLEMENTARY_TOTAL_BYTES

    def test_configured_budget_reaches_the_collector(self, tmp_path, monkeypatch):
        root = _write_skill(tmp_path)
        (root / "SKILL.md").write_text("[Guide](GUIDE.md)")
        (root / "GUIDE.md").write_text("g" * 4096)
        monkeypatch.setenv(_SUPPLEMENTARY_TOTAL_BYTES_ENV, "4095")

        with pytest.raises(_UnsafeRubricInputError):
            _collect_supplementary_content(root)

        monkeypatch.setenv(_SUPPLEMENTARY_TOTAL_BYTES_ENV, "4096")
        assert _collect_supplementary_content(root)["linked_markdown"] == {"GUIDE.md": "g" * 4096}

    @pytest.mark.parametrize("case", ["escape", "encoded_escape", "symlink", "ancestor_symlink", "hardlink"])
    def test_a_larger_budget_never_widens_the_declared_root(self, tmp_path, monkeypatch, case):
        # Raising the byte budget buys more of the skill's own guidance, never another path.
        monkeypatch.setenv(_SUPPLEMENTARY_TOTAL_BYTES_ENV, str(_MAX_CONFIGURED_SUPPLEMENTARY_TOTAL_BYTES))
        root = _write_skill(tmp_path)
        (tmp_path / "outside.md").write_text("outside guidance")
        link = "GUIDE.md"
        if case == "escape":
            link = "../outside.md"
        elif case == "encoded_escape":
            link = "%2e%2e%5coutside.md"
        elif case == "symlink":
            (root / "GUIDE.md").symlink_to(tmp_path / "outside.md")
        elif case == "ancestor_symlink":
            (tmp_path / "outside").mkdir()
            (tmp_path / "outside/guide.md").write_text("outside guidance")
            (root / "docs").symlink_to(tmp_path / "outside", target_is_directory=True)
            link = "docs/guide.md"
        else:
            os.link(tmp_path / "outside.md", root / "GUIDE.md")
        (root / "SKILL.md").write_text(f"[Guide]({link})")

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(root)

        judge.assert_not_called()
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None
        assert any(finding.check_name == "unsafe_skill_input" for finding in result.findings)

    def test_optional_previews_still_stay_small_under_a_large_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_SUPPLEMENTARY_TOTAL_BYTES_ENV, str(_MAX_CONFIGURED_SUPPLEMENTARY_TOTAL_BYTES))
        root = _write_skill(tmp_path)
        (root / "scripts").mkdir()
        (root / "scripts/main.py").write_text("x" * 20000)

        collected = _collect_supplementary_content(root)

        assert collected["scripts"]["main.py"].endswith("... [truncated]")
        assert len(collected["scripts"]["main.py"]) < 3100


class TestRubricEvalValidator:
    def test_no_skills_found_records_failed_execution_status(self, tmp_path):
        validator = RubricEvalValidator()
        result = validator.validate(tmp_path)

        assert not result.passed
        assert any("No skills found" in f.message for f in result.findings)
        rubric = result.metadata["rubric_eval"]
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert rubric["summary"] == "No skills found in target directory"
        assert rubric["checks"] == []

    def test_missing_skill_md_records_failed_execution_status(self, tmp_path):
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()

        result = RubricEvalValidator()._validate_single_skill(skill_dir)

        assert not result.passed
        assert any("SKILL.md not found" in f.message for f in result.findings)
        rubric = result.metadata["rubric_eval"]
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert rubric["summary"] == "SKILL.md not found"
        assert rubric["skill_name"] == "bad-skill"
        assert rubric["checks"] == []

    def test_symlinked_manifest_is_rejected_before_llm_request(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("SECRET OUTSIDE CONTENT", encoding="utf-8")
        (skill_dir / "SKILL.md").symlink_to(outside)

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(skill_dir)

        judge.assert_not_called()
        assert not result.passed
        assert any(f.check_name == "unsafe_skill_input" for f in result.findings)

    @pytest.mark.parametrize("unsafe_entry", ["file_symlink", "directory_symlink", "non_regular"])
    def test_unsafe_supplementary_entry_is_rejected_before_llm_request(self, tmp_path, unsafe_entry):
        skill_dir = _write_skill(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.py").write_text("SECRET OUTSIDE CONTENT", encoding="utf-8")

        if unsafe_entry == "directory_symlink":
            (skill_dir / "scripts").symlink_to(outside, target_is_directory=True)
        else:
            scripts = skill_dir / "scripts"
            scripts.mkdir()
            if unsafe_entry == "file_symlink":
                (scripts / "leak.py").symlink_to(outside / "leak.py")
            else:
                if not hasattr(os, "mkfifo"):
                    pytest.skip("FIFO creation is unavailable")
                os.mkfifo(scripts / "blocked.py")

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(skill_dir)

        judge.assert_not_called()
        assert not result.passed
        assert any(f.check_name == "unsafe_skill_input" for f in result.findings)

    def test_oversized_manifest_is_rejected_before_llm_request(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"x" * (1024 * 1024 + 1))

        with patch.object(RubricJudge, "process") as judge:
            result = RubricEvalValidator().validate(skill_dir)

        judge.assert_not_called()
        assert not result.passed
        assert any("maximum size" in f.message.lower() for f in result.findings)

    def test_llm_fallback_when_no_key(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")

        from skillevaluator.inference.types import LLMClientError

        with patch.object(RubricJudge, "completions", side_effect=LLMClientError("no provider configured")):
            validator = RubricEvalValidator()
            result = validator.validate(skill_dir)
        assert not result.passed
        assert any("unavailable" in f.message.lower() for f in result.findings)
        assert any(f.check_name == "llm_unavailable" and f.severity.value == "high" for f in result.findings)
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert "unavailable" in result.metadata["rubric_eval"]["judge_summary"].lower()

    def test_successful_evaluation(self, tmp_path):
        skill_dir = _write_skill(tmp_path)

        mock_report = {
            "overall_pass": True,
            "score": 85,
            "summary": "Good skill",
            "checks": [
                {
                    "id": criterion["id"],
                    "criterion": criterion["criterion"],
                    "pass": True,
                    "score": 9,
                    "notes": "Clear and specific",
                }
                for criterion in RUBRIC_CRITERIA
            ],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            validator = RubricEvalValidator()
            result = validator.validate(skill_dir)

        assert result.metadata.get("rubric_eval")
        assert result.metadata["rubric_eval"]["overall_score"] == 90.0
        assert result.metadata["rubric_eval"]["overall_pass"] is True
        assert result.metadata["rubric_eval"]["judge_score"] == 85
        assert result.metadata["rubric_eval"]["judge_overall_pass"] is True
        assert len(result.findings) == len(RUBRIC_CRITERIA)

    @pytest.mark.parametrize(
        ("judge_score", "judge_pass", "criterion_score", "expected_score", "expected_pass"),
        [
            (0, False, 8, 80.0, True),
            (100, True, 6, 60.0, False),
        ],
    )
    def test_model_totals_cannot_override_criterion_evidence(
        self,
        tmp_path,
        judge_score,
        judge_pass,
        criterion_score,
        expected_score,
        expected_pass,
    ):
        skill_dir = _write_skill(tmp_path)
        report = {
            "score": judge_score,
            "overall_pass": judge_pass,
            "summary": "Contradictory model aggregate",
            "checks": _complete_checks(scores={criterion["id"]: criterion_score for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert rubric["overall_score"] == expected_score
        assert rubric["overall_pass"] is expected_pass
        assert rubric["judge_score"] == judge_score
        assert rubric["judge_overall_pass"] is judge_pass
        assert result.passed is expected_pass

    def test_criterion_pass_is_derived_from_seven_point_threshold(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        zero_score_id = RUBRIC_CRITERIA[0]["id"]
        report = {
            "score": 100,
            "overall_pass": True,
            "summary": "Contradictory criterion evidence",
            "checks": _complete_checks(
                scores={zero_score_id: 0},
                passes={zero_score_id: True},
            ),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        zero_score_check = next(check for check in rubric["checks"] if check["id"] == zero_score_id)
        assert zero_score_check["pass"] is False
        assert zero_score_check["judge_pass"] is True
        assert rubric["aggregation"]["criterion_pass_score"] == 7
        assert rubric["overall_pass"] is False
        assert not result.passed

    def test_failed_local_verdict_adds_blocking_aggregate_finding(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        report = {
            "score": 100,
            "overall_pass": True,
            "summary": "Contradictory aggregate",
            "checks": _complete_checks(scores={criterion["id"]: 6 for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert not result.passed
        assert result.summary.errors == 1
        assert any(
            finding.check_name == "rubric_overall" and finding.severity.value == "high" for finding in result.findings
        )
        html = HTMLReporter(include_timestamp=False).render_all([result])
        assert '"would_block": true' in html

    def test_empty_checks_use_local_incomplete_diagnostic(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        judge_summary = "All rubric criteria passed"
        report = {"score": 100, "overall_pass": True, "summary": judge_summary, "checks": []}

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["judge_summary"] == judge_summary
        assert judge_summary not in rubric["summary"]
        assert "incomplete" in rubric["summary"].lower()
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    def test_local_score_overrides_false_model_criterion_pass(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        failed_id = RUBRIC_CRITERIA[0]["id"]
        report = {
            "score": 100,
            "overall_pass": True,
            "summary": "Contradictory pass",
            "checks": _complete_checks(
                scores={criterion["id"]: 10 for criterion in RUBRIC_CRITERIA},
                passes={failed_id: False},
            ),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert result.passed
        assert result.metadata["rubric_eval"]["overall_score"] == 100.0
        assert result.metadata["rubric_eval"]["overall_pass"] is True
        criterion_finding = next(finding for finding in result.findings if finding.metadata.get("id") == failed_id)
        assert criterion_finding.metadata["pass"] is True
        assert criterion_finding.metadata["judge_pass"] is False

    def test_overall_fields_are_optional_diagnostics(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        report = {
            "summary": "Complete criterion evidence",
            "checks": _complete_checks(scores={criterion["id"]: 7 for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert result.passed
        assert result.metadata["rubric_eval"]["overall_score"] == 70.0
        assert result.metadata["rubric_eval"]["overall_pass"] is True
        assert result.metadata["rubric_eval"]["judge_score"] is None
        assert result.metadata["rubric_eval"]["judge_overall_pass"] is None

    def test_importance_weights_drive_local_score(self, tmp_path):
        skill_dir = _write_skill(tmp_path)
        scores_by_importance = {"high": 10, "medium": 5, "low": 0}
        scores = {criterion["id"]: scores_by_importance[criterion["importance"]] for criterion in RUBRIC_CRITERIA}
        report = {
            "score": 1,
            "overall_pass": False,
            "summary": "Model aggregate is ignored",
            "checks": _complete_checks(scores=scores),
        }

        with patch.object(RubricJudge, "process", return_value=report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        # Current schema weights high/medium/low as 3/2/1: 180 / 22 * 10.
        assert result.metadata["rubric_eval"]["overall_score"] == 81.8
        assert result.metadata["rubric_eval"]["overall_pass"] is False
        assert not result.passed

    @pytest.mark.parametrize("case", ["missing", "duplicate", "unexpected"])
    def test_missing_duplicate_or_unexpected_criteria_fail_closed(self, tmp_path, case):
        skill_dir = _write_skill(tmp_path)
        checks = _complete_checks()
        if case == "missing":
            checks.pop()
        elif case == "duplicate":
            checks[-1]["id"] = checks[0]["id"]
        else:
            checks.append(
                {
                    "id": "unexpected_criterion",
                    "criterion": "Unexpected",
                    "pass": True,
                    "score": 10,
                    "notes": "Not in the schema",
                }
            )

        with patch.object(
            RubricJudge,
            "process",
            return_value={"score": 100, "overall_pass": True, "summary": "invalid", "checks": checks},
        ):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert case in rubric["summary"].lower()
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    @pytest.mark.parametrize(
        "case",
        [
            "missing_score",
            "boolean_score",
            "score_below_range",
            "score_above_range",
            "oversized_integer_score",
            "nonfinite_score",
            "missing_pass",
            "nonboolean_pass",
            "non_object_check",
        ],
    )
    def test_malformed_criterion_evidence_fails_closed(self, tmp_path, case):
        skill_dir = _write_skill(tmp_path)
        checks = _complete_checks()
        criterion_id = checks[0]["id"]
        if case == "missing_score":
            checks[0].pop("score")
        elif case == "boolean_score":
            checks[0]["score"] = True
        elif case == "score_below_range":
            checks[0]["score"] = -0.1
        elif case == "score_above_range":
            checks[0]["score"] = 10.1
        elif case == "oversized_integer_score":
            checks[0]["score"] = 10**1000
        elif case == "nonfinite_score":
            checks[0]["score"] = float("nan")
        elif case == "missing_pass":
            checks[0].pop("pass")
        elif case == "nonboolean_pass":
            checks[0]["pass"] = "true"
        else:
            checks[0] = "not an object"

        with patch.object(
            RubricJudge,
            "process",
            return_value={"score": 100, "overall_pass": True, "summary": "invalid", "checks": checks},
        ):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert rubric["checks"] == []
        if case != "non_object_check":
            assert criterion_id in rubric["summary"]
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    def test_unavailable_judging_invalidates_folder_aggregate(self, tmp_path):
        _write_skill(tmp_path, "skill-a")
        _write_skill(tmp_path, "skill-b")
        complete_report = {
            "score": 100,
            "overall_pass": True,
            "summary": "complete",
            "checks": _complete_checks(scores={criterion["id"]: 10 for criterion in RUBRIC_CRITERIA}),
        }
        unavailable_report = RubricJudge().get_fallback_response()

        with patch.object(RubricJudge, "process", side_effect=[complete_report, unavailable_report]):
            result = RubricEvalValidator(min_score=70).validate(tmp_path)

        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["overall_pass"] is False
        assert any(finding.check_name == "llm_unavailable" for finding in result.findings)

    def test_incomplete_rubric_response_fails_closed(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")
        mock_report = {
            "overall_pass": True,
            "score": 100,
            "summary": "Looks complete",
            "checks": [
                {
                    "id": RUBRIC_CRITERIA[0]["id"],
                    "criterion": RUBRIC_CRITERIA[0]["criterion"],
                    "pass": True,
                    "score": 10,
                    "notes": "Only one criterion was returned",
                }
            ],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert any(finding.check_name == "llm_invalid_response" for finding in result.findings)

    def test_min_score_threshold(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")

        mock_report = {
            "overall_pass": False,
            "score": 45,
            "summary": "Needs work",
            "checks": _complete_checks(scores={criterion["id"]: 4 for criterion in RUBRIC_CRITERIA}),
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            validator = RubricEvalValidator(min_score=60)
            result = validator.validate(skill_dir)

        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "succeeded"
        assert result.metadata["rubric_eval"]["overall_score"] == 40.0

    def test_local_scores_override_explicit_judge_failure(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\n# Test")
        mock_report = {
            "overall_pass": False,
            "score": 85,
            "summary": "A required criterion failed",
            "checks": [
                {
                    "id": criterion["id"],
                    "criterion": criterion["criterion"],
                    "pass": index != 0,
                    "score": 8,
                    "notes": "failed" if index == 0 else "passed",
                }
                for index, criterion in enumerate(RUBRIC_CRITERIA)
            ],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            result = RubricEvalValidator(min_score=70).validate(skill_dir)

        assert result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "succeeded"
        first_check = result.metadata["rubric_eval"]["checks"][0]
        assert first_check["pass"] is True
        assert first_check["judge_pass"] is False

    def test_folder_validation_fails_when_judging_is_unavailable(self, tmp_path):
        for name in ["skill-a", "skill-b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: test\n---\n# {name}")

        mock_report = {
            "overall_pass": True,
            "score": 80,
            "summary": "ok",
            "checks": [],
        }

        with patch.object(RubricJudge, "process", return_value=mock_report):
            validator = RubricEvalValidator()
            result = validator.validate(tmp_path)

        assert not result.passed
        assert result.metadata["rubric_eval"]["execution_status"] == "failed"
        assert result.metadata["rubric_eval"]["overall_score"] is None
        assert result.metadata["rubric_eval"]["overall_pass"] is False

    @pytest.mark.parametrize(
        ("answered_by", "with_fallback_block"),
        [("claude-opus-5-5", False), ("claude-opus-4-8", True), ("claude-opus-4-8", False)],
        ids=["configured-model", "fallback-in-this-call", "sticky-fallback-turn"],
    )
    def test_json_report_names_the_model_that_scored(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, answered_by: str, with_fallback_block: bool
    ):
        """A score from the server-side fallback model is labeled with that model, not the configured one.

        A turn served by sticky routing carries no fallback block; only its top-level model names the fallback.
        """
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from skillevaluator.reporting import JSONReporter

        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "claude-opus-5-5")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        for name in ("SKILL_EVAL_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        content = [{"type": "text", "text": json.dumps({"checks": _complete_checks(), "summary": "ok"})}]
        usage = {"input_tokens": 1, "output_tokens": 1}
        if with_fallback_block:
            content.insert(0, {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": answered_by}})
            usage["iterations"] = [
                {"type": "message", "model": "claude-opus-5-5"},
                {"type": "fallback_message", "model": answered_by},
            ]
        elif answered_by != "claude-opus-5-5":
            usage["iterations"] = [{"type": "fallback_message", "model": answered_by}]

        def handler(request):
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": answered_by,
                    "content": content,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": usage,
                },
            )

        validator = RubricEvalValidator()
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            validator._judge = RubricJudge(http_client=http_client)
            result = validator.validate(_write_skill(tmp_path))

        data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
        assert data["rubric_eval"]["execution_status"] == "succeeded"
        assert data["rubric_eval"]["llm_provider"] == "anthropic"
        assert data["rubric_eval"]["llm_model_used"] == answered_by
        (success,) = data["results"][0]["success_details"]
        assert success["metadata"]["llm_model_used"] == answered_by


_RETRY_MARKER = "YOUR PREVIOUS REPLY WAS REJECTED: "
_RETRY_TAIL = "Reply again with ONLY one JSON object in the shape described above.\n"
_FIRST_ID = RUBRIC_CRITERIA[0]["id"]
_LAST_ID = RUBRIC_CRITERIA[-1]["id"]


def _reply(checks: list[dict] | None = None) -> str:
    """Serialize a judge reply whose own totals disagree with its criterion scores."""
    return json.dumps(
        {
            "overall_pass": False,
            "score": 0,
            "summary": "judge summary",
            "checks": _complete_checks() if checks is None else checks,
        }
    )


def _invalid_checks(case: str) -> list[dict]:
    checks = _complete_checks()
    if case == "missing":
        checks.pop()
    elif case == "duplicate":
        checks[-1]["id"] = _FIRST_ID
    elif case == "unexpected":
        checks.append({**checks[0], "id": "made_up_criterion"})
    elif case == "score_above_range":
        checks[0]["score"] = 11
    elif case == "nonboolean_pass":
        checks[0]["pass"] = "yes"
    else:
        assert case == "empty"
        checks = []
    return checks


class _ScriptedCompletions:
    """Stand in for ``RubricJudge.completions``: return or raise each scripted reply in turn."""

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.requests.append({"system": system_prompt, "user": user_prompt, **kwargs})
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _validate_with(tmp_path: Path, *replies: str | Exception):
    scripted = _ScriptedCompletions(*replies)
    with patch.object(RubricJudge, "completions", new=scripted):
        result = RubricEvalValidator(min_score=70).validate(_write_skill(tmp_path))
    return result, scripted.requests


def _llm_findings(result) -> list:
    return [finding for finding in result.findings if finding.check_name.startswith("llm_")]


def _use_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for name in ("SKILL_EVAL_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def _anthropic_message(index: int, model: str, text: str) -> dict:
    return {
        "id": f"msg_test_{index}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class TestInvalidReplyRetry:
    """One judge reply that cannot be scored gets one more request that names the problem."""

    @pytest.mark.parametrize("padding", ["", "x" * 410 + " "])
    def test_retry_problem_redacts_credentials_before_logging_and_truncation(
        self, tmp_path, padding, caplog, monkeypatch
    ):
        secret = "sk-syntheticcredential123456789"
        checks = _complete_checks()
        checks.append({**checks[0], "id": padding + secret})

        monkeypatch.setattr(rubric_eval.logger, "propagate", True)
        result, requests = _validate_with(tmp_path, _reply(checks), _reply())

        assert "Rubric judge reply rejected" in caplog.text
        assert "sk-synthetic" not in caplog.text
        assert "sk-synthetic" not in requests[1]["user"].removeprefix(requests[0]["user"])
        assert result.passed

    @pytest.mark.parametrize("padding", ["", "x" * 410 + " "])
    def test_final_invalid_reply_diagnostics_redact_before_bounding(self, tmp_path, padding):
        checks = _complete_checks()
        checks.append({**checks[0], "id": padding + "sk-syntheticcredential123456789 " + "z" * 2000})

        result, requests = _validate_with(tmp_path, _reply(checks), _reply(checks))

        assert len(requests) == 2
        (finding,) = _llm_findings(result)
        summary = result.metadata["rubric_eval"]["summary"]
        assert finding.message == summary
        assert "sk-synthetic" not in summary
        assert len(summary) <= 500 + len(" ... [truncated]")
        assert summary.endswith(" ... [truncated]")
        assert not result.passed

    @pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x00", "\x1b[31m", "\u2028", "\u202e"])
    def test_rejection_diagnostics_are_printable_single_lines(self, tmp_path, control, caplog, monkeypatch):
        checks = _complete_checks()
        checks.append({**checks[0], "id": f"invented{control}FORGED"})
        monkeypatch.setattr(rubric_eval.logger, "propagate", True)

        result, requests = _validate_with(tmp_path, _reply(checks), _reply(checks))

        problem = requests[1]["user"].removeprefix(requests[0]["user"]).removeprefix(
            f"\n{_RETRY_MARKER}"
        ).removesuffix(f". {_RETRY_TAIL}")
        assert "invented" in problem and "FORGED" in problem
        assert problem.isprintable()
        assert "  " not in problem
        rejected = [record for record in caplog.records if "reply rejected" in record.getMessage()]
        assert rejected
        assert all(record.getMessage().isprintable() for record in rejected)
        (finding,) = _llm_findings(result)
        assert finding.message.isprintable()
        assert result.metadata["rubric_eval"]["summary"] == finding.message

    @pytest.mark.parametrize(
        ("first_reply", "problem"),
        [
            (_reply(_invalid_checks("missing")), f"missing criterion ids: {_LAST_ID}"),
            (_reply(_invalid_checks("duplicate")), f"duplicate criterion ids: {_FIRST_ID}"),
            (_reply(_invalid_checks("unexpected")), "unexpected criterion ids: made_up_criterion"),
            (
                _reply(_invalid_checks("score_above_range")),
                f"criterion {_FIRST_ID} score must be a finite number between 0 and 10",
            ),
            (_reply(_invalid_checks("nonboolean_pass")), f"criterion {_FIRST_ID} pass must be a boolean"),
            (_reply(_invalid_checks("empty")), f"expected {len(RUBRIC_CRITERIA)} criteria, received 0"),
            ("Here is my assessment of the skill.", "Could not extract valid JSON from LLM response (35 chars)"),
            ('{"summary": "cut", "checks": [', "Could not extract valid JSON from LLM response"),
        ],
        ids=["missing", "duplicate", "unexpected", "score", "pass", "empty", "prose", "unclosed-json"],
    )
    def test_invalid_reply_is_retried_once_with_the_problem_then_scored(self, tmp_path, first_reply, problem):
        result, requests = _validate_with(tmp_path, first_reply, _reply())

        assert len(requests) == 2
        first, retry = requests
        assert _RETRY_MARKER not in first["user"]
        assert retry["user"].startswith(first["user"])
        retry_note = retry["user"][len(first["user"]) :]
        assert retry_note.startswith(f"\n{_RETRY_MARKER}")
        assert problem in retry_note
        assert retry_note.endswith(f". {_RETRY_TAIL}")
        for request in requests:
            assert request["system"] == RubricJudge().get_system_prompt()
            assert request["response_schema"] == RUBRIC_JSON_SCHEMA
            assert request["schema_name"] == "rubric_evaluation"

        rubric = result.metadata["rubric_eval"]
        assert result.passed
        assert _llm_findings(result) == []
        assert rubric["execution_status"] == "succeeded"
        assert rubric["overall_score"] == 90.0
        assert rubric["overall_pass"] is True
        assert rubric["judge_score"] == 0
        assert rubric["judge_attempts"] == 2
        (success,) = result.success_details
        assert success.metadata["judge_attempts"] == 2

    def test_reply_with_an_integer_past_the_digit_limit_is_retried(self, tmp_path):
        limit = sys.get_int_max_str_digits()
        if not limit:
            pytest.skip("integer string conversion is unlimited in this interpreter")
        first_reply = f'{{"checks": [{{"id": "{_FIRST_ID}", "score": 1{"0" * limit}}}]}}'
        with pytest.raises(ValueError) as parse_error:
            json.loads(first_reply)
        assert not isinstance(parse_error.value, json.JSONDecodeError)

        result, requests = _validate_with(tmp_path, first_reply, _reply())

        assert len(requests) == 2
        first, retry = requests
        assert retry["user"] == f"{first['user']}\n{_RETRY_MARKER}{parse_error.value}. {_RETRY_TAIL}"
        rubric = result.metadata["rubric_eval"]
        assert result.passed
        assert _llm_findings(result) == []
        assert rubric["overall_score"] == 90.0
        assert rubric["judge_attempts"] == 2

    @pytest.mark.parametrize(
        ("message", "problem"),
        [
            (
                "maximum recursion depth exceeded while decoding a JSON array from a unicode string",
                "maximum recursion depth exceeded while decoding a JSON array from a unicode string",
            ),
            ("", "RecursionError"),
        ],
        ids=["message", "no-message"],
    )
    def test_reply_nested_too_deep_to_parse_is_retried(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, message, problem
    ):
        # json.loads raises RecursionError for nesting deeper than the interpreter
        # allows. That depth differs by Python version and platform, so a stand-in
        # parser raises it for this one reply.
        nested_reply = '{"checks": [[[[]]]]}'

        def extract_json(content: str) -> dict:
            if content == nested_reply:
                raise RecursionError(message)
            return _extract_json(content)

        monkeypatch.setattr(rubric_eval, "_extract_json", extract_json)

        result, requests = _validate_with(tmp_path, nested_reply, _reply())

        assert len(requests) == 2
        first, retry = requests
        assert retry["user"] == f"{first['user']}\n{_RETRY_MARKER}{problem}. {_RETRY_TAIL}"
        rubric = result.metadata["rubric_eval"]
        assert result.passed
        assert _llm_findings(result) == []
        assert rubric["judge_attempts"] == 2

    def test_valid_first_reply_is_not_retried(self, tmp_path):
        result, requests = _validate_with(tmp_path, _reply())

        assert len(requests) == 1
        assert result.passed
        assert result.metadata["rubric_eval"]["judge_attempts"] == 1

    @pytest.mark.parametrize(
        ("first_reply", "second_case", "problem"),
        [
            (_reply(_invalid_checks("missing")), "missing", f"missing criterion ids: {_LAST_ID}"),
            ("not json", "unexpected", "unexpected criterion ids: made_up_criterion"),
        ],
        ids=["missing-twice", "prose-then-unexpected"],
    )
    def test_second_invalid_reply_fails_closed(self, tmp_path, first_reply, second_case, problem):
        result, requests = _validate_with(tmp_path, first_reply, _reply(_invalid_checks(second_case)))

        assert len(requests) == 2
        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["overall_score"] is None
        assert rubric["checks"] == []
        assert rubric["judge_attempts"] == 2
        assert problem in rubric["summary"]
        (finding,) = _llm_findings(result)
        assert finding.check_name == "llm_invalid_response"
        assert finding.severity.value == "high"
        assert finding.metadata == {"judge_attempts": 2}
        assert problem in finding.message

    @pytest.mark.parametrize("padding", ["", "x" * 410 + " "], ids=["bare", "padded"])
    def test_second_invalid_reply_redacts_credentials_in_the_finding_and_summary(self, tmp_path, padding):
        secret = "sk-syntheticcredential123456789"
        checks = _complete_checks()
        checks.append({**checks[0], "id": padding + secret})

        result, requests = _validate_with(tmp_path, _reply(checks), _reply(checks))

        assert len(requests) == 2
        rubric = result.metadata["rubric_eval"]
        (finding,) = _llm_findings(result)
        assert finding.check_name == "llm_invalid_response"
        for reported in (finding.message, rubric["summary"]):
            assert "sk-synthetic" not in reported
            assert "sk-<redacted>" in reported
        assert not result.passed

    def test_second_invalid_reply_bounds_the_finding_and_summary(self, tmp_path):
        checks = _complete_checks()
        checks.append({**checks[0], "id": "x" * 20_000})

        result, _requests = _validate_with(tmp_path, _reply(checks), _reply(checks))

        rubric = result.metadata["rubric_eval"]
        (finding,) = _llm_findings(result)
        for reported in (finding.message, rubric["summary"]):
            assert reported.startswith("LLM judge returned invalid rubric evidence: unexpected criterion ids: xxx")
            assert reported.endswith("x ... [truncated]")
            assert len(reported) == 500 + len(" ... [truncated]")

    def test_judge_chosen_ids_are_collapsed_to_one_line_in_every_report_of_the_problem(self, tmp_path):
        first = _complete_checks()
        first.append({**first[0], "id": "x\n[ERROR] forged line\x1b[31m\x00\u202e end"})
        second = _complete_checks()
        second.append({**second[0], "id": "y\r\n\n[ERROR] other\x07\t\tline"})

        with patch("skillevaluator.validators.rubric_eval.logger") as log:
            result, requests = _validate_with(tmp_path, _reply(first), _reply(second))

        prefix = "LLM judge returned invalid rubric evidence: unexpected criterion ids: "
        retry_problem = f"{prefix}x [ERROR] forged line end"
        first_request, retry = requests
        assert retry["user"] == f"{first_request['user']}\n{_RETRY_MARKER}{retry_problem}. {_RETRY_TAIL}"
        (rejected,) = [call for call in log.warning.call_args_list if "rejected" in call.args[0]]
        assert rejected.args[1:] == (retry_problem,)

        reported_problem = f"{prefix}y [ERROR] other line"
        rubric = result.metadata["rubric_eval"]
        (finding,) = _llm_findings(result)
        assert finding.message == reported_problem
        assert rubric["summary"] == reported_problem

    def test_second_unparseable_reply_reports_unavailable(self, tmp_path):
        result, requests = _validate_with(tmp_path, _reply(_invalid_checks("missing")), "still not json")

        assert len(requests) == 2
        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert rubric["judge_attempts"] == 2
        (finding,) = _llm_findings(result)
        assert finding.check_name == "llm_unavailable"
        assert finding.severity.value == "high"
        assert finding.metadata == {"judge_attempts": 2}
        assert "LLM configuration or response was invalid" in finding.message

    @pytest.mark.parametrize(
        ("error", "diagnostic"),
        [
            (LLMClientError("no provider configured"), "LLM configuration or response was invalid"),
            (LLMClientRefusedError("declined"), "safety refusal"),
            (LLMClientTruncatedError("cut off", _reply()), "output-token limit"),
        ],
        ids=["request-error", "refusal", "truncated"],
    )
    def test_provider_errors_are_not_retried(self, tmp_path, error, diagnostic):
        result, requests = _validate_with(tmp_path, error, _reply())

        assert len(requests) == 1
        rubric = result.metadata["rubric_eval"]
        assert not result.passed
        assert rubric["execution_status"] == "failed"
        assert "judge_attempts" not in rubric
        (finding,) = _llm_findings(result)
        assert finding.check_name == "llm_unavailable"
        assert finding.metadata == {}
        assert diagnostic in finding.message

    def test_no_provider_sends_no_request_and_records_no_attempts(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        from skillevaluator.reporting import JSONReporter

        for name in (
            "SKILL_EVAL_LLM_PROVIDER",
            "SKILL_EVAL_LLM_MODEL",
            "SKILL_EVAL_LLM_API_KEY",
            "SKILL_EVAL_LLM_BASE_URL",
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
        ):
            monkeypatch.delenv(name, raising=False)

        with patch.object(RubricJudge, "_get_client") as get_client:
            result = RubricEvalValidator().validate(_write_skill(tmp_path))

        get_client.assert_not_called()
        data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
        assert data["rubric_eval"]["execution_status"] == "failed"
        assert "judge_attempts" not in data["rubric_eval"]
        (finding,) = [item for item in data["results"][0]["findings"] if item["check_name"].startswith("llm_")]
        assert finding["check_name"] == "llm_unavailable"
        assert "did not run" in finding["message"]
        assert "judge_attempts" not in (finding.get("metadata") or {})

    def test_provider_error_on_the_retry_reports_unavailable(self, tmp_path):
        result, requests = _validate_with(
            tmp_path, _reply(_invalid_checks("missing")), LLMClientRefusedError("declined"), _reply()
        )

        assert len(requests) == 2
        assert result.metadata["rubric_eval"]["judge_attempts"] == 2
        (finding,) = _llm_findings(result)
        assert finding.check_name == "llm_unavailable"
        assert finding.metadata == {"judge_attempts": 2}
        assert "safety refusal" in finding.message

    def test_retry_note_keeps_a_bounded_part_of_a_long_problem(self, tmp_path):
        long_id = "x" * 20_000
        checks = _complete_checks()
        checks.append({**checks[0], "id": long_id})
        with patch("skillevaluator.validators.rubric_eval.logger") as log:
            result, requests = _validate_with(tmp_path, _reply(checks), _reply())

        assert len(requests) == 2
        first, retry = requests
        retry_note = retry["user"][len(first["user"]) :]
        problem = retry_note.removeprefix(f"\n{_RETRY_MARKER}").removesuffix(f". {_RETRY_TAIL}")
        assert problem.startswith("LLM judge returned invalid rubric evidence: unexpected criterion ids: xxx")
        assert problem.endswith("x ... [truncated]")
        # The log line and the retry note keep the first 500 characters.
        assert len(problem) == 500 + len(" ... [truncated]")
        (rejected,) = [call for call in log.warning.call_args_list if "rejected" in call.args[0]]
        assert rejected.args[1:] == (problem,)
        assert result.metadata["rubric_eval"]["judge_attempts"] == 2
        assert result.passed

    @pytest.mark.parametrize(
        ("replies", "expected_requests"),
        [(("valid",), 1), (("missing", "valid"), 2)],
        ids=["valid", "missing-then-valid"],
    )
    def test_custom_criteria_are_checked_against_the_given_list(self, replies, expected_requests):
        criteria = [
            {"id": "custom_a", "criterion": "Custom A", "importance": "high"},
            {"id": "custom_b", "criterion": "Custom B", "importance": "low"},
        ]
        checks = [
            {"id": criterion["id"], "criterion": criterion["criterion"], "pass": True, "score": 9, "notes": "ok"}
            for criterion in criteria
        ]
        scripted_replies = {"valid": _reply(checks), "missing": _reply(checks[:1])}
        scripted = _ScriptedCompletions(*(scripted_replies[reply] for reply in replies))

        with patch.object(RubricJudge, "completions", new=scripted):
            report = RubricJudge().process(skill_name="demo", skill_content="# Demo", criteria=criteria)

        assert len(scripted.requests) == expected_requests
        assert report == json.loads(scripted_replies["valid"])
        if expected_requests == 2:
            retry_note = scripted.requests[1]["user"][len(scripted.requests[0]["user"]) :]
            assert "missing criterion ids: custom_b." in retry_note
            assert not any(criterion["id"] in retry_note for criterion in RUBRIC_CRITERIA)

    def test_judge_returns_the_second_reply_as_parsed(self):
        second = {"summary": "still incomplete", "checks": _invalid_checks("missing")}
        scripted = _ScriptedCompletions(_reply(_invalid_checks("missing")), json.dumps(second))
        judge = RubricJudge()
        judge.last_failure = "stale failure"

        with patch.object(RubricJudge, "completions", new=scripted):
            report = judge.process(skill_name="demo", skill_content="# Demo")

        assert report == second
        assert judge.last_attempts == 2
        assert judge.last_failure is None

    def test_each_skill_in_a_folder_records_its_own_attempts(self, tmp_path):
        for name in ("skill-a", "skill-b", "skill-c"):
            _write_skill(tmp_path, name)
        scripted = _ScriptedCompletions(
            _reply(_invalid_checks("missing")),
            _reply(),
            _reply(),
            _reply(_invalid_checks("missing")),
            _reply(_invalid_checks("missing")),
        )

        with patch.object(RubricJudge, "completions", new=scripted):
            result = RubricEvalValidator(min_score=70).validate(tmp_path)

        assert not result.passed
        assert len(scripted.requests) == 5
        successes = {detail.check_name: detail.metadata["judge_attempts"] for detail in result.success_details}
        assert successes == {"[skill-a] rubric_eval": 2, "[skill-b] rubric_eval": 1}
        (finding,) = _llm_findings(result)
        assert finding.file_path.startswith("[skill-c] ")
        assert finding.metadata == {"judge_attempts": 2}

    def test_report_names_the_model_that_answered_the_retry(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from skillevaluator.reporting import JSONReporter

        _use_anthropic(monkeypatch)
        replies = [("claude-opus-5-5", _reply(_invalid_checks("missing"))), ("claude-opus-4-8", _reply())]
        bodies: list[dict] = []

        def handler(request):
            bodies.append(json.loads(request.content))
            model, text = replies[len(bodies) - 1]
            return httpx.Response(200, json=_anthropic_message(len(bodies), model, text))

        validator = RubricEvalValidator()
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            validator._judge = RubricJudge(http_client=http_client)
            result = validator.validate(_write_skill(tmp_path))

        assert len(bodies) == 2
        assert _RETRY_MARKER not in json.dumps(bodies[0]["messages"])
        problem = f"LLM judge returned invalid rubric evidence: missing criterion ids: {_LAST_ID}."
        assert f"{_RETRY_MARKER}{problem}" in bodies[1]["messages"][-1]["content"]
        assert bodies[1]["output_config"]["format"] == bodies[0]["output_config"]["format"]
        data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
        assert data["rubric_eval"]["execution_status"] == "succeeded"
        assert data["rubric_eval"]["judge_attempts"] == 2
        assert data["rubric_eval"]["llm_model_used"] == "claude-opus-4-8"

    def test_client_retry_of_a_transient_error_is_not_counted(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")

        _use_anthropic(monkeypatch)
        bodies: list[dict] = []

        def handler(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                return httpx.Response(
                    503, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
                )
            return httpx.Response(200, json=_anthropic_message(len(bodies), "claude-opus-5-5", _reply()))

        validator = RubricEvalValidator()
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            validator._judge = RubricJudge(
                http_client=http_client, max_retries=1, retry_base_delay=0, retry_max_delay=0
            )
            result = validator.validate(_write_skill(tmp_path))

        assert len(bodies) == 2
        assert bodies[1] == bodies[0]
        rubric = result.metadata["rubric_eval"]
        assert rubric["execution_status"] == "succeeded"
        assert rubric["judge_attempts"] == 1


class TestCollectToolsScripts:
    def test_tools_directory_scripts_reach_the_judge_without_colliding_with_scripts(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "tools").mkdir()
        (skill_dir / "scripts" / "run.mjs").write_text("console.log('scripts')")
        (skill_dir / "tools" / "run.mjs").write_text("console.log('tools')")
        (skill_dir / "tools" / "helper.py").write_text("print('tools')")

        result = _collect_supplementary_content(skill_dir)

        assert set(result["scripts"]) == {"run.mjs", "tools/run.mjs", "tools/helper.py"}
        assert "scripts" in result["scripts"]["run.mjs"]
        assert "tools" in result["scripts"]["tools/run.mjs"]
