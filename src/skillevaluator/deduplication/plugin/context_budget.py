# SPDX-License-Identifier: Apache-2.0
"""Finite operator bounds and measured work shared by sequential plugin members."""

from dataclasses import dataclass

from skillevaluator.constants import CONTENT_DEDUP_MAX_CHUNKS, CONTENT_DEDUP_MAX_SCALAR_COMPARISONS

PLUGIN_CONTEXT_CHUNK_CEILING = 32 * CONTENT_DEDUP_MAX_CHUNKS
PLUGIN_CONTEXT_SCALAR_CEILING = CONTENT_DEDUP_MAX_SCALAR_COMPARISONS


def validate_plugin_context_limits(max_chunks=None, max_scalar_comparisons=None):
    for name, value, ceiling in (
        ("chunks", max_chunks, PLUGIN_CONTEXT_CHUNK_CEILING),
        ("scalar comparisons", max_scalar_comparisons, PLUGIN_CONTEXT_SCALAR_CEILING),
    ):
        if value is not None and (type(value) is not int or not 1 <= value <= ceiling):
            raise ValueError(f"Plugin context {name} must be an integer within [1, {ceiling}]")


@dataclass
class ContextWorkBudget:
    chunk_limit: int
    scalar_limit: int
    llm_call_limit: int
    prompt_char_limit: int
    chunks_collected: int = 0
    chunks_embedded: int = 0
    scalar_comparisons: int = 0
    llm_clusters_submitted: int = 0
    prompt_chars_submitted: int = 0
    vector_dimension: int | None = None

    @property
    def remaining_scalars(self):
        return self.scalar_limit - self.scalar_comparisons

    @property
    def remaining_llm_calls(self):
        return self.llm_call_limit - self.llm_clusters_submitted

    @property
    def remaining_prompt_chars(self):
        return self.prompt_char_limit - self.prompt_chars_submitted

    def snapshot(self):
        return {
            "limits": {
                "chunks": self.chunk_limit,
                "scalar_comparisons": self.scalar_limit,
                "llm_cluster_reviews": self.llm_call_limit,
                "prompt_chars": self.prompt_char_limit,
            },
            "consumed": {
                "chunks_collected": self.chunks_collected,
                "chunks_embedded": self.chunks_embedded,
                "scalar_comparisons": self.scalar_comparisons,
                "llm_clusters_submitted": self.llm_clusters_submitted,
                "prompt_chars_submitted": self.prompt_chars_submitted,
            },
            "vector_dimension": self.vector_dimension,
        }
