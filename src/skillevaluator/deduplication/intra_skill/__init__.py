# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intra-skill context deduplication pipeline.

Two-stage pipeline:
  Stage 1: Embedding clustering (batch embed → pairwise cosine → Union-Find)
  Stage 2: LLM verification (DUPLICATE / INTENTIONAL_DETAIL / RELATED_BUT_DISTINCT)
"""
