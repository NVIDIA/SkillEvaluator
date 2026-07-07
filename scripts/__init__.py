# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo tooling package.

Marks ``scripts`` as a real package so ``scripts.release`` resolves to this
repo's release tooling rather than an unrelated top-level ``scripts`` package
that may be present in the environment's site-packages. Not shipped in the
``skillevaluator`` wheel (setuptools only packages ``src/skillevaluator``).
"""
