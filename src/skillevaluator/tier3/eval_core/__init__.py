# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared evaluation core -- pure functions for skill evaluation.

Used by Harbor mode through the standalone verifier. All functions operate on
plain Python dicts so the same logic can be inlined into a Harbor container
with zero deps.
"""
