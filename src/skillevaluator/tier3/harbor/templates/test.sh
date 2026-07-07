#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
tests_dir="${HARBOR_TESTS_DIR:-/tests}"
python3 "${tests_dir}/eval.py"
