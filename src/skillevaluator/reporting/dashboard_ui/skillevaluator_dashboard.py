# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streamlit entry script for the dashboard.

Streamlit puts this script's directory first on sys.path, so the directory holds
only this uniquely named file and cannot shadow the standard library or a dependency.
"""

import sys

from skillevaluator.reporting.dashboard_app import main

main(sys.argv[1:])
