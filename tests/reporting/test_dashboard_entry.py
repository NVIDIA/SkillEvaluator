# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the packaged Streamlit entry script that the dashboard launcher starts."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("streamlit")
# st.dataframe otherwise imports these inside AppTest's timed script thread.
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
from streamlit.testing.v1 import AppTest

from skillevaluator.reporting import dashboard

# A ceiling for cold imports under xdist and coverage; passing runs finish far sooner.
_APP_TIMEOUT_SECONDS = 60


def test_entry_script_runs_dashboard_main(monkeypatch) -> None:
    # AppTest runs the script in this process, so it would otherwise read pytest's argv.
    monkeypatch.setattr(sys, "argv", [str(dashboard._APP)])
    app = AppTest.from_file(str(dashboard._APP), default_timeout=_APP_TIMEOUT_SECONDS).run()
    assert not app.exception
    assert [message.value for message in app.info] == ["Upload JSON evaluation reports to start comparing results."]
