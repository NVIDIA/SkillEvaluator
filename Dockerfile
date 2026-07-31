# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SkillEvaluator public runtime image.
#
# The image installs every public optional dependency. Tier 3 still requires
# credentials for the selected provider and agent at runtime.
FROM python:3.12-slim

ENV PIP_NO_INPUT=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . /app

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[all]" && \
    skillevaluator --help >/dev/null

ENTRYPOINT ["skillevaluator"]
CMD ["--help"]
