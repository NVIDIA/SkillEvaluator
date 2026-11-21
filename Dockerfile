# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SkillEvaluator public runtime image.
#
# The image installs every public optional dependency. Tier 3 still requires
# credentials for the selected provider and agent at runtime.
# Use a Python TLS stack that rejects the affected certificates during pip bootstrap.
FROM python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PIP_NO_INPUT=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . /app

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[all]" && \
    skillevaluator --help >/dev/null

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin skillevaluator && \
    mkdir /app/reports && \
    chown skillevaluator:skillevaluator /app/reports

USER skillevaluator

ENTRYPOINT ["skillevaluator"]
CMD ["--help"]
