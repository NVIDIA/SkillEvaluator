#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PostToolUse hook: count make invocations in a local session file.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | grep -o '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1)
[ -z "$COMMAND" ] && exit 0

COUNTER_FILE="${CLAUDE_PROJECT_DIR:-$(pwd)}/.build-notes/session.yaml"
[ ! -f "$COUNTER_FILE" ] && exit 0

if echo "$COMMAND" | grep -qE '(make|cmake)'; then
    CURRENT=$(grep 'build_attempts:' "$COUNTER_FILE" 2>/dev/null | grep -o '[0-9]*')
    NEXT=$((${CURRENT:-0} + 1))
    sed -i "s/build_attempts: [0-9]*/build_attempts: $NEXT/" "$COUNTER_FILE" 2>/dev/null
fi

exit 0
