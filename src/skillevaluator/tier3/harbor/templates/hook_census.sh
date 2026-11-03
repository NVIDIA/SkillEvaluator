#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SkillEvaluator hook execution census (POSIX sh).
#
# Usage: hook_census.sh <hook_id> <event> -- <command...>
#
# Runs <command...> with this process's stdin, stdout, and stderr untouched and
# exits with the command's exact exit status, because hook decisions depend on
# all four. After the command finishes, one JSON line is appended to the census
# file:
#
#   {"hook_id":"...","event":"...","exit_code":N,"started_at":"...Z","duration_ms":N}
#
# A single argument after "--" is a shell command string and runs with
# "/bin/sh -c", the way harnesses run a hook's "command". Two or more arguments
# run as an argv vector.
#
# The logger itself never writes to stdout or stderr, so it cannot change hook
# output. A census write that fails (missing or read-only log directory) is
# ignored. The census is best-effort evidence, not tamper-proof: anything that
# runs in the environment can also write to the log directory.

census_file=/logs/agent/skilleval-hook-census.jsonl

if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
    echo "usage: hook_census.sh <hook_id> <event> -- <command...>" >&2
    exit 64
fi
hook_id=$1
event=$2
shift 3

# Milliseconds since the epoch. GNU date supports %N; BSD and some BusyBox
# builds print it literally, so fall back to whole seconds. Prints nothing when
# no usable clock is available.
_census_now_ms() {
    _census_ns=$(date +%s%N 2>/dev/null) || _census_ns=
    case $_census_ns in
        '' | *[!0-9]*) ;;
        *)
            if [ "${#_census_ns}" -gt 12 ]; then
                echo $((_census_ns / 1000000))
                return 0
            fi
            ;;
    esac
    _census_s=$(date +%s 2>/dev/null) || _census_s=
    case $_census_s in
        '' | *[!0-9]*) return 0 ;;
    esac
    echo $((_census_s * 1000))
}

# JSON string body: control characters are dropped, backslash and double quote
# are escaped, and the value is capped at 256 bytes.
_census_json_escape() {
    printf '%s' "$1" | LC_ALL=C tr -d '\000-\037\177' | LC_ALL=C cut -c 1-256 |
        LC_ALL=C sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

_census_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) || _census_started_at=
_census_start_ms=$(_census_now_ms)

if [ "$#" -eq 1 ]; then
    /bin/sh -c "$1"
else
    "$@"
fi
_census_rc=$?

_census_end_ms=$(_census_now_ms)
_census_duration=null
if [ -n "$_census_start_ms" ] && [ -n "$_census_end_ms" ] && [ "$_census_end_ms" -ge "$_census_start_ms" ]; then
    _census_duration=$((_census_end_ms - _census_start_ms))
fi
case $_census_started_at in
    '' | *[!0-9TZ:-]*) _census_started_json=null ;;
    *) _census_started_json="\"$_census_started_at\"" ;;
esac

(
    _census_id=$(_census_json_escape "$hook_id")
    _census_event=$(_census_json_escape "$event")
    printf '{"hook_id":"%s","event":"%s","exit_code":%s,"started_at":%s,"duration_ms":%s}\n' \
        "$_census_id" "$_census_event" "$_census_rc" "$_census_started_json" "$_census_duration" \
        >>"$census_file"
) </dev/null >/dev/null 2>&1

exit "$_census_rc"
