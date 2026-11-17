# Missing-reference report

`report.json` is the unmodified output of SkillSpector 2.13.3 scanning the synthetic
`SKILL.md` in this directory, on 2026-11-10. No customer content is included.

The pinned image was
`ghcr.io/nvidia/skillevaluator-tier1@sha256:a3baec51f48145cdf9de05a5ab054d9576f62f1de0f5276b788410f5c17747d4`.
Only `SKILL.md` was mounted at `/workspace`. The entrypoint was
`/opt/skillspector/bin/skillspector scan /workspace/. --format json --no-llm`,
with network disabled and a read-only root filesystem. It exited 0.

All bundled content was inspected. The report remains `partial` / `CAUTION`
because a runtime example path and a version string were interpreted as missing
local references. Acceptance as a warning must preserve those caveats; it cannot
certify the absent targets or ignore any other inspection gap.
