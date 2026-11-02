# SkillSpector 2.13.3 read-only mount reports

These reports were captured with `skillspector scan --no-llm --format json`
from the published `v2.13.3` tag, commit
`38a1dfe93325d2f8bfcd4f8d0241eed706334915`. They are newly generated reports,
not older fixtures with a changed version. Only the temporary scan-root prefix
was normalized; finding identities, scores, timestamps, provenance and
completeness accounting are scanner output.

Both input directories contained this `SKILL.md`:

```markdown
---
name: passwd-mount-probe
description: Build a project in a container as the host user.
---

# Container build

Run `sh build.sh` to build the project in a container.
```

`build.sh` was executable (mode `0755`). For `read-only.json`:

```sh
#!/bin/sh
docker run -v /etc/passwd:/etc/passwd:ro image
```

For `separate-read.json`:

```sh
#!/bin/sh
docker run -v /etc/passwd:/etc/passwd:ro image
cat /etc/passwd
```

Each mount emits two HIGH PE3 rows with distinct `finding_id` values, one
for each path operand, plus an unrelated MEDIUM RP1 image-pinning advisory.
The separate read emits a third, untagged HIGH PE3 row on line 3. Both scans
return exit code `0`, with complete inspection and aggregate scores `39`
and `43`, respectively; SkillEvaluator still blocks the actual HIGH read.

The captures have `llm_requested=false`, `meta_analysis_applied=false`, empty
`inference_usage`, heuristic filtering and disabled semantic analyzers.
`llm_available=true` means a provider was configured, independently of
`--no-llm`. The provenance correctly says execution was not applicable.
No LLM calls were requested. The adapter tests also exercise unavailable
providers by changing only that availability flag in memory.

To regenerate, create the input directories above, keep report destinations
outside the inputs, and run:

```sh
for case in read-only separate-read; do
  OPENAI_API_KEY=fixture-only SKILLSPECTOR_PROVIDER=openai uvx \
    --from git+https://github.com/NVIDIA/SkillSpector.git@v2.13.3 \
    skillspector scan "/tmp/skillspector-2.13.3-passwd-mount/$case" \
    --no-llm --format json --output "$case.json"
done
```

The placeholder key makes provider availability reproducible without using
credentials. Findings and timestamps must be regenerated and reviewed rather
than copied across scanner releases. Partial LLM-response tests are explicitly
synthetic mutations of these reports; they are not additional captured scans.
