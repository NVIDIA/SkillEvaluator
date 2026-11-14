# Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Error: No such command` or exit `2` | Wrong command or flag for this version | `skillevaluator --version`, then `skillevaluator <command> --help` |
| `tier1 validate` evaluated the wrong thing | Directory named like a subcommand | Use `<path>/validate` |
| `--tiers 2` rejected | `validate` always includes Tier 1 | Use `skillevaluator tier2 PATH` |
| Scanner `INCOMPLETE` | SkillSpector, Gitleaks or Bandit missing or failed | `skillevaluator health-check --tier 1`; install the `security` extra; an old scanner earlier on `PATH` wins over the pinned one |
| `--skip` used and publication fails | Skipped checks cannot certify Tier 1 | Rerun without `--skip` |
| Schema: directory does not match name | Folder name differs from frontmatter `name` | Rename one so both match (kebab-case) |
| Quality below minimum | Missing expected headings, vague description, no examples, `SKILL.md` over 500 lines or 5,000 tokens | Follow each finding's suggestion; move detail into `references/` |
| Rubric fails although the score is high | One criterion below 7/10 | Read that criterion's feedback in the JSON report |
| Tier 2 needs embeddings | Anthropic and Bedrock have no embeddings | Set `SKILL_EVAL_EMBEDDING_*` for another provider |
| `<skill-dir>/evals/config.yml` rejected | Unknown key or credential in `runtime_env` | Fix the key; move secrets to the host environment |
| `Missing required fields: question` | Legacy entry without `question`, or new entry without `prompt` | Use the agentskills.io shape |
| Skill never activated | Description and prompts do not match | Name the tasks in the description; add explicit cases |
| Agent auth fails | Agent credential not exported where `skillevaluator` runs | Export it on the host; `doctor -a AGENT` |
| `AgentTimeoutError` or rate-limit errors | Slow model or too much concurrency | `--timeout-multiplier 2`, lower `--n-concurrent` |
| Custom Dockerfile build fails | Base image lacks tools the agent or grader needs, or `python3 -I` cannot import the verifier packages (`ENV PIP_USER=1`, `PIP_TARGET` or `PYTHONPATH` installs) | Fix `<skill-dir>/evals/environment/Dockerfile`; install verifier packages into `python3`'s site-packages; rerun with `--harbor-keep-jobs` to read the build log |
| Results need debugging | Harbor jobs deleted by default | Rerun with `--harbor-keep-jobs`, then `skillevaluator harbor-view JOBS_DIR` |
| Lift differs between runs | LLM judges and agents are stochastic | Increase `--n-attempts`; compare only identical setups |
