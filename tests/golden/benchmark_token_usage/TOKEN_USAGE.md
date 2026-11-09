## Token Usage

Measured agent token usage. It is not scored and changes no score or verdict. Tokens are prompt plus completion; prompt tokens include cached tokens, which are not added again. Attempts that failed before scoring are counted. A partial total covers only the attempts with token counts; nothing is estimated. Change is with skill minus baseline, as a share of the baseline, and needs complete counts for the same attempts in both arms.

| Agent | Case | With skill | Baseline | Change | Measured attempts |
|---|---|---:|---:|---:|---|
| Claude Code | All cases | 3,300 | 50,072 | N/A | skill 2/2; baseline 2/2 (1 unscored) |
| Codex | All cases | 1,210 | 1,650 | -440 (-26.7%) | skill 2/2; baseline 2/2 |
| All agents | All cases | 4,510 | 51,722 | N/A | skill 4/4; baseline 4/4 (1 unscored) |

Measured attempts: attempts with token counts, of those expected, in each arm. Unscored attempts failed before scoring.

- Claude Code: No change: baseline incomplete (1 of 2 attempts scored).
- All agents: Not every agent has a token change, so there is no change over all agents.

<details>
<summary>Show token usage per case</summary>

| Agent | Case | With skill | Baseline | Change | Measured attempts |
|---|---|---:|---:|---:|---|
| Claude Code | case-1 | 1,100 | 1,650 | N/A | skill 1/1; baseline 1/1 |
| Claude Code | case-2 | 2,200 | 48,422 | N/A | skill 1/1; baseline 1/1 (1 unscored) |
| Codex | case-1 | 550 | 770 | -220 (-28.6%) | skill 1/1; baseline 1/1 |
| Codex | case-2 | 660 | 880 | -220 (-25.0%) | skill 1/1; baseline 1/1 |

</details>
