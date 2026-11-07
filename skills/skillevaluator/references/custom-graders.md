# Custom graders and Harbor tasks

Bring your own grader (BYOG) or Harbor task (BYOT).

- `default_plus_custom`: standard dimensions stay authoritative and your
  `custom_metrics` are added. `custom_only`: your grader's `overall` decides.
- Scaffold with `skillevaluator init-custom-grader <skill-dir> --language python`
  or `skillevaluator init-harbor-task <skill-dir> --case-id case-001 --with-config`,
  replace the placeholder logic, then run
  `skillevaluator tier3 validate <skill-dir> --harbor-contract`.
- A grader reads `/logs/agent/trajectory.json`, `/tests/entry.json`, and
  `/workspace/input/`, and writes `/logs/verifier/reward.json` plus
  `/logs/verifier/reward.txt` with values from 0.0 to 1.0.
- Do not name custom metrics `security`, `skill_execution`,
  `skill_efficiency`, `accuracy`, `goal_accuracy`, `behavior_check`,
  `overall`, `details`, `metrics`, `metric_set`, or `entry_id`.
