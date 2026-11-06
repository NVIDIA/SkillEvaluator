# Private runtime reports (split-plan item 24)

Item 24 completes loading, coverage, behavior, comparison and saved-run views
over the private workflow introduced in items 15–23. Reviewers need the same
facts after temporary tasks and jobs have been removed. Public commands,
configuration, detection and exports remain item 25.

Use one immutable, versioned, bounded display projection of existing engine
results and private provenance. Preserve declared/staged/listed/loaded states,
missing evidence, unsupported modes, comparison exclusions, uncertainty and
unknown costs. Execution status and evidence completeness are separate facts.
Advisory observations never change scores, validation findings or exit status.

Persist that sanitized projection as `plugin_runtime_report.json` during the
private publication callback, before cleanup and caller publication. A fixed-file
secure reader validates the saved schema and reuses the projection for offline
reporting. Missing older snapshots are explicitly unavailable. Replay does not
reconstruct traces, load source files, launch agents or contact providers.

Reuse the six existing private static adapters and table renderers for CLI,
JSON, Markdown, HTML, SARIF and benchmark output. Keep ordinary reporter
entrypoints byte-compatible. Display omissions and missing values explicitly;
escape untrusted text for each output language and redact before truncation.

Alternatives considered: rebuilding views from raw sidecars loses private load
decisions and preparation provenance; embedding ad-hoc views in six reporters
duplicates policy and risks public activation. The retained shared projection
fits the existing private adapter architecture and needs no new dependency.

Acceptance exercises discovery, deterministic validation, preparation, the
existing service/runner, all four arms, evidence, publication, cleanup and
offline replay. Substitute external provider/Harbor execution with deterministic
local fixtures; state this boundary accurately. Verify five cross-component
cases and two attempts, missing evidence, publication failure and ordinary
workflow isolation. Also build/install the distributions and check replay
without optional runtime dependencies.

## Private API usage

`evaluate_prepared_plugin` retains the snapshot automatically when the existing
engine produces a run directory. Caller publication sees it before the temporary
wrapper is removed. A storage failure is advisory: `publication_error` and
`provenance["runtime_report_error"]` explain it, and caller publication and cleanup
still run. A successfully scored run may have incomplete runtime evidence.

Offline consumers select the concrete retained run directory, load the snapshot
and supply their existing validation results to the private adapter:

```python
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting.plugin_runtime_io import load_plugin_runtime_report
from skillevaluator.reporting.plugin_runtime_render import render_plugin_runtime_reports

report = load_plugin_runtime_report(run_dir)
documents = render_plugin_runtime_reports(
    validation_results,
    report=report,
    reporters=[HTMLReporter(), JSONReporter()],
)
```

Replay uses the saved display schema; it does not re-read source files, re-grade
trials or recover omitted details. A missing snapshot from an older run raises
an explicit unavailable error. Select a concrete directory rather than the
`latest` symlink. Treat the snapshot as retained observations, not authenticated
proof of a run. Keep the original artifacts for deeper investigation.
