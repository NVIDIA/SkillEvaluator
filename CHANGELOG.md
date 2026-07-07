# Changelog

All notable changes to Skill Evaluator are documented in this file.

## Unreleased

### Added

- Initial public release candidate.

### Security

- Removed implicit host-side pytest execution from default Tier 1
  code-integrity validation. Test evidence is now collected with contained,
  filename-only discovery that does not import or execute target-controlled
  Python code.

### Changed

- Replaced provisional `test_coverage`, `tests`, and `coverage_percent` output
  with one `test_discovery` detail. Reports now include `test_count`, supported
  filename patterns, `execution_performed=false`, and
  `coverage_measured=false`; projects must run tests and measure coverage in a
  trusted environment or explicit sandbox.
