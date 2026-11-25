# Contributing

Contributions are welcome through GitHub pull requests.

## Before Opening A Pull Request

- Discuss substantial changes in an issue first.
- Keep provider credentials, customer data, and private benchmark material out
  of the repository.
- Add focused tests for behavioral changes.
- Run `make lint`, `make test`, and `make build` locally.
- Update [CHANGELOG.md](CHANGELOG.md) when the change affects users.

For environment setup, focused tests, documentation checks and the review
workflow, read the [contributor guide](docs/developer-guide.mdx). For a runnable
public skill and task, start with the [terminal starter](examples/starter/README.md).

## TestPyPI release validation

The `pypi-test` GitHub environment uses the release reviewers and permits
deployments from `main`. Register a TestPyPI GitHub Actions Trusted Publisher
for `skillevaluator` with owner `NVIDIA`, repository `SkillEvaluator`, workflow
`release.yml`, and environment `pypi-test`.
[Set up a pending publisher](https://test.pypi.org/manage/account/publishing/).

Run **SkillEvaluator Release** from `main` with phase **Validate release**
and `publish_testpypi` selected. After approving the TestPyPI deployment,
the workflow verifies the hosted wheel and source distribution against the
retained release artifacts and smoke-tests both installs with dependencies
from production PyPI. If TestPyPI staging fails, rerun all jobs or start a
new validation run so the successful attempt retains its own artifacts.

For production, use the successful validation run's inputs in the
**Publish release** phase. The `release-kitmaker` environment approves
production publication through Kitmaker.

## Documentation And Example Contributions

- Run every command you change against the documented release or checkout and
  say which you tested. Do not claim support on platforms you did not verify.
- Keep example inputs public and portable. Include expected-output checks and
  label seeded failures and missing evidence; a fixture is not a live benchmark.
- For this starter, run `uv run python scripts/demo_starter.py --output reports/contribution-demo`
  and `uv run pytest tests/test_demo_starter.py`. Use a new output directory on
  subsequent runs. Retain reports locally and summarize exactly what ran in
  the pull request; do not commit generated reports or transcripts.
- Validate docs with the pinned Fern CLI as described in the contributor guide.
- Explain recovery from missing dependencies, credentials and incomplete runs.

## Pull Requests

Use the pull request template, keep each pull request focused, and explain the
user-visible behavior and verification performed. By submitting a contribution,
you confirm that you have the right to contribute it under the
[Apache License 2.0](LICENSE).

## Keeping Changes Small

The **Size report** CI job reports these numbers on each pull request that is not
docs-only. Not enforced yet; the `size-exception` label marks an agreed exception.

- A `fix:` adds at most 150 net lines under `src/` and 400 under `tests/`, no
  new file under `src/`, no new test module, and at most 3 new test functions.
  The function count estimates pytest discovery from Python definitions and
  matches unchanged definitions with the same qualified name across test modules.
- For all changes, prefer extending an existing test module or
  `pytest.mark.parametrize` table when it covers the same behavior.
- Big files only shrink: a file listed in `config/size_baseline.json` does not grow.
- Past 100 net new test lines, a change adds at most 3 per added source line.
- No whole captured reports as fixtures: keep one base file and build variants
  in the test. A new non-Python file under `tests/` has at most 100 lines.
- Say it once: a comment says what the code does now, and the story of the bug
  goes in the pull request.

Run it locally with `python scripts/ci/size_guard.py --title "fix: your title"`.
It compares `HEAD` with `origin/main`; pass `--base <ref>` if that is not this
repository's current `main`.

## Signing Your Work

The contribution requirement is the Developer Certificate of Origin (DCO)
sign-off below, not a Contributor License Agreement (CLA).

- We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be
    accepted. Pull requests are checked in CI by the **DCO** workflow; unsigned
    commits fail the check.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the
  [Developer Certificate of Origin](https://developercertificate.org/):

  ```text
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

Report vulnerabilities through [SECURITY.md](SECURITY.md), not public issues.
