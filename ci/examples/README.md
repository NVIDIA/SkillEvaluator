# SkillEvaluator CI examples

These examples live in the public NVIDIA/SkillEvaluator repository. You do not
need access to a personal demo repository, private pipeline or hosted report to
view them. For setup, [copy a template](../templates/README.md).

## Report preview

![SkillEvaluator report showing a failed skill with a quality score of 98.2](../../docs/assets/repository-ci-results-example.png)

This captured example shows a skill scoring **98.2/100** against a **100-point**
threshold. The scan fails because it is below the configured threshold. Quality
score and scan verdict are separate; security and other required checks also
affect the verdict. This preview is an example, not a result for your repository.

GitHub and GitLab retain the HTML report as a job artifact. Open `index.html` in
the downloaded artifact, then select a skill for its detailed findings.

## GitHub Actions summary

A shortened illustrative example:

```text
Tier 1 scan result: FAILED
1 skill · 0 passed · 1 failed
Problem: Quality score 98.2/100 is below the required 100/100
```

The full report identifies the quality checks that need attention. Improve the
skill to meet the repository's chosen threshold, then submit the change again.

## Optional GitHub PR comment

The following is a shortened illustrative comment, not a live pull request:

> **github-actions[bot]**
>
> **Tier 1 scan result: FAILED**
>
> 1 skill · 0 passed · 1 failed · 0 incomplete
>
> | Skill | Status | Quality / 100 |
> | --- | --- | --- |
> | `skills/glossary-builder` | FAIL | 98.2 / A |
>
> **Problem:** Quality score 98.2/100 is below the required 100/100.
>
> **Suggested fix:** Open the skill report and address the quality findings.
>
> **Open full report** · **View workflow run**

The last line shows link labels only. In an actual comment, they link to your
repository's hosted report and workflow run. The publisher updates one comment
on an eligible current same-repository PR after successful report publication.
Follow the [optional publisher setup](../../docs/github-actions.mdx#enable-pr-comments).

## GitLab job log

A shortened excerpt corresponding to the report preview:

```text
Skill result: "skills/glossary-builder" — FAILED
  Quality: 98.2/100, grade "A"
  Check: "QUALITY" — "failed"
    Scanner error: "Quality score 98.2/100 is below the required 100/100 ..."
```

Download `skill-evaluator-report` from the job for the complete findings. The
GitLab integration provides job artifacts; automatic MR comments and hosted
reports are not included.
