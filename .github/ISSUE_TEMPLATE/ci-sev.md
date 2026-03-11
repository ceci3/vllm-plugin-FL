---
name: CI SEV
about: Report a CI infrastructure or pipeline failure
title: "[CI SEV]: "
labels: "ci-sev"
assignees: ''
---

## Severity

<!-- Select one:
  SEV 0 - Main branch CI completely broken, all jobs failing, blocks all development
  SEV 1 - Critical job failing (e.g. functional GPU tests, build check), blocks merging
  SEV 2 - Non-critical job failing or flaky (e.g. lint, single Python version), does not fully block merging
-->

**Severity Level:**

## Failing Job

<!-- Which CI job is affected? Check all that apply. -->

- [ ] Lint (ruff check / ruff format / typos)
- [ ] Unit Tests
- [ ] Build Check
- [ ] Functional Tests - ops / compilation (GPU)
- [ ] Functional Tests - inference (GPU)
- [ ] Functional Tests - serving (GPU)
- [ ] Other: <!-- specify -->

## Description

<!-- Describe the failure. What is broken and how does it impact development? -->

## Failed Run Link

<!-- Paste the link to the failing GitHub Actions run. -->

## Error Logs

<!-- Paste the relevant error output (use <details> for long logs). -->

<details>
<summary>Error logs</summary>

```
paste logs here
```

</details>

## Likely Cause

<!-- If known, describe the suspected root cause. Examples:
  - Upstream dependency update
  - Self-hosted GPU runner offline / driver issue
  - Flaky test
  - Recent commit (link the commit)
-->

## Affected Branch / PR

<!-- Which branch or PR is affected? e.g. main, or PR #123 -->
