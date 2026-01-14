# Q&A and Patch Proposal Workflow

This directory contains the workflow for proposing and reviewing updates to the ML Study Notes.

## Workflow Overview

1. **Identify Issue**: Find an error, missing content, or improvement opportunity
2. **Create Proposal**: Submit a patch proposal in `proposals/` directory
3. **Review**: Proposal is reviewed for accuracy and consistency
4. **Apply**: Approved proposals are merged into main documentation

## Proposal Template

Create a new file in `proposals/` with the following structure:

```yaml
# File: proposals/YYYY-MM-DD-short-description.yaml

title: "Brief title of the change"
author: "Your name"
date: "YYYY-MM-DD"
status: "pending"  # pending, approved, rejected, merged

type: "fix"  # fix, enhancement, new-content

affected_files:
  - "docs/en/fundamentals/bias-variance.md"
  - "docs/zh/fundamentals/bias-variance.md"

description: |
  Detailed description of the proposed change.
  Explain why this change is needed.

changes:
  - file: "docs/en/fundamentals/bias-variance.md"
    section: "Math and Derivations"
    before: |
      Current content that needs to change
    after: |
      Proposed new content

references:
  - "Link to paper or resource supporting this change"

review_notes: |
  (Filled by reviewer)
```

## Status Flow

```
pending -> approved -> merged
       \-> rejected
```

## Validation

Before submitting a proposal:

1. Run validators: `python -m validators.main`
2. Ensure bilingual consistency (if changing content, change both EN and ZH)
3. Check quiz questions remain valid
4. Verify math renders correctly

## Example Proposal

See `proposals/example.yaml` for a complete example.
