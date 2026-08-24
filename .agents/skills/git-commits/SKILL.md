---
name: git-commits
description: Automatically create, organize, and push Git commits after completing repository changes. Use when finished work is ready for a verified commit or clean handoff.
---

# Git commits

- Automatically create commits without asking for routine confirmation.
- Commit and push at coherent, verified checkpoints rather than after every small edit.
- Before staging, inspect the working tree and distinguish task changes from pre-existing or unrelated user changes.
- Stage only files that belong to the completed task. Never include secrets, local environment files, generated artifacts, or unrelated changes.
- Split independent changes into focused commits when that improves reviewability.
- Do not commit incomplete work or changes whose relevant verification failed unless the user explicitly requests it.
- After committing, automatically push the current branch. Use an explicitly configured push remote first, then the configured upstream.
- In the common fork layout where `origin` is the contributor fork and `upstream` is the canonical repository, push the same branch to `origin` even when the local branch tracks `upstream` for updates.
- Never force-push. Do not amend, rebase, reset, rewrite history, create an upstream, or change remotes unless the user explicitly requests it.
- If no upstream exists, authentication fails, the remote rejects the push, or branch protection blocks it, stop and report the exact condition without retrying destructively.

## Commit messages

- Write every commit message bilingually.
- Use a concise English subject as the first paragraph.
- Add the equivalent Chinese translation as the second paragraph, separated from the English subject by a blank line.
- Keep the meaning and scope of both languages aligned.
- Use this form:

  ```text
  Add reusable Git commit workflow

  添加可复用的 Git 提交工作流
  ```

## Handoff

- After creating and pushing a commit, report its short hash, bilingual message, and pushed remote branch.
- If no commit is appropriate, leave the changes uncommitted without repeatedly asking whether to commit.
