---
name: feedback-commit-scope
description: "When user says commit, cross-reference worklogs and include other agents' concurrent changes too, not just Claude's own files"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b779dfd5-3b82-4c51-8e01-7b02073e1bfe
  modified: 2026-07-22T14:07:02.689Z
---

When the user says "commit" (커밋해/커밋하자) in this repo, do not narrowly scope `git add` to only files Claude itself touched this session. Read the relevant worklogs to understand what all the pending uncommitted changes are (including concurrent work from other agents, e.g. Codex, who edits this same repo in parallel per [[project_branch_rule]]-adjacent multi-agent workflow), and include that work in the commit too.

**Why:** twice in one session, Claude scoped commits narrowly to its own files only, staging/unstaging carefully to exclude a concurrent agent's in-progress changes (once accidentally including a stray pre-staged rename from the other agent, once deliberately excluding a large concurrent "proxy decomposition" changeset). The user explicitly corrected this after the second instance: they want a unified commit that reflects the real state of the repo across agents when they ask to commit, not a Claude-only slice. The user is already aware multiple agents edit this repo concurrently and is fine with a commit spanning both.

**How to apply:** before running `git commit`, read the worklogs for anything not yet committed (`git status`/`git log` to see what's new since HEAD, cross-referenced against `docs/worklogs/*.md` to understand what each pending change is *for*) so the commit message can accurately describe all the bundled work, not just Claude's own piece. Still exercise normal git safety (check diffs for secrets, don't force-push, etc.) but do not artificially exclude another agent's completed, worklogged changes from the commit just because Claude didn't author them.
