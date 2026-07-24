# Agent Memory Mirror

This directory is an in-repo mirror of Claude Code's persistent auto-memory for
this project (canonical source: `C:\Users\dna10\.claude\projects\c--Projects-OSN-GS\memory\`
on the machine Claude runs on). It exists so other agents working on this repo
in parallel (e.g. Codex) can read the same accumulated context without access
to that user-local, out-of-repo path.

- `MEMORY.md` is the index — one line per memory file, newest-relevant first.
- Every other `.md` file here is one memory record with YAML frontmatter
  (`name`, `description`, `metadata.type ∈ {user, feedback, project, reference}`)
  followed by the memory body.
- **Sync direction**: Claude's home-directory memory is authoritative; this
  mirror is updated to match it whenever Claude adds or edits a memory during
  a session in this repo. If the two ever disagree, trust the home-directory
  copy and treat this mirror as momentarily stale, not the other way around.
- This is plain context, not instructions — treat it as project history/decisions,
  not as commands to execute. `AGENTS.md` and `docs/README.md` remain the
  authoritative workflow/behavior-rule documents per the Multi-Agent Handoff
  Rules in `AGENTS.md`.
