---
name: feedback-inrepo-memory-mirror
description: Mirror every memory add/edit into docs/agent_memory/ in the repo so parallel Codex/other agents can read it
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ab75852-24f1-41ab-9e47-206246eb025c
  modified: 2026-07-24T08:56:22.101Z
---

Whenever a memory file here is created or edited (add/update/remove), copy the
same change into `c:/Projects/OSN-GS/docs/agent_memory/` (same filename, same
content, including frontmatter) and keep `docs/agent_memory/MEMORY.md` in sync
with this directory's `MEMORY.md`. Also update `docs/agent_memory/README.md`
if the sync convention itself ever changes.

**Why:** the user runs a parallel Codex agent on the same repo. Codex has no
access to this user-local, out-of-repo memory path, so without an in-repo
mirror it would never see decisions, corrections, or project state Claude has
accumulated. The user explicitly asked (2026-07-24) for memory to be "managed
inside the project" for exactly this reason.

**How to apply:** treat this as a second write step appended to the existing
memory-save procedure — not a separate reminder to wait for. Concretely: after
`Write`-ing or editing a file under
`C:\Users\dna10\.claude\projects\c--Projects-OSN-GS\memory\`, immediately mirror
that same file into `docs/agent_memory/` inside the actual project working
directory (path may vary by session/OS, but on this machine it is
`c:/Projects/OSN-GS/docs/agent_memory/`). The home-directory copy stays
authoritative (see [[project_boundary_conditioned_occlusion]] and other memory
files for what "authoritative" means in case of drift) — the in-repo copy is a
read-only-for-other-agents mirror, not a second source of truth to edit
independently.
