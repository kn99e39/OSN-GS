---
name: feedback_static_analysis_before_running
description: "when asked to find why something is slow/broken, read/analyze code first; don't default to running long empirical profiling passes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T03:52:53.299Z
---

When the user asks to investigate a performance or correctness issue ("that structure is slow, look into it"), they may specifically want static code analysis/reasoning first, not immediately launching real profiling runs. In one session (worklog 66, OSN-GS), a "GPU high VRAM but low utilization, slow" complaint was investigated by running several profiling passes (some 5-6 minutes each) before finding the root cause — the user then said this wasted the session's time and explicitly wanted code-analysis-based diagnosis instead of empirically running things.

**Why:** running experiments to diagnose is expensive (wall-clock, GPU contention with other work) and the user may already suspect they can be skipped if you just read the code carefully — grep for the operation pattern (e.g. per-element GPU tensor indexing inside a Python loop, O(N^2) patterns, batched-solver calls on huge batches) and reason about complexity/sync costs before reaching for a profiler.

**How to apply:** when asked to find a root cause, default to static analysis (read the suspect code paths, reason about complexity and GPU-sync patterns) and only run something if a static read can't resolve it or the user asks for empirical confirmation. If you do end up needing to run something, say so explicitly and keep it minimal rather than iterating through multiple profiling scripts.
