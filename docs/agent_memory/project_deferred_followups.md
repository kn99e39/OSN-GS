---
name: project-deferred-followups
description: "Training speed gap still deferred until NURBS is complete; scene quality gap on VISIBLE surfaces is NOT excused by that and is being actively root-caused"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2eedee13-be57-4cc6-9ec2-a41c63998cad
  modified: 2026-07-23T05:35:03.845Z
---

**Updated 2026-07-23 — the quality-gap deferral below was partially corrected.** The user pushed back explicitly: OSN-GS's design intent is that NURBS only guides *occluded*-surface Gaussian placement; visible/non-occluded Gaussians are meant to train from image loss alone, exactly like baseline 3DGS (`nurbs_surface_loss` already detaches Gaussian xyz, confirmed in code). So quality loss on non-occluded subjects is **not excusable** by "NURBS isn't finished yet" — that excuse only covers occluded-region shortcomings. See `docs/worklogs/70_opacity_lr_fix_and_visible_blur_investigation.md` and `71_scene_extent_basis_mismatch_and_visible_blur_root_cause.md`.

Root-caused so far (both found via direct iteration-3000 A/B against baseline on the real DATASET, same fixed camera):
- `opacity_lr` was 2x baseline's (0.05 vs 0.025) — fixed.
- **Bigger finding**: OSN-GS's Gaussian `scale` is systematically 20-50% larger than baseline's at matched iteration/Gaussian-count, directly causing the blur. Root cause: `_scene_extent()` (the worklog-63 "fix") computes extent from the **sparse point cloud** (mean-center + 90th-percentile distance), while baseline's `cameras_extent` computes it from **camera positions** (`getNerfppNorm`) — a 2.5x difference on this walkthrough-style scene (point cloud extends far past the camera path). This `scene_extent` feeds `spatial_lr_scale`, ADC's clone/split threshold (`percent_dense * scene_extent`), and world-size prune threshold — all of baseline's calibration constants assume the camera-based number, so feeding them the point-cloud-based number systematically inflates Gaussian scale. This is a basis mismatch (two independently-reasonable designs combined incorrectly), not a simple arithmetic bug — see worklog 71 for the full reasoning on why point-cloud-basis was chosen (SfM outlier robustness) and why that doesn't make it the right basis for baseline-tuned constants.
- **Not yet fixed** — user wants this refined rather than just reverted to baseline's camera-based formula, since the point-cloud basis might be genuinely better if the downstream constants were recalibrated for it. Left as an open follow-up, not reopened for implementation this session.
- Old "SSIM missing" suspicion (below, from 2026-07-21) is **wrong** — confirmed this session that OSN-GS's SSIM implementation (`osn_gs/losses/torch_losses.py`) exactly matches baseline's (window_size=11, sigma=1.5, same formula).

---

Original 2026-07-21 framing (training-speed part still holds):

1. **Training speed gap** — see [[project-baseline-comparison]] and `docs/worklogs/11_training_bottleneck_audit.md` / `10_surface_loss_runtime_audit.md`. Main causes: NURBS surface loss/backward cost, ADC's un-fused clone/split/prune tensor rebuilds (worklog 67 added `torch.cuda.empty_cache()` after ADC, addressing allocator fragmentation specifically), periodic surface maintenance, full snapshot streaming.

**Why speed is still deferred:** the visible-surface NURBS representation itself is still incomplete (Boundary-First plan is only through Phase 4; Phase 5 occluded-surface work is gated on user approval, see [[project-boundary-first-phase1]]). [[project-osn-gs-direction]] forbids trading structural correctness for throughput.

**How to apply:** quality-on-visible-surfaces is fair game to fix now (see above) — don't defer it by citing NURBS incompleteness. Speed work stays parked until the representation is functionally complete or the user explicitly reopens it. Keep timing/quality logging in place either way.
