---
name: project_gaussian_initialization_parity
description: worklog 64 - found+fixed real dominant cause of anisotropy/screen-prune blowup; covariance-KNN init forced ~25x anisotropy at iter 0
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-05T08:10:14.470Z
---

worklog 64: OSN-GS's only Gaussian-initialization path (`_canonical_initial_covariance` in `osn_gs/core/torch_pipeline.py`) hardcodes `normal_scale = tangent_scale * 0.04`, forcing every new Gaussian's iteration-0 anisotropy to ~25 by construction — this, not the loader bugs from [[project_graphdeco_lockstep_training_parity]]/[[project_fixed_loader_3k_production_replay]], is the real dominant cause of the 3k+ anisotropy/screen-prune blowup vs Graphdeco baseline (which starts at exactly anisotropy=1.0 via isotropic `distCUDA2`-based init).

Fix: added `TorchPipelineConfig.gaussian_initialization_mode` — `"baseline_compatible"` (new default, Graphdeco-equivalent isotropic init via new `_baseline_compatible_scale_rotation()`) vs `"covariance_knn"` (old planar-surfel init, kept as explicit experimental opt-in). Only affects the model's own trainable `_scaling`/`_rotation` init in `_initialize_canonical` (the production "initialize" schedule). The SEPARATE covariance used for canonical visible surface construction/reliability is untouched (still always local-PCA). `initialize_deferred` (adc_post_commit/disabled schedule) intentionally does NOT respect this flag — its first post-ADC surface reconstruction reuses the model's own scale/rotation as its only orientation evidence, so switching it to isotropic would break surface bootstrap (a regression I hit and reverted via `test_post_adc_transaction_is_detached_rng_neutral_and_observed_only`).

3-way parity harness (`scripts/devtools/gaussian_init_parity_harness.py`) confirmed: baseline_compatible tracks Graphdeco closely through step 600 + first real ADC event (anisotropy 1.576 vs baseline 1.617, count within 0.8%), while covariance_knn stays at anisotropy 26-28 throughout and triggers 3.8x more ADC splits. pytest 775→783.

**Why:** this fully explains what worklog 63's loader-fix-only replay couldn't — the loader bugs were real but minor; the init-time forced anisotropy dominates.

**How to apply:** `gaussian_initialization_mode=baseline_compatible` is now the production default (train.py, scripts/train_osn_gs_torch.py, colab_args.py all wired). Next natural follow-up (not yet done, explicitly deferred per task instruction "600-step parity 확인 전에는 3k production replay 재수행 금지"): re-run 3k/5k/10k production replay with this fix to confirm real-scale recovery, superseding worklog 63's numbers.
