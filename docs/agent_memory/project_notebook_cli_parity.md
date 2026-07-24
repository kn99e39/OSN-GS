---
name: project-notebook-cli-parity
description: OSN-GS training defaults must stay identical across the notebook and both CLI parsers; the shared recipe is VRAM-safe (low_vram on)
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f58b1e8-0abf-4c0b-a3b4-3b8396c6006c
---

OSN-GS training defaults are duplicated in THREE places and must stay in sync, or notebook-run and CLI-run produce different models: `osn_gs/interop/colab_args.py` (`build_osn_gs_train_parser`, used by `train.py` — this is what the notebook calls), `scripts/train_osn_gs_torch.py` (`build_parser`, a second standalone CLI entrypoint), and the notebook's `OSN_*` constants block in `colab_train_3dgs.ipynb`.

As of 2026-07-15 the user chose the **VRAM-safe recipe** as the shared default so a bare CLI run reproduces the notebook exactly: `densify_until_iter=15000`, `densification_interval=100`, `visible_surface_resolution_scale=4.0`, and `--low_vram` ON by default (`argparse.BooleanOptionalAction`, opt out with `--no-low_vram`). low_vram forces half-resolution training (`train_resolution_scale>=2`). See [[reference-osn-gs-docs]] → `docs/worklogs/14_notebook_cli_training_parity.md`.

**Why:** ADC used to be OFF by default on the CLI (`densify_until_iter`/`densification_interval` defaulted to 0) while the notebook turned it on, so a `python train.py ...` run silently skipped densification and produced far worse results than the notebook — a silent recipe divergence, not a VRAM tradeoff.

**How to apply:** when changing any training default, change it in all three places. Only recipe/result-affecting params must match (ADC schedule, resolutions, all voxel/NURBS/covariance/ADC numeric params, loss weights, LRs). Perf/memory-only knobs (`image_device`, `visible_surface_fit_device`, `*_chunk_size`, streaming/log/save cadence) do NOT change the trained result and are intentionally allowed to differ. Note the VRAM-safe default trains at half resolution, which handicaps OSN-GS in any full-resolution comparison against the Graphdeco baseline — see [[project-osn-gs-direction]] and `TODO.md`.
