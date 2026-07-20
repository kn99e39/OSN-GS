# Training Bottleneck Audit

Date: 2026-07-14

## Evidence

The notebook's completed 10,000-iteration run already reports the current split timings:

```text
iteration=10000 ... render_loss=0.013s surface_loss=0.056s backward=0.123s
optim=0.002s density=4.644s save=0.032s log=0.001s total=4.893s avg_iter=0.188s
```

Earlier stored output did not split `surface_loss` from `backward`, so its approximately 0.33 s normal iteration cannot be used to evaluate the current patch-minibatch path.

## Bottlenecks

1. Surface loss remains the largest steady OSN-GS-specific cost.
   - It samples up to 8,192 certain Gaussians, filters them by active patch, evaluates up to 16 rational patches, and backpropagates through their control grids every iteration.
   - The current 16-patch round-robin budget bounds this cost, but `surface_loss=0.056s` plus its contribution to `backward=0.123s` remains substantial. It is intentional structural work, not renderer work.

2. Per-iteration CUDA-to-CPU metric extraction serializes the hot path.
   - Each view performs `float(mse.detach().cpu())`.
   - Every iteration also performs `float(total.detach().cpu())`.
   - These force the CPU to wait for queued CUDA kernels, preventing useful overlap and also feed the uncertainty loss through a recreated CPU scalar. Metrics should remain device tensors until a logging cadence requires host values.

3. ADC is the dominant periodic spike.
   - It runs every 100 iterations.
   - Clone, split, and prune each rebuild full Gaussian parameter tensors. For every parameter group, Adam's moment tensors are allocated and copied to preserve state.
   - At roughly 190k Gaussians, the measured density stage is 4.644 s at iteration 10,000. This is not CPU-only ADC; it is GPU tensor allocation/copy plus GPU synchronization caused by report scalar extraction. It also increases memory traffic as SH/optimizer state grows.

4. Surface maintenance is a separate periodic global scan.
   - At the configured 1,000-iteration cadence it refreshes UV values for all certain Gaussians and evaluates residuals for every patch.
   - Local correction is rare, but correction triggers a voxel rebuild for the failed patch. Voxel normal estimation uses chunked GPU cdist/SVD, while mixed-resolution adjacency and connected-component labelling explicitly transfer region bounds/normals to CPU and use Python loops.
   - It correctly does not affect normal iterations, but it compounds ADC and streaming at 1,000-iteration boundaries.

5. Full snapshot streaming is expensive at configured checkpoints.
   - `STREAM_MAX_GAUSSIANS = 0` copies positions, scaling, rotations, opacity, and color for every Gaussian from GPU to CPU; the worker only serializes after that transfer.
   - NURBS and voxel payloads copy every patch/control grid and all voxel-region arrays as well.
   - At iteration 10,000, the same snapshot is sent twice: once because it matches the stream schedule and once from final forced streaming. This affects completion/checkpoint latency rather than the steady iteration time.

6. Timing instrumentation itself synchronizes CUDA, but only at the 100-iteration logging cadence.
   - It is necessary for stage-accurate numbers and is not the source of the 0.188 s average.

## Priority

1. Remove hot-path host scalar extraction and keep MSE/loss aggregation on CUDA; materialize metrics only when progress or snapshot metadata needs them.
2. Rework ADC tensor/Adam-state growth and prune into a single shape transaction per ADC interval, avoiding clone -> full rebuild -> split -> full rebuild -> prune -> full rebuild.
3. Decouple snapshot capture from training with a bounded pinned-memory copy policy, and avoid the duplicate final snapshot.
4. Keep the current persistent NURBS/voxel lifecycle; optimize maintenance only at its explicit cadence rather than weakening the structural objective.

No performance behavior was changed by this audit.
