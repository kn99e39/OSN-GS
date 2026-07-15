"""Ground-truth NURBS surface metrics, separated by construction concern.

The synthetic scenes expose their true surface and true patch topology
(``scenes.py``), so the generated NURBS can be scored against ground truth on
three independent concerns instead of a single conflated residual:

1. Surface Fitting Accuracy -- how geometrically close the generated surface is
   to the true surface where both exist (bidirectional / Chamfer distance).
2. Surface Support -- whether the surface exists in the right places: coverage
   (holes / under-support over the observed region) and extrapolation
   (surface hallucinated where there is no input data / over-support).
3. Patch Topology -- whether the number and boundaries of the generated patches
   match ground truth (Adjusted Rand Index of the per-Gaussian patch labels).

Distances are in scene units (xy lives in ``[-1, 1]``); support thresholds are
expressed as multiples of the median input nearest-neighbour spacing so they
adapt to point density.
"""

from __future__ import annotations

from typing import Any

import torch

from .ground_truth import gt_surface_points, observed_gt_surface_points
from .scenes import SyntheticGaussianScene

# Support thresholds as multiples of the median input NN spacing.
_OBSERVED_RADIUS_FACTOR = 3.0  # true-surface area within this radius of data counts as "observed"
_SUPPORT_TAU_FACTOR = 2.5     # coverage hole / extrapolation distance threshold


def _min_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """For each row of ``a`` the Euclidean distance to the nearest row of ``b``."""

    if a.numel() == 0 or b.numel() == 0:
        return torch.full((a.shape[0],), float("inf"))
    out = []
    for start in range(0, a.shape[0], chunk):
        out.append(torch.cdist(a[start : start + chunk], b).min(dim=1).values)
    return torch.cat(out)


def _median_nn_spacing(xy: torch.Tensor) -> float:
    if xy.shape[0] < 2:
        return 1.0
    d = torch.cdist(xy, xy)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median())


def sample_generated_surface(state: Any, per_patch: int = 28, respect_trim: bool = True) -> torch.Tensor:
    """Dense samples on every generated patch's UV domain.

    When ``respect_trim`` is set and a patch carries a UV support (trim) mask,
    only samples inside the supported region are returned -- so the measured
    surface is the trimmed surface the renderer would actually draw.
    """

    lin = torch.linspace(0.0, 1.0, per_patch)
    u, v = torch.meshgrid(lin, lin, indexing="ij")
    uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)
    samples = []
    for patch in state.surface_patches:
        uv_dev = uv.to(patch.control_grid.device)
        if respect_trim and getattr(patch, "uv_support_mask", None) is not None:
            uv_dev = uv_dev[patch.support(uv_dev)]
        if uv_dev.shape[0] == 0:
            continue
        samples.append(patch.evaluate(uv_dev).detach().cpu())
    return torch.cat(samples, dim=0) if samples else torch.empty((0, 3))


def _adjusted_rand_index(a: torch.Tensor, b: torch.Tensor) -> float:
    """Permutation-invariant agreement between two clusterings of the same points."""

    n = int(a.shape[0])
    if n == 0:
        return 1.0
    _, ai = torch.unique(a, return_inverse=True)
    _, bi = torch.unique(b, return_inverse=True)
    ka, kb = int(ai.max()) + 1, int(bi.max()) + 1
    contingency = torch.zeros(ka * kb)
    contingency.scatter_add_(0, (ai * kb + bi).long(), torch.ones(n))
    contingency = contingency.reshape(ka, kb)

    comb2 = lambda x: x * (x - 1.0) / 2.0
    sum_c = comb2(contingency).sum()
    sum_a = comb2(contingency.sum(dim=1)).sum()
    sum_b = comb2(contingency.sum(dim=0)).sum()
    total = comb2(torch.tensor(float(n)))
    if total <= 0:
        return 1.0
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if float(denom) == 0.0:
        return 1.0  # both label sets are a single cluster and agree
    return float((sum_c - expected) / denom)


def ground_truth_metrics(scene: SyntheticGaussianScene, state: Any, grid_n: int = 128) -> dict[str, Any]:
    """Score the generated NURBS against ground truth on the three concerns."""

    input_pts = scene.points.detach().cpu()
    input_xy = input_pts[:, :2]
    spacing = _median_nn_spacing(input_xy)
    observed_radius = _OBSERVED_RADIUS_FACTOR * spacing
    support_tau = _SUPPORT_TAU_FACTOR * spacing

    gen = sample_generated_surface(state)
    gt_full = gt_surface_points(scene, grid_n)
    obs_gt = observed_gt_surface_points(scene, grid_n, radius=observed_radius)

    # --- 1. Fitting accuracy: generated -> true surface (precision), over the
    #        true domain so extrapolation is scored separately under support. ---
    inside = (gen[:, :2].abs() <= 1.0 + 1e-6).all(dim=1)
    gen_inside = gen[inside]
    acc = _min_dist(gen_inside, gt_full)
    # --- completeness: observed true surface -> generated (recall). ---
    comp = _min_dist(obs_gt, gen)

    # --- 2. Support. ---
    extrap = _min_dist(gen, input_pts)  # generated -> nearest input point (3D)

    # --- 3. Topology: per-Gaussian generated vs ground-truth patch labels. ---
    gen_labels = state.model.cluster_ids.detach().cpu().clamp_min(0)
    gt_labels = scene.gt_patch_label(input_xy)
    ari = _adjusted_rand_index(gen_labels, gt_labels)

    def _rms(x: torch.Tensor) -> float:
        return float(x.square().mean().sqrt()) if x.numel() else float("nan")

    def _frac_over(x: torch.Tensor, tau: float) -> float:
        return float((x > tau).float().mean()) if x.numel() else float("nan")

    return {
        # 1. Surface Fitting Accuracy
        "accuracy_rms": _rms(acc),
        "accuracy_p95": float(torch.quantile(acc, 0.95)) if acc.numel() else float("nan"),
        "accuracy_max": float(acc.max()) if acc.numel() else float("nan"),
        "completeness_rms": _rms(comp),
        "chamfer_rms": 0.5 * (_rms(acc) + _rms(comp)),
        # 2. Surface Support
        "support_coverage_uncovered_fraction": _frac_over(comp, support_tau),
        "support_extrapolation_fraction": _frac_over(extrap, support_tau),
        "support_observed_gt_samples": int(obs_gt.shape[0]),
        "support_threshold": float(support_tau),
        # 3. Patch Topology
        "topology_gen_patch_count": len(state.surface_patches),
        "topology_gt_patch_count": int(scene.gt_patch_count),
        "topology_patch_count_delta": len(state.surface_patches) - int(scene.gt_patch_count),
        "topology_patch_count_match": len(state.surface_patches) == int(scene.gt_patch_count),
        "topology_label_ari": ari,
    }
