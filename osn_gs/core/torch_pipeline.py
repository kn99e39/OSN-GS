from __future__ import annotations

"""Torch-based OSN-GS visible surface reconstruction pipeline."""

import hashlib
import math
import time

from dataclasses import dataclass, field
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.gaussian.torch_surface_ownership import (
    SURFACE_OWNER_OCCLUDED_CHART,
    SURFACE_OWNER_UNASSIGNED,
    SURFACE_OWNER_VISIBLE_PATCH,
    UNASSIGNED_OWNER_ID,
    is_visible_patch_owned,
)
from osn_gs.surface.torch_density_preserving_representative_selection import (
    RepresentativeSelectionResult,
    select_density_preserving_representatives,
)
from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput
from osn_gs.surface.torch_full_neighborhood_evidence import (
    FullNeighborhoodEvidence,
    assign_nearest_representative,
    compute_full_neighborhood_evidence,
)
from osn_gs.surface.torch_gaussian_covariance_frame import (
    GaussianCovarianceFrame,
    covariance_from_scale_rotation,
    extract_covariance_frame,
)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    IntrinsicReliabilityResult,
    evaluate_intrinsic_reliability,
    evaluate_structural_reliability,
    evaluate_structural_reliability_from_full_evidence,
)
from osn_gs.surface.torch_nurbs import (
    TorchCurveSet,
    TorchNURBSSurface,
    project_torch_points_to_nurbs,
)
from osn_gs.surface.torch_visible_surface_construction import (
    VisibleSurfaceConstructionResult,
    construct_visible_nurbs_from_gaussians,
)
from osn_gs.surface.torch_region_owned_full_evidence import (
    collect_region_owned_evidence,
    fit_region_owned_full_evidence_patch,
)
from osn_gs.utils.torch_ops import require_torch


def _representative_knn_spacing(positions: Any, k: int = 8) -> Any:
    """Worklog 33: REPRESENTATIVE GRAPH SCALE, G1. Each representative's own
    median distance to its ``k`` nearest OTHER representatives -- a pure
    function of ``positions``, provably (and verified) exactly rigid-
    rotation/translation/uniform-scale invariant when the representative set
    itself is held fixed. Bounded-k, O(M^2) in the representative count only
    (never the full cloud) -- the same complexity class as the existing
    manifold-affinity kNN candidate search this replaces the scale for.
    """
    torch = require_torch()
    count = int(positions.shape[0])
    if count == 0:
        return positions.new_empty((0,))
    if count == 1:
        # No representative-to-representative spacing exists. Return the finite floor.
        return torch.full((1,), 1e-9, dtype=positions.dtype, device=positions.device)
    k = min(k, count - 1)
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    knn_distances, _ = torch.topk(distances, k=k, largest=False, dim=1)
    return knn_distances.median(dim=1).values.clamp_min(1e-9)


def _slice_covariance_frame(frame: GaussianCovarianceFrame, indices: Any) -> GaussianCovarianceFrame:
    """Index every field of a batched :class:`GaussianCovarianceFrame` (incl. the plain str tuple)."""
    idx_list = indices.detach().cpu().tolist()
    return GaussianCovarianceFrame(
        eigenvalues=frame.eigenvalues[indices],
        tangent_u=frame.tangent_u[indices],
        tangent_v=frame.tangent_v[indices],
        normal_candidate=frame.normal_candidate[indices],
        planarity=frame.planarity[indices],
        elongation=frame.elongation[indices],
        isotropy=frame.isotropy[indices],
        shape_class=tuple(frame.shape_class[i] for i in idx_list),
        tangent_major_scale=frame.tangent_major_scale[indices],
        tangent_minor_scale=frame.tangent_minor_scale[indices],
        normal_thickness=frame.normal_thickness[indices],
        equivalent_tangent_scale=frame.equivalent_tangent_scale[indices],
        footprint_area=frame.footprint_area[indices],
    )


def _slice_intrinsic_reliability(intrinsic: IntrinsicReliabilityResult, indices: Any) -> IntrinsicReliabilityResult:
    idx_list = indices.detach().cpu().tolist()
    return IntrinsicReliabilityResult(
        intrinsic_class=tuple(intrinsic.intrinsic_class[i] for i in idx_list),
        conditioning_score=intrinsic.conditioning_score[indices],
        planar_likelihood=intrinsic.planar_likelihood[indices],
        needle_likelihood=intrinsic.needle_likelihood[indices],
        isotropic_likelihood=intrinsic.isotropic_likelihood[indices],
        scale_validity=intrinsic.scale_validity[indices],
        reasons=tuple(intrinsic.reasons[i] for i in idx_list),
        config=intrinsic.config,
    )


@dataclass
class CanonicalConstructionWithEvidence:
    """Bundles a density-preserving-representative canonical construction (worklog 129)."""

    construction: VisibleSurfaceConstructionResult
    selection: RepresentativeSelectionResult
    evidence: FullNeighborhoodEvidence
    representative_indices: Any  # (M,) long, indices into the input full cloud
    representative_stable_ids: tuple[Any, ...]
    nearest_representative_index: Any | None = None  # (N,) long, worklog 130: reused by propagation, never recomputed
    # Worklog 67: region-owned full-cloud evidence re-fit, ADDITIVE only --
    # never read by region formation/boundary ordering/chart eligibility.
    # Keyed by (chart_type, source_region_id). Empty when construction was
    # not downsampled (representatives already ARE the full cloud, so there
    # is no extra evidence to recover).
    region_owned_full_evidence_fits: dict[tuple[str, int], Any] = field(default_factory=dict)

@dataclass
class TorchPipelineConfig:
    """Canonical visible-surface construction and Gaussian initialization."""

    sh_degree: int = 3
    # Worklog 64: which tensors seed a *newly created* visible/certain
    # Gaussian's TRAINABLE _scaling/_rotation. "baseline_compatible"
    # (default) matches Graphdeco's gaussian_model.create_from_pcd exactly --
    # isotropic scale from distCUDA2-equivalent nearest-neighbor spacing,
    # identity rotation -- so ADC/anisotropy dynamics start from the same
    # place baseline's own thresholds were tuned against. "covariance_knn"
    # keeps the pre-worklog-64 local-PCA planar-surfel init (tangent axes
    # ~25x the normal axis by construction) as an explicit experimental
    # option; it is NEVER selected implicitly. Neither mode touches the
    # SEPARATE local-PCA covariance always used for canonical visible
    # surface construction/reliability (`_canonical_initial_covariance`
    # below) -- that is surface-reconstruction geometry, not Gaussian
    # training init, and this flag intentionally does not reach it.
    gaussian_initialization_mode: str = "baseline_compatible"
    # Which PRIMITIVE the trainable model uses.
    #   "gaussian_3d" (default) -- volumetric 3D Gaussian, unchanged OSN-GS.
    #   "surfel_2d"             -- 2DGS planar surface element
    #                              (arXiv:2403.17888v3 sec. 4.1). Selects
    #                              `TorchGaussianSurfelModel` (two tangent
    #                              scales, no normal-direction scale) and the
    #                              official `create_from_pcd` initialization
    #                              (`_surfel_compatible_scale_rotation`).
    # `gaussian_initialization_mode` does not apply to "surfel_2d": a surfel
    # has no third scale for either of that flag's modes to fill in, so the
    # 2DGS branch always uses the official 2DGS initialization instead.
    primitive: str = "gaussian_3d"
    canonical_covariance_knn: int = 8
    canonical_construction_max_points: int = 2048
    covariance_knn_chunk_size: int = 0
    covariance_min_scale: float = 1e-4
    covariance_max_scale_ratio: float = 0.05
    covariance_scale_multiplier: float = 1.0
    surface_projection_chunk_size: int = 65536
    surface_projection_iterations: int = 4
    surface_trim_resolution: int = 24
    surface_trim_dilation: int = 1

@dataclass
class TorchPipelineState:
    """Canonical visible-surface state carried throughout training."""

    model: TorchGaussianModel
    base_curves: TorchCurveSet
    occlusion_curves: TorchCurveSet
    surface: TorchNURBSSurface | None
    surface_patches: list[TorchNURBSSurface]
    # Construction-time `SurfaceRegionCandidate.region_confidence`, one entry
    # per `surface_patches` index (NOT the per-Gaussian uncertain-Gaussian
    # `model.get_uncertain_confidence` -- see docs/agent_memory on the
    # uncertain_confidence rename for why these are kept separate).
    surface_patch_confidence: tuple[float, ...] = field(default_factory=tuple)
    visible_surface_construction: VisibleSurfaceConstructionResult | None = None
    visible_nurbs_state: str = "materialized"
    visible_nurbs_coverage_semantics: str = "reliable_core_only"
    visible_nurbs_adc_version: int = 0
    visible_nurbs_source_fingerprint: str = ""
    visible_nurbs_last_attempt_iteration: int = -1
    visible_nurbs_last_failure: dict[str, Any] = field(default_factory=dict)
    visible_nurbs_event_history: list[dict[str, Any]] = field(default_factory=list)
    surface_optimizer: Any | None = None
    surface_patch_residuals: dict[int, float] = field(default_factory=dict)
    surface_bad_checks: dict[int, int] = field(default_factory=dict)
    surface_topology_version: int = 0
    iteration: int = 0
    last_loss: float = 0.0
    last_psnr: float = 0.0

class TorchOSNGSPipeline:
    """Builds canonical visible NURBS state used by the trainer."""

    def __init__(self, config: TorchPipelineConfig, device: str = "cuda") -> None:
        mode = str(config.gaussian_initialization_mode).strip().lower()
        if mode not in {"baseline_compatible", "covariance_knn"}:
            raise ValueError(
                "gaussian_initialization_mode must be baseline_compatible or covariance_knn, "
                f"got {config.gaussian_initialization_mode!r}"
            )
        config.gaussian_initialization_mode = mode
        primitive = str(config.primitive).strip().lower()
        if primitive not in {"gaussian_3d", "surfel_2d"}:
            raise ValueError(
                f"primitive must be gaussian_3d or surfel_2d, got {config.primitive!r}"
            )
        config.primitive = primitive
        self.config = config
        self.device = device

    @property
    def is_surfel(self) -> bool:
        """True when this pipeline builds 2DGS planar surface elements."""

        return self.config.primitive == "surfel_2d"

    def _new_model(self) -> TorchGaussianModel:
        """Construct the primitive container this pipeline is configured for."""

        model_cls = TorchGaussianSurfelModel if self.is_surfel else TorchGaussianModel
        return model_cls(sh_degree=self.config.sh_degree, device=self.device)

    @staticmethod
    def _patch_confidence_from_regions(
        patch_count: int,
        region_to_patch: dict[Any, int],
        region_by_id: dict[Any, Any],
    ) -> tuple[float, ...]:
        """Per-patch `SurfaceRegionCandidate.region_confidence`, aligned with
        the `surface_patches`/`patch_id` index space `region_to_patch` maps
        into. This is construction-time evidence confidence, unrelated to
        `model.get_uncertain_confidence`."""

        confidence = [0.0] * patch_count
        for region_id, patch_id in region_to_patch.items():
            region = region_by_id.get(region_id)
            if region is not None:
                confidence[patch_id] = float(region.region_confidence)
        return tuple(confidence)

    def initialize(
        self,
        points: Any,
        colors: Any,
        *,
        covariance_scales: Any | None = None,
        covariance_rotations: Any | None = None,
    ) -> TorchPipelineState:
        """Build the trainable state from observed points and colors.

        This Stage 1 path fits a visible parametric surface only. It does not
        extrapolate occlusion curves, sample occluded regions, or append
        uncertain Gaussians.
        """

        torch = require_torch()
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)

        return self._initialize_canonical(
            points,
            colors,
            covariance_scales=covariance_scales,
            covariance_rotations=covariance_rotations,
        )

    def initialize_deferred(
        self,
        points: Any,
        colors: Any,
        *,
        state_label: str = "unavailable_until_adc",
        covariance_scales: Any | None = None,
        covariance_rotations: Any | None = None,
    ) -> TorchPipelineState:
        """Initialize Gaussian training without materializing a visible NURBS."""

        torch = require_torch()
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        # NOTE: unlike `_initialize_canonical`, `gaussian_initialization_mode`
        # intentionally does NOT reach the model init here. The deferred
        # ("adc_post_commit"/"disabled") schedule materializes no surface at
        # initialize time -- `reconstruct_visible_after_adc` builds the FIRST
        # canonical surface later from whatever `model.get_scaling`/
        # `get_rotation` currently hold (`covariance_from_scale_rotation`).
        # Before any real ADC event, image loss alone has not meaningfully
        # rotated a fresh Gaussian, so the local-PCA planar-surfel frame
        # computed here is the only source of usable orientation evidence for
        # that first reconstruction. Feeding it an isotropic/identity-rotation
        # init instead would starve the deferred schedule's own surface
        # bootstrap of evidence -- a surface-reconstruction change this
        # worklog's task explicitly forbids. This is a deliberate, narrower
        # scope than `_initialize_canonical`, not an oversight.
        if covariance_scales is not None or covariance_rotations is not None:
            scales, rotations, _ = self._canonical_initial_covariance(
                points,
                covariance_scales=covariance_scales,
                covariance_rotations=covariance_rotations,
            )
        else:
            construction_indices = self._canonical_construction_indices(points)
            construction_points = points[construction_indices]
            sampled = int(construction_indices.numel()) < int(points.shape[0])
            sample_scales, sample_rotations, _ = self._canonical_initial_covariance(
                construction_points,
                covariance_scales=None,
                covariance_rotations=None,
            )
            if sampled:
                nearest = self._nearest_canonical_sample_indices(points, construction_points)
                density_scale = (
                    float(construction_points.shape[0]) / float(points.shape[0])
                ) ** 0.5
                scales = (sample_scales[nearest] * density_scale).clamp_min(
                    float(self.config.covariance_min_scale)
                )
                rotations = sample_rotations[nearest]
            else:
                scales = sample_scales
                rotations = sample_rotations
        if self.is_surfel:
            # Same reasoning as `_initialize_canonical`: the local-PCA
            # planar-surfel frame above describes a 3D covariance and cannot
            # seed a two-scale surfel. The deferred schedule's own surface
            # bootstrap is not applicable to this branch either, since the
            # 2DGS arm is run with the `initialize` schedule.
            scales, rotations = self._surfel_compatible_scale_rotation(points)
        count = int(points.shape[0])
        model = self._new_model()
        model.initialize(
            positions=points,
            colors=colors,
            opacities=torch.full((count, 1), 0.12, dtype=torch.float32, device=points.device),
            scales=scales,
            rotations=rotations,
            uncertain_mask=torch.zeros((count,), dtype=torch.bool, device=points.device),
            surface_uv=torch.zeros((count, 2), dtype=torch.float32, device=points.device),
            cluster_ids=torch.full((count,), -1, dtype=torch.long, device=points.device),
            uncertain_confidence=torch.ones((count, 1), dtype=torch.float32, device=points.device),
        )
        empty_curves = self._empty_occlusion_curves(points)
        return TorchPipelineState(
            model=model,
            base_curves=empty_curves,
            occlusion_curves=self._empty_occlusion_curves(points),
            surface=None,
            surface_patches=[],
            visible_nurbs_state=state_label,
            visible_nurbs_coverage_semantics="reliable_core_only",
        )

    def _initialize_canonical(
        self,
        points: Any,
        colors: Any,
        *,
        covariance_scales: Any | None = None,
        covariance_rotations: Any | None = None,
    ) -> TorchPipelineState:
        """Build training state exclusively through canonical visible construction."""
        torch = require_torch()
        construction_indices = self._canonical_construction_indices(points)
        construction_points = points[construction_indices]
        sampled = int(construction_indices.numel()) < int(points.shape[0])
        if covariance_scales is not None or covariance_rotations is not None:
            scales, rotations, covariance = self._canonical_initial_covariance(
                points,
                covariance_scales=covariance_scales,
                covariance_rotations=covariance_rotations,
            )
            construction_covariance = covariance[construction_indices]
        else:
            sample_scales, sample_rotations, construction_covariance = (
                self._canonical_initial_covariance(
                    construction_points,
                    covariance_scales=None,
                    covariance_rotations=None,
                )
            )
            if sampled:
                nearest = self._nearest_canonical_sample_indices(
                    points, construction_points
                )
                # Local PCA is evaluated on the bounded topology sample. Copy
                # its frame to every Gaussian and compensate the scale for the
                # denser full point cloud, assuming a locally two-dimensional
                # visible surface.
                density_scale = (
                    float(construction_points.shape[0]) / float(points.shape[0])
                ) ** 0.5
                scales = (
                    sample_scales[nearest] * density_scale
                ).clamp_min(float(self.config.covariance_min_scale))
                rotations = sample_rotations[nearest]
            else:
                scales = sample_scales
                rotations = sample_rotations
        construction = construct_visible_nurbs_from_gaussians(
            construction_points,
            covariance=construction_covariance,
            stable_ids=tuple(
                int(item)
                for item in construction_indices.detach().cpu().tolist()
            ),
        )
        materialized = construction.materialized_visible_nurbs_surfaces
        if not materialized:
            raise RuntimeError(
                "Canonical visible-surface construction produced no materialized NURBS "
                f"(state={construction.construction_state!r}, "
                f"regions={construction.diagnostic_summary['region_count']}, "
                f"components={construction.diagnostic_summary['boundary_component_count']}). "
                "No legacy or voxel fallback is available."
            )
        surface_patches = [
            item.surface for item in materialized if item.surface is not None
        ]
        region_to_patch = {
            item.input.source_region_id: patch_id
            for patch_id, item in enumerate(materialized)
        }
        count = int(points.shape[0])
        cluster_ids = torch.full((count,), -1, dtype=torch.long, device=points.device)
        region_by_id = {
            region.region_id: region
            for region in construction.surface_regions.regions
        }
        for region_id, patch_id in region_to_patch.items():
            region = region_by_id.get(region_id)
            if region is None:
                continue
            indices = torch.tensor(
                [int(item) for item in region.member_ids],
                dtype=torch.long,
                device=points.device,
            )
            cluster_ids[indices] = patch_id
        patch_confidence = self._patch_confidence_from_regions(
            len(surface_patches), region_to_patch, region_by_id
        )

        cluster_ids = self._propagate_canonical_patch_ids(
            points, construction_indices, cluster_ids
        )

        surface_uv = torch.zeros((count, 2), dtype=torch.float32, device=points.device)
        for patch_id, patch in enumerate(surface_patches):
            indices = torch.nonzero(cluster_ids == patch_id, as_tuple=False).reshape(-1)
            if int(indices.numel()) == 0:
                continue
            surface_uv[indices] = project_torch_points_to_nurbs(
                points[indices],
                patch,
                iterations=int(self.config.surface_projection_iterations),
                chunk_size=int(self.config.surface_projection_chunk_size),
            )

        # `scales`/`rotations` above are the local-PCA planar-surfel frame --
        # always required to build `construction_covariance` for visible
        # surface construction (untouched by this flag). The Gaussian
        # MODEL's own trainable init is a separate decision: an explicit
        # covariance override always wins (ground truth, e.g. synthetic
        # fixtures); otherwise `gaussian_initialization_mode` picks between
        # reusing that same planar-surfel frame ("covariance_knn",
        # experimental) or Graphdeco-equivalent isotropic init
        # ("baseline_compatible", default).
        if self.is_surfel:
            # 2DGS has no third scale, so neither an explicit 3D covariance
            # override nor `gaussian_initialization_mode` can describe its
            # trainable init. The official `create_from_pcd` is used instead
            # -- see `_surfel_compatible_scale_rotation`.
            model_scales, model_rotations = self._surfel_compatible_scale_rotation(points)
        elif covariance_scales is not None or covariance_rotations is not None:
            model_scales, model_rotations = scales, rotations
        elif self.config.gaussian_initialization_mode == "covariance_knn":
            model_scales, model_rotations = scales, rotations
        else:
            model_scales, model_rotations = self._baseline_compatible_scale_rotation(points)

        model = self._new_model()
        model.initialize(
            positions=points,
            colors=colors,
            opacities=torch.full((count, 1), 0.12, dtype=torch.float32, device=points.device),
            scales=model_scales,
            rotations=model_rotations,
            uncertain_mask=torch.zeros((count,), dtype=torch.bool, device=points.device),
            surface_uv=surface_uv,
            cluster_ids=cluster_ids,
            uncertain_confidence=torch.ones((count, 1), dtype=torch.float32, device=points.device),
        )
        self._assign_uv_support_masks(model, surface_patches)
        empty_curves = self._empty_occlusion_curves(points)
        print(
            "OSN-GS canonical visible construction: "
            f"state={construction.construction_state} "
            f"regions={construction.diagnostic_summary['region_count']} "
            f"patches={len(surface_patches)} "
            f"assigned={int((cluster_ids >= 0).sum())}/{count}",
            flush=True,
        )
        return TorchPipelineState(
            model=model,
            base_curves=empty_curves,
            occlusion_curves=self._empty_occlusion_curves(points),
            surface=surface_patches[0],
            surface_patches=surface_patches,
            surface_patch_confidence=patch_confidence,
            visible_surface_construction=construction,
            visible_nurbs_state="materialized",
            visible_nurbs_coverage_semantics="reliable_core_only",
            visible_nurbs_adc_version=1,
        )

    def _canonical_construction_indices(self, points: Any) -> Any:
        """Deterministically voxel-sample large point sets for O(N^2) topology stages."""
        torch = require_torch()
        count = int(points.shape[0])
        budget = max(16, int(self.config.canonical_construction_max_points))
        if count <= budget:
            return torch.arange(count, dtype=torch.long, device=points.device)
        resolution = max(2, int(math.ceil(budget ** 0.5)))
        minimum = points.amin(dim=0)
        span = (points.amax(dim=0) - minimum).clamp_min(1e-9)
        normalized = (points - minimum) / span
        cells = torch.floor(normalized * resolution).long().clamp(0, resolution - 1)
        keys = cells[:, 0] * resolution * resolution + cells[:, 1] * resolution + cells[:, 2]
        cell_centers = (cells.to(points.dtype) + 0.5) / float(resolution)
        center_distance = (normalized - cell_centers).square().sum(dim=1)
        distance_order = torch.argsort(center_distance, stable=True)
        order = distance_order[
            torch.argsort(keys[distance_order], stable=True)
        ]
        ordered_keys = keys[order]
        first = torch.ones_like(ordered_keys, dtype=torch.bool)
        first[1:] = ordered_keys[1:] != ordered_keys[:-1]
        selected = order[first]
        if int(selected.numel()) > budget:
            positions = torch.linspace(
                0, int(selected.numel()) - 1, budget, device=points.device
            ).round().long()
            selected = selected[positions]
        return selected.sort().values

    def _construct_canonical_with_full_evidence(
        self,
        points: Any,
        covariance: Any,
        opacity: Any,
        stable_ids: Any,
    ) -> CanonicalConstructionWithEvidence:
        """Density-preserving representative selection + full-neighborhood reliability (worklog 129).

        ``points``/``covariance``/``opacity``/``stable_ids`` describe the FULL
        eligible observed cloud (already-learned per-Gaussian covariance --
        this is the O(N) path, cheap even at real ADC-trained scale). Topology
        (affinity/region formation/boundary) still runs only on the bounded
        representative subset; only representative SELECTION and CONTEXTUAL
        RELIABILITY now see the full cloud's actual density.
        """
        torch = require_torch()
        frame_full = extract_covariance_frame(covariance)
        intrinsic_full = evaluate_intrinsic_reliability(frame_full)
        stable_ids_list = list(stable_ids)
        selection = select_density_preserving_representatives(
            points,
            frame_full,
            opacity,
            stable_ids_list,
            max_points=int(self.config.canonical_construction_max_points),
        )
        rep_indices = selection.representative_indices
        rep_points = points[rep_indices]
        rep_covariance = covariance[rep_indices]
        rep_frame = _slice_covariance_frame(frame_full, rep_indices)
        rep_stable_ids = tuple(stable_ids_list[i] for i in rep_indices.detach().cpu().tolist())
        downsampled = int(rep_indices.numel()) != int(points.shape[0])

        precomputed_assignment = (
            assign_nearest_representative(points, rep_points) if downsampled else None
        )
        # Worklog 32: LOCAL EVIDENCE SCALE -- a per-representative estimate of
        # the true local full-cloud point spacing, DISTINCT from a single
        # representative Gaussian's own tangent_major_scale (worklog 31 found
        # the latter is ~8x too small on real long-horizon-trained data,
        # collapsing the local-radius bound and tangent-residual denominator
        # alike). Derived from the SAME voxel cell geometry and per-
        # representative ``source_count`` selection already computed --
        # cbrt(cell_volume / source_count) approximates the typical spacing
        # between full-cloud members inside that representative's own
        # selection-time cell, without any new O(N) or O(N^2) pass and
        # without depending on tangent_major_scale at all.
        local_evidence_scale = None
        if downsampled:
            budget = max(16, int(self.config.canonical_construction_max_points))
            resolution = max(2, int(math.ceil(budget ** 0.5)))
            # Rotation/translation/uniform-scale-invariant characteristic
            # scene length: 2 * sqrt(trace(cov(points)) / 3), an isotropic
            # RMS-radius-like measure. An axis-aligned bounding-box span
            # (what the voxel grid itself uses for CANDIDATE GROUPING, which
            # is fine to be rotation-sensitive per its own documented
            # invariance carve-out) is NOT safe to reuse here, because this
            # scale feeds a RELIABILITY decision -- using it directly broke
            # `region_count` rigid-rotation invariance (caught by existing
            # tests). trace(R cov R^T) == trace(cov) for any orthogonal R,
            # so this stays invariant under rotation while still scaling
            # linearly with uniform scale and ignoring translation.
            centered = points - points.mean(dim=0, keepdim=True)
            variance_trace = (centered.square().sum(dim=0) / max(int(points.shape[0]), 1)).sum().clamp_min(1e-12)
            characteristic_length = 2.0 * torch.sqrt(variance_trace / 3.0)
            cell_volume = float((characteristic_length / resolution).clamp_min(1e-9) ** 3)
            source_counts = torch.tensor(
                [rep.source_count for rep in selection.representatives],
                dtype=points.dtype, device=points.device,
            )
            local_evidence_scale = (cell_volume / source_counts.clamp_min(1)).pow(1.0 / 3.0)
        # Worklog 33: REPRESENTATIVE GRAPH SCALE, G1 -- this representative's
        # own median distance to its k nearest OTHER representatives. Purely
        # a function of ``rep_points`` (Euclidean distances), so it is
        # EXACTLY rigid-rotation/translation/uniform-scale invariant when the
        # representative SET is held fixed -- verified directly (worklog 33
        # frozen-representative test, zero relation mismatches on real
        # 3k/5k/10k checkpoints). Worklog 32's three earlier graph-scale
        # attempts were rejected only because the END-TO-END test they were
        # checked against also re-runs representative SELECTION (an
        # already-documented, separately-accepted non-invariance of the
        # axis-aligned voxel grid) -- this conflated selection perturbation
        # with estimator correctness. Real 3k snapshot: replaces
        # `tangent_major_scale`-based same_surface edge count of 11 with
        # 2125 (frozen test, same representative set).
        representative_graph_scale = _representative_knn_spacing(rep_points)
        evidence = compute_full_neighborhood_evidence(
            points, frame_full, opacity, intrinsic_full, rep_points, rep_frame, rep_stable_ids,
            precomputed_assignment=precomputed_assignment,
            local_evidence_scale=local_evidence_scale,
        )
        if not downsampled:
            # No downsampling occurred: the representative set IS the full
            # cloud, so a representative's full-cloud "Voronoi cell" would
            # degenerate to itself (support count ~1) -- worse evidence than
            # the plain representative-only k-NN path, not better. Full-
            # neighborhood evidence only helps once representatives are a
            # bounded subset of a denser full cloud; fall back to the
            # unchanged k-NN contextual path for the small-scene case. The
            # same degeneracy argument applies to the continuation shell
            # (worklog 130): a representative's shell would only ever contain
            # itself, so it is skipped here too.
            reliability = evaluate_structural_reliability(rep_points, rep_frame)
            construction = construct_visible_nurbs_from_gaussians(
                rep_points, covariance=rep_covariance, stable_ids=rep_stable_ids, reliability=reliability,
                candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
            )
        else:
            reliability = evaluate_structural_reliability_from_full_evidence(rep_frame, evidence)
            nearest_representative_index, _distance = precomputed_assignment
            continuation_input = ContinuationShellInput(
                full_positions=points,
                full_frame=frame_full,
                full_intrinsic=intrinsic_full,
                full_opacity=opacity,
                full_stable_ids=stable_ids_list,
                nearest_representative_index=nearest_representative_index,
                representative_mean_spacing=evidence.mean_spacing,
            )
            construction = construct_visible_nurbs_from_gaussians(
                rep_points, covariance=rep_covariance, stable_ids=rep_stable_ids, reliability=reliability,
                continuation_input=continuation_input,
                candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
            )
        bundle = CanonicalConstructionWithEvidence(
            construction=construction,
            selection=selection,
            evidence=evidence,
            representative_indices=rep_indices,
            representative_stable_ids=rep_stable_ids,
            nearest_representative_index=precomputed_assignment[0] if precomputed_assignment is not None else None,
        )
        if downsampled:
            bundle.region_owned_full_evidence_fits = self._collect_region_owned_full_evidence_fits(
                points, covariance, stable_ids_list, bundle,
            )
        return bundle

    def _collect_region_owned_full_evidence_fits(
        self, points: Any, covariance: Any, full_stable_ids: list[Any], bundle: CanonicalConstructionWithEvidence,
    ) -> dict[tuple[str, int], Any]:
        """Worklog 67: additive-only region-owned full-evidence recovery.

        Never touches ``bundle.construction`` -- region formation, boundary
        ordering, and chart eligibility (``materialized_visible_nurbs_surfaces``
        / ``materialized_parametric_chart_surfaces`` themselves) are computed
        BEFORE this runs and are read here only, never written.
        """

        torch = require_torch()
        materialized_items: list[tuple[str, Any]] = [
            ("physical", item) for item in bundle.construction.materialized_visible_nurbs_surfaces if item.surface is not None
        ] + [
            ("parametric", item) for item in bundle.construction.materialized_parametric_chart_surfaces if item.surface is not None
        ]
        if not materialized_items:
            return {}

        stable_to_local = {stable_id: local for local, stable_id in enumerate(bundle.representative_stable_ids)}
        cluster_ids_by_representative = torch.full(
            (len(bundle.representative_stable_ids),), -1, dtype=torch.long, device=points.device,
        )
        patch_id_by_key: dict[tuple[str, int], int] = {}
        for patch_id, (chart_type, item) in enumerate(materialized_items):
            key = (chart_type, item.input.source_region_id)
            patch_id_by_key[key] = patch_id
            local_indices = [
                stable_to_local[stable_id]
                for stable_id in item.input.supporting_source_ids
                if stable_id in stable_to_local
            ]
            if not local_indices:
                # `supporting_source_ids` is optional pass-through provenance
                # (worklog 55); fall back to the boundary+interior point IDs
                # this item was actually fit from, which are always present.
                fallback_ids = tuple(item.input.ordered_boundary_point_ids) + (
                    tuple(item.input.interior_reliable_point_ids) if item.input.interior_points is not None else ()
                )
                local_indices = [stable_to_local[stable_id] for stable_id in fallback_ids if stable_id in stable_to_local]
            if local_indices:
                cluster_ids_by_representative[torch.tensor(local_indices, dtype=torch.long, device=points.device)] = patch_id

        propagated, _diagnostics = self._propagate_with_evidence_gating(points, covariance, bundle, cluster_ids_by_representative)

        fits: dict[tuple[str, int], Any] = {}
        for chart_type, item in materialized_items:
            key = (chart_type, item.input.source_region_id)
            patch_id = patch_id_by_key[key]
            representative_support_count = int(
                item.input.ordered_boundary_points.shape[0]
                + (item.input.interior_points.shape[0] if item.input.interior_points is not None else 0)
            )
            full_evidence_points, full_evidence_stable_ids = collect_region_owned_evidence(
                points, full_stable_ids, propagated, patch_id,
            )
            fits[key] = fit_region_owned_full_evidence_patch(
                chart_type, item.input.source_region_id, item.input.ordered_boundary_points,
                full_evidence_points, full_evidence_stable_ids, representative_support_count,
            )
        return fits

    def _propagate_with_evidence_gating(
        self,
        points: Any,
        covariance: Any,
        bundle: CanonicalConstructionWithEvidence,
        cluster_ids_by_representative: Any,
        *,
        normal_alignment_min: float = 0.5,
        residual_max_ratio: float = 4.0,
    ) -> tuple[Any, dict[str, Any]]:
        """Assign every full Gaussian to a region only if its own evidence agrees (worklog 129 item 10).

        Unlike plain nearest-representative copy, a full Gaussian whose own
        learned normal disagrees with its nearest representative's oriented
        normal, or whose position is far off that representative's tangent
        plane, is left unassigned (``-1``) rather than forced into a region
        for coverage. ``cluster_ids_by_representative`` must already reflect
        NORMAL-ORIENTED regions (i.e. built from the same representative
        order as ``bundle.representative_indices``).
        """
        torch = require_torch()
        rep_indices = bundle.representative_indices
        rep_positions = points[rep_indices]
        full_frame = extract_covariance_frame(covariance)
        if bundle.nearest_representative_index is not None:
            nearest = bundle.nearest_representative_index
        else:
            nearest, _distance = assign_nearest_representative(points, rep_positions, chunk_size=int(self.config.surface_projection_chunk_size))

        # Eigenvector sign is ambiguous by construction (see
        # torch_gaussian_covariance_frame.py); every metric below only ever
        # uses ``.abs()`` of a dot product against this normal, so it is
        # correct without needing an oriented (signed) normal.
        rep_normal = bundle.construction.covariance_frame.normal_candidate
        rep_tangent_scale = bundle.construction.covariance_frame.tangent_major_scale

        assigned_representative_normal = rep_normal[nearest]
        assigned_representative_scale = rep_tangent_scale[nearest].clamp_min(1e-12)
        alignment = (full_frame.normal_candidate * assigned_representative_normal).sum(dim=-1).abs()
        offset = points - rep_positions[nearest]
        residual = (offset * assigned_representative_normal).sum(dim=-1).abs() / assigned_representative_scale

        candidate_cluster = cluster_ids_by_representative[nearest]
        compatible = (
            (candidate_cluster >= 0)
            & (alignment >= normal_alignment_min)
            & (residual <= residual_max_ratio)
        )
        propagated = torch.where(compatible, candidate_cluster, torch.full_like(candidate_cluster, -1))
        diagnostics = {
            "propagation_candidate_count": int(points.shape[0]),
            "propagation_assigned_count": int(compatible.sum().item()),
            "propagation_incompatible_normal_count": int(
                ((candidate_cluster >= 0) & (alignment < normal_alignment_min)).sum().item()
            ),
            "propagation_incompatible_residual_count": int(
                ((candidate_cluster >= 0) & (alignment >= normal_alignment_min) & (residual > residual_max_ratio)).sum().item()
            ),
            "propagation_unsupported_representative_count": int((candidate_cluster < 0).sum().item()),
        }
        return propagated, diagnostics

    def _propagate_canonical_patch_ids(self, points: Any, sample_indices: Any, cluster_ids: Any) -> Any:
        """Assign all points from their nearest canonical construction sample."""
        if int(sample_indices.numel()) == int(points.shape[0]):
            return cluster_ids
        sample_points = points[sample_indices]
        sample_patch_ids = cluster_ids[sample_indices]
        nearest = self._nearest_canonical_sample_indices(points, sample_points)
        return sample_patch_ids[nearest]

    def _nearest_canonical_sample_indices(self, points: Any, sample_points: Any) -> Any:
        """Return nearest bounded-sample indices without allocating a full matrix."""
        torch = require_torch()
        chunk_size = max(64, min(4096, int(self.config.surface_projection_chunk_size)))
        output = torch.empty(
            (int(points.shape[0]),), dtype=torch.long, device=points.device
        )
        for start in range(0, int(points.shape[0]), chunk_size):
            end = min(start + chunk_size, int(points.shape[0]))
            output[start:end] = torch.cdist(
                points[start:end], sample_points
            ).argmin(dim=1)
        return output
    def _canonical_initial_covariance(
        self,
        points: Any,
        *,
        covariance_scales: Any | None,
        covariance_rotations: Any | None,
    ) -> tuple[Any, Any, Any]:
        """Return planar Gaussian covariance used by both model and constructor."""
        torch = require_torch()
        count = int(points.shape[0])
        if (covariance_scales is None) != (covariance_rotations is None):
            raise ValueError("covariance_scales and covariance_rotations must be provided together.")
        if covariance_scales is not None:
            scales = torch.as_tensor(covariance_scales, dtype=torch.float32, device=points.device).reshape(count, 3)
            rotations = torch.nn.functional.normalize(
                torch.as_tensor(covariance_rotations, dtype=torch.float32, device=points.device).reshape(count, 4),
                dim=1,
                eps=1e-12,
            )
            return scales, rotations, covariance_from_scale_rotation(scales, rotations)

        if count < 4:
            raise ValueError("Canonical visible-surface construction requires at least four input points.")
        neighbor_count = min(max(3, int(self.config.canonical_covariance_knn)), count - 1)
        chunk_size = self._resolve_covariance_knn_chunk_size(points)
        local_covariance = torch.empty((count, 3, 3), dtype=torch.float32, device=points.device)
        all_indices = torch.arange(count, device=points.device)
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            distances = torch.cdist(points[start:end], points)
            distances[torch.arange(end - start, device=points.device), all_indices[start:end]] = float("inf")
            neighbors = distances.topk(neighbor_count, dim=1, largest=False).indices
            offsets = points[neighbors] - points[start:end, None, :]
            local_covariance[start:end] = offsets.transpose(1, 2) @ offsets / float(neighbor_count)

        _, eigenvectors = torch.linalg.eigh(local_covariance)
        normal = eigenvectors[:, :, 0]
        tangent_v = eigenvectors[:, :, 1]
        tangent_u = eigenvectors[:, :, 2]
        rotation_matrix = torch.stack((tangent_u, tangent_v, normal), dim=-1)
        handedness = torch.linalg.det(rotation_matrix)
        tangent_v = tangent_v * torch.where(handedness < 0.0, -1.0, 1.0).unsqueeze(1)
        rotation_matrix = torch.stack((tangent_u, tangent_v, normal), dim=-1)

        spacing = torch.sqrt(self._graphdeco_neighbor_mean_dist2(points.detach()).clamp_min(1e-12))
        tangent_scale = spacing * 0.45 * float(self.config.covariance_scale_multiplier)
        max_scale = max(
            float(self.config.covariance_min_scale),
            self._scene_scale(points) * float(self.config.covariance_max_scale_ratio),
        )
        tangent_scale = tangent_scale.clamp(min=float(self.config.covariance_min_scale), max=max_scale)
        normal_scale = (tangent_scale * 0.04).clamp_min(float(self.config.covariance_min_scale))
        # A small geometry-aligned tangent anisotropy makes the local-PCA
        # major line reproducible while keeping the splat a broad planar surfel.
        scales = torch.stack((tangent_scale * 1.05, tangent_scale, normal_scale), dim=1)
        rotations = self._quaternion_from_rotation_matrix(rotation_matrix)
        covariance = covariance_from_scale_rotation(scales, rotations)
        return scales, rotations, covariance

    def _baseline_compatible_scale_rotation(self, points: Any) -> tuple[Any, Any]:
        """Graphdeco-equivalent isotropic Gaussian init.

        Matches ``gaussian-splatting/scene/gaussian_model.py::create_from_pcd``
        tensor-for-tensor: ``dist2 = clamp_min(distCUDA2(points), 1e-7)``,
        ``scale = sqrt(dist2)`` repeated identically on all three axes (log
        activation is applied by ``TorchGaussianModel.initialize`` itself,
        exactly like baseline's ``scaling_inverse_activation`` = ``torch.log``),
        and an identity wxyz quaternion (``rot[:, 0] = 1``, no orientation
        preference). ``_graphdeco_neighbor_mean_dist2`` already implements
        the mean-squared-distance-to-3-nearest-neighbors definition
        ``distCUDA2`` uses, so it is reused verbatim here -- only the
        downstream scale/rotation construction differs from the local-PCA
        planar-surfel path in ``_canonical_initial_covariance``.
        """

        torch = require_torch()
        count = int(points.shape[0])
        dist2 = self._graphdeco_neighbor_mean_dist2(points.detach()).clamp_min(1e-7)
        iso_scale = torch.sqrt(dist2)
        scales = iso_scale.reshape(count, 1).repeat(1, 3)
        rotations = torch.zeros((count, 4), dtype=torch.float32, device=points.device)
        rotations[:, 0] = 1.0
        return scales, rotations

    def _surfel_compatible_scale_rotation(self, points: Any) -> tuple[Any, Any]:
        """Official 2DGS `create_from_pcd` initialization, tensor for tensor.

        From `hbb1/2d-gaussian-splatting` @ 335ad61,
        `scene/gaussian_model.py::create_from_pcd`::

            dist2  = clamp_min(distCUDA2(points), 1e-7)
            scales = log(sqrt(dist2))[..., None].repeat(1, 2)
            rots   = torch.rand((N, 4))

        Two points of substance:

        * the isotropic nearest-neighbor spacing seeds BOTH tangent scales and
          there is no third column to seed -- the OSN-GS covariance-derived
          "normal thickness" that `covariance_knn` would supply has no slot in
          a 2DGS primitive and is deliberately not reconstructed;
        * the rotation is RANDOM (`torch.rand`), not the identity quaternion
          3DGS/`baseline_compatible` uses. That is upstream's choice and it is
          load-bearing: identity quaternions would start every surfel's tangent
          plane world-axis-aligned, giving the normal-consistency term a
          degenerate, globally correlated starting orientation. Reproduced as
          upstream has it, including drawing from `torch.rand` (components in
          [0, 1), normalized to a unit quaternion by `initialize`) rather than
          a uniform rotation distribution.

        `_graphdeco_neighbor_mean_dist2` is the same mean-squared-distance-to-
        3-nearest-neighbors definition `distCUDA2` computes, reused verbatim
        from `_baseline_compatible_scale_rotation`; the log activation is
        applied by `TorchGaussianModel.initialize`, so linear scales are
        returned here.
        """

        torch = require_torch()
        count = int(points.shape[0])
        dist2 = self._graphdeco_neighbor_mean_dist2(points.detach()).clamp_min(1e-7)
        iso_scale = torch.sqrt(dist2)
        scales = iso_scale.reshape(count, 1).repeat(1, 2)
        rotations = torch.rand((count, 4), dtype=torch.float32, device=points.device)
        return scales, rotations

    @staticmethod
    def _quaternion_from_rotation_matrix(matrix: Any) -> Any:
        """Convert batched proper rotation matrices to normalized wxyz quaternions."""
        torch = require_torch()
        m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
        m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
        m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
        q_abs = torch.sqrt(torch.clamp(torch.stack((
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ), dim=1), min=0.0))
        candidates = torch.stack((
            torch.stack((q_abs[:, 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=1),
            torch.stack((m21 - m12, q_abs[:, 1].square(), m10 + m01, m02 + m20), dim=1),
            torch.stack((m02 - m20, m10 + m01, q_abs[:, 2].square(), m12 + m21), dim=1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[:, 3].square()), dim=1),
        ), dim=1)
        candidates = candidates / (2.0 * q_abs.unsqueeze(2).clamp_min(0.1))
        choice = q_abs.argmax(dim=1)
        quaternion = candidates[torch.arange(matrix.shape[0], device=matrix.device), choice]
        return torch.nn.functional.normalize(quaternion, dim=1, eps=1e-12)

    def reconstruct_visible_after_adc(
        self,
        state: TorchPipelineState,
        *,
        iteration: int,
        reason: str,
        event_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a detached post-commit canonical reconstruction transaction."""

        torch = require_torch()
        model = state.model
        started = time.perf_counter()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = None
        if torch.cuda.is_available():
            cuda_rng = [item.clone() for item in torch.cuda.get_rng_state_all()]
        construction = None
        source_fingerprint = self._visible_source_fingerprint(model)
        event: dict[str, Any] = {
            "iteration": int(iteration),
            "reason": str(reason),
            "source_fingerprint": source_fingerprint,
            "coverage_semantics": "reliable_core_only",
            "gaussian_count": len(model),
            **(event_metadata or {}),
        }
        if model.get_xyz.device.type == "cuda" and torch.cuda.is_available():
            event["cuda_memory_allocated_before"] = int(torch.cuda.memory_allocated(model.get_xyz.device))
            event["cuda_memory_reserved_before"] = int(torch.cuda.memory_reserved(model.get_xyz.device))
            torch.cuda.reset_peak_memory_stats(model.get_xyz.device)
        try:
            with torch.no_grad():
                eligible_mask = (
                    (~model.is_uncertain)
                    & (model.surface_owner_kind != SURFACE_OWNER_OCCLUDED_CHART)
                )
                eligible_indices = torch.nonzero(
                    eligible_mask, as_tuple=False
                ).reshape(-1)
                event["canonical_input_count"] = int(eligible_indices.numel())
                event["excluded_uncertain_count"] = int(model.is_uncertain.sum().item())
                event["excluded_occluded_owner_count"] = int(
                    (model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART).sum().item()
                )
                if int(eligible_indices.numel()) < 4:
                    raise RuntimeError("canonical_input_has_fewer_than_four_observed_gaussians")

                points = model.get_xyz.detach()[eligible_indices]
                covariance = covariance_from_scale_rotation(
                    model.get_scaling.detach()[eligible_indices],
                    model.get_rotation.detach()[eligible_indices],
                )
                eligible_opacity = model.get_opacity.detach()[eligible_indices, 0]
                eligible_stable_ids = tuple(
                    int(item)
                    for item in model.stable_gaussian_ids[eligible_indices].detach().cpu().tolist()
                )
                bundle = self._construct_canonical_with_full_evidence(
                    points, covariance, eligible_opacity, eligible_stable_ids
                )
                construction = bundle.construction
                sample_stable_ids = bundle.representative_stable_ids
                local_sample = bundle.representative_indices
                sample_model_indices = eligible_indices[local_sample]
                event["canonical_sample_count"] = len(sample_stable_ids)
                event["canonical_representative_stable_ids"] = sample_stable_ids
                event["representative_selection_mode"] = bundle.selection.diagnostics.selection_mode
                event["representative_occupied_cell_count"] = bundle.selection.diagnostics.occupied_cell_count
                event["representative_candidate_mode_count"] = bundle.selection.diagnostics.total_candidate_mode_count
                event["representative_multi_mode_cell_count"] = bundle.selection.diagnostics.multi_mode_cell_count
                event["representative_source_count_mean"] = bundle.selection.diagnostics.representative_source_count_mean
                event["representative_source_count_max"] = bundle.selection.diagnostics.representative_source_count_max
                event["full_evidence_support_count_mean"] = float(bundle.evidence.support_count.float().mean().item())
                event["full_evidence_zero_support_count"] = int((bundle.evidence.support_count == 0).sum().item())
                event.update(dict(construction.diagnostic_summary))
                event["construction_state"] = construction.construction_state
                materialized = construction.materialized_visible_nurbs_surfaces
                patches = [item.surface for item in materialized if item.surface is not None]
                event["materialized_surface_count"] = len(patches)
                if not patches:
                    self._clear_visible_nurbs_state(
                        state,
                        construction=construction,
                        lifecycle_state=construction.construction_state,
                        failure_reason="canonical_construction_produced_no_materialized_surface",
                    )
                    event["success"] = False
                    event["failure"] = dict(state.visible_nurbs_last_failure)
                    return self._finish_visible_event(state, event, started)

                region_to_patch = {
                    item.input.source_region_id: patch_id
                    for patch_id, item in enumerate(materialized)
                }
                region_by_id = {
                    region.region_id: region
                    for region in construction.surface_regions.regions
                }
                stable_to_sample = {
                    stable_id: local_id
                    for local_id, stable_id in enumerate(sample_stable_ids)
                }
                sample_cluster = torch.full(
                    (len(sample_stable_ids),), -1, dtype=torch.long, device=points.device
                )
                for region_id, patch_id in region_to_patch.items():
                    region = region_by_id.get(region_id)
                    if region is None:
                        continue
                    local_members = [
                        stable_to_sample[int(item)]
                        for item in region.member_ids
                        if int(item) in stable_to_sample
                    ]
                    if local_members:
                        sample_cluster[torch.tensor(
                            local_members, dtype=torch.long, device=points.device
                        )] = patch_id
                propagated, propagation_diagnostics = self._propagate_with_evidence_gating(
                    points, covariance, bundle, sample_cluster
                )
                event.update(propagation_diagnostics)
                candidate_cluster = model.cluster_ids.detach().clone()
                candidate_uv = model.surface_uv.detach().clone()
                candidate_kind = model.surface_owner_kind.detach().clone()
                candidate_owner_id = model.surface_owner_id.detach().clone()
                candidate_cluster[eligible_indices] = propagated
                candidate_uv[eligible_indices] = 0.0
                candidate_kind[eligible_indices] = SURFACE_OWNER_UNASSIGNED
                candidate_owner_id[eligible_indices] = UNASSIGNED_OWNER_ID
                for patch_id, patch in enumerate(patches):
                    local_indices = torch.nonzero(
                        propagated == patch_id, as_tuple=False
                    ).reshape(-1)
                    if int(local_indices.numel()) == 0:
                        continue
                    model_indices = eligible_indices[local_indices]
                    uv = project_torch_points_to_nurbs(
                        model.get_xyz.detach()[model_indices],
                        patch,
                        iterations=int(self.config.surface_projection_iterations),
                        chunk_size=int(self.config.surface_projection_chunk_size),
                    )
                    candidate_uv[model_indices] = uv
                    candidate_kind[model_indices] = SURFACE_OWNER_VISIBLE_PATCH
                    candidate_owner_id[model_indices] = patch_id
                    patch.uv_support_mask = self._uv_occupancy_mask(
                        uv,
                        int(self.config.surface_trim_resolution),
                        max(0, int(self.config.surface_trim_dilation)),
                    )

                assigned_local = propagated >= 0
                sample_assigned = int((sample_cluster >= 0).sum().item())
                full_assigned = int(assigned_local.sum().item())
                eligible_count = max(1, int(eligible_indices.numel()))
                sample_count = max(1, len(sample_stable_ids))
                opacity = model.get_opacity.detach()[eligible_indices, 0]
                opacity_total = float(opacity.sum().item())
                opacity_assigned = float(opacity[assigned_local].sum().item())
                assigned_uv = candidate_uv[eligible_indices][assigned_local]
                uv_invalid = int(
                    ((~torch.isfinite(assigned_uv).all(dim=1))
                     | (assigned_uv < 0.0).any(dim=1)
                     | (assigned_uv > 1.0).any(dim=1)).sum().item()
                ) if int(assigned_uv.numel()) else 0
                event.update({
                    "sample_assigned_count": sample_assigned,
                    "sample_coverage_ratio": sample_assigned / sample_count,
                    "full_assigned_count": full_assigned,
                    "full_coverage_ratio": full_assigned / eligible_count,
                    "opacity_weighted_full_coverage": (
                        opacity_assigned / opacity_total if opacity_total > 0.0 else 0.0
                    ),
                    "uv_invalid_count": uv_invalid,
                })

                patch_confidence = self._patch_confidence_from_regions(
                    len(patches), region_to_patch, region_by_id
                )

                # Atomic Python-level commit: all fallible construction,
                # propagation, projection, and support-mask work is complete.
                model.cluster_ids = candidate_cluster
                model.surface_uv = candidate_uv
                model.surface_owner_kind = candidate_kind
                model.surface_owner_id = candidate_owner_id
                state.surface_patches = patches
                state.surface_patch_confidence = patch_confidence
                state.surface = patches[0]
                state.visible_surface_construction = construction
                state.surface_optimizer = None
                state.surface_patch_residuals = {}
                state.surface_bad_checks = {}
                state.surface_topology_version += 1
                state.visible_nurbs_state = "materialized"
                state.visible_nurbs_coverage_semantics = "reliable_core_only"
                state.visible_nurbs_adc_version += 1
                state.visible_nurbs_source_fingerprint = source_fingerprint
                state.visible_nurbs_last_attempt_iteration = int(iteration)
                state.visible_nurbs_last_failure = {}
                event["success"] = True
                event["surface_topology_version"] = int(state.surface_topology_version)
                return self._finish_visible_event(state, event, started)
        except Exception as exc:
            self._clear_visible_nurbs_state(
                state,
                construction=construction,
                lifecycle_state="reconstruction_failed",
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
            event["success"] = False
            event["failure"] = dict(state.visible_nurbs_last_failure)
            return self._finish_visible_event(state, event, started)
        finally:
            # Constructor diagnostics are forbidden from perturbing training RNG.
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def _clear_visible_nurbs_state(
        self,
        state: TorchPipelineState,
        *,
        construction: VisibleSurfaceConstructionResult | None,
        lifecycle_state: str,
        failure_reason: str,
    ) -> None:
        """Fail closed: remove every stale visible patch and binding."""

        model = state.model
        clear_mask = (
            (~model.is_uncertain)
            & (model.surface_owner_kind != SURFACE_OWNER_OCCLUDED_CHART)
        )
        model.cluster_ids[clear_mask] = -1
        model.surface_uv[clear_mask] = 0.0
        model.surface_owner_kind[clear_mask] = SURFACE_OWNER_UNASSIGNED
        model.surface_owner_id[clear_mask] = UNASSIGNED_OWNER_ID
        state.surface = None
        state.surface_patches = []
        state.surface_patch_confidence = ()
        state.surface_optimizer = None
        state.visible_surface_construction = construction
        state.surface_patch_residuals = {}
        state.surface_bad_checks = {}
        state.visible_nurbs_state = str(lifecycle_state)
        state.visible_nurbs_coverage_semantics = "reliable_core_only"
        state.visible_nurbs_last_failure = {
            "reason": str(failure_reason),
            "construction_state": (
                construction.construction_state if construction is not None else None
            ),
        }

    def _finish_visible_event(
        self, state: TorchPipelineState, event: dict[str, Any], started: float
    ) -> dict[str, Any]:
        event["runtime_seconds"] = time.perf_counter() - started
        device = state.model.get_xyz.device
        if device.type == "cuda" and require_torch().cuda.is_available():
            event["cuda_memory_allocated_after"] = int(require_torch().cuda.memory_allocated(device))
            event["cuda_memory_reserved_after"] = int(require_torch().cuda.memory_reserved(device))
            event["cuda_peak_memory_allocated"] = int(require_torch().cuda.max_memory_allocated(device))
        state.visible_nurbs_source_fingerprint = str(event["source_fingerprint"])
        state.visible_nurbs_last_attempt_iteration = int(event["iteration"])
        state.visible_nurbs_event_history.append(dict(event))
        return event

    def _visible_source_fingerprint(self, model: TorchGaussianModel) -> str:
        torch = require_torch()
        with torch.no_grad():
            summary = torch.cat((
                model.get_xyz.detach().double().sum(dim=0),
                model.get_xyz.detach().double().square().sum(dim=0),
                model.get_scaling.detach().double().sum(dim=0),
                model.get_opacity.detach().double().sum().reshape(1),
            )).cpu().tolist()
            ids = model.stable_gaussian_ids.detach().long()
            identity = (
                len(model),
                int(ids.min().item()) if ids.numel() else -1,
                int(ids.max().item()) if ids.numel() else -1,
                int(ids.sum().item()) if ids.numel() else 0,
                tuple(float(value) for value in summary),
            )
        return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()

    def _canonical_spatial_diagnostics(
        self, points: Any, selected: Any
    ) -> dict[str, Any]:
        torch = require_torch()
        budget = max(16, int(self.config.canonical_construction_max_points))
        resolution = max(2, int(math.ceil(budget ** 0.5)))
        minimum = points.amin(dim=0)
        span = (points.amax(dim=0) - minimum).clamp_min(1e-9)
        cells = torch.floor(((points - minimum) / span) * resolution).long().clamp(0, resolution - 1)
        keys = cells[:, 0] * resolution * resolution + cells[:, 1] * resolution + cells[:, 2]
        selected_keys = keys[selected]
        return {
            "spatial_cell_resolution": resolution,
            "spatial_occupied_cell_count": int(torch.unique(keys).numel()),
            "spatial_selected_cell_count": int(torch.unique(selected_keys).numel()),
            "spatial_selected_ratio": int(selected.numel()) / max(1, int(points.shape[0])),
        }

    def maintain_surface_from_certain(
        self, state: TorchPipelineState
    ) -> dict[str, Any]:
        """Reconstruct all visible patches with the canonical Gaussian pipeline.

        The legacy local split/refit path is intentionally unavailable. A
        canonical reconstruction failure is explicit and never retains or
        synthesizes an older fallback patch.
        """
        torch = require_torch()
        model = state.model
        certain_mask = (
            (~model.is_uncertain)
            & is_visible_patch_owned(model.surface_owner_kind.detach())
        )
        certain_indices = torch.nonzero(certain_mask, as_tuple=False).reshape(-1)
        if int(certain_indices.numel()) < 4:
            raise RuntimeError("Canonical surface maintenance requires at least four certain Gaussians.")
        points = model.get_xyz.detach()[certain_indices]
        construction_local_indices = self._canonical_construction_indices(points)
        construction_points = points[construction_local_indices]
        _, _, construction_covariance = self._canonical_initial_covariance(
            construction_points, covariance_scales=None, covariance_rotations=None
        )
        stable_ids = tuple(
            int(item) for item in certain_indices[construction_local_indices].detach().cpu().tolist()
        )
        construction = construct_visible_nurbs_from_gaussians(
            construction_points,
            covariance=construction_covariance,
            stable_ids=stable_ids,
        )
        materialized = construction.materialized_visible_nurbs_surfaces
        if not materialized:
            raise RuntimeError(
                "Canonical surface maintenance produced no materialized NURBS "
                f"(state={construction.construction_state!r}). No fallback path is enabled."
            )
        patches = [item.surface for item in materialized if item.surface is not None]
        region_to_patch = {
            item.input.source_region_id: patch_id
            for patch_id, item in enumerate(materialized)
        }
        region_by_id = {
            region.region_id: region
            for region in construction.surface_regions.regions
        }
        model.cluster_ids[certain_indices] = -1
        model.surface_uv[certain_indices] = 0.0
        stable_to_model = {stable_id: stable_id for stable_id in stable_ids}
        for region_id, patch_id in region_to_patch.items():
            region = region_by_id.get(region_id)
            if region is None:
                continue
            indices = torch.tensor(
                [stable_to_model[int(item)] for item in region.member_ids],
                dtype=torch.long,
                device=model.get_xyz.device,
            )
            model.cluster_ids[indices] = patch_id
            model.surface_uv[indices] = project_torch_points_to_nurbs(
                model.get_xyz.detach()[indices],
                patches[patch_id],
                iterations=int(self.config.surface_projection_iterations),
                chunk_size=int(self.config.surface_projection_chunk_size),
            )
        propagated = self._propagate_canonical_patch_ids(
            points, construction_local_indices, model.cluster_ids[certain_indices]
        )
        model.cluster_ids[certain_indices] = propagated
        for patch_id, patch in enumerate(patches):
            indices = certain_indices[propagated == patch_id]
            if int(indices.numel()) == 0:
                continue
            model.surface_uv[indices] = project_torch_points_to_nurbs(
                model.get_xyz.detach()[indices],
                patch,
                iterations=int(self.config.surface_projection_iterations),
                chunk_size=int(self.config.surface_projection_chunk_size),
            )
        state.surface_patches = patches
        state.surface_patch_confidence = self._patch_confidence_from_regions(
            len(patches), region_to_patch, region_by_id
        )
        state.surface = patches[0]
        state.visible_surface_construction = construction
        state.surface_patch_residuals = {}
        state.surface_bad_checks = {}
        state.surface_topology_version += 1
        self._assign_uv_support_masks(model, patches)
        return {
            "patches": len(patches),
            "checked": len(patches),
            "max_residual_ratio": max(
                (float(item.interior_residual or 0.0) for item in materialized),
                default=0.0,
            ),
            "candidates": [],
            "corrected": list(range(len(patches))),
            "added_patches": len(patches),
            "uv_refreshed": int((model.cluster_ids >= 0).sum()),
            "support_masks_refreshed": len(patches),
            "topology_changed": True,
            "construction_state": construction.construction_state,
        }
    def rebuild_surface_from_certain(self, state: TorchPipelineState) -> None:
        """Compatibility wrapper that no longer rebuilds global voxel topology."""

        self.maintain_surface_from_certain(state)

    def _assign_uv_support_masks(
        self,
        model: TorchGaussianModel,
        patches: list[TorchNURBSSurface],
        patch_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Trim each patch to the UV region actually backed by observed Gaussians.

        The rectangular NURBS chart spans all of ``[0, 1]^2`` but the observed
        points usually cover an irregular sub-region; sampling the untrimmed
        corners draws surface where there is no data. This records, per patch, a
        UV occupancy mask (dilated to close gaps) so downstream consumers can
        restrict the surface to its supported footprint.
        """

        resolution = int(self.config.surface_trim_resolution)
        if resolution <= 0:
            return
        dilation = max(0, int(self.config.surface_trim_dilation))
        uv = model.surface_uv.detach()
        cluster_ids = model.cluster_ids.detach()
        # Ownership gate (docs/worklogs Occluded Chart Ownership Foundation):
        # a visible patch's UV support/trim footprint must reflect only
        # visible-patch-owned (observed) Gaussians. Without this gate, an
        # occluded-chart-owned uncertain Gaussian whose `cluster_ids`
        # compatibility projection happens to equal `patch_id` would inflate
        # this patch's support mask with a location no camera ever observed.
        visible_owned = is_visible_patch_owned(model.surface_owner_kind.detach())
        n_patches = len(patches)
        selected_patch_ids = range(n_patches) if patch_ids is None else patch_ids
        for patch_id in selected_patch_ids:
            patch = patches[patch_id]
            assigned = visible_owned & (cluster_ids == patch_id)
            if patch_id == 0:
                assigned = assigned | (visible_owned & ((cluster_ids < 0) | (cluster_ids >= n_patches)))
            patch.uv_support_mask = self._uv_occupancy_mask(uv[assigned], resolution, dilation)

    @staticmethod
    def _uv_occupancy_mask(uv: Any, resolution: int, dilation: int) -> Any:
        """Boolean ``(resolution, resolution)`` mask of occupied (then dilated) UV cells."""

        torch = require_torch()
        device = uv.device
        mask = torch.zeros((resolution, resolution), dtype=torch.bool, device=device)
        if int(uv.numel()) == 0:
            return mask
        cell_u = torch.clamp((uv[:, 0] * resolution).long(), 0, resolution - 1)
        cell_v = torch.clamp((uv[:, 1] * resolution).long(), 0, resolution - 1)
        mask[cell_u, cell_v] = True
        if dilation > 0:
            pooled = torch.nn.functional.max_pool2d(
                mask.float()[None, None], kernel_size=2 * dilation + 1, stride=1, padding=dilation
            )
            mask = pooled[0, 0] > 0.5
        return mask

    def _graphdeco_neighbor_mean_dist2(self, points: Any) -> Any:
        """Match Graphdeco ``distCUDA2``: mean squared distance to three neighbors."""

        return self._neighbor_mean_dist2(points, neighbor_count=3)

    def _neighbor_mean_dist2(self, points: Any, neighbor_count: int) -> Any:
        """Return mean squared distance to up to ``neighbor_count`` other points."""

        torch = require_torch()
        count = int(points.shape[0])
        neighbors = min(max(1, int(neighbor_count)), max(1, count - 1))
        chunk_size = self._resolve_covariance_knn_chunk_size(points)
        mean_dist2 = torch.full((count,), float("inf"), dtype=torch.float32, device=points.device)
        all_indices = torch.arange(count, device=points.device)
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            chunk = points[start:end]
            distances = torch.cdist(chunk, points).square()
            local = all_indices[start:end]
            distances[torch.arange(end - start, device=points.device), local] = float("inf")
            closest = distances.topk(neighbors, dim=1, largest=False).values
            mean_dist2[start:end] = closest.mean(dim=1)
        finite = torch.isfinite(mean_dist2)
        if not bool(finite.any()):
            fallback = self._scene_scale(points) * 0.001
            mean_dist2.fill_(max(float(self.config.covariance_min_scale) ** 2, float(fallback) ** 2))
        else:
            fill = mean_dist2[finite].median()
            mean_dist2 = torch.where(finite, mean_dist2, fill)
        return mean_dist2

    def _resolve_covariance_knn_chunk_size(self, points: Any) -> int:
        configured = int(self.config.covariance_knn_chunk_size)
        if configured > 0:
            return configured
        torch = require_torch()
        count = max(1, int(points.shape[0]))
        if points.device.type == "cuda" and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(points.device)
            workspace_bytes = max(64 * 1024 * 1024, int(free_bytes * 0.10))
            bytes_per_query = count * 4 * 2
            chunk_size = max(16, min(4096, int(workspace_bytes // max(bytes_per_query, 1))))
            self.config.covariance_knn_chunk_size = chunk_size
            print(
                "OSN-GS covariance KNN chunk: "
                f"auto={chunk_size} free_vram={free_bytes / (1024 ** 3):.2f}GB "
                f"total_vram={total_bytes / (1024 ** 3):.2f}GB points={count}",
                flush=True,
            )
            return chunk_size
        chunk_size = min(1024, count)
        self.config.covariance_knn_chunk_size = chunk_size
        print(f"OSN-GS covariance KNN chunk: auto={chunk_size} device={points.device}", flush=True)
        return chunk_size

    def _scene_scale(self, points: Any) -> float:
        torch = require_torch()
        if points.numel() == 0:
            return 1.0
        span = points.max(dim=0).values - points.min(dim=0).values
        return max(float(torch.linalg.norm(span).detach().cpu()), 1e-6)

    def _empty_occlusion_curves(self, points: Any) -> TorchCurveSet:
        """Return an explicit empty Stage 2 placeholder."""

        torch = require_torch()
        return TorchCurveSet(
            control_points=torch.empty((0, 3, 3), dtype=torch.float32, device=self.device),
            observed=torch.zeros((0,), dtype=torch.bool, device=self.device),
        )

def _uv_support_payload(surface: TorchNURBSSurface) -> dict[str, Any] | None:
    """Serialize a patch's UV trim mask for the renderer, or ``None`` if untrimmed."""

    mask = getattr(surface, "uv_support_mask", None)
    if mask is None:
        return None
    return {
        "resolution": [int(mask.shape[0]), int(mask.shape[1])],
        "mask": mask.detach().cpu().bool().tolist(),
    }


def nurbs_intermediate_payload(state: TorchPipelineState) -> dict[str, Any]:
    """Serialize every canonical visible NURBS patch for files and tools."""

    surface = state.surface
    construction = state.visible_surface_construction
    if surface is None:
        return {
            "type": "visible_nurbs_intermediate",
            "iteration": int(state.iteration),
            "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
            "patches": [],
            "base_curves": state.base_curves.control_points.detach().cpu().tolist(),
            "occlusion_curves": state.occlusion_curves.control_points.detach().cpu().tolist(),
            "metadata": {
                "source": "osn_gs_canonical_visible_surface_construction",
                "constructor_mode": "canonical_gaussian_visible_surface",
                "gaussian_count": len(state.model),
                "uncertain_count": int(state.model.is_uncertain.sum().item()),
                "materialized_surface_count": 0,
                "visible_nurbs_state": state.visible_nurbs_state,
                "coverage_semantics": state.visible_nurbs_coverage_semantics,
                "source_fingerprint": state.visible_nurbs_source_fingerprint,
                "last_failure": dict(state.visible_nurbs_last_failure),
                "final_output_remains_gaussian": True,
                "construction_state": (
                    construction.construction_state if construction is not None else state.visible_nurbs_state
                ),
                "construction_diagnostics": (
                    dict(construction.diagnostic_summary) if construction is not None else {}
                ),
            },
        }
    return {
        "type": "visible_nurbs_intermediate",
        "iteration": int(state.iteration),
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "degree_u": int(surface.degree_u),
        "degree_v": int(surface.degree_v),
        "knots_u": surface.knots_u.detach().cpu().tolist(),
        "knots_v": surface.knots_v.detach().cpu().tolist(),
        "observed_v_max": float(surface.observed_v_max),
        "control_grid_shape": list(surface.control_grid.shape),
        "control_grid": surface.control_grid.detach().cpu().tolist(),
        "weights": surface.weights.detach().cpu().tolist(),
        "uv_support": _uv_support_payload(surface),
        "base_curves": state.base_curves.control_points.detach().cpu().tolist(),
        "occlusion_curves": state.occlusion_curves.control_points.detach().cpu().tolist(),
        "patches": [
            {
                "patch_id": patch_id,
                "control_grid_shape": [int(value) for value in patch.control_grid.shape],
                "control_grid": patch.control_grid.detach().cpu().tolist(),
                "weights": patch.weights.detach().cpu().tolist(),
                "degree_u": int(patch.degree_u),
                "degree_v": int(patch.degree_v),
                "knots_u": patch.knots_u.detach().cpu().tolist(),
                "knots_v": patch.knots_v.detach().cpu().tolist(),
                "uv_support": _uv_support_payload(patch),
            }
            for patch_id, patch in enumerate(state.surface_patches)
        ],
        "metadata": {
            "source": "osn_gs_canonical_visible_surface_construction",
            "constructor_mode": "canonical_gaussian_visible_surface",
            "gaussian_count": len(state.model),
            "uncertain_count": int(state.model.is_uncertain.sum().item()),
            "surface_topology_version": int(state.surface_topology_version),
            "patch_residual_ratios": dict(state.surface_patch_residuals),
            "materialized_surface_count": len(state.surface_patches),
            "visible_nurbs_state": state.visible_nurbs_state,
            "coverage_semantics": state.visible_nurbs_coverage_semantics,
            "source_fingerprint": state.visible_nurbs_source_fingerprint,
            "last_failure": dict(state.visible_nurbs_last_failure),
            "final_output_remains_gaussian": True,
            "construction_state": (
                construction.construction_state
                if construction is not None
                else "checkpoint_restored"
            ),
            "construction_diagnostics": (
                dict(construction.diagnostic_summary)
                if construction is not None
                else {}
            ),
        },
    }
