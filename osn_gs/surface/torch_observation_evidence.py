from __future__ import annotations

"""Read-only per-camera observation evidence and multi-view free-space query.

Phase C of the boundary-conditioned occlusion methodology
(docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md section 5).
Isolated benchmark/prototype only: not imported by osn_gs/core/torch_pipeline.py
or osn_gs/core/torch_trainer.py, and does not alter any default rendering
behavior. Empty-voxel evidence here is recorded only as "no observed support"
and is never itself an occlusion-candidate signal (that is Phase E's job).

Depth-approximation caveat: the CUDA backend only ever exposes ``E[1/z]``
(alpha/transmittance-composited inverse depth), never ``1/z`` at a single
surface. Recovering a linear view-depth via ``1/E[1/z]`` is an approximation
(``1/E[1/z] != E[z]`` in general for multi-Gaussian compositing) -- tight only
where a pixel's underlying depth distribution has low variance.
``CameraViewEvidence.depth_is_approximate`` records this per view so downstream
phases never treat CUDA and fallback evidence as equally confident.

On-surface aggregate semantics (Gate C round 2, docs/worklogs/79; supersedes
the docs/worklogs/78 draft that reused ``conflicting_evidence`` for the
on-surface-alone case too): ``STATUS_ON_OBSERVED_SURFACE`` is its own aggregate
state, produced when ``on_surface_in`` is the ONLY non-empty per-view evidence
list for a sample (no camera says free, no camera says behind). Whenever
``on_surface_in`` co-occurs with ``free_space_confirmed_by`` and/or
``behind_surface_in``, the aggregate is ``conflicting_evidence`` -- a
resolved-surface reading is itself contradicting evidence against BOTH a pure
free-space claim and a pure occluded-candidate claim, not just the former.

**Invariant**: a ``SampleEvidence`` whose ``on_surface_in`` is non-empty can
NEVER have ``status`` equal to ``STATUS_KNOWN_FREE_SPACE`` or
``STATUS_OCCLUDED_CANDIDATE``. See ``classify_world_samples`` for the exact
priority order and ``tests/test_observation_evidence.py``'s
``test_on_surface_only_aggregates_as_on_observed_surface`` /
``test_multi_view_on_surface_and_behind_are_conflicting``.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.gaussian_rasterizer import OSNGaussianRasterizer
from osn_gs.render.torch_fallback import TorchCamera, _auto_chunk_size
from osn_gs.surface.torch_voxel_hierarchy import STATE_EMPTY, TorchVoxelGaussianHierarchy
from osn_gs.utils.torch_ops import require_torch

# Per-view classification: one camera's verdict for one world sample.
VIEW_STATUS_KNOWN_FREE_SPACE = "known_free_space"
VIEW_STATUS_ON_OBSERVED_SURFACE = "on_observed_surface"
VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE = "behind_first_observed_surface"
VIEW_STATUS_UNOBSERVED = "unobserved"
VIEW_STATUS_OUTSIDE_VALID_VIEW = "outside_valid_view"
VIEW_STATUSES = {
    VIEW_STATUS_KNOWN_FREE_SPACE,
    VIEW_STATUS_ON_OBSERVED_SURFACE,
    VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE,
    VIEW_STATUS_UNOBSERVED,
    VIEW_STATUS_OUTSIDE_VALID_VIEW,
}

# Aggregate (cross-camera) classification: a deliberately different, smaller
# enum from the per-view one -- aggregation preserves genuine multi-view
# disagreement (conflicting_evidence) instead of collapsing it.
STATUS_KNOWN_FREE_SPACE = "known_free_space"
STATUS_ON_OBSERVED_SURFACE = "on_observed_surface"
STATUS_OCCLUDED_CANDIDATE = "occluded_candidate"
STATUS_UNOBSERVED = "unobserved"
STATUS_OUTSIDE_VALID_VIEW = "outside_valid_view"
STATUS_CONFLICTING_EVIDENCE = "conflicting_evidence"
SAMPLE_STATUSES = {
    STATUS_KNOWN_FREE_SPACE,
    STATUS_ON_OBSERVED_SURFACE,
    STATUS_OCCLUDED_CANDIDATE,
    STATUS_UNOBSERVED,
    STATUS_OUTSIDE_VALID_VIEW,
    STATUS_CONFLICTING_EVIDENCE,
}

# Invariant (Gate C round 2, docs/worklogs/79): a SampleEvidence whose
# on_surface_in is non-empty can NEVER aggregate to STATUS_KNOWN_FREE_SPACE or
# STATUS_OCCLUDED_CANDIDATE. Any camera resolving the sample as sitting on its
# own observed surface is itself evidence against both "confirmed free" and
# "confirmed occluded" claims. If on_surface_in is the ONLY non-empty evidence
# list, the aggregate is STATUS_ON_OBSERVED_SURFACE. If on_surface_in
# co-occurs with EITHER free_space_confirmed_by OR behind_surface_in (or
# both), the aggregate is STATUS_CONFLICTING_EVIDENCE -- see
# classify_world_samples for the exact priority order.

# The only value an empty-voxel query result may ever take. Structurally kept
# out of SAMPLE_STATUSES/VIEW_STATUSES: empty voxels are "no observed support",
# never occlusion-candidate evidence.
VOXEL_SUPPORT_NONE_OBSERVED = "no_observed_support"


@dataclass
class CameraViewEvidence:
    """One camera's rendered depth/coverage evidence for one scene snapshot."""

    camera_index: int
    image_height: int
    image_width: int
    world_view_transform: Any
    full_proj_transform: Any
    view_depth: Any  # (H, W) float, linear camera-space z
    valid_depth_mask: Any  # (H, W) bool
    coverage_alpha: Any | None  # (H, W) float in [0, 1], fallback only
    backend_source: str  # "cuda" | "fallback"
    coverage_kind: str  # "alpha_fraction" | "binary_contribution_mask"
    depth_kind: str  # "direct_linear" | "inverted_expected_reciprocal"
    depth_is_approximate: bool


@dataclass
class ObservationEvidence:
    """Read-only per-scene observation evidence, passed instead of a full TorchScene."""

    views: list[CameraViewEvidence]
    near: float
    far: float
    depth_epsilon: float
    topology_version: str
    camera_set_version: str

    def payload(self) -> dict[str, Any]:
        return {
            "near": self.near,
            "far": self.far,
            "depth_epsilon": self.depth_epsilon,
            "topology_version": self.topology_version,
            "camera_set_version": self.camera_set_version,
            "views": [
                {
                    "camera_index": view.camera_index,
                    "image_height": view.image_height,
                    "image_width": view.image_width,
                    "backend_source": view.backend_source,
                    "coverage_kind": view.coverage_kind,
                    "depth_kind": view.depth_kind,
                    "depth_is_approximate": view.depth_is_approximate,
                }
                for view in self.views
            ],
        }


@dataclass
class SampleViewStatus:
    """One camera's per-view verdict for one world sample."""

    camera_index: int
    status: str
    sample_view_depth: float
    observed_surface_depth: float | None
    valid: bool

    def __post_init__(self) -> None:
        if self.status not in VIEW_STATUSES:
            raise ValueError(f"Unknown per-view sample status: {self.status!r}")


@dataclass
class SampleEvidence:
    """Aggregated multi-view status for one world sample.

    The five camera-index lists are always populated (never gated behind an
    opt-in flag) so genuine multi-view disagreement -- e.g. one camera seeing
    free space past a point another camera calls behind its own resolved
    surface -- is never silently lost behind a single scalar ``status``.
    """

    status: str
    reason: str
    free_space_confirmed_by: list[int]
    on_surface_in: list[int]
    behind_surface_in: list[int]
    unobserved_in: list[int]
    outside_in: list[int]
    per_view: list[SampleViewStatus] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in SAMPLE_STATUSES:
            raise ValueError(f"Unknown aggregate sample status: {self.status!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "free_space_confirmed_by": list(self.free_space_confirmed_by),
            "on_surface_in": list(self.on_surface_in),
            "behind_surface_in": list(self.behind_surface_in),
            "unobserved_in": list(self.unobserved_in),
            "outside_in": list(self.outside_in),
            "per_view": [
                {
                    "camera_index": item.camera_index,
                    "status": item.status,
                    "sample_view_depth": item.sample_view_depth,
                    "observed_surface_depth": item.observed_surface_depth,
                    "valid": item.valid,
                }
                for item in self.per_view
            ],
        }


@dataclass
class EmptyVoxelSupportResult:
    """AABB-overlap query against empty voxel leaves only.

    ``support`` can structurally only ever equal ``VOXEL_SUPPORT_NONE_OBSERVED``
    -- there is no other branch in ``query_empty_voxel_support`` capable of
    producing anything else, and this type is never passed into
    ``classify_world_samples``/``SampleEvidence``. Callers wanting both facts
    for the same point call the two query functions independently.
    """

    query_min: Any
    query_max: Any
    overlapping_empty_leaf_ids: list[str]
    support: str

    def payload(self) -> dict[str, Any]:
        return {
            "query_min": self.query_min.detach().cpu().tolist(),
            "query_max": self.query_max.detach().cpu().tolist(),
            "overlapping_empty_leaf_ids": list(self.overlapping_empty_leaf_ids),
            "support": self.support,
        }


def _to_hw(tensor: Any) -> Any:
    """Drop a leading singleton channel dim so depth/mask tensors are (H, W)."""

    if tensor.dim() == 3 and int(tensor.shape[0]) == 1:
        return tensor.squeeze(0)
    return tensor


def _project_points(world_points: Any, world_view_transform: Any, full_proj_transform: Any) -> tuple[Any, Any, Any, Any]:
    """Project world points into one camera's screen space.

    Returns ``(view_depth, pixel_row, pixel_col, behind_camera)``, all (N,).
    Pixel indices are left as floats (not yet rounded/clamped) so callers can
    test in-frustum bounds before snapping to the nearest sampled pixel.
    """

    torch = require_torch()
    n = int(world_points.shape[0])
    ones = torch.ones((n, 1), dtype=world_points.dtype, device=world_points.device)
    homogeneous = torch.cat([world_points, ones], dim=1)
    view = homogeneous @ world_view_transform.to(dtype=world_points.dtype, device=world_points.device)
    view_depth = view[:, 2]
    clip = homogeneous @ full_proj_transform.to(dtype=world_points.dtype, device=world_points.device)
    w = clip[:, 3]
    behind_camera = w <= 1e-8
    safe_w = w.clamp_min(1e-8)
    ndc_x = clip[:, 0] / safe_w
    ndc_y = clip[:, 1] / safe_w
    return view_depth, ndc_x, ndc_y, behind_camera


def _fallback_view_depth(camera: TorchCamera, model: TorchGaussianModel) -> tuple[Any, Any, Any]:
    """Evidence-only camera-space depth/coverage for the fallback backend.

    Parallel to, and never a replacement for, ``torch_fallback.fallback_render``:
    that production renderer keeps its pre-existing camera-pose-ignoring
    behavior (raw world-frame xy/z) completely unchanged. This helper reruns
    the identical Gaussian-weight accumulation formula, but on actual
    camera-space projected xy/z, so Phase C evidence has a real view depth.
    """

    torch = require_torch()
    device = model.device
    height = int(camera.image_height)
    width = int(camera.image_width)

    xyz = model.get_xyz
    view_depth_per_point, ndc_x, ndc_y, _ = _project_points(
        xyz, camera.world_view_transform, camera.full_proj_transform
    )
    xy = torch.stack([ndc_x, ndc_y], dim=1).clamp(-1.2, 1.2)
    scales = model.get_scaling[:, :2].mean(dim=-1).clamp(min=1e-3, max=0.5)
    opacity = model.get_opacity[:, 0].clamp(0.0, 1.0)

    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx, yy], dim=-1)

    gaussian_count = int(xy.shape[0])
    chunk_size = _auto_chunk_size(height, width, gaussian_count)

    weight_sum = torch.zeros((1, height, width), dtype=xyz.dtype, device=device)
    depth_accum = torch.zeros((1, height, width), dtype=xyz.dtype, device=device)

    for start in range(0, gaussian_count, chunk_size):
        end = min(start + chunk_size, gaussian_count)
        chunk_xy = xy[start:end]
        chunk_scales = scales[start:end]
        chunk_opacity = opacity[start:end]
        chunk_depth = view_depth_per_point[start:end]

        dist2 = (pixels[None, :, :, :] - chunk_xy[:, None, None, :]).square().sum(dim=-1)
        weights = torch.exp(-dist2 / (2.0 * chunk_scales[:, None, None].square())) * chunk_opacity[:, None, None]

        weight_sum = weight_sum + weights.sum(dim=0, keepdim=True)
        depth_accum = depth_accum + (weights * chunk_depth[:, None, None]).sum(dim=0, keepdim=True)

    alpha = weight_sum.squeeze(0).clamp(0.0, 1.0)
    denom = weight_sum.clamp(min=1e-6)
    view_depth = (depth_accum / denom).squeeze(0)
    valid_mask = alpha > 1e-3
    return view_depth, valid_mask, alpha


def build_observation_evidence(
    cameras: list[TorchCamera],
    model: TorchGaussianModel,
    rasterizer: OSNGaussianRasterizer,
    *,
    near: float = 1e-3,
    far: float = 1e6,
    depth_epsilon: float = 1e-2,
    topology_version: str | None = None,
    camera_set_version: str | None = None,
) -> ObservationEvidence:
    """Build read-only per-camera observation evidence for the current scene.

    CUDA views reuse the rasterizer's own render call for depth/valid-mask
    (no re-derivation of the vendored kernel math); fallback views bypass
    ``fallback_render`` entirely and use ``_fallback_view_depth`` instead,
    since production fallback rendering never applies the camera transform to
    its own "depth" output (see module docstring / grounding facts).
    """

    torch = require_torch()
    views: list[CameraViewEvidence] = []
    for index, camera in enumerate(cameras):
        use_cuda = rasterizer.has_cuda_backend and rasterizer.config.prefer_cuda
        if use_cuda:
            render_pkg = rasterizer.render(camera, model)
            backend_source = rasterizer.backend_source
        else:
            render_pkg = None
            backend_source = "fallback"

        if backend_source == "cuda" and render_pkg is not None:
            raw_invdepth = _to_hw(render_pkg["depth"]).detach()
            valid_mask = _to_hw(render_pkg["valid_depth_mask"]).detach().bool()
            view_depth = torch.full_like(raw_invdepth, float("inf"))
            view_depth[valid_mask] = 1.0 / raw_invdepth[valid_mask].clamp_min(1e-8)
            coverage_alpha = None
            coverage_kind = "binary_contribution_mask"
            depth_kind = "inverted_expected_reciprocal"
            depth_is_approximate = True
        else:
            view_depth, valid_mask, coverage_alpha = _fallback_view_depth(camera, model)
            view_depth = view_depth.detach()
            valid_mask = valid_mask.detach()
            coverage_alpha = coverage_alpha.detach()
            coverage_kind = "alpha_fraction"
            depth_kind = "direct_linear"
            depth_is_approximate = False

        views.append(
            CameraViewEvidence(
                camera_index=index,
                image_height=int(camera.image_height),
                image_width=int(camera.image_width),
                world_view_transform=camera.world_view_transform,
                full_proj_transform=camera.full_proj_transform,
                view_depth=view_depth,
                valid_depth_mask=valid_mask,
                coverage_alpha=coverage_alpha,
                backend_source=backend_source,
                coverage_kind=coverage_kind,
                depth_kind=depth_kind,
                depth_is_approximate=depth_is_approximate,
            )
        )

    resolved_topology_version = topology_version if topology_version is not None else _topology_version(model)
    resolved_camera_set_version = (
        camera_set_version if camera_set_version is not None else _camera_set_version(cameras)
    )
    return ObservationEvidence(
        views=views,
        near=near,
        far=far,
        depth_epsilon=depth_epsilon,
        topology_version=resolved_topology_version,
        camera_set_version=resolved_camera_set_version,
    )


def classify_world_samples(evidence: ObservationEvidence, world_points: Any) -> list[SampleEvidence]:
    """Classify each world sample against every camera, then aggregate.

    Per-view rule (one camera, one sample):
      1. Screen-space projection outside the image or behind the camera plane
         -> ``outside_valid_view``.
      2. Nearest-pixel lookup invalid, or view depth outside ``[near, far]``
         -> ``unobserved``.
      3. Else compare to the pixel's resolved ``view_depth`` within
         ``depth_epsilon``: strictly nearer -> ``known_free_space``; within
         the tolerance band -> ``on_observed_surface``; strictly farther ->
         ``behind_first_observed_surface`` (a ray-geometric fact only -- NOT a
         claim that the sample is inside occluded geometry).

    Aggregation preserves disagreement instead of collapsing it. Given the
    three non-empty-or-not per-view evidence buckets (free, behind, on-surface):
      - Two or more buckets non-empty -> ``conflicting_evidence`` (covers
        free+behind, free+on-surface, behind+on-surface, and all three).
      - Only ``behind_surface_in`` non-empty -> ``occluded_candidate``.
      - Only ``free_space_confirmed_by`` non-empty -> ``known_free_space``.
      - Only ``on_surface_in`` non-empty -> ``on_observed_surface`` (its own
        aggregate state -- never folded into ``unobserved`` or
        ``known_free_space``; a sample whose ``on_surface_in`` is non-empty can
        never aggregate to ``known_free_space`` or ``occluded_candidate``).
      - Only ``unobserved_in`` non-empty -> ``unobserved``.
      - Every camera ``outside_valid_view`` -> ``outside_valid_view``.
    """

    torch = require_torch()
    world_points = torch.as_tensor(world_points, dtype=torch.float32)
    n = int(world_points.shape[0])
    per_sample_views: list[list[SampleViewStatus]] = [[] for _ in range(n)]

    for view in evidence.views:
        device = view.view_depth.device
        points = world_points.to(device=device, dtype=view.view_depth.dtype)
        view_depth, ndc_x, ndc_y, behind_camera = _project_points(
            points, view.world_view_transform, view.full_proj_transform
        )
        pixel_col = (ndc_x * 0.5 + 0.5) * view.image_width
        pixel_row = (ndc_y * 0.5 + 0.5) * view.image_height
        row_idx = pixel_row.round().long()
        col_idx = pixel_col.round().long()
        in_bounds = (
            (row_idx >= 0)
            & (row_idx < view.image_height)
            & (col_idx >= 0)
            & (col_idx < view.image_width)
            & (~behind_camera)
        )
        clamped_row = row_idx.clamp(0, view.image_height - 1)
        clamped_col = col_idx.clamp(0, view.image_width - 1)
        observed_depth = view.view_depth[clamped_row, clamped_col]
        valid_at_pixel = view.valid_depth_mask[clamped_row, clamped_col]

        for i in range(n):
            sample_depth = float(view_depth[i])
            if not bool(in_bounds[i]):
                per_sample_views[i].append(
                    SampleViewStatus(view.camera_index, VIEW_STATUS_OUTSIDE_VALID_VIEW, sample_depth, None, False)
                )
                continue
            obs_depth = float(observed_depth[i])
            is_valid = bool(valid_at_pixel[i]) and (evidence.near <= sample_depth <= evidence.far)
            if not is_valid:
                per_sample_views[i].append(
                    SampleViewStatus(view.camera_index, VIEW_STATUS_UNOBSERVED, sample_depth, None, False)
                )
                continue
            diff = sample_depth - obs_depth
            if diff < -evidence.depth_epsilon:
                status = VIEW_STATUS_KNOWN_FREE_SPACE
            elif diff > evidence.depth_epsilon:
                status = VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE
            else:
                status = VIEW_STATUS_ON_OBSERVED_SURFACE
            per_sample_views[i].append(
                SampleViewStatus(view.camera_index, status, sample_depth, obs_depth, True)
            )

    results: list[SampleEvidence] = []
    for i in range(n):
        views_i = per_sample_views[i]
        free_ids = [v.camera_index for v in views_i if v.status == VIEW_STATUS_KNOWN_FREE_SPACE]
        on_ids = [v.camera_index for v in views_i if v.status == VIEW_STATUS_ON_OBSERVED_SURFACE]
        behind_ids = [v.camera_index for v in views_i if v.status == VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE]
        unobserved_ids = [v.camera_index for v in views_i if v.status == VIEW_STATUS_UNOBSERVED]
        outside_ids = [v.camera_index for v in views_i if v.status == VIEW_STATUS_OUTSIDE_VALID_VIEW]

        # Any two-or-more-way combination among {free, behind, on-surface} is a
        # conflict -- none of the three may silently win over another (Gate C
        # round 2, docs/worklogs/79). on-surface alone (no free, no behind) is
        # its own aggregate state, never folded into unobserved or free space.
        evidence_kinds = sum([bool(free_ids), bool(behind_ids), bool(on_ids)])
        if evidence_kinds >= 2:
            status = STATUS_CONFLICTING_EVIDENCE
            if free_ids and behind_ids and on_ids:
                reason = "free_space_behind_and_on_surface_conflict"
            elif free_ids and behind_ids:
                reason = "free_space_and_occluded_conflict"
            elif free_ids and on_ids:
                reason = "free_space_and_on_surface_conflict"
            else:
                reason = "behind_and_on_surface_conflict"
        elif behind_ids:
            status, reason = STATUS_OCCLUDED_CANDIDATE, "behind_first_observed_surface_only"
        elif free_ids:
            status, reason = STATUS_KNOWN_FREE_SPACE, "free_space_confirmed"
        elif on_ids:
            status, reason = STATUS_ON_OBSERVED_SURFACE, "on_observed_surface_only"
        elif unobserved_ids:
            status, reason = STATUS_UNOBSERVED, "no_camera_confirms_free_or_occluded"
        else:
            status, reason = STATUS_OUTSIDE_VALID_VIEW, "outside_all_camera_views"

        results.append(
            SampleEvidence(
                status=status,
                reason=reason,
                free_space_confirmed_by=free_ids,
                on_surface_in=on_ids,
                behind_surface_in=behind_ids,
                unobserved_in=unobserved_ids,
                outside_in=outside_ids,
                per_view=views_i,
            )
        )
    return results


def _aabb_overlaps(a_min: Any, a_max: Any, b_min: Any, b_max: Any) -> bool:
    torch = require_torch()
    b_min = b_min.to(dtype=a_min.dtype, device=a_min.device)
    b_max = b_max.to(dtype=a_max.dtype, device=a_max.device)
    return bool(torch.all(a_min <= b_max)) and bool(torch.all(b_min <= a_max))


def query_empty_voxel_support(
    hierarchy: TorchVoxelGaussianHierarchy, query_min: Any, query_max: Any
) -> EmptyVoxelSupportResult:
    """AABB-overlap query against empty voxel leaves only.

    Kept structurally separate from ``classify_world_samples``/``SampleEvidence``
    -- there is no combined type, so callers wanting both facts for the same
    point call the two query functions independently.
    """

    torch = require_torch()
    query_min_t = torch.as_tensor(query_min, dtype=torch.float32)
    query_max_t = torch.as_tensor(query_max, dtype=torch.float32)
    overlapping = [
        leaf.node_id
        for leaf in hierarchy.leaves_in_state(STATE_EMPTY)
        if _aabb_overlaps(query_min_t, query_max_t, leaf.aabb_min, leaf.aabb_max)
    ]
    return EmptyVoxelSupportResult(
        query_min=query_min_t,
        query_max=query_max_t,
        overlapping_empty_leaf_ids=overlapping,
        support=VOXEL_SUPPORT_NONE_OBSERVED,
    )


def _tensor_digest(label: str, tensor: Any) -> str:
    """Content fingerprint for one labeled tensor field.

    Hashes the field label, exact shape, and dtype ALONGSIDE the raw content
    bytes -- not bytes alone (Gate C round 2, docs/worklogs/79 fix: the prior
    version hashed only ``tobytes()``, so two tensors with identical byte
    content but different shape, e.g. a ``(16,)`` vs a ``(4, 4)`` float32
    tensor holding the same 16 numbers, would have collided; likewise two
    different fields holding same-shape/dtype tensors could not be told apart
    from the digest alone without the label). Device-independent and
    deterministic: ``.cpu().contiguous()`` normalizes memory layout before
    reading bytes, so a CUDA tensor and a CPU tensor with the same logical
    content always produce the identical digest, and repeated calls on the
    same tensor always produce the identical digest.
    """

    cpu_tensor = tensor.detach().cpu().contiguous()
    header = f"{label}|{tuple(cpu_tensor.shape)}|{cpu_tensor.dtype}".encode("utf-8")
    return hashlib.sha256(header + cpu_tensor.numpy().tobytes()).hexdigest()


def _topology_version(model: TorchGaussianModel, hierarchy: TorchVoxelGaussianHierarchy | None = None) -> str:
    """Content fingerprint, not a mutation counter (no such counter exists on
    ``TorchGaussianModel``/``TorchVoxelGaussianHierarchy`` today, and adding one
    would touch production model state, which this isolated module must not do).

    Covers every model field that actually changes rendered evidence: position,
    scale, rotation, and opacity (the effective ``get_*`` properties actually
    consumed by the rasterizer, not just the raw underlying parameters) --
    Gate C follow-up, worklog 78. Color/SH is intentionally excluded: it does
    not affect depth/coverage evidence.
    """

    parts = [
        str(len(model)),
        _tensor_digest("xyz", model.get_xyz),
        _tensor_digest("scaling", model.get_scaling),
        _tensor_digest("rotation", model.get_rotation),
        _tensor_digest("opacity", model.get_opacity),
    ]
    if hierarchy is not None:
        parts.append(str(len(hierarchy.nodes)))
        parts.append(repr(sorted(hierarchy.state_counts().items())))
    return "|".join(parts)


def _camera_set_version(cameras: list[TorchCamera]) -> str:
    """Covers every camera field that changes projection/depth: transforms,
    image dimensions, and a stable per-camera identity (Gate C follow-up,
    worklog 78 added ``full_proj_transform``, which the original version
    omitted despite it determining screen position; Gate C round 2, worklog 79
    added the explicit list-index identity below).

    Stable-identity fallback rule: ``TorchCamera.image_name`` defaults to the
    shared literal ``"camera"`` and is never ``None`` (plain ``str`` field, not
    ``str | None``), so it cannot reliably distinguish two cameras that both
    left it at that default. Rather than special-case a "name is absent"
    branch that can't actually occur, this function ALWAYS also includes each
    camera's zero-based position within the input list as an explicit,
    unconditionally-available identity component -- not merely relying on the
    implicit ordering of the joined string, so the identity is visible as its
    own token and documented as a deliberate disambiguator, not an accident of
    ``"|".join`` order.
    """

    parts = [str(len(cameras))]
    for index, camera in enumerate(cameras):
        parts.append(f"camera_index={index}")
        parts.append(f"image_name={camera.image_name}")
        parts.append(str(int(camera.image_height)))
        parts.append(str(int(camera.image_width)))
        for label, transform in (
            ("world_view_transform", camera.world_view_transform),
            ("full_proj_transform", camera.full_proj_transform),
        ):
            parts.append(_tensor_digest(label, transform) if transform is not None else f"{label}=none")
    return "|".join(parts)


def evidence_cache_key(evidence: ObservationEvidence) -> str:
    """POST-BUILD result fingerprint of an already-constructed ``ObservationEvidence``.

    Explicit contract (Gate C round 2, docs/worklogs/79): this function can
    only be called AFTER ``build_observation_evidence()`` has returned. It is
    NOT a pre-render cache-lookup key and cannot be computed from
    model/cameras/config alone ahead of time, because a view's actual
    ``backend_source`` (and therefore its ``depth_kind``/``coverage_kind``) is
    only known once rendering has actually been attempted -- a CUDA failure
    mid-render silently falls back (see ``OSNGaussianRasterizer.render``), so
    two calls with identical inputs can still resolve to different backends.
    Use this to compare/deduplicate two evidences you already built (e.g. "is
    this newly-built evidence identical in content to the one I cached last
    time"), never to decide whether a re-render is needed before building one.

    Covers every field that can change the actual evidence content:
    model/camera content (``topology_version``/``camera_set_version``), the
    query config (``near``/``far``/``depth_epsilon``), and each view's resolved
    backend/depth convention (``backend_source``/``depth_kind``/
    ``coverage_kind``). This is a key-generation convention only -- no cache
    dict/singleton is added here; callers own any actual caching.
    """

    config_parts = [
        f"near={evidence.near:.9g}",
        f"far={evidence.far:.9g}",
        f"depth_epsilon={evidence.depth_epsilon:.9g}",
    ]
    for view in evidence.views:
        config_parts.append(
            f"{view.camera_index}:{view.backend_source}:{view.depth_kind}:{view.coverage_kind}"
        )
    return "::".join([evidence.topology_version, evidence.camera_set_version, "|".join(config_parts)])
