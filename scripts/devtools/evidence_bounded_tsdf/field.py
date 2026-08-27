from __future__ import annotations

"""Worklog 127 -- EVIDENCE-BOUNDED PROJECTIVE TSDF (directive sections 3, 5, 6).

    s_v(x)   = d_v(p) - z_v(x)          raw projective signed distance
    phi_v(x) = clamp(s_v(x)/mu, -1, +1) normalized ONLY for fusion
    authority                            iff the view can query x AND |s_v(x)| <= mu

`s_v > 0` is the CAMERA-FACING side of the renderer median surface and
`s_v < 0` is behind it. This is NOT object inside/outside and this field is NOT
a watertight object SDF; it is a projective TSDF over the renderer's own
visible-surface observations.

THE SPARSE AUTHORITY CONTRACT (directive section 6) is the whole point of this
module. A voxel that no view could place inside its truncation band has

    state = UNKNOWN

and is represented by BEING ABSENT from the sparse store. It is never given
+1, never -1, never 0, never a nearest value; nothing is diffused, smoothed,
closed, propagated or filled anywhere in this file.

`z_v(x)` and the relevance rule are byte-for-byte the same computation as
worklog 120-123's frozen `observed_occluded.shared.project_queries`, and
`MIDDEPTH_OFFSET` is the same canonical `out_others` channel candidate B reads.
They are re-implemented here rather than imported so this module keeps ZERO
imports from the historical/diagnostic families (directive section 8); the
test-suite asserts bitwise equality against the frozen implementations instead.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

# Canonical renderer constants -- see osn_gs/render/vendor/diff_surfel_rasterization/.
MIDDEPTH_OFFSET = 5          # forward.cu's own median ("mid") depth channel
CANONICAL_NEAR_N = 0.2       # auxiliary.h's own near plane

# Voxel key packing. One int64 per voxel, |index| < KEY_BOUND on every axis.
KEY_BOUND = 1 << 19
_AXIS_SPAN = KEY_BOUND << 1
STRIDE_Z = 1
STRIDE_Y = _AXIS_SPAN
STRIDE_X = _AXIS_SPAN * _AXIS_SPAN


def median_depth_map(out_others: torch.Tensor) -> torch.Tensor:
    """The canonical kernel's own median-depth channel, (H, W)."""

    return out_others[MIDDEPTH_OFFSET]


# ---------------------------------------------------------------- projection
@dataclass
class ProjectedQuery:
    depth: torch.Tensor        # (N,) float32 camera-space z -- the frozen query-depth quantity
    pixel_index: torch.Tensor  # (N,) int64 row * W + col, -1 where the view cannot query x
    relevant: torch.Tensor     # (N,) bool


def project_world_points(
    positions: torch.Tensor, world_view_transform: torch.Tensor,
    full_proj_transform: torch.Tensor, width: int, height: int,
) -> ProjectedQuery:
    """Identical arithmetic to the frozen `shared.project_queries`: row-vector
    homogeneous convention, ndc2Pix(v, S) = ((v + 1) * S - 1) * 0.5, camera-space
    z depth, and the three-way relevance rule with the canonical near plane."""

    count = int(positions.shape[0])
    ones = torch.ones((count, 1), dtype=torch.float32, device=positions.device)
    homogeneous = torch.cat([positions.to(torch.float32), ones], dim=1)

    depth = (homogeneous @ world_view_transform)[:, 2].contiguous()
    clip = homogeneous @ full_proj_transform
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 0, w, torch.full_like(w, 1.0))
    pixel_x = ((clip[:, 0] / safe_w + 1.0) * width - 1.0) * 0.5
    pixel_y = ((clip[:, 1] / safe_w + 1.0) * height - 1.0) * 0.5
    col = torch.round(pixel_x).to(torch.int64)
    row = torch.round(pixel_y).to(torch.int64)

    invalid = w <= 0
    below_near = (~invalid) & (depth < CANONICAL_NEAR_N)
    outside = (~invalid) & (~below_near) & ((col < 0) | (col >= width) | (row < 0) | (row >= height))
    relevant = ~(invalid | below_near | outside)
    index = torch.where(relevant, row * width + col, torch.full_like(col, -1))
    return ProjectedQuery(depth=depth, pixel_index=index, relevant=relevant)


def unproject_pixels(camera: Any, pixel_index: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """World position of a renderer event, from its own (pixel, camera-space
    depth). Used ONLY to enumerate candidate voxels -- never to decide a field
    value, which is always the forward projection above."""

    width = int(camera.image_width)
    height = int(camera.image_height)
    row = (pixel_index // width).to(torch.float32)
    col = (pixel_index % width).to(torch.float32)
    ndc_x = (2.0 * col + 1.0) / width - 1.0
    ndc_y = (2.0 * row + 1.0) / height - 1.0
    tan_x = math.tan(float(camera.FoVx) * 0.5)
    tan_y = math.tan(float(camera.FoVy) * 0.5)
    z = depth.reshape(-1)
    view_point = torch.stack([ndc_x * tan_x * z, ndc_y * tan_y * z, z, torch.ones_like(z)], dim=1)
    inverse = torch.linalg.inv(camera.world_view_transform.transpose(0, 1).to(torch.float32))
    return (view_point @ inverse.transpose(0, 1))[:, :3].contiguous()


# ------------------------------------------------------------------ voxel keys
def voxel_index_of(positions: torch.Tensor, h: float) -> torch.Tensor:
    return torch.floor(positions.to(torch.float32) / h).to(torch.int64)


def encode_keys(index: torch.Tensor, margin: int = 0) -> tuple[torch.Tensor, int]:
    """Pack (N, 3) integer voxel indices into sorted-comparable int64 keys.
    Voxels within `margin` of the representable bound are DROPPED (and counted)
    so later dilation can never wrap between axes."""

    limit = KEY_BOUND - margin - 1
    inside = (index.abs() <= limit).all(dim=1)
    kept = index[inside]
    keys = ((kept[:, 0] + KEY_BOUND) * _AXIS_SPAN + (kept[:, 1] + KEY_BOUND)) * _AXIS_SPAN + (kept[:, 2] + KEY_BOUND)
    return keys, int((~inside).sum().item())


def decode_keys(keys: torch.Tensor) -> torch.Tensor:
    iz = keys % _AXIS_SPAN - KEY_BOUND
    rest = keys // _AXIS_SPAN
    iy = rest % _AXIS_SPAN - KEY_BOUND
    ix = rest // _AXIS_SPAN - KEY_BOUND
    return torch.stack([ix, iy, iz], dim=1)


def voxel_centers(keys: torch.Tensor, h: float) -> torch.Tensor:
    return (decode_keys(keys).to(torch.float32) + 0.5) * h


def union_sorted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sorted union of a sorted `a` and an arbitrary `b`, without materializing
    a full concatenation of duplicates."""

    if a.numel() == 0:
        return torch.unique(b)
    if b.numel() == 0:
        return a
    position = torch.searchsorted(a, b)
    clamped = position.clamp(max=a.numel() - 1)
    already = (position < a.numel()) & (a[clamped] == b)
    fresh = b[~already]
    if fresh.numel() == 0:
        return a
    return torch.unique(torch.cat([a, fresh]))


def neighbour_shell(keys: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Keys at L-infinity distance <= `radius` of `keys`, EXCLUDING `keys`."""

    grown = dilate_linf(keys, radius)
    position = torch.searchsorted(keys, grown)
    clamped = position.clamp(max=max(keys.numel() - 1, 0))
    inside = (position < keys.numel()) & (keys[clamped] == grown)
    return grown[~inside]


def dilate_linf(keys: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact L-infinity ball dilation of a sorted key set, done separably
    (three axis passes) so peak allocation stays close to the result size.

    Pure bookkeeping: it enlarges the set of voxels that will be TESTED. It
    cannot give a voxel authority -- only `fuse_views` can, and only through the
    truncation rule."""

    current = keys
    for stride in (STRIDE_X, STRIDE_Y, STRIDE_Z):
        source = current
        for step in range(1, radius + 1):
            for sign in (step, -step):
                current = union_sorted(current, source + sign * stride)
    return current


# ------------------------------------------------------------------- the field
@dataclass
class SparseProjectiveTSDF:
    """value + authority mask + support_count, exactly as directive section 6
    demands. Absence from `keys` IS the UNKNOWN state."""

    keys: torch.Tensor            # (M,) int64 ascending -- authoritative voxels ONLY
    value: torch.Tensor           # (M,) float32 -- mean of authoritative phi_v
    support_count: torch.Tensor   # (M,) int32   -- number of contributing views
    h: float
    mu: float

    def __len__(self) -> int:
        return int(self.keys.numel())

    def lookup(self, query_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(value, support_count, has_authority). Rows without authority come
        back as NaN / 0 / False -- never as a filled-in field value."""

        if self.keys.numel() == 0:
            nan = torch.full(query_keys.shape, float("nan"), dtype=torch.float32, device=query_keys.device)
            zero = torch.zeros(query_keys.shape, dtype=torch.int32, device=query_keys.device)
            return nan, zero, torch.zeros(query_keys.shape, dtype=torch.bool, device=query_keys.device)
        position = torch.searchsorted(self.keys, query_keys)
        clamped = position.clamp(max=self.keys.numel() - 1)
        found = (position < self.keys.numel()) & (self.keys[clamped] == query_keys)
        gathered = self.value[clamped]
        value = torch.where(found, gathered, torch.full_like(gathered, float("nan")))
        gathered_count = self.support_count[clamped]
        count = torch.where(found, gathered_count, torch.zeros_like(gathered_count))
        return value, count, found


def projective_signed_distance(depth: torch.Tensor, median_depth_at_pixel: torch.Tensor) -> torch.Tensor:
    """s_v(x) = d_v(p) - z_v(x)."""

    return median_depth_at_pixel - depth


def truncated_phi(signed_distance: torch.Tensor, mu: float) -> torch.Tensor:
    return torch.clamp(signed_distance / mu, -1.0, 1.0)


def view_authority(
    centers: torch.Tensor, camera: Any, median_flat: torch.Tensor, mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(authoritative, signed_distance) of one view over arbitrary world points."""

    width, height = int(camera.image_width), int(camera.image_height)
    projected = project_world_points(centers, camera.world_view_transform, camera.full_proj_transform, width, height)
    index = projected.pixel_index.clamp(min=0)
    median = median_flat[index]
    valid = projected.relevant & (median > 0.0)
    signed = projective_signed_distance(projected.depth, median)
    return valid & (signed.abs() <= mu), signed


def fuse_views(
    candidate_keys: torch.Tensor,
    views: Iterable[tuple[Any, torch.Tensor]],
    *, h: float, mu: float, chunk_size: int = 8_000_000, progress=None,
) -> SparseProjectiveTSDF:
    """UNIFORM fusion (directive section 5): every authoritative observation has
    weight exactly 1. No angle, opacity, confidence, visibility, normal,
    component or region weighting exists anywhere in this function, and there is
    NO minimum-view threshold -- one observation is enough."""

    view_list = list(views)
    # `SparseProjectiveTSDF.lookup` binary-searches `keys`, so the store's key
    # order is an invariant of the type, not of the caller. Enforce it here
    # rather than trusting every call site; it is a pure reordering.
    if candidate_keys.numel() > 1 and not bool((candidate_keys[1:] >= candidate_keys[:-1]).all()):
        candidate_keys = torch.sort(candidate_keys).values
    total = int(candidate_keys.numel())
    device = candidate_keys.device
    value_sum = torch.zeros((total,), dtype=torch.float32, device=device)
    counts = torch.zeros((total,), dtype=torch.int32, device=device)

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        centers = voxel_centers(candidate_keys[start:stop], h)
        chunk_sum = torch.zeros((stop - start,), dtype=torch.float32, device=device)
        chunk_count = torch.zeros((stop - start,), dtype=torch.int32, device=device)
        for camera, median_flat in view_list:
            authoritative, signed = view_authority(centers, camera, median_flat, mu)
            chunk_sum += torch.where(authoritative, truncated_phi(signed, mu), torch.zeros_like(signed))
            chunk_count += authoritative.to(torch.int32)
        value_sum[start:stop] = chunk_sum
        counts[start:stop] = chunk_count
        if progress is not None:
            progress(f"fusion {stop:,}/{total:,} candidate voxels")

    authoritative = counts > 0
    keys = candidate_keys[authoritative]
    return SparseProjectiveTSDF(
        keys=keys,
        value=(value_sum[authoritative] / counts[authoritative].to(torch.float32)),
        support_count=counts[authoritative], h=h, mu=mu,
    )


def grow_field_to_closure(
    seed_keys: torch.Tensor,
    views,
    *, h: float, mu: float, max_rounds: int = 12, chunk_size: int = 8_000_000, progress=None,
) -> tuple[SparseProjectiveTSDF, dict[str, Any]]:
    """Enumerate the authoritative set to CLOSURE instead of guessing a
    dilation radius.

    Start from the voxels of the renderer's own median events, fuse, then
    repeatedly test only the 1-voxel shell around the authoritative set. The
    loop stops when a whole shell yields no new authoritative voxel, at which
    point the result is a fixed point: every voxel adjacent to the field has
    been tested and rejected by the truncation rule itself.

    This changes nothing about the field's DEFINITION -- a voxel is
    authoritative iff some view puts it within mu, exactly as before. It only
    replaces an arbitrary enumeration radius with a self-certifying one, and it
    reports the certificate (`closed`, `final_shell_tested`, `final_shell_new`)
    so the closure is a measured fact rather than an assumption.

    Note the authoritative set is connected through the shell by construction of
    the truncation band, so a shell that yields nothing cannot hide a detached
    authoritative island: any such island would have to be reachable from a
    median event's own voxel, which is a seed.
    """

    view_list = list(views)
    field = fuse_views(seed_keys, view_list, h=h, mu=mu, chunk_size=chunk_size, progress=None)
    if progress is not None:
        progress(f"closure round 0: {int(seed_keys.numel()):,} seed voxels -> {len(field):,} authoritative")
    history: list[dict[str, int]] = [{"round": 0, "tested": int(seed_keys.numel()), "authoritative_total": len(field)}]
    closed = False
    round_index = 0
    frontier = field.keys
    for round_index in range(1, max_rounds + 1):
        # Only voxels adjacent to LAST round's new authoritative voxels can be
        # newly authoritative. Every other neighbour of the field was already
        # tested in an earlier round and rejected, and a rejection is permanent
        # because the field definition never changes. Testing the frontier's
        # shell instead of the whole field's shell therefore reaches the SAME
        # fixed point while touching a set proportional to the growth front
        # rather than to the whole field.
        shell = neighbour_shell(frontier, radius=1)
        if shell.numel():
            position = torch.searchsorted(field.keys, shell)
            clamped = position.clamp(max=max(field.keys.numel() - 1, 0))
            known = (position < field.keys.numel()) & (field.keys[clamped] == shell)
            shell = shell[~known]
        if shell.numel() == 0:
            closed = True
            history.append({"round": round_index, "tested": 0, "new_authoritative": 0, "authoritative_total": len(field)})
            break
        grown = fuse_views(shell, view_list, h=h, mu=mu, chunk_size=chunk_size, progress=None)
        if progress is not None:
            progress(
                f"closure round {round_index}: tested {int(shell.numel()):,} shell voxels, "
                f"+{len(grown):,} authoritative (total {len(field) + len(grown):,})"
            )
        history.append({
            "round": round_index, "tested": int(shell.numel()),
            "new_authoritative": len(grown), "authoritative_total": len(field) + len(grown),
            "new_authoritative_fraction_of_total": (
                len(grown) / max(len(field) + len(grown), 1)
            ),
        })
        if len(grown) == 0:
            closed = True
            break
        frontier = grown.keys
        merged = torch.cat([field.keys, grown.keys])
        order = torch.argsort(merged)
        field = SparseProjectiveTSDF(
            keys=merged[order],
            value=torch.cat([field.value, grown.value])[order],
            support_count=torch.cat([field.support_count, grown.support_count])[order],
            h=h, mu=mu,
        )
        del shell, grown, merged, order
    report: dict[str, Any] = {
        "closed": closed,
        "rounds_run": round_index,
        "max_rounds": max_rounds,
        "rounds": history,
        "closure_certificate": (
            "the final 1-voxel shell around the authoritative set was tested in full and produced no new "
            "authoritative voxel, so the enumeration is a fixed point of the truncation rule -- no radius was chosen"
        ) if closed else (
            "NOT CLOSED at max_rounds. The residual growth is reported per round and attributed below; the "
            "authoritative set is a strict SUBSET of the true one, so anything it omits is omitted surface, "
            "never invented surface."
        ),
    }
    if not closed and len(field):
        # Attribute the residual growth rather than hiding it. It is dominated by
        # a handful of extreme far-field median events (sky/grazing surfel-plane
        # intersections at depths of hundreds of world units) whose pixel
        # footprint is thousands of voxels wide, so their truncation band is a
        # very wide single-view sheet that keeps creeping outward one voxel a
        # round. Every voxel in it is still an EXACT truncation-rule pass.
        single = field.support_count == 1
        radius = torch.linalg.norm(voxel_centers(field.keys, h), dim=1)
        cuts = [10.0, 20.0, 50.0, 100.0]
        recent = history[-1].get("new_authoritative", 0) if history else 0
        report["residual_growth_attribution"] = {
            "authoritative_voxels_with_support_count_1": int(single.sum().item()),
            "fraction_support_count_1": float(single.to(torch.float64).mean().item()),
            "last_round_new_authoritative": int(recent),
            "last_round_new_as_fraction_of_total": float(recent) / max(len(field), 1),
            "voxel_distance_from_world_origin": {
                f"fraction_beyond_{cut:g}_world_units": float((radius > cut).to(torch.float64).mean().item())
                for cut in cuts
            },
            "note": (
                "the un-closed frontier is far-field single-view band, not scene body; the reconstruction is a "
                "strict SUBSET of the true authoritative set, so what it omits is omitted surface, never "
                "invented surface"
            ),
        }
    return field, report
