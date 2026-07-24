from __future__ import annotations

"""Surface-agnostic bilinear Coons (transfinite) patch seed.

Phase F prerequisite (docs/Urgent_Work/OSN_GS_Phase_F_Constrained_Occluded_Chart_Design.md
section 9). Given four boundary curves sampled on a common grid, produce a
bilinearly-blended transfinite interpolation surface. Used only to SEED the
constrained single-chart NURBS fit -- never accepted as the final surface.

This module knows nothing about NURBS, candidates, or occlusion: it operates on
plain sampled boundary curves.
"""

from typing import Any

from osn_gs.utils.torch_ops import require_torch


def coons_bilinear_patch(
    curve_v0: Any,
    curve_v1: Any,
    curve_u0: Any,
    curve_u1: Any,
    *,
    atol: float = 1e-6,
) -> Any:
    """Bilinear Coons transfinite patch as an ``(Nu, Nv, 3)`` sampled grid.

    - ``curve_v0`` / ``curve_v1``: the two ``v=0`` / ``v=1`` boundaries, each
      ``(Nu, 3)``, parameterized by ``u`` on a common grid.
    - ``curve_u0`` / ``curve_u1``: the two ``u=0`` / ``u=1`` boundaries, each
      ``(Nv, 3)``, parameterized by ``v`` on a common grid.

    Corners must agree (``curve_v0[0] == curve_u0[0]`` etc.) within ``atol``;
    callers that build connectors as straight lines between the support-chain
    endpoints get this for free. The standard formula is::

        S(u,v) = (1-v) Cv0(u) + v Cv1(u)
               + (1-u) Cu0(v) + u Cu1(v)
               - [ (1-u)(1-v) P00 + u(1-v) P10 + (1-u)v P01 + uv P11 ]
    """

    torch = require_torch()
    nu = int(curve_v0.shape[0])
    nv = int(curve_u0.shape[0])
    if int(curve_v1.shape[0]) != nu or int(curve_u1.shape[0]) != nv:
        raise ValueError("Opposite boundary curves must share sample counts.")
    if nu < 2 or nv < 2:
        raise ValueError("Coons patch needs at least two samples per boundary.")

    dtype, device = curve_v0.dtype, curve_v0.device
    p00, p10 = curve_v0[0], curve_v0[-1]
    p01, p11 = curve_v1[0], curve_v1[-1]
    # Corner-consistency check (diagnostic-grade, not silently reconciled).
    for corner_a, corner_b, name in (
        (curve_v0[0], curve_u0[0], "v0/u0 start"),
        (curve_v0[-1], curve_u1[0], "v0/u1 start"),
        (curve_v1[0], curve_u0[-1], "v1/u0 end"),
        (curve_v1[-1], curve_u1[-1], "v1/u1 end"),
    ):
        if float((corner_a - corner_b).norm()) > atol:
            raise ValueError(f"Coons corner mismatch at {name} exceeds atol={atol}.")

    u = torch.linspace(0.0, 1.0, nu, dtype=dtype, device=device)
    v = torch.linspace(0.0, 1.0, nv, dtype=dtype, device=device)
    uu = u[:, None]  # (Nu, 1)
    vv = v[None, :]  # (1, Nv)

    ruled_v = (1.0 - vv)[..., None] * curve_v0[:, None, :] + vv[..., None] * curve_v1[:, None, :]
    ruled_u = (1.0 - uu)[..., None] * curve_u0[None, :, :] + uu[..., None] * curve_u1[None, :, :]
    bilinear = (
        ((1.0 - uu) * (1.0 - vv))[..., None] * p00
        + (uu * (1.0 - vv))[..., None] * p10
        + ((1.0 - uu) * vv)[..., None] * p01
        + (uu * vv)[..., None] * p11
    )
    return ruled_v + ruled_u - bilinear
