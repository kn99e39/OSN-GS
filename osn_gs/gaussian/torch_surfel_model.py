from __future__ import annotations

"""2DGS surface-element (surfel) primitive container.

Implements Section 4.1 of

    Huang, Yu, Chen, Geiger, Gao. "2D Gaussian Splatting for Geometrically
    Accurate Radiance Fields." SIGGRAPH 2024 / ACM TOG. arXiv:2403.17888v3

on top of OSN-GS's `TorchGaussianModel`, matching the official implementation
`hbb1/2d-gaussian-splatting` @ 335ad61 (`scene/gaussian_model.py`).

Parameterization
----------------

Each surfel k is::

    center      p_k in R^3                          -> `_xyz`         (N, 3)
    orientation R = [t_u, t_v, t_w]                 -> `_rotation`    (N, 4) wxyz quaternion
    normal      t_w = t_u x t_v                     -> DERIVED, third column of R
    scales      (s_u, s_v)                          -> `_scaling`     (N, 2), log domain
    opacity     alpha                               -> `_opacity`     (N, 1), logit
    appearance  SH                                  -> `_features_dc` / `_features_rest`

with the local surface parameterization (paper eq. 4)::

    P(u, v) = p_k + s_u t_u u + s_v t_v v = H (u, v, 1, 1)^T

and the local Gaussian kernel (paper eq. 6)::

    G(u, v) = exp(-(u^2 + v^2) / 2)

THE THIRD SCALE DOES NOT EXIST. `_scaling` has exactly two columns
(`scale_dim = 2`), so there is no tensor entry a gradient step, a densification
rule, a checkpoint load, or an opacity reset could push away from zero. This is
the structural difference from "a 3D Gaussian with a small third scale": the
normal direction has no trainable extent at all. The paper's scale matrix
S = diag(s_u, s_v, 0) is realized by the two-column tensor; the official CUDA
`scale_to_mat` leaves the transform's third column at unit length precisely
because it carries the UNIT normal t_w of eq. 5's H, not a variance.

The normal is intrinsic: it is read off the orientation, never stored or
optimized separately, so it cannot disagree with the tangent frame.

Legacy OSN-GS analysis code that wants a 3x3-covariance-shaped tensor must go
through `osn_gs/gaussian/torch_surfel_analysis_adapter.py`. That adapter is
explicitly read-only and is never the training representation.

Fidelity: PAPER_FAITHFUL and OFFICIAL_CODE_FAITHFUL. `get_splat2world` is the
official `GaussianModel.get_covariance`/`build_covariance_from_scaling_rotation`
verbatim in behavior.
"""

from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.utils.torch_ops import require_torch


def build_rotation(quaternions: Any) -> Any:
    """wxyz quaternions -> (N, 3, 3) rotation matrices with columns [t_u, t_v, t_w].

    Behavioral port of the official `utils/general_utils.py::build_rotation`
    (device follows the input instead of being hard-coded to cuda). Column
    convention verified against the vendored CUDA `quat_to_rotmat`
    (`cuda_rasterizer/auxiliary.h`), which is column-major glm and produces the
    same matrix, so `matrix[:, :, 2]` is the same `t_w` the rasterizer uses.
    """

    torch = require_torch()
    r = quaternions
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    matrix = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - w * z)
    matrix[:, 0, 2] = 2 * (x * z + w * y)
    matrix[:, 1, 0] = 2 * (x * y + w * z)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - w * x)
    matrix[:, 2, 0] = 2 * (x * z - w * y)
    matrix[:, 2, 1] = 2 * (y * z + w * x)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def build_scaling_rotation(scales: Any, quaternions: Any) -> Any:
    """R @ diag(s), the official `build_scaling_rotation` (device-portable)."""

    torch = require_torch()
    rotation = build_rotation(quaternions)
    diagonal = torch.zeros((scales.shape[0], 3, 3), dtype=scales.dtype, device=scales.device)
    diagonal[:, 0, 0] = scales[:, 0]
    diagonal[:, 1, 1] = scales[:, 1]
    diagonal[:, 2, 2] = scales[:, 2]
    return rotation @ diagonal


class TorchGaussianSurfelModel(TorchGaussianModel):
    """2D Gaussian surface element bundle. See module docstring."""

    # Two trainable tangent scales. There is no third axis to grow.
    scale_dim = 2
    ply_scale_properties = ("scale_0", "scale_1")

    @property
    def get_rotation_matrix(self) -> Any:
        """(N, 3, 3) orientation R = [t_u, t_v, t_w] from the normalized quaternion."""

        return build_rotation(self.get_rotation)

    @property
    def get_tangent_u(self) -> Any:
        return self.get_rotation_matrix[:, :, 0]

    @property
    def get_tangent_v(self) -> Any:
        return self.get_rotation_matrix[:, :, 1]

    @property
    def get_normal(self) -> Any:
        """Intrinsic surfel normal t_w = t_u x t_v, i.e. R's third column.

        Derived, never stored: it is definitionally orthogonal to the tangent
        plane. The rasterizer additionally flips its sign toward the camera
        (`DUAL_VISIABLE` in the vendored `forward.cu`); the un-flipped
        orientation is what this property returns.
        """

        return self.get_rotation_matrix[:, :, 2]

    def get_splat2world(self, scaling_modifier: float = 1.0) -> Any:
        """Official `build_covariance_from_scaling_rotation` output, (N, 4, 4).

        Behavioral port of the official 2DGS
        `GaussianModel.get_covariance`/`build_covariance_from_scaling_rotation`.
        The matrix is stored ROW-wise: row 0 is ``s_u t_u``, row 1 is
        ``s_v t_v``, row 2 is the UNIT normal ``t_w``, row 3 is
        ``(p_k, 1)``, so a local point multiplies from the left.

        Row 2 is a placeholder, NOT part of the paper's H. The paper's eq. 5
        has a zero third column; upstream fills the slot with the unit normal
        (`scale_to_mat` leaves ``S[2][2] = 1``, i.e. it is not a learnable
        scale) and every consumer then discards it:

        * the CUDA `compute_transmat` builds a THREE-column `glm::mat3x4`
          from ``L[0], L[1], p_k`` -- row 2 never exists there;
        * the official `compute_cov3D_python` path indexes
          ``splat2world[:, [0, 1, 3]]``, dropping it explicitly.

        Use `splat_to_world_uv1` for the paper's H itself. This method exists
        so the port stays checkable against upstream line by line.
        """

        torch = self.torch
        scaling = self.get_scaling
        stacked = torch.cat([scaling * float(scaling_modifier), torch.ones_like(scaling[:, :1])], dim=-1)
        rs = build_scaling_rotation(stacked, self._rotation).permute(0, 2, 1)
        transform = torch.zeros((self._xyz.shape[0], 4, 4), dtype=torch.float32, device=self.device)
        transform[:, :3, :3] = rs
        transform[:, 3, :3] = self.get_xyz
        transform[:, 3, 3] = 1
        return transform

    def splat_to_world_uv1(self, scaling_modifier: float = 1.0) -> Any:
        """Paper eq. 5's H as the (N, 3, 4) map ``(u, v, 1) -> world``.

        Rows ``[0, 1, 3]`` of `get_splat2world`, i.e. ``s_u t_u``, ``s_v t_v``,
        ``(p_k, 1)`` -- the same three rows the CUDA `compute_transmat`
        assembles and the same three the official `compute_cov3D_python` path
        selects. Left-multiplying ``(u, v, 1)`` gives::

            P(u, v) = p_k + s_u t_u u + s_v t_v v

        with the paper's zero third column realized by the row simply not
        being there, so no normal-direction extent can enter.
        """

        return self.get_splat2world(scaling_modifier)[:, [0, 1, 3], :]

    def get_covariance(self, scaling_modifier: float = 1.0) -> Any:
        """Official API name for `get_splat2world`. Kept for port fidelity.

        NOTE this is NOT a 3x3 covariance despite the inherited upstream name:
        2DGS's `GaussianModel.get_covariance` returns the 4x4 splat-to-world
        transform. Anything wanting a covariance-shaped tensor for legacy
        OSN-GS structural analysis must use
        `osn_gs.gaussian.torch_surfel_analysis_adapter`.
        """

        return self.get_splat2world(scaling_modifier)
