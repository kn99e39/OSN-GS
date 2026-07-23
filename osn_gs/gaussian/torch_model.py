from __future__ import annotations

"""3DGS-style Torch Gaussian parameter container.

Exposes the same core properties the original Inria 3DGS `GaussianModel`
gives the rasterizer (`get_xyz`, `get_features`, `get_opacity`,
`get_scaling`, `get_rotation`), so the CUDA rasterizer adapter can consume
this model the same way.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from osn_gs.utils.torch_ops import inverse_sigmoid, quaternion_identity, require_torch, rgb_to_sh_dc, sh_dc_to_rgb


@dataclass
class GaussianParameterGroups:
    """Learning rate per Gaussian parameter group."""

    # Gaussian center position.
    xyz_lr: float = 1.6e-4
    xyz_lr_final: float = 1.6e-6
    xyz_lr_delay_mult: float = 0.01
    xyz_lr_max_steps: int = 30000
    # SH DC color coefficient.
    feature_lr: float = 2.5e-3
    # raw opacity logit. Matches original 3DGS (arguments/__init__.py: opacity_lr=0.025).
    opacity_lr: float = 2.5e-2
    # log scale parameter.
    scaling_lr: float = 5.0e-3
    # quaternion rotation.
    rotation_lr: float = 1.0e-3
    # OSN-GS-specific uncertain confidence logit.
    confidence_lr: float = 1.0e-3


class TorchGaussianModel:
    """Holds certain/uncertain Gaussians as a single bundle of parameter tensors."""

    def __init__(self, sh_degree: int = 3, device: str = "cuda") -> None:
        torch = require_torch()
        self.torch = torch
        self.device = device
        # active_sh_degree increases gradually during training.
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        # Parameters below are empty tensors until initialize() is called;
        # the optimizer is built after initialize() sets real parameters.
        self._xyz = torch.nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=device))
        self._features_dc = torch.nn.Parameter(torch.empty((0, 1, 3), dtype=torch.float32, device=device))
        self._features_rest = torch.nn.Parameter(
            torch.empty((0, (sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=device)
        )
        self._scaling = torch.nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=device))
        self._rotation = torch.nn.Parameter(torch.empty((0, 4), dtype=torch.float32, device=device))
        self._opacity = torch.nn.Parameter(torch.empty((0, 1), dtype=torch.float32, device=device))
        self._confidence = torch.nn.Parameter(torch.empty((0, 1), dtype=torch.float32, device=device))

        # OSN-GS metadata. Not optimizer targets, but needed by loss/policy code.
        self.is_uncertain = torch.empty((0,), dtype=torch.bool, device=device)
        self.surface_uv = torch.empty((0, 2), dtype=torch.float32, device=device)
        self.cluster_ids = torch.empty((0,), dtype=torch.long, device=device)
        self.optimizer: Any | None = None
        self.spatial_lr_scale: float = 1.0
        self.xyz_gradient_accum = torch.empty((0, 1), dtype=torch.float32, device=device)
        self.denom = torch.empty((0, 1), dtype=torch.float32, device=device)
        self.max_radii2D = torch.empty((0,), dtype=torch.float32, device=device)

    @property
    def get_xyz(self) -> Any:
        # Gaussian center the rasterizer receives gradients on directly.
        return self._xyz

    @property
    def get_scaling(self) -> Any:
        # 3DGS convention: raw scale is stored in log domain and exponentiated for use.
        return self.torch.exp(self._scaling)

    @property
    def get_rotation(self) -> Any:
        # Quaternion is normalized before use as a rotation parameter.
        return self.torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> Any:
        # Opacity is stored as a logit and mapped to [0, 1] with sigmoid.
        return self.torch.sigmoid(self._opacity)

    @property
    def get_confidence(self) -> Any:
        # OSN-GS-specific: structural reliability of an uncertain Gaussian.
        return self.torch.sigmoid(self._confidence)

    @property
    def surface_patch_ids(self) -> Any:
        """Persistent NURBS patch binding for every Gaussian."""

        return self.cluster_ids

    @property
    def get_features(self) -> Any:
        # SH feature tensor expected by the CUDA rasterizer.
        return self.torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_features_dc(self) -> Any:
        return self._features_dc

    @property
    def get_features_rest(self) -> Any:
        return self._features_rest

    @property
    def rgb(self) -> Any:
        # Only used for current color initialization; converts SH DC back to RGB.
        return self.torch.clamp(sh_dc_to_rgb(self._features_dc[:, 0, :]), 0.0, 1.0)

    def __len__(self) -> int:
        return int(self._xyz.shape[0])

    def oneup_sh_degree(self) -> None:
        # Raises SH degree for coarse-to-fine color representation, matching original 3DGS.
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def initialize(
        self,
        positions: Any,
        colors: Any,
        opacities: Any | None = None,
        scales: Any | None = None,
        uncertain_mask: Any | None = None,
        surface_uv: Any | None = None,
        cluster_ids: Any | None = None,
        confidence: Any | None = None,
    ) -> None:
        """Initialize Gaussian parameter tensors from raw values.

        Reused by prune/rebuild/append call sites. This method does not
        preserve optimizer state; the initial construction stage just needs
        to handle shape changes simply.
        """

        torch = self.torch
        # Coerce every input to a device-local float tensor.
        positions = torch.as_tensor(positions, dtype=self.torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=self.torch.float32, device=self.device)
        count = positions.shape[0]

        # Missing opacity/scale falls back to a small Gaussian close to the 3DGS initial value.
        if opacities is None:
            opacities = torch.full((count, 1), 0.1, dtype=self.torch.float32, device=self.device)
        else:
            opacities = torch.as_tensor(opacities, dtype=self.torch.float32, device=self.device).reshape(count, 1)
        if scales is None:
            scales = torch.full((count, 3), 0.02, dtype=self.torch.float32, device=self.device)
        else:
            scales = torch.as_tensor(scales, dtype=self.torch.float32, device=self.device).reshape(count, 3)

        # uncertain_mask is the core flag separating certain/uncertain in loss and density policy.
        if uncertain_mask is None:
            uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        else:
            uncertain_mask = torch.as_tensor(uncertain_mask, dtype=torch.bool, device=self.device).reshape(count)

        # surface_uv stores which NURBS surface parameter an uncertain Gaussian was derived from.
        if surface_uv is None:
            surface_uv = torch.zeros((count, 2), dtype=self.torch.float32, device=self.device)
        else:
            surface_uv = torch.as_tensor(surface_uv, dtype=self.torch.float32, device=self.device).reshape(count, 2)

        # cluster_ids is a hook for color prior / ADC pattern transfer.
        if cluster_ids is None:
            cluster_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        else:
            cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device).reshape(count)

        # Default confidence: certain=1, uncertain=0.25.
        if confidence is None:
            confidence = torch.where(
                uncertain_mask[:, None],
                torch.full((count, 1), 0.25, dtype=self.torch.float32, device=self.device),
                torch.ones((count, 1), dtype=self.torch.float32, device=self.device),
            )
        else:
            confidence = torch.as_tensor(confidence, dtype=self.torch.float32, device=self.device).reshape(count, 1)

        rest_dim = (self.max_sh_degree + 1) ** 2 - 1

        # Store unconstrained raw parameters so constrained values can be recovered via the get_* properties for training stability.
        self._xyz = torch.nn.Parameter(positions.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(rgb_to_sh_dc(colors).reshape(count, 1, 3).requires_grad_(True))
        self._features_rest = torch.nn.Parameter(torch.zeros((count, rest_dim, 3), device=self.device).requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.log(torch.clamp(scales, min=1e-6)).requires_grad_(True))
        self._rotation = torch.nn.Parameter(quaternion_identity(count, self.device).requires_grad_(True))
        self._opacity = torch.nn.Parameter(inverse_sigmoid(opacities).requires_grad_(True))
        self._confidence = torch.nn.Parameter(inverse_sigmoid(confidence).requires_grad_(True))
        self.is_uncertain = uncertain_mask
        self.surface_uv = surface_uv
        self.cluster_ids = cluster_ids
        self._reset_density_stats(count)

    def _reset_density_stats(self, count: int | None = None) -> None:
        """Reset ADC accumulators after shape-changing Gaussian edits."""

        if count is None:
            count = len(self)
        self.xyz_gradient_accum = self.torch.zeros((count, 1), dtype=self.torch.float32, device=self.device)
        self.denom = self.torch.zeros((count, 1), dtype=self.torch.float32, device=self.device)
        self.max_radii2D = self.torch.zeros((count,), dtype=self.torch.float32, device=self.device)

    def replace_tensors(
        self,
        xyz: Any,
        features_dc: Any,
        features_rest: Any,
        opacity: Any,
        scaling: Any,
        rotation: Any,
        confidence: Any,
        uncertain_mask: Any,
        surface_uv: Any,
        cluster_ids: Any,
        optimizer_keep_indices: Any | None = None,
    ) -> None:
        """Replace Gaussian tensors while preserving Adam rows when possible."""

        torch = self.torch
        xyz = torch.as_tensor(xyz, dtype=self.torch.float32, device=self.device)
        count = int(xyz.shape[0])
        rest_dim = (self.max_sh_degree + 1) ** 2 - 1
        old_count = len(self)
        old_params = self._optimizer_named_params()
        old_gradients = {
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in old_params.items()
        }
        keep_indices = None
        if optimizer_keep_indices is not None:
            keep_indices = torch.as_tensor(optimizer_keep_indices, dtype=torch.long, device=self.device).reshape(-1)
        self._xyz = torch.nn.Parameter(xyz.detach().clone().requires_grad_(True))
        self._features_dc = torch.nn.Parameter(torch.as_tensor(features_dc, dtype=self.torch.float32, device=self.device).reshape(count, 1, 3).detach().clone().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(torch.as_tensor(features_rest, dtype=self.torch.float32, device=self.device).reshape(count, rest_dim, 3).detach().clone().requires_grad_(True))
        self._opacity = torch.nn.Parameter(torch.as_tensor(opacity, dtype=self.torch.float32, device=self.device).reshape(count, 1).detach().clone().requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.as_tensor(scaling, dtype=self.torch.float32, device=self.device).reshape(count, 3).detach().clone().requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.as_tensor(rotation, dtype=self.torch.float32, device=self.device).reshape(count, 4).detach().clone().requires_grad_(True))
        self._confidence = torch.nn.Parameter(torch.as_tensor(confidence, dtype=self.torch.float32, device=self.device).reshape(count, 1).detach().clone().requires_grad_(True))
        self.is_uncertain = torch.as_tensor(uncertain_mask, dtype=torch.bool, device=self.device).reshape(count)
        self.surface_uv = torch.as_tensor(surface_uv, dtype=self.torch.float32, device=self.device).reshape(count, 2)
        self.cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device).reshape(count)
        self._preserve_optimizer_state(old_params, keep_indices, old_count)
        self._preserve_parameter_gradients(old_gradients, keep_indices, old_count)
        self._reset_density_stats(count)

    def _preserve_parameter_gradients(
        self, old_gradients: dict[str, Any], keep_indices: Any | None, old_count: int
    ) -> None:
        new_params = self._optimizer_named_params()
        for name, new_param in new_params.items():
            old_grad = old_gradients.get(name)
            if old_grad is None:
                continue
            preserved = self.torch.zeros_like(new_param)
            if keep_indices is None:
                rows = min(old_count, int(new_param.shape[0]))
                preserved[:rows] = old_grad[:rows]
            else:
                rows = min(int(keep_indices.numel()), int(new_param.shape[0]))
                if rows > 0:
                    preserved[:rows] = old_grad[keep_indices[:rows]]
            new_param.grad = preserved

    def _optimizer_named_params(self) -> dict[str, Any]:
        return {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "confidence": self._confidence,
        }

    def _preserve_optimizer_state(self, old_params: dict[str, Any], keep_indices: Any | None, old_count: int) -> None:
        if self.optimizer is None:
            return
        torch = self.torch
        new_params = self._optimizer_named_params()
        for group in self.optimizer.param_groups:
            name = group.get("name")
            old_param = old_params.get(name)
            new_param = new_params.get(name)
            if old_param is None or new_param is None:
                continue
            group["params"] = [new_param]
            old_state = self.optimizer.state.pop(old_param, {})
            new_state = {}
            for key, value in old_state.items():
                if torch.is_tensor(value) and value.shape[:1] == (old_count,):
                    preserved = torch.zeros_like(new_param.data)
                    if keep_indices is None:
                        rows = min(old_count, int(new_param.shape[0]))
                        if rows > 0:
                            preserved[:rows] = value[:rows]
                    else:
                        rows = min(int(keep_indices.numel()), int(new_param.shape[0]))
                        if rows > 0:
                            preserved[:rows] = value[keep_indices[:rows]]
                    new_state[key] = preserved
                else:
                    new_state[key] = value
            self.optimizer.state[new_param] = new_state

    def append_gaussians_raw(
        self,
        xyz: Any,
        features_dc: Any,
        features_rest: Any,
        opacity: Any,
        scaling: Any,
        rotation: Any,
        confidence: Any | None = None,
        uncertain_mask: Any | None = None,
        surface_uv: Any | None = None,
        cluster_ids: Any | None = None,
    ) -> None:
        """Append raw Gaussian parameters, used by ADC clone/split."""

        torch = self.torch
        xyz = torch.as_tensor(xyz, dtype=self.torch.float32, device=self.device)
        count = int(xyz.shape[0])
        if count == 0:
            return
        if confidence is None:
            confidence = torch.full((count, 1), 12.0, dtype=self.torch.float32, device=self.device)
        if uncertain_mask is None:
            uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        if surface_uv is None:
            surface_uv = torch.zeros((count, 2), dtype=self.torch.float32, device=self.device)
        if cluster_ids is None:
            cluster_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        keep_indices = torch.arange(len(self), dtype=torch.long, device=self.device)
        self.replace_tensors(
            xyz=torch.cat([self._xyz.detach(), xyz.detach()], dim=0),
            features_dc=torch.cat([self._features_dc.detach(), torch.as_tensor(features_dc, dtype=self.torch.float32, device=self.device).reshape(count, 1, 3).detach()], dim=0),
            features_rest=torch.cat([self._features_rest.detach(), torch.as_tensor(features_rest, dtype=self.torch.float32, device=self.device).reshape(count, (self.max_sh_degree + 1) ** 2 - 1, 3).detach()], dim=0),
            opacity=torch.cat([self._opacity.detach(), torch.as_tensor(opacity, dtype=self.torch.float32, device=self.device).reshape(count, 1).detach()], dim=0),
            scaling=torch.cat([self._scaling.detach(), torch.as_tensor(scaling, dtype=self.torch.float32, device=self.device).reshape(count, 3).detach()], dim=0),
            rotation=torch.cat([self._rotation.detach(), torch.as_tensor(rotation, dtype=self.torch.float32, device=self.device).reshape(count, 4).detach()], dim=0),
            confidence=torch.cat([self._confidence.detach(), torch.as_tensor(confidence, dtype=self.torch.float32, device=self.device).reshape(count, 1).detach()], dim=0),
            uncertain_mask=torch.cat([self.is_uncertain, torch.as_tensor(uncertain_mask, dtype=torch.bool, device=self.device).reshape(count)], dim=0),
            surface_uv=torch.cat([self.surface_uv, torch.as_tensor(surface_uv, dtype=self.torch.float32, device=self.device).reshape(count, 2)], dim=0),
            cluster_ids=torch.cat([self.cluster_ids, torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device).reshape(count)], dim=0),
            optimizer_keep_indices=keep_indices,
        )

    def training_setup(self, groups: GaussianParameterGroups) -> None:
        """Build the Adam optimizer with a per-parameter-group learning rate."""

        torch = self.torch
        params = [
            {"params": [self._xyz], "lr": groups.xyz_lr, "name": "xyz"},
            {"params": [self._features_dc], "lr": groups.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": groups.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": groups.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": groups.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": groups.rotation_lr, "name": "rotation"},
            {"params": [self._confidence], "lr": groups.confidence_lr, "name": "confidence"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self._parameter_groups = groups

    def update_learning_rate(self, iteration: int) -> float:
        """Apply the original 3DGS-style exponential position LR schedule."""

        groups = getattr(self, "_parameter_groups", GaussianParameterGroups())
        maximum = max(1, int(groups.xyz_lr_max_steps))
        progress = min(max(float(iteration) / maximum, 0.0), 1.0)
        delay = groups.xyz_lr_delay_mult + (1.0 - groups.xyz_lr_delay_mult) * math.sin(
            0.5 * math.pi * min(progress / 0.01, 1.0)
        )
        scale = max(float(self.spatial_lr_scale), 1e-20)
        initial = max(float(groups.xyz_lr) * scale, 1e-20)
        final = max(float(groups.xyz_lr_final) * scale, 1e-20)
        lr = delay * math.exp(math.log(initial) * (1.0 - progress) + math.log(final) * progress)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get("name") == "xyz":
                    group["lr"] = lr
                    break
        return float(lr)

    def reset_opacity(self, maximum: float = 0.01) -> None:
        """Clamp opacity and clear only its Adam moments."""

        with self.torch.no_grad():
            reset = self.torch.minimum(self.get_opacity, self.torch.full_like(self.get_opacity, float(maximum)))
            self._opacity.copy_(inverse_sigmoid(reset))
        if self.optimizer is not None:
            state = self.optimizer.state.get(self._opacity)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key].zero_()

    def append_uncertain(self, positions: Any, colors: Any, surface_uv: Any, cluster_ids: Any, opacity: float, scale: float) -> None:
        """Append uncertain Gaussians to the existing model.

        The current pipeline mainly relies on rebuild -> initialize, but this
        method exists for future online surface sampling or ADC-driven
        uncertain densification.
        """

        torch = self.torch
        positions = torch.as_tensor(positions, dtype=self.torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=self.torch.float32, device=self.device)
        surface_uv = torch.as_tensor(surface_uv, dtype=self.torch.float32, device=self.device)
        cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device)
        count = positions.shape[0]
        if count == 0:
            return
        # Simplest correct approach: re-initialize the whole tensor set instead of tracking shape deltas.
        self.initialize(
            positions=torch.cat([self._xyz.detach(), positions], dim=0),
            colors=torch.cat([self.rgb.detach(), colors], dim=0),
            opacities=torch.cat(
                [self.get_opacity.detach(), torch.full((count, 1), opacity, device=self.device)], dim=0
            ),
            scales=torch.cat(
                [self.get_scaling.detach(), torch.full((count, 3), scale, device=self.device)], dim=0
            ),
            uncertain_mask=torch.cat([self.is_uncertain, torch.ones((count,), dtype=torch.bool, device=self.device)]),
            surface_uv=torch.cat([self.surface_uv, surface_uv], dim=0),
            cluster_ids=torch.cat([self.cluster_ids, cluster_ids], dim=0),
            confidence=torch.cat(
                [self.get_confidence.detach(), torch.full((count, 1), 0.25, device=self.device)], dim=0
            ),
        )

    def prune(self, keep_mask: Any) -> None:
        """Remove Gaussians where keep_mask is False."""

        torch = self.torch
        keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=self.device)
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
        self.replace_tensors(
            xyz=self._xyz.detach()[keep_mask],
            features_dc=self._features_dc.detach()[keep_mask],
            features_rest=self._features_rest.detach()[keep_mask],
            opacity=self._opacity.detach()[keep_mask],
            scaling=self._scaling.detach()[keep_mask],
            rotation=self._rotation.detach()[keep_mask],
            confidence=self._confidence.detach()[keep_mask],
            uncertain_mask=self.is_uncertain[keep_mask],
            surface_uv=self.surface_uv[keep_mask],
            cluster_ids=self.cluster_ids[keep_mask],
            optimizer_keep_indices=keep_indices,
        )

    def save_ply(self, path: str | Path) -> None:
        """Save renderer-compatible ASCII PLY using a vectorized writer."""

        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = np.column_stack(
            [
                self._xyz.detach().cpu().numpy(),
                self._features_dc.detach().cpu().numpy()[:, 0, :],
                self._opacity.detach().cpu().numpy(),
                self._scaling.detach().cpu().numpy(),
                self.get_rotation.detach().cpu().numpy(),
                self.is_uncertain.detach().cpu().numpy().astype(np.int32),
                self.get_confidence.detach().cpu().numpy(),
            ]
        )
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(self)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "property int uncertain\nproperty float confidence\nend_header"
        )
        formats = ["%.9g"] * 14 + ["%d", "%.9g"]
        np.savetxt(path, columns, fmt=formats, header=header, comments="", encoding="utf-8")
