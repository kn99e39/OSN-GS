from __future__ import annotations

"""3DGS-style Torch Gaussian parameter container.

기존 Inria 3DGS의 `GaussianModel`이 rasterizer와 약속하는 핵심 property
(`get_xyz`, `get_features`, `get_opacity`, `get_scaling`, `get_rotation`)를
OSN-GS 내부 모델도 제공한다. 덕분에 CUDA rasterizer adapter가 이 모델을
거의 같은 방식으로 넘길 수 있다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osn_gs.utils.torch_ops import inverse_sigmoid, quaternion_identity, require_torch, rgb_to_sh_dc, sh_dc_to_rgb


@dataclass
class GaussianParameterGroups:
    """Gaussian parameter group별 learning rate."""

    # Gaussian center 위치.
    xyz_lr: float = 1.6e-4
    # SH DC color coefficient.
    feature_lr: float = 2.5e-3
    # raw opacity logit.
    opacity_lr: float = 2.5e-2
    # log scale parameter.
    scaling_lr: float = 5.0e-3
    # quaternion rotation.
    rotation_lr: float = 1.0e-3
    # OSN-GS에서 추가한 uncertain confidence logit.
    confidence_lr: float = 1.0e-3


class TorchGaussianModel:
    """Certain/uncertain Gaussian을 하나의 parameter tensor 묶음으로 보관한다."""

    def __init__(self, sh_degree: int = 3, device: str = "cuda") -> None:
        torch = require_torch()
        self.torch = torch
        self.device = device
        # active_sh_degree는 학습 중 점진적으로 증가한다.
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        # 아래 Parameter들은 initialize 전에는 빈 tensor다.
        # initialize 이후 optimizer가 이 Parameter들을 잡는다.
        self._xyz = torch.nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=device))
        self._features_dc = torch.nn.Parameter(torch.empty((0, 1, 3), dtype=torch.float32, device=device))
        self._features_rest = torch.nn.Parameter(
            torch.empty((0, (sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=device)
        )
        self._scaling = torch.nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=device))
        self._rotation = torch.nn.Parameter(torch.empty((0, 4), dtype=torch.float32, device=device))
        self._opacity = torch.nn.Parameter(torch.empty((0, 1), dtype=torch.float32, device=device))
        self._confidence = torch.nn.Parameter(torch.empty((0, 1), dtype=torch.float32, device=device))

        # OSN-GS metadata. optimizer 대상은 아니지만 loss/policy에서 필요하다.
        self.is_uncertain = torch.empty((0,), dtype=torch.bool, device=device)
        self.surface_uv = torch.empty((0, 2), dtype=torch.float32, device=device)
        self.cluster_ids = torch.empty((0,), dtype=torch.long, device=device)
        self.optimizer: Any | None = None

    @property
    def get_xyz(self) -> Any:
        # Rasterizer가 직접 gradient를 흘리는 Gaussian center.
        return self._xyz

    @property
    def get_scaling(self) -> Any:
        # 3DGS convention: raw scale은 log domain에 두고 exp로 양수화한다.
        return self.torch.exp(self._scaling)

    @property
    def get_rotation(self) -> Any:
        # quaternion은 normalize해서 rotation parameter로 사용한다.
        return self.torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> Any:
        # opacity도 logit으로 들고 sigmoid로 [0, 1] 범위에 둔다.
        return self.torch.sigmoid(self._opacity)

    @property
    def get_confidence(self) -> Any:
        # OSN-GS 전용: uncertain Gaussian의 구조 신뢰도.
        return self.torch.sigmoid(self._confidence)

    @property
    def get_features(self) -> Any:
        # CUDA rasterizer가 기대하는 SH feature tensor.
        return self.torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_features_dc(self) -> Any:
        return self._features_dc

    @property
    def get_features_rest(self) -> Any:
        return self._features_rest

    @property
    def rgb(self) -> Any:
        # 현재 color 초기화/저장은 SH DC만 RGB로 되돌려 사용한다.
        return self.torch.clamp(sh_dc_to_rgb(self._features_dc[:, 0, :]), 0.0, 1.0)

    def __len__(self) -> int:
        return int(self._xyz.shape[0])

    def oneup_sh_degree(self) -> None:
        # 기존 3DGS와 동일하게 coarse-to-fine color 표현을 위해 SH degree를 올린다.
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
        """Gaussian parameter tensor들을 새 값으로 초기화한다.

        prune/rebuild/append에서도 이 함수를 재사용한다. 이 방식은 optimizer state를
        보존하지는 않지만, 연구 초기 단계에서 shape 변경을 단순하게 처리할 수 있다.
        """

        torch = self.torch
        # 모든 입력은 device-local float tensor로 통일한다.
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        count = positions.shape[0]

        # opacity/scale이 없으면 3DGS 초기값에 가까운 작은 Gaussian으로 시작한다.
        if opacities is None:
            opacities = torch.full((count, 1), 0.1, dtype=torch.float32, device=self.device)
        else:
            opacities = torch.as_tensor(opacities, dtype=torch.float32, device=self.device).reshape(count, 1)
        if scales is None:
            scales = torch.full((count, 3), 0.02, dtype=torch.float32, device=self.device)
        else:
            scales = torch.as_tensor(scales, dtype=torch.float32, device=self.device).reshape(count, 3)

        # uncertain_mask는 certain/uncertain을 loss와 density policy에서 분기하는 핵심 flag다.
        if uncertain_mask is None:
            uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        else:
            uncertain_mask = torch.as_tensor(uncertain_mask, dtype=torch.bool, device=self.device).reshape(count)

        # surface_uv는 uncertain Gaussian이 NURBS surface의 어느 parameter에 묶였는지 저장한다.
        if surface_uv is None:
            surface_uv = torch.zeros((count, 2), dtype=torch.float32, device=self.device)
        else:
            surface_uv = torch.as_tensor(surface_uv, dtype=torch.float32, device=self.device).reshape(count, 2)

        # cluster_ids는 color prior/ADC pattern transfer를 위한 hook이다.
        if cluster_ids is None:
            cluster_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        else:
            cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device).reshape(count)

        # 기본 confidence는 certain=1, uncertain=0.25로 둔다.
        if confidence is None:
            confidence = torch.where(
                uncertain_mask[:, None],
                torch.full((count, 1), 0.25, dtype=torch.float32, device=self.device),
                torch.ones((count, 1), dtype=torch.float32, device=self.device),
            )
        else:
            confidence = torch.as_tensor(confidence, dtype=torch.float32, device=self.device).reshape(count, 1)

        rest_dim = (self.max_sh_degree + 1) ** 2 - 1

        # 학습 안정성을 위해 실제 constrained 값 대신 unconstrained raw parameter를 둔다.
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

    def training_setup(self, groups: GaussianParameterGroups) -> None:
        """Parameter group별 learning rate로 Adam optimizer를 만든다."""

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

    def append_uncertain(self, positions: Any, colors: Any, surface_uv: Any, cluster_ids: Any, opacity: float, scale: float) -> None:
        """기존 model 뒤에 uncertain Gaussian을 추가한다.

        현재 pipeline은 rebuild 시 initialize를 주로 쓰지만, 나중에 online surface
        sampling이나 ADC 기반 uncertain densification을 넣을 때 이 함수가 사용된다.
        """

        torch = self.torch
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        surface_uv = torch.as_tensor(surface_uv, dtype=torch.float32, device=self.device)
        cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device)
        count = positions.shape[0]
        if count == 0:
            return
        # shape이 바뀌므로 단순하게 전체 tensor를 재초기화한다.
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
        """keep_mask가 False인 Gaussian을 제거한다."""

        torch = self.torch
        keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=self.device)
        self.initialize(
            positions=self._xyz.detach()[keep_mask],
            colors=self.rgb.detach()[keep_mask],
            opacities=self.get_opacity.detach()[keep_mask],
            scales=self.get_scaling.detach()[keep_mask],
            uncertain_mask=self.is_uncertain[keep_mask],
            surface_uv=self.surface_uv[keep_mask],
            cluster_ids=self.cluster_ids[keep_mask],
            confidence=self.get_confidence.detach()[keep_mask],
        )

    def save_ply(self, path: str | Path) -> None:
        """현재 Gaussian들을 ASCII PLY로 저장한다.

        표준 3DGS PLY 전체 속성을 모두 쓰지는 않지만, x/y/z, RGB, opacity,
        scale, uncertain flag를 저장해 시각화와 디버깅에는 충분하게 둔다.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self._xyz.detach().cpu()
        colors = (self.rgb.detach().cpu().clamp(0.0, 1.0) * 255.0).to(self.torch.uint8)
        opacity = self.get_opacity.detach().cpu()
        scales = self.get_scaling.detach().cpu()
        uncertain = self.is_uncertain.detach().cpu().to(self.torch.int32)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(self)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property float opacity\n")
            handle.write("property float scale_x\nproperty float scale_y\nproperty float scale_z\n")
            handle.write("property int uncertain\nend_header\n")
            for idx in range(len(self)):
                x, y, z = xyz[idx].tolist()
                r, g, b = colors[idx].tolist()
                op = float(opacity[idx, 0])
                sx, sy, sz = scales[idx].tolist()
                flag = int(uncertain[idx])
                handle.write(f"{x} {y} {z} {r} {g} {b} {op} {sx} {sy} {sz} {flag}\n")
