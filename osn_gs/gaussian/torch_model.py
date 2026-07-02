from __future__ import annotations

"""3DGS-style Torch Gaussian parameter container.

湲곗〈 Inria 3DGS??`GaussianModel`??rasterizer? ?쎌냽?섎뒗 ?듭떖 property
(`get_xyz`, `get_features`, `get_opacity`, `get_scaling`, `get_rotation`)瑜?
OSN-GS ?대? 紐⑤뜽???쒓났?쒕떎. ?뺣텇??CUDA rasterizer adapter媛 ??紐⑤뜽??
嫄곗쓽 媛숈? 諛⑹떇?쇰줈 ?섍만 ???덈떎.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osn_gs.utils.torch_ops import inverse_sigmoid, quaternion_identity, require_torch, rgb_to_sh_dc, sh_dc_to_rgb


@dataclass
class GaussianParameterGroups:
    """Gaussian parameter group蹂?learning rate."""

    # Gaussian center ?꾩튂.
    xyz_lr: float = 1.6e-4
    # SH DC color coefficient.
    feature_lr: float = 2.5e-3
    # raw opacity logit.
    opacity_lr: float = 2.5e-2
    # log scale parameter.
    scaling_lr: float = 5.0e-3
    # quaternion rotation.
    rotation_lr: float = 1.0e-3
    # OSN-GS?먯꽌 異붽???uncertain confidence logit.
    confidence_lr: float = 1.0e-3


class TorchGaussianModel:
    """Certain/uncertain Gaussian???섎굹??parameter tensor 臾띠쓬?쇰줈 蹂닿??쒕떎."""

    def __init__(self, sh_degree: int = 3, device: str = "cuda") -> None:
        torch = require_torch()
        self.torch = torch
        self.device = device
        # active_sh_degree???숈뒿 以??먯쭊?곸쑝濡?利앷??쒕떎.
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        # ?꾨옒 Parameter?ㅼ? initialize ?꾩뿉??鍮?tensor??
        # initialize ?댄썑 optimizer媛 ??Parameter?ㅼ쓣 ?〓뒗??
        self._xyz = torch.nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=device))
        self._features_dc = torch.nn.Parameter(torch.empty((0, 1, 3), dtype=torch.float32, device=device))
        self._features_rest = torch.nn.Parameter(
            torch.empty((0, (sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=device)
        )
        self._scaling = torch.nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=device))
        self._rotation = torch.nn.Parameter(torch.empty((0, 4), dtype=torch.float32, device=device))
        self._opacity = torch.nn.Parameter(torch.empty((0, 1), dtype=torch.float32, device=device))
        self._confidence = torch.nn.Parameter(torch.empty((0, 1), dtype=torch.float32, device=device))

        # OSN-GS metadata. optimizer ??곸? ?꾨땲吏留?loss/policy?먯꽌 ?꾩슂?섎떎.
        self.is_uncertain = torch.empty((0,), dtype=torch.bool, device=device)
        self.surface_uv = torch.empty((0, 2), dtype=torch.float32, device=device)
        self.cluster_ids = torch.empty((0,), dtype=torch.long, device=device)
        self.optimizer: Any | None = None

    @property
    def get_xyz(self) -> Any:
        # Rasterizer媛 吏곸젒 gradient瑜??섎━??Gaussian center.
        return self._xyz

    @property
    def get_scaling(self) -> Any:
        # 3DGS convention: raw scale? log domain???먭퀬 exp濡??묒닔?뷀븳??
        return self.torch.exp(self._scaling)

    @property
    def get_rotation(self) -> Any:
        # quaternion? normalize?댁꽌 rotation parameter濡??ъ슜?쒕떎.
        return self.torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> Any:
        # opacity??logit?쇰줈 ?ㅺ퀬 sigmoid濡?[0, 1] 踰붿쐞???붾떎.
        return self.torch.sigmoid(self._opacity)

    @property
    def get_confidence(self) -> Any:
        # OSN-GS ?꾩슜: uncertain Gaussian??援ъ“ ?좊ː??
        return self.torch.sigmoid(self._confidence)

    @property
    def get_features(self) -> Any:
        # CUDA rasterizer媛 湲곕??섎뒗 SH feature tensor.
        return self.torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_features_dc(self) -> Any:
        return self._features_dc

    @property
    def get_features_rest(self) -> Any:
        return self._features_rest

    @property
    def rgb(self) -> Any:
        # ?꾩옱 color 珥덇린????μ? SH DC留?RGB濡??섎룎???ъ슜?쒕떎.
        return self.torch.clamp(sh_dc_to_rgb(self._features_dc[:, 0, :]), 0.0, 1.0)

    def __len__(self) -> int:
        return int(self._xyz.shape[0])

    def oneup_sh_degree(self) -> None:
        # 湲곗〈 3DGS? ?숈씪?섍쾶 coarse-to-fine color ?쒗쁽???꾪빐 SH degree瑜??щ┛??
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
        """Gaussian parameter tensor?ㅼ쓣 ??媛믪쑝濡?珥덇린?뷀븳??

        prune/rebuild/append?먯꽌?????⑥닔瑜??ъ궗?⑺븳?? ??諛⑹떇? optimizer state瑜?
        蹂댁〈?섏????딆?留? ?곌뎄 珥덇린 ?④퀎?먯꽌 shape 蹂寃쎌쓣 ?⑥닚?섍쾶 泥섎━?????덈떎.
        """

        torch = self.torch
        # 紐⑤뱺 ?낅젰? device-local float tensor濡??듭씪?쒕떎.
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        count = positions.shape[0]

        # opacity/scale???놁쑝硫?3DGS 珥덇린媛믪뿉 媛源뚯슫 ?묒? Gaussian?쇰줈 ?쒖옉?쒕떎.
        if opacities is None:
            opacities = torch.full((count, 1), 0.1, dtype=torch.float32, device=self.device)
        else:
            opacities = torch.as_tensor(opacities, dtype=torch.float32, device=self.device).reshape(count, 1)
        if scales is None:
            scales = torch.full((count, 3), 0.02, dtype=torch.float32, device=self.device)
        else:
            scales = torch.as_tensor(scales, dtype=torch.float32, device=self.device).reshape(count, 3)

        # uncertain_mask??certain/uncertain??loss? density policy?먯꽌 遺꾧린?섎뒗 ?듭떖 flag??
        if uncertain_mask is None:
            uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        else:
            uncertain_mask = torch.as_tensor(uncertain_mask, dtype=torch.bool, device=self.device).reshape(count)

        # surface_uv??uncertain Gaussian??NURBS surface???대뒓 parameter??臾띠??붿? ??ν븳??
        if surface_uv is None:
            surface_uv = torch.zeros((count, 2), dtype=torch.float32, device=self.device)
        else:
            surface_uv = torch.as_tensor(surface_uv, dtype=torch.float32, device=self.device).reshape(count, 2)

        # cluster_ids??color prior/ADC pattern transfer瑜??꾪븳 hook?대떎.
        if cluster_ids is None:
            cluster_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        else:
            cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device).reshape(count)

        # 湲곕낯 confidence??certain=1, uncertain=0.25濡??붾떎.
        if confidence is None:
            confidence = torch.where(
                uncertain_mask[:, None],
                torch.full((count, 1), 0.25, dtype=torch.float32, device=self.device),
                torch.ones((count, 1), dtype=torch.float32, device=self.device),
            )
        else:
            confidence = torch.as_tensor(confidence, dtype=torch.float32, device=self.device).reshape(count, 1)

        rest_dim = (self.max_sh_degree + 1) ** 2 - 1

        # ?숈뒿 ?덉젙?깆쓣 ?꾪빐 ?ㅼ젣 constrained 媛????unconstrained raw parameter瑜??붾떎.
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
        """Parameter group蹂?learning rate濡?Adam optimizer瑜?留뚮뱺??"""

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
        """湲곗〈 model ?ㅼ뿉 uncertain Gaussian??異붽??쒕떎.

        ?꾩옱 pipeline? rebuild ??initialize瑜?二쇰줈 ?곗?留? ?섏쨷??online surface
        sampling?대굹 ADC 湲곕컲 uncertain densification???ｌ쓣 ?????⑥닔媛 ?ъ슜?쒕떎.
        """

        torch = self.torch
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        surface_uv = torch.as_tensor(surface_uv, dtype=torch.float32, device=self.device)
        cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long, device=self.device)
        count = positions.shape[0]
        if count == 0:
            return
        # shape??諛붾뚮?濡??⑥닚?섍쾶 ?꾩껜 tensor瑜??ъ큹湲고솕?쒕떎.
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
        """keep_mask媛 False??Gaussian???쒓굅?쒕떎."""

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
        """Save Gaussians as a Graphdeco-style PLY for external renderers.

        The WebRenderer PLY loader requires `x`, `y`, `z`, `f_dc_0..2`, and
        raw `opacity`. It also understands raw log-scale `scale_0..2` and
        quaternion `rot_0..3`, so OSN-GS writes those names directly instead
        of the earlier debug-only RGB/scale_x fields.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self._xyz.detach().cpu()
        features_dc = self._features_dc.detach().cpu()[:, 0, :]
        opacity = self._opacity.detach().cpu()
        scales = self._scaling.detach().cpu()
        rotation = self.get_rotation.detach().cpu()
        confidence = self.get_confidence.detach().cpu()
        uncertain = self.is_uncertain.detach().cpu().to(self.torch.int32)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(self)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
            handle.write("property float opacity\n")
            handle.write("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            handle.write("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            handle.write("property int uncertain\n")
            handle.write("property float confidence\n")
            handle.write("end_header\n")
            for idx in range(len(self)):
                x, y, z = xyz[idx].tolist()
                f0, f1, f2 = features_dc[idx].tolist()
                op = float(opacity[idx, 0])
                s0, s1, s2 = scales[idx].tolist()
                r0, r1, r2, r3 = rotation[idx].tolist()
                flag = int(uncertain[idx])
                conf = float(confidence[idx, 0])
                handle.write(
                    f"{x} {y} {z} {f0} {f1} {f2} {op} "
                    f"{s0} {s1} {s2} {r0} {r1} {r2} {r3} {flag} {conf}\n"
                )
