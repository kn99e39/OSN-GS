from __future__ import annotations

"""Numpy prototype OSN-GS pipeline.

Torch 구현 전에 알고리즘 골격을 빠르게 확인하기 위해 만든 경로다.
현재 실험/결과 생성은 `torch_pipeline.py`가 담당하지만, 이 파일은
아이디어를 작은 배열 연산으로 이해하는 참고 구현으로 남겨둔다.
"""

from dataclasses import dataclass

import numpy as np

from osn_gs.core.state import OSNGSState
from osn_gs.gaussian.color_clusters import colors_for_cluster_ids, fit_color_clusters
from osn_gs.gaussian.certain_gaussians import CertainGaussianSet
from osn_gs.gaussian.uncertain_gaussians import UncertainGaussianSet
from osn_gs.surface.base_curves import fit_base_curves
from osn_gs.surface.nurbs_surface import build_surface_from_curves
from osn_gs.surface.occlusion_curves import predict_occlusion_curves
from osn_gs.surface.point_cloud import from_certain_gaussians
from osn_gs.surface.sampling import sample_occluded_surface
from osn_gs.utils.geometry import pairwise_distances


@dataclass
class PipelineConfig:
    """Prototype surface/uncertain Gaussian 생성 설정."""

    base_curve_count: int = 4
    occlusion_offset_scale: float = 0.25
    uncertain_samples_u: int = 8
    uncertain_samples_v: int = 2
    color_cluster_count: int = 4
    uncertain_opacity: float = 0.2
    uncertain_scale: float = 0.02


class OSNGSPipeline:
    """certain Gaussian에서 NURBS surface와 uncertain Gaussian을 만드는 numpy prototype."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def initialize(self, certain_gaussians: CertainGaussianSet) -> OSNGSState:
        # certain Gaussian center를 관측 surface point cloud로 변환한다.
        point_cloud = from_certain_gaussians(certain_gaussians)
        # point cloud에서 base curve를 fitting한다.
        base_curves = fit_base_curves(point_cloud, curve_count=self.config.base_curve_count)
        # base curve를 occluded 영역으로 외삽한다.
        occlusion_curves = predict_occlusion_curves(
            base_curves,
            offset_scale=self.config.occlusion_offset_scale,
        )
        # base/occlusion curve로 NURBS-like surface를 만든다.
        nurbs_surface = build_surface_from_curves(base_curves, occlusion_curves)
        # surface 위에 uncertain Gaussian을 배치한다.
        uncertain_gaussians = self._sample_uncertain(certain_gaussians, nurbs_surface)
        return OSNGSState(
            certain_gaussians=certain_gaussians,
            uncertain_gaussians=uncertain_gaussians,
            base_curves=base_curves,
            occlusion_curves=occlusion_curves,
            nurbs_surface=nurbs_surface,
        )

    def rebuild_surface(self, state: OSNGSState) -> None:
        # 현재 certain Gaussian만 사용해 surface hypothesis를 다시 만든다.
        point_cloud = from_certain_gaussians(state.certain_gaussians)
        state.base_curves = fit_base_curves(point_cloud, curve_count=self.config.base_curve_count)
        state.occlusion_curves = predict_occlusion_curves(
            state.base_curves,
            offset_scale=self.config.occlusion_offset_scale,
        )
        state.nurbs_surface = build_surface_from_curves(state.base_curves, state.occlusion_curves)
        state.uncertain_gaussians = self._sample_uncertain(state.certain_gaussians, state.nurbs_surface)

    def _sample_uncertain(self, certain_gaussians: CertainGaussianSet, nurbs_surface) -> UncertainGaussianSet:
        # NURBS occluded 영역에서 Gaussian center 후보를 샘플링한다.
        positions, uv = sample_occluded_surface(
            nurbs_surface,
            samples_u=self.config.uncertain_samples_u,
            samples_v=self.config.uncertain_samples_v,
        )
        # certain 색상 cluster를 만들고, uncertain에 nearest cluster 색상을 할당한다.
        clusters = fit_color_clusters(certain_gaussians.colors, k=self.config.color_cluster_count)
        nearest = pairwise_distances(positions, certain_gaussians.positions).argmin(axis=1)
        cluster_ids = clusters.assignments[nearest]
        colors = colors_for_cluster_ids(clusters, cluster_ids)
        n = len(positions)
        return UncertainGaussianSet(
            positions=positions,
            colors=colors,
            opacities=np.full(n, self.config.uncertain_opacity, dtype=np.float32),
            scales=np.full((n, 3), self.config.uncertain_scale, dtype=np.float32),
            confidence=np.full(n, 0.25, dtype=np.float32),
            surface_uv=uv,
            cluster_ids=cluster_ids,
        )
