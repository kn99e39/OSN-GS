from __future__ import annotations

"""Torch training losses for OSN-GS.

여기 있는 loss들은 "렌더링 품질"과 "NURBS 기반 구조 가설"을 동시에
최적화하기 위한 최소 구성이다. 추후 ablation을 쉽게 하기 위해 trainer에서
직접 수식을 쓰지 않고 함수로 분리했다.
"""

from typing import Any

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.utils.torch_ops import require_torch


_SSIM_WINDOW_CACHE: dict[Any, Any] = {}


def _ssim_window(window_size: int, sigma: float, channel: int, device: Any, dtype: Any) -> Any:
    """Cached separable Gaussian window matching the original 3DGS SSIM."""

    torch = require_torch()
    key = (window_size, sigma, channel, str(device), str(dtype))
    cached = _SSIM_WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = (gauss[:, None] @ gauss[None, :])[None, None]
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    _SSIM_WINDOW_CACHE[key] = window
    return window


def ssim(image: Any, target: Any, window_size: int = 11, sigma: float = 1.5) -> Any:
    """Structural similarity, matching the original 3DGS ``utils/loss_utils.ssim``.

    Accepts ``(C, H, W)`` or ``(N, C, H, W)`` and returns the mean SSIM.
    """

    torch = require_torch()
    functional = torch.nn.functional
    if image.dim() == 3:
        image, target = image[None], target[None]
    channel = int(image.shape[-3])
    window = _ssim_window(window_size, sigma, channel, image.device, image.dtype)
    pad = window_size // 2
    mu1 = functional.conv2d(image, window, padding=pad, groups=channel)
    mu2 = functional.conv2d(target, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = functional.conv2d(image * image, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(image * target, window, padding=pad, groups=channel) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def image_reconstruction_loss(
    image: Any,
    target: Any,
    lambda_dssim: float = 0.2,
) -> tuple[Any, Any]:
    """렌더 이미지와 GT 이미지의 reconstruction loss (original 3DGS와 동일 구성).

    ``(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)``. D-SSIM(구조 유사도)이
    3DGS 품질의 선명도를 좌우한다. MSE는 loss에 직접 쓰지 않고 PSNR 계산용으로만
    함께 반환한다.
    """

    l1 = (image - target).abs().mean()
    mse = (image - target).square().mean()
    dssim = 1.0 - ssim(image, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim, mse


def nurbs_surface_loss(
    state: TorchPipelineState,
    weight: float = 0.01,
    max_patches: int = 0,
) -> Any:
    """Fit persistent NURBS patches to the observed Gaussians (one-way).

    Direction matters: the visible (certain) Gaussians are the observation and
    the NURBS is the derived intermediate. This term therefore pulls the surface
    toward the Gaussians and never the reverse -- the Gaussian positions are
    detached, so visible Gaussians stay optimized by the image loss alone, as in
    baseline 3DGS. Surface geometry only supplies positions to *uncertain*
    Gaussians sampled on inferred/occluded regions (``uncertain_anchor_loss``).

    A zero budget retains the full-patch loss. A positive budget evaluates a
    deterministic rotating subset, so large multi-patch scenes stay surface-aware
    without a Python GPU synchronization for every patch on every iteration.
    """

    torch = require_torch()
    patches = state.surface_patches or [state.surface]
    patch_count = len(patches)
    if patch_count == 0:
        return torch.zeros((), dtype=torch.float32, device=state.model.device)

    budget = max(0, int(max_patches))
    if budget == 0 or budget >= patch_count:
        active_patch_ids = list(range(patch_count))
    else:
        start_patch = (int(state.iteration) * budget) % patch_count
        active_patch_ids = [
            (start_patch + offset) % patch_count for offset in range(budget)
        ]

    certain = ~state.model.is_uncertain
    indices = torch.nonzero(certain, as_tuple=False).reshape(-1)
    if int(indices.numel()) > 8192:
        sample = torch.linspace(
            0, indices.numel() - 1, steps=8192, device=indices.device
        ).long()
        indices = indices[sample]

    smoothness = torch.stack(
        [patches[patch_id].smoothness() for patch_id in active_patch_ids]
    ).mean()
    if int(indices.numel()) == 0:
        return weight * smoothness

    # Detached: the observed Gaussians are the fitting target for the surface,
    # so no gradient from this term may flow back into their positions.
    xyz = state.model.get_xyz[indices].detach()
    uv = state.model.surface_uv[indices]
    patch_ids = state.model.surface_patch_ids[indices]
    active_ids = torch.tensor(
        active_patch_ids, dtype=patch_ids.dtype, device=patch_ids.device
    )
    valid = (patch_ids >= 0) & (patch_ids < patch_count)
    active_mask = valid & torch.isin(patch_ids, active_ids)
    active_xyz = xyz[active_mask]
    active_uv = uv[active_mask]
    active_patch_ids_tensor = patch_ids[active_mask]
    anchors = torch.zeros_like(active_xyz)

    # Empty local groups are valid tensor operations. This avoids bool(mask.any())
    # and the per-patch CPU/GPU synchronization it would otherwise introduce.
    for patch_id in active_patch_ids:
        local_indices = torch.nonzero(
            active_patch_ids_tensor == patch_id, as_tuple=False
        ).reshape(-1)
        local_anchors = patches[patch_id].evaluate(active_uv[local_indices])
        anchors = anchors.index_copy(0, local_indices, local_anchors)

    active_scale = (
        state.model.get_scaling[indices][active_mask]
        .detach()
        .mean(dim=1)
        .clamp_min(1e-4)
    )
    squared_error = (active_xyz - anchors).square().sum(dim=1)
    fit = (squared_error / active_scale.square()).clamp_max(100.0).sum()
    fit = fit / active_mask.sum().to(dtype=fit.dtype).clamp_min(1.0)
    return weight * (fit + 0.1 * smoothness)


def uncertain_anchor_loss(state: TorchPipelineState, weight: float = 0.01) -> Any:
    """uncertain Gaussian이 자신이 샘플링된 surface anchor에서 멀어지는 것을 억제한다."""

    torch = require_torch()
    if not state.model.is_uncertain.any():
        return torch.zeros((), dtype=torch.float32, device=state.model.device)

    # uncertain Gaussian만 surface uv anchor를 갖는다.
    uncertain_xyz = state.model.get_xyz[state.model.is_uncertain]
    uv = state.model.surface_uv[state.model.is_uncertain]
    anchors = state.surface.evaluate(uv)

    # confidence가 낮을수록 surface prior를 더 강하게 적용한다.
    confidence = state.model.get_confidence[state.model.is_uncertain].detach()
    return weight * ((uncertain_xyz - anchors).square() * (1.0 - confidence)).mean()


def uncertain_confidence_loss(state: TorchPipelineState, residual_mse: Any, weight: float = 0.05) -> Any:
    """image residual을 uncertain confidence의 감독 신호로 사용한다.

    residual이 낮으면 surface hypothesis가 이미지와 충돌하지 않는다는 뜻이므로
    confidence target을 높이고, residual이 높으면 confidence를 낮춘다.
    """

    torch = require_torch()
    if not state.model.is_uncertain.any():
        return torch.zeros((), dtype=torch.float32, device=state.model.device)
    confidence = state.model.get_confidence[state.model.is_uncertain]

    # exp(-mse)는 0~1 범위의 부드러운 confidence target으로 쓰기 좋다.
    target_confidence = torch.exp(-residual_mse.detach()).clamp(0.0, 1.0)
    return weight * (confidence - target_confidence).square().mean()
