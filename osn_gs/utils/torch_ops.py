from __future__ import annotations

"""Torch 관련 작은 유틸리티.

PyTorch import를 파일 top-level에서 하지 않고 `require_torch()`로 지연시키면,
문서 빌드나 AST 문법 검사처럼 torch가 없어도 되는 상황에서 코드베이스를
읽기 쉬워진다.
"""

import math
from typing import Any


def require_torch() -> Any:
    """PyTorch를 lazy import하고, 없으면 OSN-GS 관점의 에러 메시지를 낸다."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("OSN-GS torch training requires PyTorch.") from exc
    return torch


def default_device(prefer_cuda: bool = True) -> str:
    """CUDA가 가능하면 cuda, 아니면 cpu를 반환한다."""

    torch = require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def inverse_sigmoid(x: Any, eps: float = 1e-6) -> Any:
    """[0, 1] 값을 logit domain으로 옮긴다."""

    torch = require_torch()
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def quaternion_identity(count: int, device: str) -> Any:
    """3DGS rotation 초기값으로 쓰는 identity quaternion batch."""

    torch = require_torch()
    rotation = torch.zeros((count, 4), dtype=torch.float32, device=device)
    rotation[:, 0] = 1.0
    return rotation


def rgb_to_sh_dc(rgb: Any) -> Any:
    """RGB를 SH DC coefficient로 근사 변환한다."""

    return (rgb - 0.5) / 0.28209479177387814


def sh_dc_to_rgb(dc: Any) -> Any:
    """SH DC coefficient를 RGB로 되돌린다."""

    return dc * 0.28209479177387814 + 0.5


def psnr_from_mse(mse: float) -> float:
    """MSE scalar에서 PSNR을 계산한다."""

    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)
