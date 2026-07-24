#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""프레임워크: 원본 Graphdeco 3DGS의 held-out test-camera 선택 및 해상도 결정 로직을
그대로 이식(vendor)한 모듈.

Why this exists (OSN-GS, not upstream): the OSN-GS vs baseline 3DGS quality
A/B (``TODO.md``'s top section) needs the SAME held-out test cameras and the
SAME effective training resolution on both sides, or the comparison is not
fair. Rather than re-derive this logic by hand (risking a subtle off-by-one
or rounding mismatch), the self-contained pieces of upstream logic that
decide this are copied verbatim from ``gaussian-splatting/scene/
dataset_readers.py`` (``readColmapSceneInfo``'s llffhold branch and
``getNerfppNorm``'s camera-based scene-extent radius) and
``gaussian-splatting/utils/camera_utils.py`` (``loadCam``'s resolution
branch), preserving the original control flow byte-for-byte so a future diff
against upstream stays meaningful. Only lightly repackaged into standalone
functions operating on plain values (image name list / width / height /
camera centers) instead of upstream's own ``CameraInfo``/``args`` objects,
since OSN-GS has its own COLMAP reader (``osn_gs/data/colmap_scene.py``) and
does not import upstream's scene-loading module wholesale.

Per ``AGENTS.md``'s vendoring rule ("avoid editing external reference
projects; vendor code into OSN-GS and point runtime paths only at OSN-GS"),
this file lives under OSN-GS's own tree and ``osn_gs/data/colmap_scene.py``
imports from here -- the ``gaussian-splatting/`` checkout itself is never
imported or modified at runtime.
"""

from __future__ import annotations

import os

from typing import Sequence


def estimate_camera_extent(camera_centers: Sequence[Sequence[float]]) -> float:
    """Verbatim port of ``getNerfppNorm``'s radius computation
    (``gaussian-splatting/scene/dataset_readers.py:50-71``).

    Given the world-space camera centers of a scene's TRAIN cameras, returns
    upstream's ``cameras_extent``: 1.1x the max distance from the mean camera
    center. Unlike a point-cloud-based scene extent, this is computed purely
    from (noise-free, bundle-adjusted) camera poses, so it needs no
    outlier-robust statistic the way a raw SfM point cloud does -- but it can
    diverge sharply from the reconstructed geometry's actual spread on
    walkthrough-style captures where observed content extends well past the
    camera path (see ``docs/worklogs/71_scene_extent_basis_mismatch_and_visible_blur_root_cause.md``).
    All of baseline 3DGS's own scale-sensitive constants (``percent_dense``,
    ``opacity_reset``'s implicit size assumptions, world-size prune ratio)
    are calibrated against this exact quantity, so any OSN-GS code path that
    reuses those constants unmodified must feed them this, not a
    point-cloud-based extent.
    """

    import numpy as np

    centers = np.asarray([np.asarray(c, dtype=np.float64).reshape(3) for c in camera_centers])
    if centers.shape[0] == 0:
        return 1.0
    center = centers.mean(axis=0)
    dist = np.linalg.norm(centers - center, axis=1)
    return float(dist.max()) * 1.1


def select_llff_holdout_test_names(
    image_names: list[str],
    scene_path: str | os.PathLike | None = None,
    eval: bool = True,
    llffhold: int = 8,
) -> list[str]:
    """Verbatim port of ``readColmapSceneInfo``'s test-camera selection
    (``gaussian-splatting/scene/dataset_readers.py:181-193``).

    Given the FULL list of COLMAP image names for the scene (unsorted is
    fine -- sorted internally, matching upstream), returns the subset of
    names upstream would mark ``is_test=True``. Every ``llffhold``-th name
    in SORTED order is held out -- index 0, ``llffhold``, ``2*llffhold``,
    ... -- so this must be called on ALL image names in the scene (not a
    pre-truncated subset), exactly as upstream does before any
    ``max_images``-style truncation.

    ``scene_path`` is only consulted for the ``llffhold=0`` fallback (reads
    ``sparse/0/test.txt``), matching upstream's own fallback path.
    """

    if not eval:
        return []

    # Upstream auto-overrides llffhold=8 for any path containing "360"
    # (their own heuristic for auto-detecting mip-NeRF-360-style datasets).
    # Kept verbatim for fidelity even though it rarely matters here.
    if scene_path is not None and "360" in str(scene_path):
        llffhold = 8

    if llffhold:
        cam_names = sorted(image_names)
        test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
    else:
        if scene_path is None:
            raise ValueError("scene_path is required when llffhold=0 (reads sparse/0/test.txt)")
        test_txt = os.path.join(str(scene_path), "sparse/0", "test.txt")
        with open(test_txt, "r") as file:
            test_cam_names_list = [line.strip() for line in file]
    return test_cam_names_list


def resolve_graphdeco_resolution(
    orig_w: int,
    orig_h: int,
    resolution: int = -1,
    resolution_scale: float = 1.0,
) -> tuple[int, int, float]:
    """Verbatim port of ``loadCam``'s resolution decision
    (``gaussian-splatting/utils/camera_utils.py:44-63``).

    Returns ``(width, height, downscale_factor)`` where ``downscale_factor``
    is upstream's ``scale`` (so callers needing just the factor, e.g. to
    resize a COLMAP intrinsic focal length consistently, do not have to
    re-derive it from the rounded output resolution). Default args
    (``resolution=-1``, ``resolution_scale=1.0``) reproduce upstream's own
    default behavior: auto-downscale to <=1.6K width, otherwise unchanged.
    """

    if resolution in (1, 2, 4, 8):
        width = round(orig_w / (resolution_scale * resolution))
        height = round(orig_h / (resolution_scale * resolution))
        # Upstream does not expose a single "scale" in this branch; derive
        # the equivalent factor for callers that need it (e.g. focal length
        # rescaling) -- mathematically identical to resolution_scale*resolution.
        downscale_factor = float(resolution_scale * resolution)
        return width, height, downscale_factor

    if resolution == -1:
        if orig_w > 1600:
            global_down = orig_w / 1600
        else:
            global_down = 1.0
    else:
        global_down = orig_w / resolution

    scale = float(global_down) * float(resolution_scale)
    width = int(orig_w / scale)
    height = int(orig_h / scale)
    return width, height, scale
