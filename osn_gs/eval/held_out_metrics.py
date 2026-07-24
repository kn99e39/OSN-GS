from __future__ import annotations

"""Held-out (test-camera) PSNR/SSIM evaluation.

Companion to ``osn_gs/data/vendor/graphdeco_scene_split.py``: once a scene is
loaded with ``load_colmap_scene_with_eval_split`` (train/test camera split
identical to the Graphdeco baseline's ``--eval --llffhold``), this module
renders the held-out test cameras with a trained model and scores them
against their ground-truth images -- the "동일 holdout camera의 PSNR/SSIM"
acceptance metric in ``TODO.md``'s baseline quality-gap A/B.

Deliberately NOT wired into ``TorchOSNGSTrainer.train`` itself: the trainer
only ever sees the train-only ``TorchScene``, so held-out images are never
sampled during training (as required). Evaluation happens as a separate
post-training pass in the CLI script, using the same trained
``TorchGaussianModel``/rasterizer the trainer already built.
"""

from typing import Any

from osn_gs.utils.torch_ops import psnr_from_mse, require_torch


def final_iteration_opacity_reset_applies(
    iteration: int,
    opacity_reset_interval: int,
    densify_until_iter: int,
) -> bool:
    """Whether the completed model was opacity-reset at ``iteration``.

    The reset happens inside the training loop before a post-training
    held-out evaluation can render the model. Keep this scheduling predicate
    pure so both CLI entry points can report the evaluation state without
    changing training or evaluation semantics.
    """

    return (
        int(opacity_reset_interval) > 0
        and int(iteration) > 0
        and int(iteration) < int(densify_until_iter)
        and int(iteration) % int(opacity_reset_interval) == 0
    )


def evaluate_held_out_cameras(
    rasterizer: Any,
    model: Any,
    test_cameras: list[Any],
    test_images: list[Any],
    device: str,
    background: Any | None = None,
) -> dict[str, Any]:
    """Render every held-out camera and return mean/per-camera PSNR/SSIM.

    ``test_cameras``/``test_images`` come straight from
    ``EvalSplitScene.test_cameras``/``test_images`` -- same convention as a
    ``TorchScene``'s own ``cameras``/``images`` (CPU-staged ``(3, H, W)``
    image tensors, one camera per image, same order).
    """

    torch = require_torch()
    # Lazy import, and osn_gs.core BEFORE osn_gs.losses: there is a
    # pre-existing circular import at package-init time
    # (osn_gs.losses.torch_losses imports TorchPipelineState from
    # osn_gs.core.torch_pipeline; osn_gs.core.__init__ imports the trainer,
    # which imports osn_gs.losses.torch_losses). Whichever package a process
    # imports FIRST resolves the cycle; production entrypoints always import
    # osn_gs.core first so this never surfaces there, but this module could
    # otherwise be the very first OSN-GS import in a process (e.g. a
    # standalone eval script or this file's own unit tests). Importing
    # osn_gs.core first here reproduces that same safe order defensively.
    import osn_gs.core  # noqa: F401
    from osn_gs.losses.torch_losses import ssim as compute_ssim

    if len(test_cameras) != len(test_images):
        raise ValueError(
            f"test_cameras ({len(test_cameras)}) and test_images ({len(test_images)}) must be the same length"
        )
    if background is None:
        background = torch.zeros((3,), dtype=torch.float32, device=device)

    per_camera: list[dict[str, Any]] = []
    with torch.no_grad():
        for camera, target in zip(test_cameras, test_images):
            render_pkg = rasterizer.render(camera, model, background)
            rendered = render_pkg["render"].to(device=device, dtype=torch.float32)
            ground_truth = target.to(device=device, dtype=torch.float32)
            mse = float(torch.nn.functional.mse_loss(rendered, ground_truth).detach().cpu())
            camera_psnr = psnr_from_mse(mse)
            camera_ssim = float(compute_ssim(rendered, ground_truth).detach().cpu())
            per_camera.append(
                {"image_name": camera.image_name, "psnr": camera_psnr, "ssim": camera_ssim, "mse": mse}
            )

    finite_psnr = [entry["psnr"] for entry in per_camera if entry["psnr"] != float("inf")]
    psnr_mean = sum(finite_psnr) / len(finite_psnr) if finite_psnr else float("inf")
    ssim_mean = sum(entry["ssim"] for entry in per_camera) / len(per_camera) if per_camera else float("nan")
    return {
        "camera_count": len(per_camera),
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
        "per_camera": per_camera,
    }
