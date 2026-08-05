"""Worklog 62: run the UNMODIFIED Graphdeco baseline's own train.py::training()
with a non-invasive runtime monkeypatch around GaussianModel.densify_and_prune/
reset_opacity/densify_and_clone/densify_and_split, logging the exact same
per-ADC-event stats the OSN-GS side already logs (gaussian count, clone/
split/prune counts, opacity distribution) -- for a first-divergence audit.

Does not modify any file under gaussian-splatting/ -- monkeypatches
GaussianModel methods at runtime only, in this process.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "gaussian-splatting"


def _load_baseline_train_module():
    sys.path.insert(0, str(BASELINE_ROOT))
    spec = importlib.util.spec_from_file_location("baseline_train", BASELINE_ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _log_state(model, tag: str, extra: str = "") -> None:
    with torch.no_grad():
        scale = model.get_scaling.detach().cpu().numpy()
        sorted_scale = np.sort(scale, axis=1)
        s_min = sorted_scale[:, 0]
        opacity = model.get_opacity.detach().reshape(-1).cpu().numpy()
        anisotropy = sorted_scale[:, 2] / np.clip(s_min, 1e-12, None)
    print(
        f"BASELINE-ADC-STATE tag={tag} n={model.get_xyz.shape[0]} "
        f"opacity_median={float(np.median(opacity)):.6g} opacity_below_0.005={int((opacity < 0.005).sum())} "
        f"s_min_median={float(np.median(s_min)):.6g} aniso_median={float(np.median(anisotropy)):.4g} "
        f"aniso_p99={float(np.percentile(anisotropy, 99)):.4g} {extra}",
        flush=True,
    )


def instrument(gaussian_model_cls) -> None:
    original_densify_and_prune = gaussian_model_cls.densify_and_prune
    original_reset_opacity = gaussian_model_cls.reset_opacity
    original_prune_points = gaussian_model_cls.prune_points
    original_densify_and_clone = gaussian_model_cls.densify_and_clone
    original_densify_and_split = gaussian_model_cls.densify_and_split

    def patched_densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_before = self.get_xyz.shape[0]
        original_densify_and_clone(self, grads, grad_threshold, scene_extent)
        cloned = int(self.get_xyz.shape[0]) - n_before
        self._last_cloned = cloned

    def patched_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_before = self.get_xyz.shape[0]
        original_densify_and_split(self, grads, grad_threshold, scene_extent, N)
        delta = int(self.get_xyz.shape[0]) - n_before  # net (children added - parents pruned)
        self._last_split_net = delta

    def patched_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        n_before = self.get_xyz.shape[0]
        self._last_cloned = 0
        self._last_split_net = 0
        prune_mask_before = (self.get_opacity < min_opacity).squeeze()
        opacity_prune_candidates = int(prune_mask_before.sum().item())
        original_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii)
        n_after = self.get_xyz.shape[0]
        print(
            f"BASELINE-ADC iteration={self._iteration_hint} n_before={n_before} n_after={n_after} "
            f"cloned~{self._last_cloned} split_net~{self._last_split_net} "
            f"opacity_prune_candidates_pre_grow={opacity_prune_candidates} "
            f"max_grad={max_grad} min_opacity={min_opacity} extent={extent} max_screen_size={max_screen_size}",
            flush=True,
        )
        _log_state(self, f"post_densify_and_prune@{self._iteration_hint}")

    def patched_reset_opacity(self):
        _log_state(self, f"pre_reset@{getattr(self, '_iteration_hint', '?')}")
        original_reset_opacity(self)
        _log_state(self, f"post_reset@{getattr(self, '_iteration_hint', '?')}")

    gaussian_model_cls.densify_and_prune = patched_densify_and_prune
    gaussian_model_cls.densify_and_clone = patched_densify_and_clone
    gaussian_model_cls.densify_and_split = patched_densify_and_split
    gaussian_model_cls.reset_opacity = patched_reset_opacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3600)
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--llffhold", type=int, default=8)  # accepted, baseline uses a fixed modulo internally
    args = parser.parse_args()

    module = _load_baseline_train_module()
    from scene.gaussian_model import GaussianModel
    instrument(GaussianModel)

    # Hook: stamp the current iteration onto the model so the patched methods
    # can label their log lines -- train.py's own loop doesn't pass iteration
    # into these methods, so we intercept it via add_densification_stats
    # (called every iteration while densifying) as a lightweight iteration
    # tracker.
    original_add_stats = GaussianModel.add_densification_stats
    counter = {"n": 0}

    def patched_add_stats(self, viewspace_point_tensor, update_filter):
        counter["n"] += 1
        if counter["n"] <= 3 or counter["n"] % 100 == 0:
            print(f"BASELINE-DEBUG add_densification_stats call #{counter['n']}", flush=True)
        original_add_stats(self, viewspace_point_tensor, update_filter)

    GaussianModel.add_densification_stats = patched_add_stats
    print(f"BASELINE-DEBUG patched densify_and_prune={GaussianModel.densify_and_prune.__name__}", flush=True)

    from argparse import ArgumentParser as BaselineArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.general_utils import safe_state

    baseline_parser = BaselineArgumentParser()
    lp = ModelParams(baseline_parser)
    op = OptimizationParams(baseline_parser)
    pp = PipelineParams(baseline_parser)
    baseline_parser.add_argument('--debug_from', type=int, default=-1)
    baseline_parser.add_argument('--detect_anomaly', action='store_true', default=False)
    baseline_parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--quiet", action="store_true")
    baseline_parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--start_checkpoint", type=str, default=None)

    argv = [
        "-s", args.source_path, "-m", args.model_path, "--eval",
        "--iterations", str(args.iterations),
        "--save_iterations", str(args.iterations),
    ]
    parsed = baseline_parser.parse_args(argv)
    # NOT safe_state(True) -- baseline's own quiet=True mode installs a
    # no-op stdout wrapper (utils/general_utils.py:safe_state) that silently
    # swallows every print() from this point on, including our own
    # instrumentation lines. safe_state(False) keeps the (seeded,
    # deterministic: random/numpy/torch all seed(0)) RNG setup but leaves
    # stdout untouched.
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    # Patch GaussianModel.__init__-adjacent iteration hint via training_report
    # call site is too invasive to hook cleanly; instead wrap the module's
    # own `training` iteration loop indirectly by monkeypatching `range` is
    # not feasible -- simplest robust hook: patch `add_densification_stats`
    # is called every iter, but iteration isn't passed either. Use a global
    # counter driven by the `training_report` call, which DOES receive
    # `iteration` every loop -- patch that instead.
    original_training_report = module.training_report

    def patched_training_report(tb_writer, iteration, *rest, **kwargs):
        # training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
        #                  testing_iterations, scene, render, renderArgs, train_test_exp)
        scene = rest[5] if len(rest) > 5 else kwargs.get("scene")
        if scene is not None:
            scene.gaussians._iteration_hint = iteration
        return original_training_report(tb_writer, iteration, *rest, **kwargs)

    module.training_report = patched_training_report

    dataset_args = lp.extract(parsed)
    opt_args = op.extract(parsed)
    print(
        f"BASELINE-DEBUG dataset.sh_degree={dataset_args.sh_degree} opt.iterations={opt_args.iterations} "
        f"opt.densify_from_iter={opt_args.densify_from_iter} opt.densification_interval={opt_args.densification_interval} "
        f"opt.densify_until_iter={opt_args.densify_until_iter}",
        flush=True,
    )
    try:
        module.training(
            dataset_args, opt_args, pp.extract(parsed),
            parsed.test_iterations, parsed.save_iterations, parsed.checkpoint_iterations,
            parsed.start_checkpoint, parsed.debug_from,
        )
    except Exception:
        import traceback
        print("BASELINE-DEBUG training() raised:", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
