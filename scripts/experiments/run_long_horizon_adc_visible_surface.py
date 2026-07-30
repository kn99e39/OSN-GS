"""Read-only long-horizon ADC visible-surface maturation runner."""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.interop.colab_args import build_osn_gs_train_parser, output_dir_from_args, surface_fit_config_kwargs
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.utils.torch_ops import default_device

def counts(items): return dict(sorted(Counter(map(str, items)).items()))
def summary(values):
    values = sorted(float(x) for x in values)
    return {"mean": sum(values)/len(values), "median": values[len(values)//2], "max": values[-1]} if values else {"mean": 0., "median": 0., "max": 0.}

class MaturationTrainer(TorchOSNGSTrainer):
    def __init__(self, *args, snapshot_iterations: set[int], **kwargs):
        super().__init__(*args, **kwargs); self.snapshot_iterations = snapshot_iterations
    @staticmethod
    def _append_visible_nurbs_event(output_dir, event):
        row = dict(event); row.pop("canonical_representative_stable_ids", None)
        TorchOSNGSTrainer._append_visible_nurbs_event(output_dir, row)
    @staticmethod
    def _jsonl(path, row):
        row = dict(row); row.pop("canonical_representative_stable_ids", None)
        with path.open("a", encoding="utf-8", newline="\n") as f: f.write(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n")
    def _run_visible_nurbs_update(self, state, output_dir, **kwargs):
        event = super()._run_visible_nurbs_update(state, output_dir, **kwargs)
        construction = state.visible_surface_construction
        event.update(self.metrics(state, construction))
        self._jsonl(output_dir / "visible_surface_maturation.jsonl", event)
        if event["reason"] == "training_terminal" or int(event["iteration"]) in self.snapshot_iterations:
            self.snapshot(output_dir, state, construction, event)
        return event
    def metrics(self, state, c):
        model = state.model
        result = {"training_maturity": {"loss": float(state.last_loss), "psnr": float(state.last_psnr)}, "observed_gaussian_count": int((~model.is_uncertain).sum().item()), "uncertain_gaussian_count": int(model.is_uncertain.sum().item())}
        if c is None: return result
        r, regions, candidates, components, compatibility = c.reliability, c.surface_regions.regions, c.boundary_halfedge_candidates, c.ordered_boundary_components, c.boundary_compatibility
        degrees = Counter()
        for edge in compatibility:
            if edge.decision == "accepted": degrees[edge.source_half_edge_id] += 1; degrees[edge.target_half_edge_id] += 1
        result.update({
          "reliability": {"intrinsic": counts(r.intrinsic.intrinsic_class), "contextual": counts(r.contextual.contextual_class), "final": counts(r.reliability_class), "rejection_reasons": counts(reason for group in r.reasons for reason in group)},
          "region": {"pairwise_same_surface_edge_count": sum(x.manifold_relation == "same_surface" for x in c.manifold_affinity.edges), "region_count": len(regions), "size": summary(len(x.member_ids) for x in regions), "largest_region": max((len(x.member_ids) for x in regions), default=0), "core_members": sum(len(x.core_member_ids) for x in regions), "consensus_attached": sum(len(x.member_ids)-len(x.core_member_ids) for x in regions), "unresolved_members": len(c.surface_regions.unresolved_membership_ids)},
          "boundary_query": {"eligible_nodes": sum(len(x.member_ids) for x in regions), "reason_counts": counts(x.boundary_reason for x in candidates), "genuine_candidate_count": sum(x.boundary_reason == "observed_support_termination" for x in candidates)},
          "boundary_linking": {"compatibility_edge_count": len(compatibility), "accepted_edge_count": sum(x.decision == "accepted" for x in compatibility), "candidate_degree": counts(degrees.values()), "failure_stage": c.diagnostic_summary.get("boundary_failure_stage")},
          "components": counts(x.ordering_state for x in components),
          "materialization": {"attempt_count": len(c.materialization_attempts), "states": counts(x.state for x in c.materialization_attempts), "materialized_count": len(c.materialized_visible_nurbs_surfaces)},
        })
        return result
    def snapshot(self, output_dir, state, c, event):
        target = output_dir / "maturation_snapshots"; target.mkdir(parents=True, exist_ok=True)
        ids = list(event.get("canonical_representative_stable_ids", ()))
        by_id = {int(v): i for i, v in enumerate(state.model.stable_gaussian_ids.detach().cpu().tolist())}
        indices = [by_id[x] for x in ids if x in by_id]
        payload: dict[str, Any] = {"event": {k:v for k,v in event.items() if k != "canonical_representative_stable_ids"}, "representative_stable_ids": ids}
        if indices:
            ix = self.torch.tensor(indices, device=state.model.get_xyz.device, dtype=self.torch.long)
            payload["representatives"] = {"positions": state.model.get_xyz.detach()[ix].cpu().tolist(), "scales": state.model.get_scaling.detach()[ix].cpu().tolist(), "rotations": state.model.get_rotation.detach()[ix].cpu().tolist()}
        if c is not None:
            payload.update({"reliability_class": list(c.reliability.reliability_class), "region_membership": [{"region_id": x.region_id, "members": list(x.member_ids), "core_members": list(x.core_member_ids)} for x in c.surface_regions.regions], "accepted_topology": [list(x) for x in c.accepted_local_topology], "continuation_candidates": [x.__dict__ for x in c.boundary_halfedge_candidates], "components": [x.__dict__ for x in c.ordered_boundary_components], "materialization": [{"state": x.state, "input": x.input.__dict__} for x in c.materialization_attempts]})
        label = "terminal" if event["reason"] == "training_terminal" else f"iteration_{int(event['iteration']):06d}"
        (target / f"{label}.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")

def main():
    args = build_osn_gs_train_parser().parse_args()
    if not args.source_path: args.source_path = str(ROOT / "DATASET")
    if args.iterations == 1000: args.iterations = 15000
    args.visible_nurbs_update_schedule = "adc_post_commit"; args.surface_loss_patch_budget = 0
    device = args.device or default_device(prefer_cuda=True); image_device = args.image_device or ("auto" if device == "cuda" else device)
    if args.low_vram and not args.image_device: image_device = "cpu"
    density = TorchDensityControlConfig(densify_from_iter=max(0,args.densify_from_iter), densify_until_iter=max(0,args.densify_until_iter), densification_interval=max(0,args.densification_interval), densify_grad_threshold=float(args.densify_grad_threshold), max_gaussians=max(0,args.adc_max_gaussians), percent_dense=float(args.adc_percent_dense), prune_opacity_threshold=float(args.adc_prune_opacity_threshold), split_samples=max(1,args.adc_split_samples), max_screen_size=float(args.adc_max_screen_size), max_scale_ratio=float(args.adc_max_scale_ratio), opacity_reset_interval=max(0,args.opacity_reset_interval), screen_size_prune_from_iter=max(0,args.screen_size_prune_from_iter), preserve_adc_gradients=not args.adc_drop_survivor_gradients)
    pipeline = TorchPipelineConfig(canonical_covariance_knn=max(3,args.canonical_covariance_knn), canonical_construction_max_points=max(16,args.canonical_construction_max_points), covariance_knn_chunk_size=max(0,args.covariance_knn_chunk_size), covariance_min_scale=max(0.,args.covariance_min_scale), covariance_max_scale_ratio=max(0.,args.covariance_max_scale_ratio), covariance_scale_multiplier=max(0.,args.covariance_scale_multiplier), **surface_fit_config_kwargs(args))
    config = TorchTrainingConfig(iterations=args.iterations, surface_rebuild_interval=max(0,args.surface_update_interval), visible_nurbs_update_schedule="adc_post_commit", surface_loss_patch_budget=0, density_control_interval=args.density_control_interval, save_interval=0, save_iterations=tuple(sorted(set(x for x in args.save_iterations if x > 0))), progress_log_interval=max(1,args.progress_log_interval), timing_log_interval=max(0,args.timing_log_interval), write_output_files=not args.disable_output_files, resume_checkpoint=args.resume_checkpoint, prefer_cuda=device == "cuda", train_resolution_scale=max(1,args.train_resolution_scale), position_lr_extent_mode=args.position_lr_extent_mode, density_control=density)
    events = list(range(max(1,density.densify_from_iter), min(config.iterations,density.densify_until_iter)+1, max(1,density.densification_interval)))
    targets = {events[0], events[-1]} if events else set()
    for f in (.25,.50,.75): targets.add(min(events, key=lambda x: abs(x-(density.densify_from_iter+f*(density.densify_until_iter-density.densify_from_iter)))))
    out = output_dir_from_args(args); out.mkdir(parents=True, exist_ok=True)
    (out / "maturation_experiment_config.json").write_text(json.dumps({"iterations":config.iterations,"densify_from_iter":density.densify_from_iter,"densify_until_iter":density.densify_until_iter,"densification_interval":density.densification_interval,"snapshot_iterations":sorted(targets),"source_path":args.source_path,"max_images":args.max_images,"image_downscale":args.image_downscale,"train_resolution_scale":config.train_resolution_scale}, indent=2), encoding="utf-8")
    trainer = MaturationTrainer(pipeline_config=pipeline, training_config=config, rasterizer_config=GaussianRasterizerConfig(prefer_cuda=not args.disable_cuda_rasterizer), device=device, snapshot_iterations=targets)
    scene = load_colmap_scene(args.source_path, device=device, image_device=image_device, image_dir_name=args.images, sparse_dir_name=args.sparse_dir, image_downscale=args.image_downscale, max_images=args.max_images)
    trainer.train(scene, out)
if __name__ == "__main__": main()
