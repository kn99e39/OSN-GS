"""Construction-stage metrics and JSON export for the synthetic benchmark."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json
import torch
from osn_gs.surface.torch_nurbs import TorchNURBSSurface

def _num(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if is_dataclass(x):
        return {k: _num(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    return x

def _grid_metrics(grid: torch.Tensor) -> dict[str, float]:
    flat = grid.reshape(-1, 3)
    extent = (flat.amax(0) - flat.amin(0)).norm()
    edges = []
    if grid.shape[0] > 1: edges.append((grid[1:] - grid[:-1]).norm(dim=-1).reshape(-1))
    if grid.shape[1] > 1: edges.append((grid[:, 1:] - grid[:, :-1]).norm(dim=-1).reshape(-1))
    edge = torch.cat(edges) if edges else torch.zeros(1, device=grid.device)
    return {"extent": float(extent.cpu()), "edge_median": float(edge.median().cpu()), "edge_min": float(edge.min().cpu())}

def _uv_support(uv: torch.Tensor, bins: int = 32) -> dict[str, float]:
    cell = torch.clamp((uv * bins).long(), 0, bins - 1)
    occupied = torch.unique(cell[:, 0] * bins + cell[:, 1]).numel()
    return {"point_count": int(uv.shape[0]), "occupied_cells": int(occupied), "coverage_ratio": float(occupied / (bins * bins))}

def _mapping_metrics(surface: TorchNURBSSurface, resolution: int = 24) -> dict[str, float]:
    t = torch.linspace(0., 1., resolution, device=surface.control_grid.device)
    u, v = torch.meshgrid(t, t, indexing="ij")
    _, du, dv = surface.evaluate_with_derivatives(torch.stack((u.flatten(), v.flatten()), 1))
    jac = torch.cross(du, dv, dim=1).norm(dim=1)
    median = jac.median().clamp_min(1e-12)
    normal = torch.nn.functional.normalize(torch.cross(du, dv, dim=1), dim=1, eps=1e-12)
    reference = torch.nn.functional.normalize(normal.sum(0), dim=0, eps=1e-12)
    return {"jacobian_median": float(median.cpu()), "jacobian_min": float(jac.min().cpu()), "degenerate_fraction": float((jac <= median * 1e-3).float().mean().cpu()), "fold_fraction": float((normal @ reference < -0.1).float().mean().cpu())}

def summarize(state: Any) -> list[dict[str, Any]]:
    out = []
    for patch_id, (patch, diag) in enumerate(zip(state.surface_patches, state.surface_fit_diagnostics)):
        point = diag.fitting_points.to(patch.control_grid.device)
        stages = [{"stage": "idw_seed", "grid": _grid_metrics(diag.idw_seed_control_grid), "uv_support": _uv_support(diag.initial_uv), "mapping": _mapping_metrics(TorchNURBSSurface(diag.idw_seed_control_grid, torch.ones_like(diag.idw_seed_control_grid[...,0]), patch.degree_u, patch.degree_v))}]
        for index, round_diag in enumerate(diag.rounds, 1):
            stage_surface = TorchNURBSSurface(round_diag.control_grid_after_lsq.to(patch.control_grid.device), torch.ones_like(patch.weights), patch.degree_u, patch.degree_v)
            residual = (stage_surface.evaluate(round_diag.uv_after_footpoint.to(patch.control_grid.device)) - point).norm(dim=1)
            stages.append({"stage": f"round_{index}", "grid": _grid_metrics(round_diag.control_grid_after_lsq), "uv_support": _uv_support(round_diag.uv_after_footpoint), "fit_rms": float(residual.square().mean().sqrt().cpu()), "mapping": _mapping_metrics(stage_surface)})
        out.append({"patch_id": patch_id, "fit_point_count": int(point.shape[0]), "point_weights": _num(diag.point_weights), "initial_uv": _num(diag.initial_uv), "idw_seed_control_grid": _num(diag.idw_seed_control_grid), "rounds": _num(diag.rounds), "final_control_grid": _num(diag.final_control_grid), "final_weights": _num(diag.final_weights), "final_gaussian_indices": _num(diag.final_gaussian_indices), "final_gaussian_uv": _num(diag.final_gaussian_uv), "uv_support_mask": _num(getattr(patch, "uv_support_mask", None)), "stages": stages, "final_mapping": _mapping_metrics(patch)})
    return out

def export(state: Any, path: Path) -> list[dict[str, Any]]:
    result = summarize(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"patches": result}, indent=2), encoding="utf-8")
    path.with_name("uv_support.json").write_text(json.dumps({"patches": result}, indent=2), encoding="utf-8")
    panels = []
    for item in result:
        uv = item["initial_uv"]
        dots = "".join(f'<circle cx="{20 + 180 * q[0]:.1f}" cy="{220 - 180 * q[1]:.1f}" r="1.5"/>' for q in uv)
        labels = " | ".join(f'{stage["stage"]}: coverage={stage["uv_support"]["coverage_ratio"]:.3f}, degenerate={stage["mapping"]["degenerate_fraction"]:.3f}, fold={stage["mapping"]["fold_fraction"]:.3f}' for stage in item["stages"])
        panels.append(f'<g transform="translate({item["patch_id"] * 240},0)"><rect x="20" y="20" width="180" height="200" fill="none" stroke="#777"/><g fill="#24b7d9">{dots}</g><text x="20" y="245">patch {item["patch_id"]}: fitting UV support</text><foreignObject x="20" y="252" width="220" height="90"><div xmlns="http://www.w3.org/1999/xhtml" style="font:10px sans-serif;white-space:normal">{labels}</div></foreignObject></g>')
    width = max(280, 240 * len(result))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="350" viewBox="0 0 {width} 350"><rect width="100%" height="100%" fill="white"/>' + "".join(panels) + "</svg>"
    path.with_name("uv_support.svg").write_text(svg, encoding="utf-8")
    return result
