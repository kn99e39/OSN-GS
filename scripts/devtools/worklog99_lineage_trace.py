"""Worklog 100 section 5 -- trace the Worklog 99 giant-region lineage.

Uses Worklog 99's own merge_provenance graph to find the merge chain that
FIRST connects a patio-side seed surfel to a hedge/background-side seed
surfel, then recomputes the region-conditioned BILATERAL statistics for
those exact same interfaces (at the membership state they had in Worklog 99
at that point), using the SAME evidence primitives the new Worklog 100
algorithm uses (`_region_conditioned_bilateral_residuals`). Answers:

    - Were those old WL99 merges supported primarily from only one side?
    - Does the bilateral certificate now stop that connection?
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import load_primitive_model, checkpoint_primitive, PRIMITIVE_SURFEL_2D  # noqa: E402
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, build_candidate_graph
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig, partition_surfels_region_coherent
from osn_gs.surface.torch_interface_coherent_region_merge import InterfaceCoherentMergeConfig, partition_surfels_interface_coherent
from osn_gs.surface.torch_bilateral_interface_region_merge import (
    BilateralInterfaceMergeConfig,
    _compute_initial_residual_threshold,
    _region_conditioned_bilateral_residuals,
)
from osn_gs.surface.torch_discontinuity_first_surfel_partition import _knn, _auto_chunk_size
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel


def _progress(message: str) -> None:
    print(f"[wl99 lineage trace] {message}", flush=True)


def main() -> None:
    checkpoint = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000"
    _progress(f"loading checkpoint {checkpoint}")
    model, payload = load_primitive_model(checkpoint, device="cuda")
    primitive = checkpoint_primitive(payload)
    assert primitive == PRIMITIVE_SURFEL_2D and int(getattr(model, "scale_dim", 3)) == 2

    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)

    with torch.no_grad():
        from dataclasses import replace as _dc_replace

        full_orientation = derive_surface_orientation_from_surfel(model)
        orientation = _dc_replace(
            full_orientation,
            gaussian_ids=full_orientation.gaussian_ids[visible_selector],
            positions=full_orientation.positions[visible_selector],
            tangent_axis_u=full_orientation.tangent_axis_u[visible_selector],
            tangent_axis_v=full_orientation.tangent_axis_v[visible_selector],
            surface_normal=full_orientation.surface_normal[visible_selector],
            tangent_scale_u=full_orientation.tangent_scale_u[visible_selector],
            tangent_scale_v=full_orientation.tangent_scale_v[visible_selector],
        )
        positions = orientation.positions

        local_config = CoverageFirstPartitionConfig()
        region_config = RegionCoherenceConfig(local=local_config, require_positional_continuity=True)

        _progress("replaying Worklog 99 to get merge_provenance + initial regions")
        config_wl99 = InterfaceCoherentMergeConfig(local=local_config, region=region_config)
        partition_wl99 = partition_surfels_interface_coherent(orientation, config_wl99, progress=_progress)
        initial_region_ids = partition_wl99.initial_region_ids
        initial_region_count = partition_wl99.initial_region_count
        provenance = list(partition_wl99.merge_provenance)
        _progress(f"WL99: initial_region_count={initial_region_count} merges={len(provenance)}")

        # --- pick a patio-side seed and a hedge-side seed by POSITION -------
        # The scene camera looks roughly along -Z with the patio as the
        # near/low surface and the hedge/wall as the tall background; we
        # identify seeds by extreme Y (up axis in this dataset's convention)
        # among surfels that ended up in Worklog 99's largest final subset,
        # which is exactly the component under investigation.
        largest_subset_id = 0  # subset_ids are ordered descending by size
        members = torch.nonzero(partition_wl99.subset_ids == largest_subset_id, as_tuple=False).reshape(-1)
        member_positions = positions[members]
        y = member_positions[:, 1]
        patio_seed = members[torch.argmax(y)].item()  # convention-dependent; verified by printing coordinates below
        hedge_seed = members[torch.argmin(y)].item()
        _progress(
            f"seed candidates: node {patio_seed} pos={positions[patio_seed].tolist()}, "
            f"node {hedge_seed} pos={positions[hedge_seed].tolist()}"
        )

        patio_region = int(initial_region_ids[patio_seed].item())
        hedge_region = int(initial_region_ids[hedge_seed].item())
        _progress(f"seed initial regions: patio_region={patio_region} hedge_region={hedge_region}")

        # --- BFS over the WL99 provenance graph to find the connecting chain
        adjacency: dict[int, list[tuple[int, int]]] = {}  # region -> [(neighbor_region, provenance_index)]
        for index, record in enumerate(provenance):
            a, b = record["region_a"], record["region_b"]
            adjacency.setdefault(a, []).append((b, index))
            adjacency.setdefault(b, []).append((a, index))

        visited = {patio_region: None}
        queue = deque([patio_region])
        found = patio_region == hedge_region
        while queue and not found:
            current = queue.popleft()
            for neighbor, prov_index in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                visited[neighbor] = (current, prov_index)
                if neighbor == hedge_region:
                    found = True
                    break
                queue.append(neighbor)

        if not found:
            _progress("patio_region and hedge_region are NOT connected via WL99's own merge_provenance graph.")
            path_records = []
        else:
            # walk back from hedge_region to patio_region
            chain = []
            node = hedge_region
            while node != patio_region:
                parent, prov_index = visited[node]
                chain.append(prov_index)
                node = parent
            chain.reverse()
            path_records = [provenance[index] for index in chain]
            _progress(f"lineage chain length: {len(path_records)} merges")
            for record in path_records:
                _progress(
                    f"  round={record['round']} region_a={record['region_a']} region_b={record['region_b']} "
                    f"fraction_smooth_continuation={record['fraction_smooth_continuation']:.3f} "
                    f"edge_count={record['edge_count']}"
                )

        # --- for each lineage interface, recompute BILATERAL statistics ----
        # at the membership state WL99 had immediately BEFORE that merge was
        # applied (replay WL99's own recorded merges up to but not including
        # this one), using the shared graph and the NEW evidence primitives.
        graph = build_candidate_graph(orientation, local_config, progress=_progress)
        bilateral_config = BilateralInterfaceMergeConfig(local=local_config, region=region_config)
        k_shape = min(bilateral_config.resolved_shape_operator_neighbor_count(), positions.shape[0] - 1)
        chunk_size = _auto_chunk_size(positions.shape[0], positions.device)
        full_neighbor_index, _ = _knn(positions, k_shape, chunk_size, _progress)
        residual_threshold = _compute_initial_residual_threshold(
            orientation, full_neighbor_index, initial_region_ids, graph, bilateral_config
        )
        _progress(f"global bilateral residual threshold: {residual_threshold:.6f}")

        class _DSU:
            def __init__(self, n: int) -> None:
                self.parent = list(range(n))

            def find(self, x: int) -> int:
                root = x
                while self.parent[root] != root:
                    root = self.parent[root]
                while self.parent[x] != root:
                    self.parent[x], x = root, self.parent[x]
                return root

            def union(self, a: int, b: int) -> None:
                ra, rb = self.find(a), self.find(b)
                if ra == rb:
                    return
                if ra > rb:
                    ra, rb = rb, ra
                self.parent[rb] = ra

        # Replay the FULL Worklog 99 provenance sequence (all 5572 merges, in
        # their original order) rather than just the lineage-chain subset:
        # a provenance record's `region_a`/`region_b` are DSU ROOTS at the
        # moment of that merge, which by then may already have absorbed many
        # OTHER regions from earlier/same-round merges NOT on the lineage
        # path. Snapshotting membership from only the chain's own 5 merges
        # left region "0" artificially tiny (its true round-3 membership
        # included ~1100+ other regions absorbed in rounds 1-2) and produced
        # zero surviving edges for every step but the first -- a real bug,
        # caught by inspecting the trace output before trusting it.
        dsu = _DSU(initial_region_count)
        chain_index_set = set(chain) if found else set()
        chain_step_of_prov_index = {prov_index: step for step, prov_index in enumerate(chain)} if found else {}

        left_all, right_all = graph.candidate_edges[:, 0], graph.candidate_edges[:, 1]
        spatial_mask = graph.spatial_edge_mask

        lineage_report: list[dict] = [None] * len(path_records)  # type: ignore[list-item]
        for prov_index, record in enumerate(provenance):
            region_a, region_b = record["region_a"], record["region_b"]
            if prov_index not in chain_index_set:
                dsu.union(region_a, region_b)
                continue

            step = chain_step_of_prov_index[prov_index]
            current_root = torch.tensor(
                [dsu.find(r) for r in range(initial_region_count)], dtype=torch.int64, device=positions.device
            )
            node_root = current_root[initial_region_ids]
            root_a = current_root[region_a]
            root_b = current_root[region_b]
            edge_root_left = node_root[left_all]
            edge_root_right = node_root[right_all]
            this_interface_mask = spatial_mask & (
                ((edge_root_left == root_a) & (edge_root_right == root_b))
                | ((edge_root_left == root_b) & (edge_root_right == root_a))
            )
            edge_left = left_all[this_interface_mask]
            edge_right = right_all[this_interface_mask]
            if int(edge_left.shape[0]) == 0:
                lineage_report[step] = {"step": step, "record": record, "note": "no surviving edges to re-evaluate"}
                dsu.union(region_a, region_b)
                continue

            evidence = _region_conditioned_bilateral_residuals(
                orientation, full_neighbor_index, node_root, edge_left, edge_right, bilateral_config
            )
            left_is_a = node_root[edge_left] == root_a
            smooth_a_to_b = torch.where(
                left_is_a,
                (~evidence["unsupported_left"]) & (evidence["r_left_own_model"] <= residual_threshold),
                (~evidence["unsupported_right"]) & (evidence["r_right_own_model"] <= residual_threshold),
            )
            smooth_b_to_a = torch.where(
                left_is_a,
                (~evidence["unsupported_right"]) & (evidence["r_right_own_model"] <= residual_threshold),
                (~evidence["unsupported_left"]) & (evidence["r_left_own_model"] <= residual_threshold),
            )
            positional_ok = evidence["normal_offset_ratio"] <= bilateral_config.parallel_sheet_normal_over_tangent_ratio
            bilateral_smooth = smooth_a_to_b & smooth_b_to_a & positional_ok

            step_report = {
                "step": step,
                "wl99_record": record,
                "edge_count": int(edge_left.shape[0]),
                "smooth_a_to_b_fraction": float(smooth_a_to_b.float().mean().item()),
                "smooth_b_to_a_fraction": float(smooth_b_to_a.float().mean().item()),
                "bilateral_smooth_fraction": float(bilateral_smooth.float().mean().item()),
                "one_sided_only": bool(
                    (float(smooth_a_to_b.float().mean().item()) >= 0.5) != (float(smooth_b_to_a.float().mean().item()) >= 0.5)
                ),
                "would_bilateral_certificate_accept": bool(bilateral_smooth.float().mean().item() >= 0.5),
            }
            lineage_report[step] = step_report
            _progress(
                f"step {step}: region_a={region_a} region_b={region_b} edges={step_report['edge_count']} "
                f"smooth_a_to_b={step_report['smooth_a_to_b_fraction']:.3f} "
                f"smooth_b_to_a={step_report['smooth_b_to_a_fraction']:.3f} "
                f"bilateral={step_report['bilateral_smooth_fraction']:.3f} "
                f"one_sided_only={step_report['one_sided_only']} "
                f"would_accept={step_report['would_bilateral_certificate_accept']}"
            )
            dsu.union(region_a, region_b)

    output_path = REPO_ROOT / "output/osn_gs_bilateral_interface_region_merge/worklog99_lineage_trace.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "patio_seed": patio_seed,
                "hedge_seed": hedge_seed,
                "patio_seed_position": positions[patio_seed].tolist(),
                "hedge_seed_position": positions[hedge_seed].tolist(),
                "patio_initial_region": patio_region,
                "hedge_initial_region": hedge_region,
                "connected_in_wl99": found,
                "lineage_chain_length": len(path_records),
                "lineage_steps": lineage_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _progress(f"report -> {output_path}")


if __name__ == "__main__":
    main()
